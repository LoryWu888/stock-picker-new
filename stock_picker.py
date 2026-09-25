#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
收盘后候选池选股器（务实版）
- 盘后运行，生成次日观察名单（单文件HTML）
- 数据源：akshare 代码列表 + 腾讯批量行情 + 腾讯日K历史
- 输出：单文件自包含HTML，浏览器本地打开即可查看
"""
import sys, os, json, math, warnings, time, re
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings('ignore')

try:
    import akshare as ak
    import pandas as pd
    import numpy as np
    import requests
except ImportError as e:
    print(f"[ERROR] 缺少依赖: {e}")
    print("请运行: python -m pip install akshare pandas numpy requests")
    sys.exit(1)

# ============================================================
# 配置常量
# ============================================================
TOP_N = 50
HIST_DAYS = 320
MIN_LIST_DAYS = 120
MIN_AMT = 50_000_000  # 当日最低成交额 5000万
MAX_PRICE = 500.0
VOL_MA_DAYS = 5
BATCH_SIZE = 600       # 腾讯批量接口每次请求数量
MAX_WORKERS = 8       # 历史K线并行线程数

# 大盘择时配置
MT_RET3_THRESHOLD = -1.5   # 条件A：近3日累计跌幅阈值(%)
MT_DROP2_THRESHOLD = -1.0  # 条件D：近2日单日跌幅阈值(%)
MT_ATR_PANIC = 0.8         # 条件F：ATR/收盘价恐慌阈值(%)
MT_LOOKBACK = 30           # 择时数据拉取天数（需≥20以支持布林带计算）

# ============================================================
# 工具函数
# ============================================================
def _safe_div(a, b, default=0.0):
    return a / b if b and b != 0 else default

def ema(s, span):
    return s.ewm(span=span, adjust=False).mean()

def macd(close):
    ema12 = ema(close, 12)
    ema26 = ema(close, 26)
    dif = ema12 - ema26
    dea = ema(dif, 9)
    macd_bar = (dif - dea) * 2
    return dif, dea, macd_bar

def rsi(close, window=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def bollinger(close, window=20, std=2):
    ma = close.rolling(window=window).mean()
    sigma = close.rolling(window=window).std()
    return ma, ma - std * sigma, ma + std * sigma

def atr(high, low, close, window=14):
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()

def compute_kdj(high, low, close, n=9, m1=3, m2=3):
    low_n = low.rolling(window=n).min()
    high_n = high.rolling(window=n).max()
    rsv = 100 * (close - low_n) / (high_n - low_n).replace(0, np.nan)
    k = rsv.ewm(alpha=1/m1, adjust=False).mean()
    d = k.ewm(alpha=1/m2, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j

# ============================================================
# 数据获取：腾讯批量行情
# ============================================================
def fetch_tencent_batch(codes):
    """
    codes: list of str like ['sh600519','sz000001',...]
    返回 dict: code -> {name, price, change_pct, open, high, low, prev_close, volume, amount, turnover}
    """
    url = f"https://qt.gtimg.cn/q={','.join(codes)}"
    try:
        r = requests.get(url, timeout=20)
        text = r.text.strip()
        if not text:
            return {}
        result = {}
        for line in text.split(';'):
            line = line.strip()
            if not line or '="' not in line:
                continue
            parts = line.split('="')
            if len(parts) < 2:
                continue
            code_key = parts[0].strip()
            if not code_key.startswith('v_'):
                continue
            code = code_key[2:]
            data_str = parts[1].rstrip('"')
            fields = data_str.split('~')
            if len(fields) < 45:
                continue
            try:
                result[code] = {
                    'name': fields[1],
                    'price': float(fields[3]) if fields[3] else 0,
                    'prev_close': float(fields[4]) if fields[4] else 0,
                    'open': float(fields[5]) if fields[5] else 0,
                    'volume': int(fields[6]) if fields[6] else 0,  # 手
                    'amount': float(fields[37]) if fields[37] else 0,  # 万元
                    'high': float(fields[33]) if fields[33] else 0,
                    'low': float(fields[34]) if fields[34] else 0,
                    'change_pct': float(fields[32]) if fields[32] else 0,
                    'turnover': float(fields[38]) if len(fields) > 38 and fields[38] else 0,
                    'pe': float(fields[39]) if len(fields) > 39 and fields[39] else 0,
                    'pb': float(fields[46]) if len(fields) > 46 and fields[46] else 0,
                }
            except (ValueError, IndexError):
                continue
        return result
    except Exception as e:
        print(f"[WARN] 腾讯批量接口失败: {e}")
        return {}

# ============================================================
# 数据获取：腾讯日K历史
# ============================================================
def fetch_tencent_kline(code, days=HIST_DAYS):
    """
    返回 DataFrame [date, open, close, high, low, volume]
    """
    try:
        url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={code},day,,,{days},qfq"
        r = requests.get(url, timeout=15)
        data = r.json()
        if data.get('code', -1) != 0:
            return None
        key = code
        raw = data['data'][key].get('qfqday', [])
        if not raw or len(raw) < 30:
            return None
        df = pd.DataFrame(raw, columns=['date','open','close','high','low','volume'])
        for c in ['open','close','high','low']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        df = df.dropna()
        if len(df) < 30:
            return None
        return df
    except Exception as e:
        return None

# ============================================================
# 板块强度（新浪行业，akshare可用）
# ============================================================
def get_board_strength():
    try:
        df = ak.stock_sector_spot()
        if '板块' in df.columns and '涨跌幅' in df.columns:
            df = df[['板块', '涨跌幅']].dropna()
            return df.set_index('板块')['涨跌幅'].to_dict()
    except Exception as e:
        print(f"[WARN] 板块数据获取失败: {e}")
    return {}

# ============================================================
# 大盘指数日K线拉取（用于计算相对强度）
# ============================================================
def fetch_index_kline(symbol='sh000001', days=10):
    try:
        url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={symbol},day,,,{days},qfq"
        r = requests.get(url, timeout=10)
        data = r.json()
        if data.get('code', -1) != 0:
            return None
        raw = data['data'][symbol].get('qfqday', [])
        if not raw:
            raw = data['data'][symbol].get('day', [])
        if not raw:
            return None
        df = pd.DataFrame(raw, columns=['date','open','close','high','low','volume'])
        for c in ['open','close','high','low','volume']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna()
        return df
    except Exception:
        return None

# ============================================================
# 大盘择时过滤
# ============================================================
def market_timing_filter():
    """
    判断大盘是否处于弱势/可选股环境。
    条件（满足任一即弱市）：
      A. 近3个交易日累计跌幅 < MT_RET3_THRESHOLD (默认-1.5%)
      B. 收盘价 < MA10
      C. MA5 < MA10
      D. 连续2日单日跌幅均 < MT_DROP2_THRESHOLD (默认-1.0%)
      E. 收盘价 < 布林带下轨（大盘超跌）
      F. 大盘ATR占比 > MT_ATR_PANIC（恐慌放大，默认0.8%）
    返回 (is_weak, details_dict)
    """
    df = fetch_index_kline('sh000001', days=MT_LOOKBACK)
    if df is None or len(df) < 10:
        return False, {'error': '数据不足'}
    
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma10'] = df['close'].rolling(10).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    df['boll_mid'], df['boll_low'], df['boll_up'] = bollinger(df['close'])
    df['atr14'] = atr(df['high'], df['low'], df['close'], 14)
    df = df.dropna()
    if len(df) < 1:
        return False, {'error': '计算指标后数据不足'}
    
    latest = df.iloc[-1]
    prev3 = df.iloc[-4] if len(df) >= 4 else df.iloc[0]
    prev2 = df.iloc[-3] if len(df) >= 3 else df.iloc[0]
    prev1 = df.iloc[-2] if len(df) >= 2 else df.iloc[0]
    
    # 条件A：近3日累计跌幅
    ret3 = (latest['close'] - prev3['close']) / prev3['close'] * 100
    cond_a = ret3 < MT_RET3_THRESHOLD
    
    # 条件B：收盘 < MA10
    cond_b = latest['close'] < latest['ma10']
    
    # 条件C：MA5 < MA10
    cond_c = latest['ma5'] < latest['ma10']
    
    # 条件D：近2日连续急跌（当日相对昨收跌幅）
    drop_today = (latest['close'] - prev1['close']) / prev1['close'] * 100
    drop_yesterday = (prev1['close'] - prev2['close']) / prev2['close'] * 100
    cond_d = (drop_today < MT_DROP2_THRESHOLD) and (drop_yesterday < MT_DROP2_THRESHOLD)
    
    # 条件E：收盘 < 布林带下轨
    cond_e = latest['close'] < latest['boll_low']
    
    # 条件F：ATR恐慌放大
    atr_pct = (latest['atr14'] / latest['close'] * 100) if latest['close'] > 0 else 0
    cond_f = atr_pct > MT_ATR_PANIC
    
    is_weak = cond_a or cond_b or cond_c or cond_d or cond_e or cond_f
    
    # 计算弱市强度等级
    active_count = sum([cond_a, cond_b, cond_c, cond_d, cond_e, cond_f])
    if active_count >= 4:
        level = '极强'
    elif active_count >= 2:
        level = '较强'
    else:
        level = '偏弱'
    
    return is_weak, {
        'level': level,
        'ret3': round(ret3, 2),
        'cond_a': cond_a,
        'cond_b': cond_b,
        'cond_c': cond_c,
        'drop_today': round(drop_today, 2),
        'drop_yesterday': round(drop_yesterday, 2),
        'cond_d': cond_d,
        'cond_e': cond_e,
        'cond_f': cond_f,
        'close': round(latest['close'], 2),
        'ma5': round(latest['ma5'], 2),
        'ma10': round(latest['ma10'], 2),
        'ma20': round(latest['ma20'], 2),
        'boll_low': round(latest['boll_low'], 2),
        'atr_pct': round(atr_pct, 2),
        'active_count': active_count
    }

# ============================================================
# 个股技术指标评分（新增相对强度因子）
# ============================================================
def score_stock(df_hist, idx_ret=0.0):
    if df_hist is None or len(df_hist) < 30:
        return None
    close = df_hist['close'].astype(float)
    high = df_hist['high'].astype(float)
    low = df_hist['low'].astype(float)
    vol = df_hist['volume'].astype(float)
    
    if len(close) < 30:
        return None
    
    df_hist['ma5'] = close.rolling(5).mean()
    df_hist['ma10'] = close.rolling(10).mean()
    df_hist['ma20'] = close.rolling(20).mean()
    df_hist['ma60'] = close.rolling(60).mean()
    df_hist['dif'], df_hist['dea'], df_hist['macd_bar'] = macd(close)
    df_hist['rsi14'] = rsi(close, 14)
    df_hist['boll_mid'], df_hist['boll_low'], df_hist['boll_up'] = bollinger(close)
    df_hist['atr14'] = atr(high, low, close, 14)
    df_hist['k'], df_hist['d'], df_hist['j'] = compute_kdj(high, low, close)
    df_hist['vol_ma5'] = vol.rolling(5).mean()
    df_hist['vol_ma20'] = vol.rolling(20).mean()
    
    d = df_hist.iloc[-1].to_dict()
    p = float(d['close'])
    prev = df_hist.iloc[-2].to_dict() if len(df_hist) >= 2 else d
    prev_close = float(prev['close'])
    
    ma20 = d['ma20']
    ma60 = d['ma60']
    
    # --- 趋势因子 (0~20) ---
    trend = 0
    if p > ma20: trend += 5
    if ma20 > ma60: trend += 5
    if d['dif'] > d['dea']: trend += 5
    if d['macd_bar'] > 0: trend += 5
    
    # --- 动量因子 (0~20) ---
    mom = 0
    rsi_val = d['rsi14']
    if 40 < rsi_val < 70: mom += 10
    if rsi_val > 50: mom += 5
    if d['macd_bar'] > float(prev.get('macd_bar', 0)): mom += 5
    
    # --- 量能因子 (0~20) ---
    vol_score = 0
    v_ratio = _safe_div(d['volume'], d['vol_ma5'], 0)
    if v_ratio > 1.0: vol_score += 10
    if v_ratio > 1.3: vol_score += 10
    elif v_ratio > 0.8: vol_score += 5
    
    # --- 突破因子 (0~25) ---
    breakout = 0
    recent_high = df_hist['high'].tail(20).max()
    recent_low = df_hist['low'].tail(20).min()
    if p > ma20: breakout += 5
    if p >= recent_high * 0.995: breakout += 10
    if recent_high > recent_low:
        pos = (p - recent_low) / (recent_high - recent_low)
        if pos > 0.66: breakout += 10
    
    # --- 波动/风控因子 (0~15) ---
    risk = 0
    atr_val = d['atr14']
    atr_pct = _safe_div(atr_val, p, 0) * 100
    if 1.5 < atr_pct < 5.0: risk += 10
    if d['boll_up'] > p > d['boll_mid']: risk += 5
    
    # --- 相对强度因子 (0~15) --- 弱市核心
    rel_strength = 0
    if len(close) >= 6:
        individual_ret = (p - float(close.iloc[-6])) / float(close.iloc[-6]) * 100
        # 个股收益减去大盘收益，差值越大得分越高
        diff = individual_ret - idx_ret
        if diff > 0: rel_strength += 5
        if diff > 3: rel_strength += 5
        if diff > 6: rel_strength += 5
        if diff > 10: rel_strength += 5  # 大盘暴跌它暴涨，最高15分
    
    total = trend + mom + vol_score + breakout + risk + rel_strength
    
    return {
        'trend': trend, 'momentum': mom, 'volume': vol_score,
        'breakout': breakout, 'risk_ctrl': risk,
        'rel_strength': rel_strength,
        'total': total,
        'close': p, 'prev_close': prev_close,
        'ma20': ma20, 'ma60': ma60,
        'dif': d['dif'], 'dea': d['dea'], 'macd_bar': d['macd_bar'],
        'rsi14': d['rsi14'],
        'boll_mid': d['boll_mid'], 'boll_up': d['boll_up'], 'boll_low': d['boll_low'],
        'atr14': d['atr14'],
        'k': d['k'], 'd': d['d'], 'j': d['j'],
        'vol_ratio': v_ratio,
        'recent_high': recent_high, 'recent_low': recent_low,
        'vol_ma5': d['vol_ma5'], 'vol_ma20': d['vol_ma20']
    }

# ============================================================
# 历史胜率回测
# ============================================================
def backtest_signal(df_hist, lookback=60, hold_days=5):
    if df_hist is None or len(df_hist) < lookback + hold_days + 30:
        return None
    
    buy_signals = []
    sell_signals = []
    
    for i in range(lookback, 0, -1):
        sub = df_hist.iloc[:len(df_hist)-i].copy()
        if len(sub) < 30:
            continue
        sc = score_stock(sub)
        if sc is None:
            continue
        future_idx = len(df_hist) - i + hold_days - 1
        if future_idx >= len(df_hist):
            continue
        future_close = float(df_hist.iloc[future_idx]['close'])
        entry_close = float(sub.iloc[-1]['close'])
        ret = (future_close - entry_close) / entry_close * 100
        
        if sc['total'] >= 55:
            buy_signals.append(ret)
        elif sc['total'] <= -55:
            sell_signals.append(ret)
    
    def _stat(arr):
        if not arr:
            return 0, 0.0, 0.0
        wins = sum(1 for x in arr if x > 0)
        return len(arr), round(wins/len(arr)*100, 1), round(sum(arr)/len(arr), 2)
    
    b_n, b_win, b_avg = _stat(buy_signals)
    s_n, s_win, s_avg = _stat(sell_signals)
    return {
        'buy_count': b_n, 'buy_win_rate': b_win, 'buy_avg_ret': b_avg,
        'sell_count': s_n, 'sell_win_rate': s_win, 'sell_avg_ret': s_avg,
    }

# ============================================================
# 交易区间生成
# ============================================================
def make_trade_zones(score_dict):
    c = score_dict['close']
    a = score_dict['atr14']
    buy_near = round(max(score_dict['ma20'], c - a * 0.5), 2)
    buy_far = round(c - a * 1.2, 2)
    stop = round(c - a * 1.5, 2)
    target1 = round(c + a * 2.0, 2)
    target2 = round(c + a * 3.5, 2)
    return {
        'buy_zone': [buy_far, buy_near],
        'stop': stop,
        'target1': target1,
        'target2': target2,
    }

# ============================================================
# HTML 生成器（单文件自包含）
# ============================================================
def generate_html(payload, board_dict, dt_str, market_status):
    board_list = sorted(board_dict.items(), key=lambda x: x[1], reverse=True)[:15]
    
    html = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<meta name="apple-mobile-web-app-capable" content="yes">
<title>''' + ('弱市候选池 · 次日观察名单 (' + dt_str + ')' if market_status.get('is_weak') else '震荡市观察名单 · 次日观察名单 (' + dt_str + ')') + '''</title>
<style>
:root{--bg:#0d1117;--panel:#161b22;--panel2:#1c2333;--line:#21262d;--line2:#30363d;--tx:#e6edf3;--tx2:#8b949e;--tx3:#6e7681;--up:#f85149;--down:#3fb950;--flat:#8b949e;--acc:#58a6ff;--acc2:#1f6feb;--gold:#d29922;--r1:#ff9da6;}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--tx);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;font-size:14px;line-height:1.6;padding:12px;max-width:1200px;margin:0 auto}
header{text-align:center;padding:16px 0;border-bottom:1px solid var(--line);margin-bottom:16px}
h1{font-size:22px;font-weight:800;letter-spacing:1px}
.sub{font-size:12px;color:var(--tx2);margin-top:4px}
.market-status{padding:12px 16px;border-radius:8px;margin:12px 0;font-size:13px}
.market-status.weak{background:#f8514922;border:1px solid #f8514955;color:#ff9da6}
.market-status.neutral{background:#d2992222;border:1px solid #d2992255;color:#d29922}
.market-status .status-title{font-weight:700;font-size:14px;margin-bottom:4px}
.market-status .status-detail{font-size:12px;color:var(--tx2);line-height:1.6}
.board-row{display:flex;gap:6px;flex-wrap:wrap;margin:12px 0;padding:10px;background:var(--panel);border-radius:8px;border:1px solid var(--line)}
.board-tag{font-size:11px;padding:3px 8px;border-radius:12px;background:var(--panel2);border:1px solid var(--line2);color:var(--tx2)}
.board-tag b{color:var(--tx)}
table{width:100%;border-collapse:collapse;margin-top:12px;font-size:12px}
th{position:sticky;top:0;background:var(--panel);color:var(--tx2);font-weight:600;text-align:left;padding:8px;border-bottom:2px solid var(--line2);white-space:nowrap}
td{padding:8px;border-bottom:1px solid var(--line);vertical-align:top}
tr:hover td{background:var(--panel2)}
.up{color:var(--up)!important} .down{color:var(--down)!important} .flat{color:var(--flat)!important}
.score-bar{display:inline-block;width:40px;height:6px;border-radius:3px;background:var(--line2);vertical-align:middle;margin-left:4px;position:relative;overflow:hidden}
.score-bar i{display:block;height:100%;border-radius:3px;background:linear-gradient(90deg,var(--acc),var(--acc2))}
.signal{display:inline-block;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:700}
.signal.buy{background:#f8514922;color:#ff9da6;border:1px solid #f8514933}
.signal.strong{background:#f8514944;color:#fff;border:1px solid #f8514988}
.signal.hold{background:var(--line2);color:var(--tx2);border:1px solid var(--line)}
.note{font-size:11px;color:var(--tx3);margin-top:4px;line-height:1.5}
.num{font-family:"SF Mono",Consolas,Menlo,monospace;font-variant-numeric:tabular-nums}
.zone{display:inline-block;font-size:11px;padding:1px 6px;border-radius:4px;background:var(--panel2);margin-right:4px}
.footer{text-align:center;padding:20px 0;color:var(--tx3);font-size:11px;border-top:1px solid var(--line);margin-top:20px}
@media(max-width:768px){body{padding:8px} h1{font-size:18px} table{font-size:11px} th,td{padding:5px} .hide-m{display:none}}
</style>
</head>
<body>
<header>
<h1>''' + ('弱市候选池' if market_status.get('is_weak') else '震荡市观察名单') + ''' · 次日观察名单</h1>
<div class="sub">''' + ('数据截至 ' + dt_str + ' · 共筛选 ' + str(len(payload)) + ' 只 · 评分模型：趋势/动量/量能/突破/风控/相对强度' + ('' if market_status.get('is_weak') else ' ⚠ 当前大盘偏强，建议观望或持有指数ETF')) + '''</div>
</header>
''' + ('<div class="market-status weak">' if market_status.get('is_weak') else '<div class="market-status neutral">') + '''
<div class="status-title">''' + ('🔴 弱市环境：建议主动选股、控制仓位、严格止损' if market_status.get('is_weak') else '🟡 震荡/偏强市：建议观望或持有指数ETF') + '''</div>
<div class="status-detail">
上证指数近3日涨跌：''' + str(round(market_status.get("ret3", 0), 2)) + '''% ''' + ('(条件A: 跌幅>1.5% ✅)' if market_status.get("cond_a") else '(条件A: 跌幅≤1.5%)') + '''<br>
收盘 ''' + str(market_status.get("close", "—")) + ''' vs MA10 ''' + str(market_status.get("ma10", "—")) + ''' ''' + ('(条件B: 跌破MA10 ✅)' if market_status.get("cond_b") else '(条件B: 站上MA10)') + '''<br>
MA5 ''' + str(market_status.get("ma5", "—")) + ''' vs MA10 ''' + str(market_status.get("ma10", "—")) + ''' ''' + ('(条件C: MA5<MA10 ✅)' if market_status.get("cond_c") else '(条件C: MA5>MA10)') + '''<br>
近2日跌幅：''' + str(market_status.get("drop_yesterday", "—")) + '''% / ''' + str(market_status.get("drop_today", "—")) + '''% ''' + ('(条件D: 连续急跌 ✅)' if market_status.get("cond_d") else '(条件D: 非连续急跌)') + '''<br>
收盘 ''' + str(market_status.get("close", "—")) + ''' vs 布林下轨 ''' + str(market_status.get("boll_low", "—")) + ''' ''' + ('(条件E: 跌破下轨 ✅)' if market_status.get("cond_e") else '(条件E: 未破下轨)') + '''<br>
ATR占比 ''' + str(market_status.get("atr_pct", "—")) + '''% ''' + ('(条件F: 恐慌放大 ✅)' if market_status.get("cond_f") else '(条件F: 波动正常)') + '''<br>
<b>弱市等级：''' + str(market_status.get("level", "—")) + ''' · 触发 ''' + str(market_status.get("active_count", 0)) + '''/6 个条件</b>
</div>
</div>

<div style="font-size:13px;color:var(--tx2);margin-bottom:8px"><b>当日强势板块 Top 15</b></div>
<div class="board-row">
'''
    for name, chg in board_list:
        color = 'up' if chg > 0 else 'down' if chg < 0 else 'flat'
        html += f'<span class="board-tag"><b>{name}</b> <span class="{color}">{chg:+.2f}%</span></span>'
    html += '</div>\n'
    
    html += '''
<table>
<thead>
<tr>
<th>排名</th>
<th>代码</th>
<th>名称</th>
<th>现价</th>
<th>涨跌</th>
<th class="hide-m">成交额</th>
<th class="hide-m">换手</th>
<th>综合<br>评分</th>
<th class="hide-m">因子分解</th>
<th>历史胜率<br>(买入信号)</th>
<th>技术区间</th>
</tr>
</thead>
<tbody>
'''
    
    for idx, r in enumerate(payload, 1):
        zones = r['zones']
        chg_cls = 'up' if r['change_pct'] > 0 else 'down' if r['change_pct'] < 0 else 'flat'
        score = r['final_score']
        sig = 'strong' if score >= 55 else 'buy' if score >= 35 else 'hold'
        sig_text = '强买' if score >= 55 else '买入' if score >= 35 else '观察'
        bar_w = min(100, max(5, score))
        
        amt_str = f"{r['amount']/10000:.0f}万" if r['amount'] < 1e8 else f"{r['amount']/1e8:.1f}亿"
        
        html += f'''<tr>
<td class="num">{idx}</td>
<td class="num">{r['code']}</td>
<td><b>{r['name']}</b></td>
<td class="num">{r['close']:.2f}</td>
<td class="num {chg_cls}">{r['change_pct']:+.2f}%</td>
<td class="hide-m num">{amt_str}</td>
<td class="hide-m num">{r['turnover']:.1f}%</td>
<td><span class="signal {sig}">{sig_text}</span><br><span class="num" style="font-weight:700;font-size:13px">{score:.1f}</span><span class="score-bar"><i style="width:{bar_w}%"></i></span></td>
<td class="hide-m" style="font-size:11px;color:var(--tx2)">
趋势:{r['trend']} 动量:{r['momentum']}<br>
量能:{r['volume']} 突破:{r['breakout']}<br>
风控:{r['risk_ctrl']} 相对强:{r.get('rel_strength', 0)}</td>
<td style="font-size:11px">
<b style="color:var(--{'up' if r['buy_win_rate']>=50 else 'down'})">{r['buy_win_rate']}%</b> ({r['buy_count']}次)<br>
<span class="note">均收益 {r['buy_avg_ret']:+.2f}%</span>
</td>
<td style="font-size:11px">
<span class="zone">买入 {zones['buy_zone'][0]:.2f}~{zones['buy_zone'][1]:.2f}</span><br>
<span class="zone">止损 {zones['stop']:.2f}</span><br>
<span class="zone">目标1 {zones['target1']:.2f}</span>
<span class="zone">目标2 {zones['target2']:.2f}</span>
</td>
</tr>
'''
    
    html += '''</tbody></table>
<div class="footer">
<p>数据来源：腾讯财经公开接口 · 技术指标仅作参考，不构成投资建议</p>
<p>使用方式：次日开盘前浏览本名单，将感兴趣的股票加入《股票AI追踪助手》自选股，盘中接收买卖点提醒</p>
<p>历史胜率 = 近60个交易日中出现同等评分信号后，持有5日的方向正确率</p>
</div>
</body>
</html>'''
    
    return html

# ============================================================
# 主流程
# ============================================================
def main():
    print("[1/6] 获取全市场股票代码...")
    try:
        code_df = ak.stock_info_a_code_name()
        codes = code_df['code'].tolist()
        print(f"  共 {len(codes)} 只股票")
    except Exception as e:
        print(f"[FATAL] 获取代码列表失败: {e}")
        sys.exit(1)
    
    # 添加前缀（仅沪深主板）
    def add_prefix(c):
        # 沪市主板：6开头，排除688/689（科创板）
        if c.startswith('6') and not c.startswith('68'):
            return f'sh{c}'
        # 深市主板：0开头（排除300/301创业板）
        if c.startswith('0'):
            return f'sz{c}'
        # 其余（创业板3、科创板68、北交所4/8）丢弃
        return None
    
    full_codes = [add_prefix(c) for c in codes]
    full_codes = [c for c in full_codes if c is not None]
    
    # 获取板块强度
    print("[2/6] 获取板块强度...")
    board_dict = get_board_strength()
    print(f"  获取 {len(board_dict)} 个板块")
    
    # 批量获取实时行情
    print("[3/6] 批量获取实时行情...")
    all_quotes = {}
    total_batches = math.ceil(len(full_codes) / BATCH_SIZE)
    for i in range(total_batches):
        batch = full_codes[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        batch_data = fetch_tencent_batch(batch)
        all_quotes.update(batch_data)
        if (i + 1) % 5 == 0 or i + 1 == total_batches:
            print(f"  批次 {i+1}/{total_batches}，累计有效 {len(all_quotes)}")
        time.sleep(0.15)
    
    print(f"  有效行情数据 {len(all_quotes)} 只")
    
    # 获取大盘近5日收益（用于相对强度计算）
    print("[3.5/6] 计算大盘相对强度基准...")
    idx_df = fetch_index_kline('sh000001', days=10)
    if idx_df is not None and len(idx_df) >= 6:
        idx_ret = (float(idx_df.iloc[-1]['close']) - float(idx_df.iloc[-6]['close'])) / float(idx_df.iloc[-6]['close']) * 100
    else:
        idx_ret = 0.0
    print(f"  上证指数近5日收益: {idx_ret:+.2f}%")
    
    # 基础过滤
    filtered = []
    for code, q in all_quotes.items():
        if q['price'] <= 0 or q['price'] > MAX_PRICE:
            continue
        if q['amount'] * 10000 < MIN_AMT:  # 腾讯 amount 是万元
            continue
        if 'ST' in q['name'] or '*ST' in q['name']:
            continue
        if q['turnover'] <= 0:
            continue
        filtered.append(code)
    
    print(f"[4/6] 基础过滤后 {len(filtered)} 只（成交额≥{MIN_AMT/1e8:.0f}亿、非ST）")
    
    if not filtered:
        print("[WARN] 过滤后无股票，请检查数据")
        sys.exit(0)
    
    # 逐只获取历史K线并评分
    print("[5/6] 逐只计算技术指标（约需 10~20 分钟）...")
    results = []
    processed = 0
    
    def process_one(code):
        hist = fetch_tencent_kline(code, days=HIST_DAYS)
        if hist is None or len(hist) < MIN_LIST_DAYS:
            return None
        sc = score_stock(hist, idx_ret=idx_ret)
        if sc is None:
            return None
        bt = backtest_signal(hist, lookback=60, hold_days=5)
        if bt is None:
            return None
        zones = make_trade_zones(sc)
        q = all_quotes[code]
        return {
            'code': code, 'name': q['name'],
            'close': q['price'], 'change_pct': q['change_pct'],
            'amount': q['amount'] * 10000, 'turnover': q['turnover'],
            'pe': q['pe'], 'pb': q['pb'],
            'final_score': sc['total'],
            'trend': sc['trend'], 'momentum': sc['momentum'],
            'volume': sc['volume'], 'breakout': sc['breakout'], 'risk_ctrl': sc['risk_ctrl'],
            'buy_count': bt['buy_count'], 'buy_win_rate': bt['buy_win_rate'], 'buy_avg_ret': bt['buy_avg_ret'],
            'sell_count': bt['sell_count'], 'sell_win_rate': bt['sell_win_rate'], 'sell_avg_ret': bt['sell_avg_ret'],
            'zones': zones
        }
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_one, c): c for c in filtered}
        for future in as_completed(futures):
            res = future.result()
            processed += 1
            if res:
                results.append(res)
            if processed % 200 == 0 or processed == len(filtered):
                print(f"  已处理 {processed}/{len(filtered)}，有效评分 {len(results)}")
    
    print(f"[5/6] 有效评分股票 {len(results)} 只")
    
    if not results:
        print("[WARN] 无有效结果")
        sys.exit(0)
    
    # 排序取 Top N
    results.sort(key=lambda x: x['final_score'], reverse=True)
    
    # 大盘择时过滤
    print("[5.5/6] 大盘择时判断...")
    is_weak, mkt_status = market_timing_filter()
    if is_weak:
        top = results[:TOP_N]
        print(f"  🔴 弱市环境，生成候选池 Top {len(top)}")
    else:
        top = results[:20]  # 非弱市仅展示Top 20参考
        print(f"  🟡 非弱市环境，生成候选池 Top {len(top)}（仅供参考，建议观望）")
    
    dt_str = datetime.now().strftime('%m-%d %H:%M')
    market_status = {'is_weak': is_weak, **mkt_status}
    html_content = generate_html(top, board_dict, dt_str, market_status)
    
    out_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(out_dir, f'stock_picker_{datetime.now().strftime("%Y%m%d")}.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"[OK] HTML 已生成: {html_path}")
    print(f"[OK] 候选池评分区间: {top[-1]['final_score']:.1f} ~ {top[0]['final_score']:.1f}")

    # 生成候选池JSON文件
    # 转换numpy类型为Python原生类型
    def _convert(v):
        import numpy as np
        if isinstance(v, np.bool_):
            return bool(v)
        if hasattr(v, 'item'):
            return v.item()
        if isinstance(v, dict):
            return {k: _convert(vv) for k, vv in v.items()}
        if isinstance(v, list):
            return [_convert(vv) for vv in v]
        return v
    
    pool_data = {
        "date": datetime.now().strftime("%Y%m%d"),
        "is_weak": bool(is_weak),
        "market_status": _convert(market_status),
        "stocks": [
            {
                "code": r["code"],
                "name": r["name"],
                "close": r["close"],
                "final_score": r["final_score"],
                "buy_win_rate": r["buy_win_rate"],
                "zones": r["zones"]
            }
            for r in top
        ]
    }
    json_path = os.path.join(out_dir, f'stock_picker_pool_{datetime.now().strftime("%Y%m%d")}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(pool_data, f, ensure_ascii=False, indent=2)
    print(f"[OK] 候选池JSON已生成: {json_path}")

    print("\n预览 Top 10:")
    for i, r in enumerate(top[:10], 1):
        print(f"  {i:2d}. {r['code']} {r['name']:6s} 评分:{r['final_score']:5.1f}  "
              f"买胜率:{r['buy_win_rate']}% ({r['buy_count']}次)  "
              f"买入区:{r['zones']['buy_zone'][0]:.2f}~{r['zones']['buy_zone'][1]:.2f}")

if __name__ == '__main__':
    main()
