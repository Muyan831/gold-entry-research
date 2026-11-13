# -*- coding: utf-8 -*-
"""
黄金 H1 短线进场规则回测（真实数据，不编造）
数据源：yfinance GC=F（黄金期货，与 XAUUSD 走势基本同步），1小时 K，最近2年。
点差成本：每笔交易扣 0.20 价格点（美元/盎司）。
进场约定：信号在根K收盘时确认，于该根收盘价进场（无未来函数）。
离场约定：见各规则；SL/TP 使用后续K的 high/low 逐根检测，避免未来函数。
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

SPREAD = 0.20   # 点差成本（每笔，价格点）
SYMBOL = "GC=F"
PERIOD = "730d"  # 最近2年（yfinance 1h 上限约730天）
INTERVAL = "1h"

print("=" * 70)
print("下载数据：", SYMBOL, INTERVAL, PERIOD)
df = yf.download(SYMBOL, period=PERIOD, interval=INTERVAL, progress=False, auto_adjust=False)
if df is None or len(df) == 0:
    raise SystemExit("下载失败，无数据")
print("下载完成：", len(df), "根K，", df.index[0], "~", df.index[-1])

# 扁平化列名（yfinance 可能返回 MultiIndex，level 0 为 Price 字段）
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df.columns = [str(c).lower() for c in df.columns]

print("价格区间：low=%.2f  high=%.2f" % (df["low"].min(), df["high"].max()))

df = df[["open", "high", "low", "close"]].dropna()

close = df["close"]
high = df["high"]
low = df["low"]

df["ma20"] = SMAIndicator(close, window=20).sma_indicator()
df["ma50"] = SMAIndicator(close, window=50).sma_indicator()
df["atr14"] = AverageTrueRange(high, low, close, window=14).average_true_range()
df["rsi14"] = RSIIndicator(close, window=14).rsi()
df["hh20"] = high.rolling(20).max()
df["ll20"] = low.rolling(20).min()
df["hour_utc"] = df.index.hour
df["in_session"] = (df["hour_utc"] >= 14) & (df["hour_utc"] < 20)

WARMUP = 60
n = len(df)


def run_rule(entries, sl_dist_fn, tp_dist_fn=None, exit_reverse_fn=None,
             session_filter=None, label=""):
    """单规则回测。entries: list of (bar_index, direction)。"""
    entry_set = {i: d for i, d in entries}
    trades = []
    in_pos = False
    pos = {}

    i = WARMUP
    while i < n:
        if not in_pos:
            if i in entry_set:
                if session_filter is not None and not session_filter(i):
                    i += 1
                    continue
                direction = entry_set[i]
                entry_price = float(close.iloc[i])
                sld = sl_dist_fn(i)
                if sld is None or np.isnan(sld) or sld <= 0:
                    i += 1
                    continue
                sl = entry_price - sld if direction == 1 else entry_price + sld
                tp = None
                if tp_dist_fn is not None:
                    tpd = tp_dist_fn(i)
                    if tpd is not None and not np.isnan(tpd) and tpd > 0:
                        tp = entry_price + tpd if direction == 1 else entry_price - tpd
                pos = {"dir": direction, "entry": entry_price, "sl": sl, "tp": tp,
                       "entry_bar": i}
                in_pos = True
                i += 1
                continue
            i += 1
            continue
        else:
            cur_high = float(high.iloc[i])
            cur_low = float(low.iloc[i])
            exited = False
            exit_price = None
            if pos["dir"] == 1:
                if cur_low <= pos["sl"]:
                    exit_price = pos["sl"]; exited = True
                elif pos["tp"] is not None and cur_high >= pos["tp"]:
                    exit_price = pos["tp"]; exited = True
                elif exit_reverse_fn is not None and exit_reverse_fn(i):
                    exit_price = float(close.iloc[i]); exited = True
            else:
                if cur_high >= pos["sl"]:
                    exit_price = pos["sl"]; exited = True
                elif pos["tp"] is not None and cur_low <= pos["tp"]:
                    exit_price = pos["tp"]; exited = True
                elif exit_reverse_fn is not None and exit_reverse_fn(i):
                    exit_price = float(close.iloc[i]); exited = True

            if exited:
                gross = (exit_price - pos["entry"]) if pos["dir"] == 1 else (pos["entry"] - exit_price)
                pnl = gross - SPREAD
                trades.append({"dir": pos["dir"], "entry": pos["entry"], "exit": exit_price,
                               "pnl": pnl, "win": pnl > 0})
                in_pos = False
                pos = {}
                i += 1
                continue
            i += 1
    return trades


def stats(trades, label):
    if not trades:
        return {"规则": label, "交易数": 0, "胜率%": 0.0, "均盈亏比": 0.0,
                "总收益(点)": 0.0, "最大连亏": 0, "扣点差": "是(0.20)"}
    wins = [t for t in trades if t["win"]]
    losses = [t for t in trades if not t["win"]]
    win_rate = len(wins) / len(trades) * 100
    avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0.0
    avg_loss = np.mean([abs(t["pnl"]) for t in losses]) if losses else 0.0
    payoff = (avg_win / avg_loss) if avg_loss > 0 else None
    total = sum(t["pnl"] for t in trades)
    max_loss_streak = 0
    cur = 0
    for t in trades:
        if not t["win"]:
            cur += 1
            max_loss_streak = max(max_loss_streak, cur)
        else:
            cur = 0
    return {"规则": label, "交易数": len(trades), "胜率%": round(win_rate, 2),
            "均盈亏比": round(payoff, 2) if payoff is not None else None,
            "总收益(点)": round(total, 2), "最大连亏": max_loss_streak,
            "扣点差": "是(0.20)"}


def sl_dist(i):
    atr = df["atr14"].iloc[i]
    return 2.0 * atr if pd.notna(atr) else None


def tp_dist_1to2(i):
    atr = df["atr14"].iloc[i]
    return 4.0 * atr if pd.notna(atr) else None


def rev_long(i):
    return close.iloc[i] < df["ma20"].iloc[i]


def rev_short(i):
    return close.iloc[i] > df["ma20"].iloc[i]


# ---------------- 信号构建 ----------------
def build_entries_rule1():
    sig = []
    for i in range(WARMUP, n - 1):
        c = close.iloc[i]
        m20 = df["ma20"].iloc[i]; m50 = df["ma50"].iloc[i]
        if pd.notna(m20) and pd.notna(m50) and c > m20 and m20 > m50:
            sig.append((i, 1))
    return sig


def build_entries_rule2():
    sig = []
    for i in range(WARMUP, n - 1):
        c = close.iloc[i]
        m20 = df["ma20"].iloc[i]; m50 = df["ma50"].iloc[i]
        if pd.notna(m20) and pd.notna(m50) and c < m20 and m20 < m50:
            sig.append((i, -1))
    return sig


def build_entries_rule3():
    sig = []
    for i in range(WARMUP, n - 1):
        rsi_prev = df["rsi14"].iloc[i - 1]
        if pd.notna(rsi_prev) and rsi_prev < 30 and close.iloc[i] > close.iloc[i - 1]:
            sig.append((i, 1))
    return sig


def build_entries_rule4():
    sig = []
    for i in range(WARMUP, n - 1):
        if high.iloc[i] > df["hh20"].iloc[i - 1]:
            sig.append((i, 1))
    return sig


def build_entries_rule5():
    sig = []
    for i in range(WARMUP, n - 1):
        if low.iloc[i] < df["ll20"].iloc[i - 1]:
            sig.append((i, -1))
    return sig


def build_entries_rule6():
    sig = []
    for i in range(WARMUP, n - 1):
        c = close.iloc[i]
        m20 = df["ma20"].iloc[i]; m50 = df["ma50"].iloc[i]
        if (pd.notna(m20) and pd.notna(m50) and c > m20 and m20 > m50
                and bool(df["in_session"].iloc[i])):
            sig.append((i, 1))
    return sig


def build_entries_rule7():
    return build_entries_rule1()


def build_entries_rule8():
    sig = []
    for i in range(WARMUP, n - 1):
        c = close.iloc[i]
        m20 = df["ma20"].iloc[i]; m50 = df["ma50"].iloc[i]; atr = df["atr14"].iloc[i]
        if (pd.notna(m20) and pd.notna(m50) and pd.notna(atr)
                and m20 > m50 and c > m20
                and abs(c - m20) <= 0.2 * atr):
            sig.append((i, 1))
    return sig


results = []

# Rule 1
tr = run_rule(build_entries_rule1(), sl_dist, None,
              exit_reverse_fn=lambda i: rev_long(i),
              label="1) 趋势多(收盘>MA20且MA20>MA50，至反向信号离场，SL=2ATR)")
results.append(stats(tr, "1) 趋势多(收盘>MA20且MA20>MA50，至反向信号离场，SL=2ATR)"))

# Rule 2
tr = run_rule(build_entries_rule2(), sl_dist, None,
              exit_reverse_fn=lambda i: rev_short(i),
              label="2) 趋势空")
results.append(stats(tr, "2) 趋势空(收盘<MA20且MA20<MA50，至反向信号离场，SL=2ATR)"))

# Rule 3
tr = run_rule(build_entries_rule3(), sl_dist, tp_dist_1to2, None,
              label="3) RSI超卖反弹多")
results.append(stats(tr, "3) RSI14<30后下一根收盘转阳→做多(SL=2ATR, TP=4ATR)"))

# Rule 4
tr = run_rule(build_entries_rule4(), sl_dist, tp_dist_1to2, None,
              label="4) 突破20根高点多")
results.append(stats(tr, "4) 突破近20根H1最高→做多(SL=2ATR, TP=4ATR)"))

# Rule 5
tr = run_rule(build_entries_rule5(), sl_dist, tp_dist_1to2, None,
              label="5) 跌破20根低点空(H1近似)")
results.append(stats(tr, "5) 跌破近20根最低→做空(H1近似，SL=2ATR, TP=4ATR)"))

# Rule 6
tr = run_rule(build_entries_rule6(), sl_dist, None,
              exit_reverse_fn=lambda i: rev_long(i),
              session_filter=lambda i: bool(df["in_session"].iloc[i]),
              label="6) 仅14-20UTC执行规则1")
results.append(stats(tr, "6) 仅14-20 UTC执行规则1(趋势多，至反向，SL=2ATR)"))

# Rule 7
tr = run_rule(build_entries_rule7(), sl_dist, tp_dist_1to2, None,
              label="7) 规则1+SL+1:2")
results.append(stats(tr, "7) 规则1+ATR止损+1:2盈亏比(SL=2ATR, TP=4ATR，硬离场)"))

# Rule 8
tr = run_rule(build_entries_rule8(), sl_dist, tp_dist_1to2, None,
              label="8) 均线回踩多")
results.append(stats(tr, "8) 均线回踩(价在MA20±0.2ATR且趋势同向→做多, SL=2ATR, TP=4ATR)"))

# ---------------- 输出 ----------------
print("\n" + "=" * 70)
print("黄金 H1 短线进场规则回测结果（数据：%s %s，%d 根K，%s ~ %s）"
      % (SYMBOL, INTERVAL, n, df.index[0].date(), df.index[-1].date()))
print("每笔交易已扣点差 0.20 点。盈亏比 = 平均盈利/平均亏损（绝对值）。\n")
print(f"{'规则':<36} {'交易数':>5} {'胜率%':>7} {'均盈亏比':>9} {'总收益(点)':>11} {'最大连亏':>7} {'扣点差':>10}")
print("-" * 70)
for r in results:
    payoff_disp = "∞" if r["均盈亏比"] is None else str(r["均盈亏比"])
    print(f"{r['规则']:<36} {r['交易数']:>5} {r['胜率%']:>7} {payoff_disp:>9} {r['总收益(点)']:>11} {r['最大连亏']:>7} {r['扣点差']:>10}")
print("=" * 70)

ranked = [r for r in results if r["交易数"] > 0]
ranked.sort(key=lambda r: r["总收益(点)"], reverse=True)
print("\nTop3 推荐（按总收益点数）：")
for k, r in enumerate(ranked[:3], 1):
    exp = r["总收益(点)"] / max(r["交易数"], 1)
    print(f"  {k}. {r['规则']} | 交易{r['交易数']} 胜率{r['胜率%']}% 均盈亏比{r['均盈亏比']} 总收益{r['总收益(点)']}点 最大连亏{r['最大连亏']}  单笔期望≈{exp:.2f}点")

print("\n期望判断（单笔扣点差后总收益/交易数）：")
for r in results:
    if r["交易数"] > 0:
        exp = r["总收益(点)"] / r["交易数"]
        flag = "正期望 ✔" if exp > 0 else "负期望 ✘"
        print(f"  {r['规则']}: {exp:+.2f} 点/笔  {flag}")
    else:
        print(f"  {r['规则']}: 无交易")

pd.DataFrame(results).to_csv("/home/rx22/gold_rule_backtest.csv", index=False, encoding="utf-8-sig")
print("\n已保存汇总至 /home/rx22/gold_rule_backtest.csv")
