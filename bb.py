"""
볼린저밴드 과매도 반등 스캐너
- S&P 500 + 내 관심종목에서
- 상승 추세 종목이 볼린저밴드 하단을 과하게 이탈한 뒤 반등을 확인한 순간을 찾아
- 목표가(중심선)·손절가(이탈 저가)·손익비와 함께 디스코드로 보냅니다.
"""
import datetime as dt
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

# ───────────── 설정 (여기만 고치면 됩니다) ─────────────
GROUPS = os.environ.get("SCAN_GROUPS", "sp500,watchlist").split(",")  # 앞 그룹 우선

BB_LEN = int(os.environ.get("BB_LEN", "20"))                    # 볼린저밴드 기간
BB_STD = float(os.environ.get("BB_STD", "2.0"))                 # 볼린저밴드 표준편차 배수
PCTB_MAX = float(os.environ.get("PCTB_MAX", "-0.1"))            # ① 이탈일 %B가 이 값보다 작아야 함 (하단을 확실히 뚫음)
LOOKBACK = int(os.environ.get("LOOKBACK", "1"))                 # ④ 이탈일이 오늘 기준 며칠 전까지 허용되는지 (1 = 어제)

RSI2_MAX = float(os.environ.get("RSI2_MAX", "5"))               # ② 이탈일 RSI(2) < 5
RSI14_MAX = float(os.environ.get("RSI14_MAX", "30"))            # ② 또는 RSI(14) < 30

REQUIRE_VOL_SPIKE = os.environ.get("REQUIRE_VOL_SPIKE", "1") == "1"
VOL_SPIKE = float(os.environ.get("VOL_SPIKE", "1.5"))           # ⑤ 이탈일 거래량 ≥ 50일 평균 × 1.5
REQUIRE_NO_GAP = os.environ.get("REQUIRE_NO_GAP", "1") == "1"
GAP_DOWN_PCT = float(os.environ.get("GAP_DOWN_PCT", "8"))       # ⑥ 이탈일·오늘 시가가 전일 종가보다 8% 이상 낮으면 제외
MAX_ABOVE_200 = float(os.environ.get("MAX_ABOVE_200", "15"))    # ③⑦ 오늘 종가가 200일선 위 0~15% 이내

MIN_PRICE = float(os.environ.get("MIN_PRICE", "5"))
MIN_DOLLAR_VOL = float(os.environ.get("MIN_DOLLAR_VOL", "5000000"))  # 50일 평균 거래대금($)

REALERT_DAYS = int(os.environ.get("REALERT_DAYS", "5"))         # 같은 종목은 이 기간(달력일) 안에 다시 알리지 않음
WEBHOOK = os.environ.get("DISCORD_WEBHOOK_URL", "")
FORCE = os.environ.get("FORCE", "0") == "1"                     # 테스트용 수동 실행
# ──────────────────────────────────────────────────────

NY = ZoneInfo("America/New_York")
STATE = Path(__file__).parent / "state"
STATE.mkdir(exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (bb-scanner)"}
GROUP_LABEL = {"sp500": "S&P 500", "watchlist": "Watchlist"}


# ───────────── 종목 목록 ─────────────
def _fetch_sp500() -> pd.DataFrame:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = requests.get(url, headers=UA, timeout=30).text
    t = pd.read_html(io.StringIO(html))[0]
    return pd.DataFrame({"ticker": t["Symbol"], "name": t["Security"], "industry": t["GICS Sub-Industry"]})


def _load_watchlist() -> pd.DataFrame:
    p = Path(__file__).parent / "watchlist.txt"
    rows = []
    if p.exists():
        for ln in p.read_text(encoding="utf-8").splitlines():
            ln = ln.split("#")[0].strip()
            if not ln:
                continue
            tk, _, name = ln.partition(",")
            tk = re.sub(r"\.([A-Z])$", r"-\1", tk.strip().upper())  # MOG.A → MOG-A
            rows.append({"ticker": tk, "name": name.strip(), "industry": "Watchlist"})
    return pd.DataFrame(rows, columns=["ticker", "name", "industry"]).drop_duplicates("ticker")


def load_universe(group: str) -> pd.DataFrame:
    if group == "watchlist":
        return _load_watchlist()
    cache = STATE / f"universe_{group}.csv"
    fresh = cache.exists() and (time.time() - cache.stat().st_mtime) < 7 * 86400
    if not fresh:
        try:
            df = {"sp500": _fetch_sp500}[group]()
            if len(df) < 50:
                raise ValueError("종목 수가 비정상적으로 적음")
            df.to_csv(cache, index=False)
        except Exception as e:
            print(f"[경고] {group} 목록 갱신 실패: {e}")
    if not cache.exists():
        return pd.DataFrame(columns=["ticker", "name", "industry"])
    df = pd.read_csv(cache).fillna("")
    df["ticker"] = df["ticker"].astype(str).str.strip().str.replace(".", "-", regex=False)
    df = df[df["ticker"].str.fullmatch(r"[A-Z\-]{1,6}")]
    return df.drop_duplicates("ticker")


# ───────────── 시세 ─────────────
def download(tickers: list[str]):
    import yfinance as yf
    for i in range(0, len(tickers), 200):
        chunk = tickers[i:i + 200]
        try:
            data = yf.download(chunk, period="1y", interval="1d", group_by="ticker",
                               auto_adjust=False, threads=True, progress=False)
        except Exception as e:
            print(f"[경고] 시세 다운로드 실패({chunk[0]}…): {e}")
            continue
        if not isinstance(data.columns, pd.MultiIndex):
            data = pd.concat({chunk[0]: data}, axis=1)
        have = set(data.columns.get_level_values(0))
        for t in chunk:
            if t in have:
                yield t, data[t]
        time.sleep(1)


# ───────────── 신호 판별 ─────────────
def rsi(close: pd.Series, n: int) -> pd.Series:
    """와일더 방식 RSI (트레이딩뷰와 같은 계산)"""
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100)


def find_signal(df: pd.DataFrame, today: dt.date | None) -> dict | None:
    df = df.dropna(subset=["Open", "Close", "High", "Low", "Volume"])
    if len(df) < 210 or (today is not None and df.index[-1].date() != today):
        return None
    o, c, h, l, v = df["Open"], df["Close"], df["High"], df["Low"], df["Volume"]
    last = float(c.iloc[-1])
    if last < MIN_PRICE or (c * v).tail(50).mean() < MIN_DOLLAR_VOL:
        return None

    mid = c.rolling(BB_LEN).mean()
    sd = c.rolling(BB_LEN).std(ddof=0)                    # 트레이딩뷰와 같은 방식
    upper, lower = mid + BB_STD * sd, mid - BB_STD * sd
    pctb = (c - lower) / (upper - lower)
    r2, r14 = rsi(c, 2), rsi(c, 14)
    vavg = v.rolling(50).mean().shift(1)                   # 당일 제외 50일 평균
    s200 = float(c.rolling(200).mean().iloc[-1])

    # ③⑦ 상승 추세 종목 + 200일선과의 거리
    above200 = (last / s200 - 1) * 100
    if not (0 < above200 <= MAX_ABOVE_200):
        return None

    # 이탈일 후보: 오늘(반전형) 또는 최근 LOOKBACK일(복귀형)
    for back in range(0, LOOKBACK + 1):
        e = len(df) - 1 - back                              # 이탈일 위치
        # ① 과도한 이탈: 이탈일 종가가 하단 아래, %B < 기준
        if not pctb.iloc[e] < PCTB_MAX:
            continue
        # ② 단기 과매도
        if not (r2.iloc[e] < RSI2_MAX or r14.iloc[e] < RSI14_MAX):
            continue
        # ④ 반등 확인
        if back == 0:
            # 반전형: 오늘 하단을 뚫고 마감했지만 종가가 당일 범위 상단 절반
            rng = h.iloc[-1] - l.iloc[-1]
            if rng <= 0 or (last - l.iloc[-1]) / rng < 0.5:
                continue
            kind = "TAIL"   # 반전형: 오늘 이탈 + 아래꼬리
        else:
            # 복귀형: 이탈 이후 오늘 처음으로 하단 안으로 복귀 마감
            if not last > lower.iloc[-1]:
                continue
            if any(c.iloc[j] > lower.iloc[j] for j in range(e + 1, len(df) - 1)):
                continue
            kind = "BACK"   # 복귀형: 이탈 후 밴드 안 복귀
        # ⑤ 투매 거래량 (이탈일)
        vol_x = float(v.iloc[e] / vavg.iloc[e]) if vavg.iloc[e] else float("nan")
        if REQUIRE_VOL_SPIKE and not vol_x >= VOL_SPIKE:
            continue
        # ⑥ 악재 갭 제외 (이탈일부터 오늘까지 중 가장 큰 갭하락)
        gaps = [(o.iloc[j] / c.iloc[j - 1] - 1) * 100 for j in range(e, len(df))]
        worst_gap = float(min(gaps))
        if REQUIRE_NO_GAP and worst_gap <= -GAP_DOWN_PCT:
            continue

        target = float(mid.iloc[-1])
        stop = float(l.iloc[e:].min())                      # 이탈 이후 최저가
        if target <= last or stop >= last:
            continue                                        # 이미 중심선 회복했거나 손절가가 의미 없음
        rr = (target - last) / (last - stop)
        return {"kind": kind, "pctb": float(pctb.iloc[e]), "rsi2": float(r2.iloc[e]), "rsi14": float(r14.iloc[e]),
                "vol_x": vol_x, "gap": worst_gap, "above200": float(above200),
                "last": last, "target": target, "stop": stop, "rr": float(rr)}
    return None


# ───────────── 기타 ─────────────
def company_name(ticker: str) -> str:
    cache_file = STATE / "names.json"
    try:
        cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    except Exception:
        cache = {}
    if ticker in cache:
        return cache[ticker]
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        name = info.get("shortName") or info.get("longName") or ""
    except Exception:
        name = ""
    cache[ticker] = name
    cache_file.write_text(json.dumps(cache))
    return name


def post(content: str):
    if not WEBHOOK:
        print(content)
        return
    for _ in range(5):
        r = requests.post(WEBHOOK, json={"content": content}, timeout=30)
        if r.status_code == 429:
            time.sleep(float(r.json().get("retry_after", 2)) + 0.5)
            continue
        r.raise_for_status()
        time.sleep(0.6)
        return


def send_table(title: str, rows: list[dict]):
    head = (f"{'TICKER':<7}{'NAME':<14}{'TYPE':<5}{'%B':>6}{'RSI2':>5}{'RSI14':>6}"
            f"{'VOLx':>5}{'GAP%':>6}{'200%':>6}{'LAST':>8}{'TARGET':>8}{'STOP':>8}{'R:R':>5}")
    lines = [
        f"{r['ticker']:<7}{r['name'][:12]:<14}{r['kind']:<5}{r['pctb']:>6.2f}{r['rsi2']:>5.0f}{r['rsi14']:>6.0f}"
        f"{r['vol_x']:>5.1f}{r['gap']:>6.1f}{r['above200']:>6.1f}{r['last']:>8.2f}{r['target']:>8.2f}{r['stop']:>8.2f}{r['rr']:>5.1f}"
        for r in rows
    ]
    first = True
    while lines:
        block, size = [], 0
        while lines and size + len(lines[0]) + 1 < 1700:
            size += len(lines[0]) + 1
            block.append(lines.pop(0))
        header = f"**{title}**\n" if first else ""
        post(f"{header}```\n{head}\n" + "\n".join(block) + "\n```")
        first = False


# ───────────── 실행 ─────────────
def main():
    now = dt.datetime.now(NY)
    if now.weekday() >= 5 and not FORCE:
        print("주말. 종료.")
        return
    today = now.date()

    sent_file = STATE / "sent.json"
    try:
        sent = json.loads(sent_file.read_text()) if sent_file.exists() else {}
    except Exception:
        sent = {}
    cutoff = (today - dt.timedelta(days=REALERT_DAYS)).isoformat()
    sent = {k: d for k, d in sent.items() if d >= cutoff}

    universes = {g.strip(): load_universe(g.strip()) for g in GROUPS if g.strip()}
    meta, group_of = {}, {}
    for g, u in universes.items():
        for row in u.itertuples():
            if row.ticker not in meta:
                meta[row.ticker] = row
                group_of[row.ticker] = g
    print(f"스캔 대상 {len(meta)}종목 " + ", ".join(f"{g}={len(u)}" for g, u in universes.items()))
    if not meta:
        post("⚠️ 볼린저 스캐너: 종목 목록을 불러오지 못했습니다. Actions 로그를 확인하세요.")
        return

    found: dict[str, list] = {}
    scanned = 0
    for ticker, df in download(list(meta)):
        scanned += 1
        s = find_signal(df, None if FORCE else today)
        if not s:
            continue
        if not FORCE:
            if ticker in sent:
                continue
            sent[ticker] = today.isoformat()
        m = meta[ticker]
        found.setdefault(group_of[ticker], []).append(
            {"ticker": ticker, "name": str(m.name) or company_name(ticker), **s})

    total = sum(len(v) for v in found.values())
    print(f"시세 확인 {scanned}종목, 신규 신호 {total}건")
    for g in universes:
        rows = found.get(g)
        if rows:
            rows.sort(key=lambda r: -r["rr"])                # 손익비 좋은 순
            send_table(f"{now:%b} {now.day} | Bollinger Oversold Rebound | {GROUP_LABEL.get(g, g)}", rows)

    if scanned < len(meta) * 0.5:
        post(f"⚠️ 볼린저 스캐너: 시세를 {scanned}/{len(meta)}종목만 받았습니다. 일시적 오류일 수 있습니다.")
    if FORCE:
        post(f"✅ 볼린저 스캐너 테스트 완료: {scanned}종목 확인, 신호 {total}건 (최근 거래일 기준)")
    else:
        sent_file.write_text(json.dumps(sent))


if __name__ == "__main__":
    sys.exit(main())
