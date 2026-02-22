"""
MEXC vs Yahoo Finance Spread Monitor Bot
Version: 7.0.0
- SQLite database — all settings persist across restarts
- Dynamic Yahoo price: regular / pre-market / after-hours
- Market state change notifications
- /check command for real-time debug
- Redesigned alert: higher price always on top
- bid1 for SHORT, ask1 for LONG (no lastPrice)
"""

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("spread_bot")

BOT_TOKEN: str = os.getenv("BOT_TOKEN", "")
if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is missing from .env")

ADMIN_ID: int = 868931721
FETCH_INTERVAL: int = int(os.getenv("FETCH_INTERVAL", "60"))
SPREAD_THRESHOLD: float = float(os.getenv("SPREAD_THRESHOLD", "0.5"))
ALERT_STEP: float = float(os.getenv("ALERT_STEP", "0.2"))

TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"
MEXC_TICKER_URL = "https://futures.mexc.com/api/v1/contract/ticker"

MEXC_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://futures.mexc.com/",
    "Origin": "https://futures.mexc.com",
}

# ---------------------------------------------------------------------------
# Symbol map
# ---------------------------------------------------------------------------

SYMBOL_MAP: dict[str, tuple[str, str]] = {
    "TSLASTOCK_USDT":  ("TSLA",  "TSLA"),
    "NVDASTOCK_USDT":  ("NVDA",  "NVDA"),
    "AAPLSTOCK_USDT":  ("AAPL",  "AAPL"),
    "MSFTSTOCK_USDT":  ("MSFT",  "MSFT"),
    "AMZNSTOCK_USDT":  ("AMZN",  "AMZN"),
    "METASTOCK_USDT":  ("META",  "META"),
    "AMDSTOCK_USDT":   ("AMD",   "AMD"),
    "GOOGLSTOCK_USDT": ("GOOGL", "GOOGL"),
    "COINBASE_USDT":   ("COIN",  "COIN"),
    "CVNASTOCK_USDT":  ("CVNA",  "CVNA"),
    "AMATSTOCK_USDT":  ("AMAT",  "AMAT"),
    "QCOMSTOCK_USDT":  ("QCOM",  "QCOM"),
    "CRMSTOCK_USDT":   ("CRM",   "CRM"),
    "SHOPSTOCK_USDT":  ("SHOP",  "SHOP"),
    "VZSTOCK_USDT":    ("VZ",    "VZ"),
    "INTCSTOCK_USDT":  ("INTC",  "INTC"),
    "QQQSTOCK_USDT":   ("QQQ",   "QQQ"),
    "CSCOSTOCK_USDT":  ("CSCO",  "CSCO"),
    "JNJSTOCK_USDT":   ("JNJ",   "JNJ"),
    "FUTUSTOCK_USDT":  ("FUTU",  "FUTU"),
    "XOMSTOCK_USDT":   ("XOM",   "XOM"),
    "RDDTSTOCK_USDT":  ("RDDT",  "RDDT"),
    "SPOTSTOCK_USDT":  ("SPOT",  "SPOT"),
    "NFLXSTOCK_USDT":  ("NFLX",  "NFLX"),
    "SMCISTOCK_USDT":  ("SMCI",  "SMCI"),
    "ORCLSTOCK_USDT":  ("ORCL",  "ORCL"),
    "ASMLSTOCK_USDT":  ("ASML",  "ASML"),
    "ACNSTOCK_USDT":   ("ACN",   "ACN"),
    "UNHSTOCK_USDT":   ("UNH",   "UNH"),
    "NOWSTOCK_USDT":   ("NOW",   "NOW"),
    "LLYSTOCK_USDT":   ("LLY",   "LLY"),
    "LRCXSTOCK_USDT":  ("LRCX",  "LRCX"),
    "IBMSTOCK_USDT":   ("IBM",   "IBM"),
    "COSTSTOCK_USDT":  ("COST",  "COST"),
    "JDSTOCK_USDT":    ("JD",    "JD"),
    "JPMSTOCK_USDT":   ("JPM",   "JPM"),
    "GSSTOCK_USDT":    ("GS",    "GS"),
    "MASTOCK_USDT":    ("MA",    "MA"),
    "KOSTOCK_USDT":    ("KO",    "KO"),
    "WMTSTOCK_USDT":   ("WMT",   "WMT"),
    "GESTOCK_USDT":    ("GE",    "GE"),
    "MUSTOCK_USDT":    ("MU",    "MU"),
    "VSTOCK_USDT":     ("V",     "V"),
    "NKESTOCK_USDT":   ("NKE",   "NKE"),
    "PEPSTOCK_USDT":   ("PEP",   "PEP"),
    "BASTOCK_USDT":    ("BA",    "BA"),
    "ROBINHOOD_USDT":  ("HOOD",  "HOOD"),
    "FIGSTOCK_USDT":   ("FIG",   "FIG"),
}

DISPLAY_TO_MEXC: dict[str, str] = {v[1].upper(): k for k, v in SYMBOL_MAP.items()}
ALL_YAHOO_TICKERS = list({v[0] for v in SYMBOL_MAP.values()})

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

MARKET_REGULAR = "Regular Market"
MARKET_PRE     = "Pre-Market"
MARKET_AFTER   = "After-Hours"
MARKET_CLOSED  = "Closed"


@dataclass
class YahooSnapshot:
    ticker:        str
    active_price:  float
    active_state:  str
    regular_price: float
    pre_price:     Optional[float] = None
    post_price:    Optional[float] = None
    updated_at:    float = field(default_factory=time.time)


@dataclass
class MexcSnapshot:
    symbol:     str
    bid1:       float
    ask1:       float
    last:       float
    fair:       float
    updated_at: float = field(default_factory=time.time)


@dataclass
class SpreadRecord:
    display:          str
    mexc_sym:         str
    yahoo_ticker:     str
    direction:        str
    actionable_price: float
    yahoo_price:      float
    spread_pct:       float
    bid1:             float
    ask1:             float
    fair:             float
    market_state:     str
    updated_at:       float = field(default_factory=time.time)



# ---------------------------------------------------------------------------
# Database — Upstash Redis (via upstash-redis HTTP client)
# ---------------------------------------------------------------------------
# Uses upstash_redis library which connects over HTTPS — no SSL issues.
# Requires in .env:
#   UPSTASH_REDIS_REST_URL   = https://xxx-xxx.upstash.io
#   UPSTASH_REDIS_REST_TOKEN = AXxx...
#
# Keys used:
#   settings:global_threshold      JSON float
#   settings:alert_step            JSON float
#   settings:muted_until           JSON float
#   settings:muteall_active        JSON bool
#   settings:muteall_exceptions    JSON list
#   thresholds:<SYMBOL>            float as string
#   subscriber:<chat_id>           JSON {name, username, joined_at}
# ---------------------------------------------------------------------------

from upstash_redis import Redis as UpstashRedis

_redis: Optional[UpstashRedis] = None


def init_db() -> None:
    global _redis
    url   = os.getenv("UPSTASH_REDIS_REST_URL", "")
    token = os.getenv("UPSTASH_REDIS_REST_TOKEN", "")
    if not url or not token:
        raise RuntimeError(
            "UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN are required in .env"
        )
    _redis = UpstashRedis(url=url, token=token)
    # Ping to verify connection
    _redis.ping()
    logger.info("Upstash Redis connected OK — %s", url)


def _r() -> UpstashRedis:
    if _redis is None:
        raise RuntimeError("Redis not initialised — call init_db() first")
    return _redis


# --- settings ---

def _db_get(key: str, default):
    try:
        val = _r().get(f"settings:{key}")
        if val is not None:
            return json.loads(val) if isinstance(val, str) else val
    except Exception as exc:
        logger.error("redis_get %s: %s", key, exc)
    return default


def _db_set(key: str, value) -> None:
    try:
        _r().set(f"settings:{key}", json.dumps(value))
    except Exception as exc:
        logger.error("redis_set %s: %s", key, exc)


def db_save_all_settings() -> None:
    _db_set("global_threshold",   global_threshold)
    _db_set("alert_step",         alert_step)
    _db_set("muted_until",        muted_until)
    _db_set("muteall_active",     muteall_active)
    _db_set("muteall_exceptions", list(muteall_exceptions))


def db_load_all_settings() -> None:
    global global_threshold, alert_step, muted_until, muteall_active, muteall_exceptions
    global symbol_thresholds, subscribers
    global_threshold   = float(_db_get("global_threshold",   SPREAD_THRESHOLD))
    alert_step         = float(_db_get("alert_step",         ALERT_STEP))
    # Safety: never allow alert_step of 0 — it causes every cycle to alert
    if alert_step <= 0:
        alert_step = ALERT_STEP
        logger.warning("alert_step was 0 or negative — reset to default %.2f%%", ALERT_STEP)
    muted_until        = float(_db_get("muted_until",        0.0))
    muteall_active     = bool(_db_get("muteall_active",      False))
    muteall_exceptions = set(_db_get("muteall_exceptions",   []) or [])
    symbol_thresholds  = db_load_symbol_thresholds()
    subscribers        = db_load_subscribers()
    logger.info(
        "Redis loaded — threshold=%.2f%% step=%.2f%% muted=%s muteall=%s "
        "exceptions=%s custom_th=%d subs=%d",
        global_threshold, alert_step,
        "yes" if muted_until > time.time() else "no",
        muteall_active, list(muteall_exceptions),
        len(symbol_thresholds), len(subscribers),
    )


# --- symbol thresholds ---

def db_load_symbol_thresholds() -> dict[str, float]:
    try:
        keys = _r().keys("thresholds:*")
        if not keys:
            return {}
        result = {}
        for key in keys:
            symbol = key.split(":", 1)[1]
            val    = _r().get(key)
            if val is not None:
                result[symbol] = float(val)
        return result
    except Exception as exc:
        logger.error("redis_load_thresholds: %s", exc)
        return {}


def db_save_symbol_threshold(symbol: str, threshold: float) -> None:
    try:
        _r().set(f"thresholds:{symbol}", str(threshold))
    except Exception as exc:
        logger.error("redis_save_threshold %s: %s", symbol, exc)


def db_delete_symbol_threshold(symbol: str) -> None:
    try:
        _r().delete(f"thresholds:{symbol}")
    except Exception as exc:
        logger.error("redis_delete_threshold %s: %s", symbol, exc)


# --- subscribers ---

def db_load_subscribers() -> dict[int, dict]:
    try:
        keys = _r().keys("subscriber:*")
        if not keys:
            return {}
        result = {}
        for key in keys:
            chat_id = int(key.split(":", 1)[1])
            val     = _r().get(key)
            if val:
                data = json.loads(val) if isinstance(val, str) else val
                result[chat_id] = data
        return result
    except Exception as exc:
        logger.error("redis_load_subscribers: %s", exc)
        return {}


def db_save_subscriber(chat_id: int, info: dict) -> None:
    try:
        _r().set(f"subscriber:{chat_id}", json.dumps(info))
    except Exception as exc:
        logger.error("redis_save_subscriber %d: %s", chat_id, exc)


def db_delete_subscriber(chat_id: int) -> None:
    try:
        _r().delete(f"subscriber:{chat_id}")
    except Exception as exc:
        logger.error("redis_delete_subscriber %d: %s", chat_id, exc)


# ---------------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------------

yahoo_cache:       dict[str, YahooSnapshot] = {}
mexc_cache:        dict[str, MexcSnapshot]  = {}
spread_cache:      dict[str, SpreadRecord]  = {}
peak_spread:       dict[str, float]         = {}
last_market_state: dict[str, str]           = {}

state_lock = threading.Lock()

muted_until:        float    = 0.0
muteall_active:     bool     = False
muteall_exceptions: set[str] = set()

global_threshold:  float          = SPREAD_THRESHOLD
alert_step:        float          = ALERT_STEP
symbol_thresholds: dict[str, float] = {}

subscribers:      dict[int, dict] = {}
subscribers_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def is_muted() -> bool:
    return time.time() < muted_until


def mute_remaining() -> str:
    r = muted_until - time.time()
    if r <= 0:
        return ""
    h, m = int(r // 3600), int((r % 3600) // 60)
    return f"{h}h {m}m" if h else f"{m}m {int(r%60)}s"


def get_threshold(mexc_sym: str) -> float:
    display = SYMBOL_MAP.get(mexc_sym, ("", ""))[1].upper()
    return symbol_thresholds.get(display, global_threshold)


def resolve_symbol(user_input: str) -> Optional[str]:
    s = user_input.upper().strip()
    if s in SYMBOL_MAP:
        return s
    if s in DISPLAY_TO_MEXC:
        return DISPLAY_TO_MEXC[s]
    return None


def is_symbol_muted(mexc_sym: str) -> bool:
    if is_muted():
        return True
    if muteall_active:
        display = SYMBOL_MAP.get(mexc_sym, ("", ""))[1].upper()
        return display not in muteall_exceptions
    return False


# ---------------------------------------------------------------------------
# Telegram helpers
# ---------------------------------------------------------------------------


def tg_send(
    chat_id: int,
    text: str,
    keyboard: Optional[dict] = None,
    message_id: Optional[int] = None,
) -> Optional[int]:
    if message_id:
        payload = {
            "chat_id": chat_id, "message_id": message_id,
            "text": text, "parse_mode": "HTML",
        }
        if keyboard:
            payload["reply_markup"] = keyboard
        try:
            resp = requests.post(
                f"{TELEGRAM_API}/editMessageText", json=payload, timeout=10
            ).json()
            return resp.get("result", {}).get("message_id")
        except Exception as exc:
            logger.error("Edit error: %s", exc)
        return None

    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
    if keyboard:
        payload["reply_markup"] = keyboard
    try:
        resp = requests.post(
            f"{TELEGRAM_API}/sendMessage", json=payload, timeout=10
        ).json()
        return resp.get("result", {}).get("message_id")
    except Exception as exc:
        logger.error("Send error: %s", exc)
    return None


def tg_answer_callback(callback_id: str, text: str = "") -> None:
    try:
        requests.post(
            f"{TELEGRAM_API}/answerCallbackQuery",
            json={"callback_query_id": callback_id, "text": text},
            timeout=5,
        )
    except Exception:
        pass


def tg_get_updates(offset: int) -> list[dict]:
    try:
        resp = requests.get(
            f"{TELEGRAM_API}/getUpdates",
            params={"offset": offset, "timeout": 20},
            timeout=25,
        )
        return resp.json().get("result", [])
    except Exception as exc:
        logger.error("getUpdates error: %s", exc)
        return []


def broadcast(text: str, keyboard: Optional[dict] = None) -> None:
    dead = []
    with subscribers_lock:
        targets = list(subscribers.keys())
    for chat_id in targets:
        try:
            resp = requests.post(
                f"{TELEGRAM_API}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": text,
                    "parse_mode": "HTML",
                    **({"reply_markup": keyboard} if keyboard else {}),
                },
                timeout=10,
            ).json()
            if not resp.get("ok"):
                err = resp.get("description", "")
                if any(x in err for x in ("blocked", "not found", "deactivated")):
                    dead.append(chat_id)
        except Exception:
            pass
        time.sleep(0.05)
    if dead:
        with subscribers_lock:
            for cid in dead:
                subscribers.pop(cid, None)
                db_delete_subscriber(cid)
        logger.info("Removed %d dead subscribers", len(dead))


# ---------------------------------------------------------------------------
# Yahoo Finance — dynamic price parsing
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Finnhub — price fetching (replaces Yahoo Finance)
# ---------------------------------------------------------------------------
# Yahoo Finance blocks Railway IPs. Finnhub is free (60 req/min),
# reliable, and supports pre/post market prices.
#
# Free API key: https://finnhub.io (sign up, copy key to .env)
# Env var: FINNHUB_TOKEN=your_key_here
#
# Finnhub /quote returns:
#   c  = current price (regular market)
#   pc = previous close
# Finnhub doesn't have pre/post market on free tier, but we can detect
# extended hours by comparing current price to previous close + timestamp.
#
# For extended hours detection we use the marketStatus field from
# Finnhub's market status endpoint, plus the quote timestamp.
# ---------------------------------------------------------------------------

FINNHUB_TOKEN: str = os.getenv("FINNHUB_TOKEN", "")
FINNHUB_BASE  = "https://finnhub.io/api/v1"


def _finnhub_get(path: str, params: dict) -> dict:
    """Make a Finnhub API call. Raises on error."""
    params["token"] = FINNHUB_TOKEN
    resp = requests.get(f"{FINNHUB_BASE}{path}", params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def _parse_finnhub_quote(ticker: str, data: dict) -> Optional[YahooSnapshot]:
    """
    Parse Finnhub /quote response into YahooSnapshot.

    Finnhub /quote fields:
      c  = current price
      pc = previous close
      t  = timestamp of last trade

    Extended hours detection:
      Finnhub free tier doesn't give separate pre/post prices.
      We detect market state by checking if the quote timestamp
      falls outside regular NYSE hours (9:30-16:00 ET Mon-Fri).
    """
    try:
        price = float(data.get("c") or 0)
        if price <= 0:
            return None

        # Determine market state from timestamp
        import datetime, zoneinfo
        ts = data.get("t", 0)
        et = zoneinfo.ZoneInfo("America/New_York")
        dt = datetime.datetime.fromtimestamp(ts, tz=et) if ts else datetime.datetime.now(tz=et)

        market_open  = dt.replace(hour=9,  minute=30, second=0, microsecond=0)
        market_close = dt.replace(hour=16, minute=0,  second=0, microsecond=0)
        is_weekday   = dt.weekday() < 5  # Mon-Fri

        if not is_weekday or dt >= market_close:
            active_state = MARKET_AFTER
        elif dt < market_open:
            active_state = MARKET_PRE
        else:
            active_state = MARKET_REGULAR

        # For after/pre hours, Finnhub `c` is already the extended price
        # (it reflects the last trade regardless of session)
        regular_price = float(data.get("pc") or price)  # use prev close as "regular"
        if active_state == MARKET_REGULAR:
            regular_price = price

        logger.info(
            "Finnhub %-6s | %-14s active=%8.4f",
            ticker, active_state, price,
        )

        return YahooSnapshot(
            ticker=ticker,
            active_price=price,
            active_state=active_state,
            regular_price=regular_price,
            pre_price=price  if active_state == MARKET_PRE   else None,
            post_price=price if active_state == MARKET_AFTER else None,
        )
    except Exception as exc:
        logger.error("Finnhub parse %s: %s", ticker, exc)
        return None


def parse_yahoo_snapshot(ticker: str) -> Optional[YahooSnapshot]:
    """Single-ticker fetch via Finnhub."""
    try:
        data = _finnhub_get("/quote", {"symbol": ticker})
        return _parse_finnhub_quote(ticker, data)
    except Exception as exc:
        logger.error("Finnhub %s error: %s", ticker, exc)
        return None


def fetch_all_yahoo() -> dict[str, YahooSnapshot]:
    """
    Fetch all tickers from Finnhub in parallel.
    Free tier: 60 req/min — 48 tickers well within limit.
    Uses 8 workers to stay safely under rate limit.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time as _time

    if not FINNHUB_TOKEN:
        logger.error("FINNHUB_TOKEN not set — cannot fetch prices")
        return {}

    result:      dict[str, YahooSnapshot] = {}
    result_lock = threading.Lock()

    def _fetch_one(ticker: str) -> None:
        snap = parse_yahoo_snapshot(ticker)
        if snap:
            with result_lock:
                result[ticker] = snap

    # 8 workers, small delay between batches to stay under 60 req/min
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_fetch_one, t): t for t in ALL_YAHOO_TICKERS}
        for fut in as_completed(futures):
            exc = fut.exception()
            if exc:
                logger.warning("Finnhub worker error: %s", exc)

    logger.info("Finnhub: got %d/%d tickers", len(result), len(ALL_YAHOO_TICKERS))
    return result

# ---------------------------------------------------------------------------
# MEXC — bid1 / ask1 / fair price
# ---------------------------------------------------------------------------


def fetch_mexc_data() -> dict[str, MexcSnapshot]:
    try:
        resp    = requests.get(MEXC_TICKER_URL, headers=MEXC_HEADERS, timeout=15)
        payload = resp.json()
        if not payload.get("success"):
            return {}

        result: dict[str, MexcSnapshot] = {}
        for t in payload.get("data", []):
            sym = t.get("symbol", "")
            if sym not in SYMBOL_MAP:
                continue
            try:
                bid1 = float(t.get("bid1") or t.get("bidPrice") or 0)
                ask1 = float(t.get("ask1") or t.get("askPrice") or 0)
                last = float(t.get("lastPrice") or t.get("last") or 0)
                fair = float(
                    t.get("indexPrice") or t.get("fairPrice")
                    or t.get("markPrice") or t.get("fair_price")
                    or t.get("index_price") or last
                )
                if bid1 > 0 and ask1 > 0:
                    result[sym] = MexcSnapshot(
                        symbol=sym, bid1=bid1, ask1=ask1, last=last, fair=fair
                    )
            except (TypeError, ValueError):
                pass

        logger.info("MEXC: %d symbols with bid1/ask1", len(result))
        return result

    except Exception as exc:
        logger.error("MEXC fetch error: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Spread calculation
# ---------------------------------------------------------------------------


def calculate_actionable_spread(
    bid1: float,
    ask1: float,
    yahoo_price: float,
) -> Optional[tuple[str, float, float]]:
    """
    SHORT: sell into bid1  →  spread = (bid1 - yahoo) / yahoo * 100
    LONG:  buy at ask1     →  spread = (yahoo - ask1) / yahoo * 100
    Returns (direction, actionable_price, spread_pct) or None.
    """
    short_spread = (bid1 - yahoo_price) / yahoo_price * 100
    if short_spread > 0:
        return "SHORT", bid1, short_spread

    long_spread = (yahoo_price - ask1) / yahoo_price * 100
    if long_spread > 0:
        return "LONG", ask1, long_spread

    return None


# ---------------------------------------------------------------------------
# Alert message builder — higher price ALWAYS on top
# ---------------------------------------------------------------------------


def build_alert_keyboard(mexc_sym: str, yahoo_ticker: str) -> dict:
    return {
        "inline_keyboard": [[
            {"text": "📊 MEXC Futures",
             "url": f"https://futures.mexc.com/exchange/{mexc_sym}"},
            {"text": "📈 Yahoo Finance",
             "url": f"https://finance.yahoo.com/quote/{yahoo_ticker}"},
        ]]
    }


def build_alert_message(rec: SpreadRecord, is_growing: bool = False) -> str:
    growing_tag = "  ⬆️ <b>GROWING</b>" if is_growing else ""

    if rec.direction == "SHORT":
        # MEXC bid1 > Yahoo — MEXC on top
        signal_icon  = "🔴"
        signal_label = "SHORT"
        top_label    = f"MEXC Futures (bid1)"
        top_price    = rec.actionable_price
        bot_label    = f"Yahoo Finance ({rec.market_state})"
        bot_price    = rec.yahoo_price
    else:
        # Yahoo > MEXC ask1 — Yahoo on top
        signal_icon  = "🟢"
        signal_label = "LONG"
        top_label    = f"Yahoo Finance ({rec.market_state})"
        top_price    = rec.yahoo_price
        bot_label    = f"MEXC Futures (ask1)"
        bot_price    = rec.actionable_price

    return (
        f"{signal_icon} <b>{signal_label} {rec.display}</b> | "
        f"<b>{rec.spread_pct:.2f}%</b> Spread{growing_tag}\n\n"
        f"📌 <b>{top_label}:</b>  <code>${top_price:.4f}</code>\n"
        f"📌 <b>{bot_label}:</b>  <code>${bot_price:.4f}</code>\n\n"
        f"📐 <b>MEXC Fair Price:</b>  <code>${rec.fair:.4f}</code>\n"
        f"💰 <b>Diff:</b>  <code>${abs(rec.actionable_price - rec.yahoo_price):.4f}</code>"
    )


def send_spread_alert(
    rec: SpreadRecord,
    is_growing: bool = False,
    chat_id: Optional[int] = None,
) -> None:
    text     = build_alert_message(rec, is_growing)
    keyboard = build_alert_keyboard(rec.mexc_sym, rec.yahoo_ticker)
    if chat_id:
        tg_send(chat_id, text, keyboard)
    else:
        broadcast(text, keyboard)
    logger.info(
        "Alert %s %s spread=%.2f%% growing=%s state=%s",
        rec.direction, rec.display, rec.spread_pct, is_growing, rec.market_state,
    )


# ---------------------------------------------------------------------------
# Market state change notifications
# ---------------------------------------------------------------------------


def check_and_notify_state_changes(snapshots: dict[str, YahooSnapshot]) -> None:
    for ticker, snap in snapshots.items():
        prev = last_market_state.get(ticker)
        if prev is None:
            last_market_state[ticker] = snap.active_state
            continue
        if prev != snap.active_state:
            logger.info("State change %s: %s → %s", ticker, prev, snap.active_state)
            last_market_state[ticker] = snap.active_state

            if snap.active_state == MARKET_PRE:
                icon  = "🌅"
                label = "Pre-Market trading has begun"
            elif snap.active_state == MARKET_AFTER:
                icon  = "🌙"
                label = "After-Hours trading has begun"
            elif snap.active_state == MARKET_REGULAR:
                icon  = "🔔"
                label = "Regular Market is now open"
            else:
                icon  = "🔕"
                label = "Market closed"

            msg = (
                f"{icon} <b>Market State Changed: {ticker}</b>\n\n"
                f"<b>{prev}</b> → <b>{snap.active_state}</b>\n"
                f"{label}\n\n"
                f"📌 Regular close: <code>${snap.regular_price:.4f}</code>\n"
                f"📡 Now tracking:  <code>${snap.active_price:.4f}</code>"
            )
            threading.Thread(target=broadcast, args=(msg,), daemon=True).start()


# ---------------------------------------------------------------------------
# /check command
# ---------------------------------------------------------------------------


def handle_check(chat_id: int, args: list[str]) -> None:
    if not args:
        # Summary of all active spreads
        if not spread_cache:
            tg_send(chat_id, "⏳ No data yet — wait for first fetch cycle.")
            return

        active = sorted(
            [r for r in spread_cache.values() if r.spread_pct >= get_threshold(r.mexc_sym)],
            key=lambda r: r.spread_pct, reverse=True,
        )
        if not active:
            tg_send(
                chat_id,
                f"✅ No active spreads above threshold ({global_threshold}%) right now.\n"
                "Use <code>/check TSLA</code> to inspect a specific symbol."
            )
            return

        lines = []
        for r in active:
            si = {"Regular Market": "🟡", "Pre-Market": "🌅", "After-Hours": "🌙"}.get(r.market_state, "⚫")
            di = "🔴" if r.direction == "SHORT" else "🟢"
            lines.append(f"{di} <b>{r.display}</b> {r.spread_pct:.2f}% [{si} {r.market_state}]")

        tg_send(
            chat_id,
            f"🔍 <b>Active Spreads ({len(active)})</b>\n\n"
            + "\n".join(lines)
            + "\n\nUse <code>/check SYMBOL</code> for full details."
        )
        return

    # Detailed single-symbol check
    mexc_sym = resolve_symbol(args[0].upper())
    if not mexc_sym:
        tg_send(chat_id, f"❌ Symbol <code>{args[0].upper()}</code> not found.")
        return

    yahoo_ticker, display = SYMBOL_MAP[mexc_sym]
    yahoo_snap = yahoo_cache.get(yahoo_ticker)
    mexc_snap  = mexc_cache.get(mexc_sym)
    spread_rec = spread_cache.get(mexc_sym)

    # Yahoo section
    if yahoo_snap:
        state_label = {
            MARKET_REGULAR: "🟡 Regular Market",
            MARKET_PRE:     "🌅 Pre-Market",
            MARKET_AFTER:   "🌙 After-Hours",
            MARKET_CLOSED:  "🔕 Closed",
        }.get(yahoo_snap.active_state, yahoo_snap.active_state)

        yahoo_lines = (
            f"<b>Yahoo Finance</b>\n"
            f"  State:          <b>{state_label}</b>\n"
            f"  Active price:   <code>${yahoo_snap.active_price:.4f}</code>  ← used for spread\n"
            f"  Regular close:  <code>${yahoo_snap.regular_price:.4f}</code>\n"
        )
        if yahoo_snap.pre_price and yahoo_snap.pre_price != yahoo_snap.regular_price:
            yahoo_lines += f"  Pre-market:     <code>${yahoo_snap.pre_price:.4f}</code>\n"
        if yahoo_snap.post_price and yahoo_snap.post_price != yahoo_snap.regular_price:
            yahoo_lines += f"  After-hours:    <code>${yahoo_snap.post_price:.4f}</code>\n"
    else:
        yahoo_lines = "<b>Yahoo Finance</b>\n  ⏳ No data yet\n"

    # MEXC section
    if mexc_snap:
        mexc_lines = (
            f"\n<b>MEXC Futures</b>\n"
            f"  bid1 (sell to): <code>${mexc_snap.bid1:.4f}</code>  ← used for SHORT\n"
            f"  ask1 (buy at):  <code>${mexc_snap.ask1:.4f}</code>  ← used for LONG\n"
            f"  Fair price:     <code>${mexc_snap.fair:.4f}</code>\n"
            f"  Last price:     <code>${mexc_snap.last:.4f}</code>  (ignored)\n"
        )
    else:
        mexc_lines = "\n<b>MEXC Futures</b>\n  ⏳ No data yet\n"

    # Spread section
    if spread_rec:
        th       = get_threshold(mexc_sym)
        above    = "🚨 YES" if spread_rec.spread_pct >= th else "✅ NO"
        dir_icon = "🔴 SHORT" if spread_rec.direction == "SHORT" else "🟢 LONG"
        if spread_rec.direction == "SHORT" and mexc_snap and yahoo_snap:
            calc = f"(bid1 ${mexc_snap.bid1:.4f} − Yahoo ${yahoo_snap.active_price:.4f}) / Yahoo × 100"
        elif spread_rec.direction == "LONG" and mexc_snap and yahoo_snap:
            calc = f"(Yahoo ${yahoo_snap.active_price:.4f} − ask1 ${mexc_snap.ask1:.4f}) / Yahoo × 100"
        else:
            calc = "N/A"
        spread_lines = (
            f"\n<b>Actionable Spread</b>\n"
            f"  Direction:    <b>{dir_icon}</b>\n"
            f"  Spread:       <b>{spread_rec.spread_pct:.4f}%</b>\n"
            f"  Threshold:    {th}%\n"
            f"  Above alert:  {above}\n"
            f"  Formula:      <code>{calc}</code>\n"
        )
    else:
        spread_lines = "\n<b>Actionable Spread</b>\n  No actionable spread right now.\n"

    age = int(time.time() - (yahoo_snap.updated_at if yahoo_snap else time.time()))
    tg_send(
        chat_id,
        f"🔍 <b>Check: {display}</b>\n\n"
        + yahoo_lines + mexc_lines + spread_lines
        + f"\n<i>Data age: {age}s</i>"
    )


# ---------------------------------------------------------------------------
# Admin UI
# ---------------------------------------------------------------------------


def build_admin_keyboard() -> dict:
    muteall_label = "🔕 MuteAll ON" if muteall_active else "🔔 MuteAll OFF"
    mute_label    = f"🔇 ({mute_remaining()})" if is_muted() else "🔊 Unmuted"
    return {
        "inline_keyboard": [
            [
                {"text": muteall_label, "callback_data": "toggle_muteall"},
                {"text": mute_label,    "callback_data": "show_mute"},
            ],
            [
                {"text": "📊 Spreads",     "callback_data": "view_spreads"},
                {"text": "⚙️ Thresholds",  "callback_data": "view_thresholds"},
            ],
            [
                {"text": "👥 Subscribers", "callback_data": "view_subscribers"},
                {"text": "🔄 Refresh",     "callback_data": "refresh_admin"},
            ],
            [
                {"text": f"📶 Global: {global_threshold}%", "callback_data": "show_global"},
                {"text": f"📶 Step: {alert_step}%",         "callback_data": "show_step"},
            ],
        ]
    }


def build_admin_text() -> str:
    above  = sum(1 for s, r in spread_cache.items() if r.spread_pct >= get_threshold(s))
    shorts = sum(1 for r in spread_cache.values() if r.direction == "SHORT")
    longs  = sum(1 for r in spread_cache.values() if r.direction == "LONG")
    states: dict[str, int] = {}
    for snap in yahoo_cache.values():
        states[snap.active_state] = states.get(snap.active_state, 0) + 1
    state_str = "  ".join(f"{k}:{v}" for k, v in states.items())
    muteall_text = ""
    if muteall_active:
        exc_list = ", ".join(sorted(muteall_exceptions)) if muteall_exceptions else "none"
        muteall_text = f"\n🔕 MuteAll ON — exceptions: <b>{exc_list}</b>"
    mute_text = f"\n🔇 Muted: <b>{mute_remaining()}</b>" if is_muted() else ""
    with subscribers_lock:
        sub_count = len(subscribers)
    return (
        f"🎛 <b>Admin Control Panel</b>\n\n"
        f"📡 Symbols tracked:    <b>{len(SYMBOL_MAP)}</b>\n"
        f"📊 With live data:     <b>{len(spread_cache)}</b>  (<i>/missing for details</i>)\n"
        f"🚨 Above threshold:    <b>{above}</b>\n"
        f"🔴 SHORT opps:         <b>{shorts}</b>\n"
        f"🟢 LONG opps:          <b>{longs}</b>\n"
        f"📶 Global threshold:   <b>±{global_threshold}%</b>\n"
        f"📶 Alert step:         <b>{alert_step}%</b>\n"
        f"⚙️ Custom thresholds:  <b>{len(symbol_thresholds)}</b>\n"
        f"👥 Subscribers:        <b>{sub_count}</b>\n"
        f"🕐 Market states:      {state_str}"
        f"{muteall_text}{mute_text}\n\n"
        f"<i>Updated: {time.strftime('%H:%M:%S')}</i>"
    )


def send_admin_panel(chat_id: int, message_id: Optional[int] = None) -> None:
    tg_send(chat_id, build_admin_text(), build_admin_keyboard(), message_id=message_id)


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def handle_start(chat_id: int, user: dict) -> None:
    name     = user.get("first_name", "User")
    username = user.get("username", "")
    info = {
        "name":      name,
        "username":  f"@{username}" if username else "no username",
        "joined_at": time.time(),
    }
    with subscribers_lock:
        already = chat_id in subscribers
        if not already:
            info["joined_at"] = time.time()
        else:
            info["joined_at"] = subscribers[chat_id].get("joined_at", time.time())
        subscribers[chat_id] = info
    db_save_subscriber(chat_id, info)
    verb = "You're already subscribed" if already else "You're now subscribed"
    tg_send(
        chat_id,
        f"👋 <b>MEXC vs Yahoo Spread Monitor</b>\n\n"
        f"✅ {verb}.\n\n"
        f"📡 Monitoring <b>{len(SYMBOL_MAP)}</b> symbols\n"
        f"🚨 Threshold: <b>±{global_threshold}%</b>\n"
        f"🔄 Interval: <b>{FETCH_INTERVAL}s</b>\n\n"
        "<b>Commands:</b>\n"
        "/prices — live spread table\n"
        "/check — debug current prices\n"
        "/check TSLA — debug one symbol\n"
        "/status — bot status\n"
        "/stop — unsubscribe\n"
        "/help — all commands\n"
    )
    if not already:
        with subscribers_lock:
            total = len(subscribers)
        tg_send(
            ADMIN_ID,
            f"🆕 <b>New subscriber!</b>\n"
            f"👤 {name} @{username or 'none'} | <code>{chat_id}</code>\n"
            f"👥 Total: <b>{total}</b>",
        )


def handle_stop(chat_id: int) -> None:
    with subscribers_lock:
        if chat_id not in subscribers:
            tg_send(chat_id, "ℹ️ You are not subscribed.")
            return
        name  = subscribers[chat_id]["name"]
        subscribers.pop(chat_id)
        total = len(subscribers)
    db_delete_subscriber(chat_id)
    tg_send(chat_id, "✅ Unsubscribed. Send /start to resubscribe.")
    tg_send(ADMIN_ID, f"👋 <b>Left:</b> {name} | <code>{chat_id}</code>\n👥 Remaining: <b>{total}</b>")


def handle_prices(chat_id: int) -> None:
    """
    Show ALL 48 symbols — regardless of whether they have a spread or not.
    Grouped into 3 sections:
      1. Actionable spreads (sorted by spread % descending)
      2. No spread / price inside bid-ask (sorted alphabetically)
      3. No data (MEXC or Yahoo failed)
    """
    lines = []

    # ── Section 1: actionable spreads ────────────────────────────────────────
    actionable = sorted(
        [r for r in spread_cache.values() if r.direction != "NONE"],
        key=lambda r: r.spread_pct, reverse=True
    )
    for r in actionable:
        th         = get_threshold(r.mexc_sym)
        alert_tag  = "🚨" if r.spread_pct >= th else ("🟡" if r.spread_pct >= th * 0.5 else "⚪")
        dir_icon   = "🔴" if r.direction == "SHORT" else "🟢"
        state_icon = {"Pre-Market": "🌅", "After-Hours": "🌙"}.get(r.market_state, "")
        custom     = "⚙️" if SYMBOL_MAP[r.mexc_sym][1].upper() in symbol_thresholds else ""
        lines.append(
            f"{alert_tag}{dir_icon}{state_icon}{custom} <b>{r.display}</b>: "
            f"{r.spread_pct:+.2f}%  (b:{r.bid1:.2f} a:{r.ask1:.2f} Y:{r.yahoo_price:.2f})"
        )

    # ── Section 2: no spread (price is inside bid/ask, no trade possible) ────
    no_spread = sorted(
        [r for r in spread_cache.values() if r.direction == "NONE"],
        key=lambda r: r.display
    )
    if no_spread:
        lines.append("")
        lines.append("— <i>No spread (price inside bid/ask)</i> —")
        for r in no_spread:
            state_icon = {"Pre-Market": "🌅", "After-Hours": "🌙"}.get(r.market_state, "")
            custom     = "⚙️" if SYMBOL_MAP[r.mexc_sym][1].upper() in symbol_thresholds else ""
            lines.append(
                f"⚫{state_icon}{custom} <b>{r.display}</b>: "
                f"(b:{r.bid1:.2f} a:{r.ask1:.2f} Y:{r.yahoo_price:.2f})"
            )

    # ── Section 3: no data at all ─────────────────────────────────────────────
    shown_syms = set(spread_cache.keys())
    missing = sorted(
        [(display, mexc_sym) for mexc_sym, (_, display) in SYMBOL_MAP.items()
         if mexc_sym not in shown_syms],
        key=lambda x: x[0]
    )
    if missing:
        lines.append("")
        lines.append("— <i>No data available</i> —")
        for display, mexc_sym in missing:
            yahoo_ticker = SYMBOL_MAP[mexc_sym][0]
            has_mexc  = "✅" if mexc_sym      in mexc_cache  else "❌ no MEXC"
            has_yahoo = "✅" if yahoo_ticker  in yahoo_cache else "❌ no Yahoo"
            lines.append(f"❌ <b>{display}</b>: MEXC {has_mexc}  Yahoo {has_yahoo}")

    if not lines:
        tg_send(chat_id, "⏳ No data yet — wait for first fetch cycle.")
        return

    total_shown = len(actionable) + len(no_spread)
    chunks = [lines[i:i+30] for i in range(0, len(lines), 30)]
    for i, chunk in enumerate(chunks):
        header = (
            f"📊 <b>All Prices</b>  "
            f"({total_shown} tracked / {len(missing)} missing / {len(SYMBOL_MAP)} total)\n\n"
            if i == 0 else "📊 <b>All Prices</b> (cont.)\n\n"
        )
        tg_send(
            chat_id,
            header + "\n".join(chunk)
            + ("\n\n🚨 Alert  🟡 Half  ⚪ Normal  ⚫ No spread  ❌ No data\n"
               "🔴 SHORT  🟢 LONG  🌅 Pre  🌙 AH  ⚙️ Custom" if i == len(chunks)-1 else "")
        )


def handle_status(chat_id: int) -> None:
    # What /prices actually shows
    actionable = [r for r in spread_cache.values() if r.direction != "NONE"]
    no_spread  = [r for r in spread_cache.values() if r.direction == "NONE"]
    above      = sum(1 for s, r in spread_cache.items() if r.spread_pct >= get_threshold(s))
    shorts     = sum(1 for r in actionable if r.direction == "SHORT")
    longs      = sum(1 for r in actionable if r.direction == "LONG")

    # Symbols that have raw data but not yet processed into spread_cache
    in_spread  = set(spread_cache.keys())
    has_data   = sum(1 for s, (yt, _) in SYMBOL_MAP.items()
                     if s in mexc_cache and yt in yahoo_cache)
    no_data    = len(SYMBOL_MAP) - has_data

    with subscribers_lock:
        sub_count = len(subscribers)
    mute_text    = f"\n🔇 Muted: <b>{mute_remaining()}</b>" if is_muted() else ""
    muteall_text = "\n🔕 MuteAll: <b>ON</b>" if muteall_active else ""
    tg_send(
        chat_id,
        f"📊 <b>Bot Status</b>\n\n"
        f"📡 Total symbols:         <b>{len(SYMBOL_MAP)}</b>\n"
        f"✅ Have price data:       <b>{has_data}</b>  (MEXC + Yahoo OK)\n"
        f"❌ No data:               <b>{no_data}</b>  (MEXC or Yahoo failed)\n"
        f"——\n"
        f"🔴 SHORT opportunity:     <b>{shorts}</b>  (MEXC bid1 > Yahoo)\n"
        f"🟢 LONG opportunity:      <b>{longs}</b>  (MEXC ask1 < Yahoo)\n"
        f"⚫ No spread yet:         <b>{len(no_spread)}</b>  (price inside bid/ask)\n"
        f"🚨 Above alert threshold: <b>{above}</b>\n"
        f"——\n"
        f"📶 Threshold:  <b>±{global_threshold}%</b>  |  Step: <b>{alert_step}%</b>\n"
        f"🔄 Interval:   <b>{FETCH_INTERVAL}s</b>\n"
        f"👥 Subscribers: <b>{sub_count}</b>"
        f"{muteall_text}{mute_text}"
    )


def handle_help(chat_id: int) -> None:
    admin_section = ""
    if chat_id == ADMIN_ID:
        admin_section = (
            "\n\n<b>🔐 Admin</b>\n"
            "/admin — control panel\n"
            "/threshold 0.5 — set threshold\n"
            "/step 0.2 — set alert step\n"
            "/mute 30 — mute N minutes\n"
            "/unmute — unmute\n"
            "/muteall GS NKE — mute all except listed\n"
            "/unmuteall — unmute all\n"
            "\n"
            "/setthreshold GS 0.3 — custom threshold\n"
            "/delthreshold GS — remove custom threshold\n"
            "/thresholds — view all custom thresholds\n"
            "\n"
            "/subscribers — list subscribers\n"
            "/test    — send test alert\n"
            "/missing — symbols with no data\n"
        )
    tg_send(
        chat_id,
        "<b>📖 Commands</b>\n\n"
        "<b>User</b>\n"
        "/prices — all prices\n"
        "/check  — active spreads\n"
        "/check TSLA — debug one symbol\n"
        "/status — bot status\n"
        "/help   — this message\n"
        "/start · /stop — subscribe/unsubscribe"
        f"{admin_section}\n\n"
        "<b>Legend</b>\n"
        "🔴 SHORT · 🟢 LONG · 🚨 Alert · 🟡 Half · ⚪ Normal\n"
        "🌅 Pre-market · 🌙 After-hours · ⚫ No spread · ❌ No data\n\n"
        "<b>Spread logic</b>\n"
        "SHORT → MEXC bid1 vs Yahoo  |  LONG → MEXC ask1 vs Yahoo\n"
        "Alert fires at threshold, repeats every +step%, resets on drop."
    )


def handle_admin_cmd(chat_id: int) -> None:
    if chat_id != ADMIN_ID:
        tg_send(chat_id, "⛔ Admin only.")
        return
    send_admin_panel(chat_id)


def handle_threshold_cmd(chat_id: int, args: list[str]) -> None:
    global global_threshold
    if chat_id != ADMIN_ID:
        tg_send(chat_id, "⛔ Admin only.")
        return
    if not args:
        tg_send(chat_id, f"Current: <b>±{global_threshold}%</b>\nUsage: <code>/threshold 0.5</code>")
        return
    try:
        val = float(args[0])
        if not 0.01 <= val <= 100:
            raise ValueError
    except ValueError:
        tg_send(chat_id, "❌ Must be 0.01–100.")
        return
    old = global_threshold
    global_threshold = val
    db_save_all_settings()
    with state_lock:
        peak_spread.clear()
    tg_send(chat_id, f"✅ Threshold: <b>{old}%</b> → <b>{val}%</b>")


def handle_step_cmd(chat_id: int, args: list[str]) -> None:
    global alert_step
    if chat_id != ADMIN_ID:
        tg_send(chat_id, "⛔ Admin only.")
        return
    if not args:
        tg_send(chat_id, f"Current: <b>{alert_step}%</b>\nUsage: <code>/step 0.2</code>")
        return
    try:
        val = float(args[0])
        if not 0.01 <= val <= 10:
            raise ValueError
    except ValueError:
        tg_send(chat_id, "❌ Must be 0.01–10.")
        return
    old = alert_step
    alert_step = val
    db_save_all_settings()
    tg_send(chat_id, f"✅ Step: <b>{old}%</b> → <b>{val}%</b>")


def handle_setthreshold(chat_id: int, args: list[str]) -> None:
    if chat_id != ADMIN_ID:
        tg_send(chat_id, "⛔ Admin only.")
        return
    if len(args) < 2:
        tg_send(chat_id, "Usage: <code>/setthreshold GS 0.3</code>")
        return
    mexc_sym = resolve_symbol(args[0].upper())
    if not mexc_sym:
        tg_send(chat_id, f"❌ <code>{args[0].upper()}</code> not found.")
        return
    try:
        val = float(args[1])
        if not 0.01 <= val <= 100:
            raise ValueError
    except ValueError:
        tg_send(chat_id, "❌ Must be 0.01–100.")
        return
    display = SYMBOL_MAP[mexc_sym][1].upper()
    symbol_thresholds[display] = val
    db_save_symbol_threshold(display, val)
    with state_lock:
        peak_spread.pop(mexc_sym, None)
    tg_send(chat_id, f"✅ <b>{display}</b> → <b>±{val}%</b>")


def handle_delthreshold(chat_id: int, args: list[str]) -> None:
    if chat_id != ADMIN_ID:
        tg_send(chat_id, "⛔ Admin only.")
        return
    if not args:
        tg_send(chat_id, "Usage: <code>/delthreshold GS</code>")
        return
    mexc_sym = resolve_symbol(args[0].upper())
    display  = SYMBOL_MAP.get(mexc_sym or "", ("", args[0].upper()))[1].upper() if mexc_sym else args[0].upper()
    if display not in symbol_thresholds:
        tg_send(chat_id, f"ℹ️ <b>{display}</b> has no custom threshold.")
        return
    symbol_thresholds.pop(display)
    db_delete_symbol_threshold(display)
    tg_send(chat_id, f"✅ Removed. Using global: <b>±{global_threshold}%</b>")


def handle_thresholds(chat_id: int) -> None:
    if chat_id != ADMIN_ID:
        tg_send(chat_id, "⛔ Admin only.")
        return
    if not symbol_thresholds:
        tg_send(chat_id, f"⚙️ No custom thresholds.\nGlobal: <b>±{global_threshold}%</b>")
        return
    lines = [f"  ⚙️ <b>{s}</b>: ±{t}%" for s, t in sorted(symbol_thresholds.items())]
    tg_send(
        chat_id,
        f"⚙️ <b>Custom Thresholds ({len(symbol_thresholds)}):</b>\n\n"
        + "\n".join(lines)
        + f"\n\n📶 Global: <b>±{global_threshold}%</b>"
    )


def handle_mute(chat_id: int, args: list[str]) -> None:
    global muted_until
    if chat_id != ADMIN_ID:
        tg_send(chat_id, "⛔ Admin only.")
        return
    if not args:
        s = f"🔇 Muted — {mute_remaining()} remaining." if is_muted() else "Not muted."
        tg_send(chat_id, f"{s}\nUsage: <code>/mute MINUTES</code>")
        return
    try:
        minutes = float(args[0])
        if minutes <= 0:
            raise ValueError
    except ValueError:
        tg_send(chat_id, "❌ Positive number required.")
        return
    muted_until = time.time() + minutes * 60
    db_save_all_settings()
    h, m = int(minutes // 60), int(minutes % 60)
    tg_send(chat_id, f"🔇 Muted for <b>{'%dh %dm' % (h,m) if h else '%dm' % m}</b>.")


def handle_unmute(chat_id: int) -> None:
    global muted_until
    if chat_id != ADMIN_ID:
        tg_send(chat_id, "⛔ Admin only.")
        return
    if not is_muted():
        tg_send(chat_id, "ℹ️ Not muted.")
        return
    muted_until = 0.0
    db_save_all_settings()
    tg_send(chat_id, "🔊 <b>Alerts restored!</b>")


def handle_muteall(chat_id: int, args: list[str]) -> None:
    global muteall_active, muteall_exceptions
    if chat_id != ADMIN_ID:
        tg_send(chat_id, "⛔ Admin only.")
        return
    if not args:
        tg_send(chat_id, "Usage: <code>/muteall GS NKE TSLA</code>")
        return
    resolved, failed = [], []
    for a in args:
        mexc = resolve_symbol(a.upper())
        if mexc:
            resolved.append(SYMBOL_MAP[mexc][1].upper())
        else:
            failed.append(a.upper())
    muteall_active     = True
    muteall_exceptions = set(resolved)
    db_save_all_settings()
    exc_text  = ", ".join(f"<b>{s}</b>" for s in sorted(resolved)) if resolved else "none"
    fail_text = f"\n⚠️ Not found: {', '.join(failed)}" if failed else ""
    tg_send(chat_id,
        f"🔕 <b>MuteAll ON</b> — only alerting: {exc_text}{fail_text}\n"
        "Use /unmuteall to restore.")


def handle_unmuteall(chat_id: int) -> None:
    global muteall_active, muteall_exceptions
    if chat_id != ADMIN_ID:
        tg_send(chat_id, "⛔ Admin only.")
        return
    muteall_active     = False
    muteall_exceptions = set()
    db_save_all_settings()
    tg_send(chat_id, "🔔 <b>MuteAll disabled!</b>")


def handle_subscribers(chat_id: int) -> None:
    if chat_id != ADMIN_ID:
        tg_send(chat_id, "⛔ Admin only.")
        return
    with subscribers_lock:
        if not subscribers:
            tg_send(chat_id, "📭 No subscribers yet.")
            return
        lines = []
        for i, (cid, info) in enumerate(subscribers.items(), 1):
            joined = time.strftime("%d.%m.%Y", time.localtime(info.get("joined_at", 0)))
            lines.append(
                f"{i}. <b>{info['name']}</b> {info['username']}\n"
                f"   <code>{cid}</code> | {joined}"
            )
        total = len(subscribers)
    for chunk in [lines[i:i+30] for i in range(0, len(lines), 30)]:
        tg_send(chat_id, f"👥 <b>Subscribers ({total}):</b>\n\n" + "\n\n".join(chunk))


def handle_test(chat_id: int) -> None:
    if chat_id != ADMIN_ID:
        tg_send(chat_id, "⛔ Admin only.")
        return
    if not spread_cache:
        tg_send(chat_id, "⏳ No data yet.")
        return
    best = max(spread_cache.values(), key=lambda r: r.spread_pct)
    send_spread_alert(best, chat_id=chat_id)
    tg_send(chat_id,
        f"🧪 Test sent!\n"
        f"<b>{best.display}</b> {best.direction} {best.spread_pct:.2f}% [{best.market_state}]")



def handle_missing(chat_id: int) -> None:
    """
    Show exactly which symbols are missing from /prices and why.

    Three separate counts:
    - spread_cache = what /prices actually shows (ground truth)
    - mexc_cache   = raw MEXC data from last fetch
    - yahoo_cache  = raw Yahoo data from last fetch

    A symbol can be in both raw caches but MISSING from spread_cache if:
    - Yahoo price failed the sanity check (< $0.50 or > $100k)
    - The monitoring loop cycle hasn't processed it yet
    - It was skipped due to a transient fetch error this cycle
    """
    if chat_id != ADMIN_ID:
        tg_send(chat_id, "⛔ Admin only.")
        return

    no_mexc, no_yahoo, bad_yahoo, missing_spread = [], [], [], []

    for mexc_sym, (yahoo_ticker, display) in sorted(SYMBOL_MAP.items(),
                                                     key=lambda x: x[1][1]):
        has_mexc   = mexc_sym in mexc_cache
        has_yahoo  = yahoo_ticker in yahoo_cache
        in_prices  = mexc_sym in spread_cache

        if not has_mexc:
            no_mexc.append(f"  ❌ <b>{display}</b> — no MEXC bid1/ask1")
            continue

        if not has_yahoo:
            no_yahoo.append(f"  ❌ <b>{display}</b> ({yahoo_ticker}) — Yahoo fetch failed")
            continue

        # Both caches have data — check Yahoo price sanity
        snap = yahoo_cache[yahoo_ticker]
        if snap.active_price < 0.5 or snap.active_price > 100_000:
            bad_yahoo.append(
                f"  ⚠️ <b>{display}</b> ({yahoo_ticker}) — "
                f"suspicious price <code>${snap.active_price:.4f}</code>"
            )
            continue

        # Has valid data in both caches but not showing in /prices
        if not in_prices:
            mexc = mexc_cache.get(mexc_sym)
            yahoo = yahoo_cache.get(yahoo_ticker)
            bid1 = mexc.bid1 if mexc else 0
            ask1 = mexc.ask1 if mexc else 0
            yp   = yahoo.active_price if yahoo else 0
            missing_spread.append(
                f"  ⚠️ <b>{display}</b> — in caches but not in spread_cache "
                f"(b:{bid1:.2f} a:{ask1:.2f} Y:{yp:.2f})"
            )

    in_prices_count = len(spread_cache)
    total_ok = len(SYMBOL_MAP) - len(no_mexc) - len(no_yahoo) - len(bad_yahoo) - len(missing_spread)

    lines = [
        f"📊 <b>Data Coverage</b>\n"
        f"  /prices shows: <b>{in_prices_count}/{len(SYMBOL_MAP)}</b> symbols\n"
        f"  MEXC cache:    <b>{len(mexc_cache)}/{len(SYMBOL_MAP)}</b>\n"
        f"  Yahoo cache:   <b>{len(yahoo_cache)}/{len(ALL_YAHOO_TICKERS)}</b> unique tickers\n"
    ]

    if no_mexc:
        lines.append(f"\n<b>No MEXC data ({len(no_mexc)}):</b>")
        lines.extend(no_mexc)

    if no_yahoo:
        lines.append(f"\n<b>No Yahoo data ({len(no_yahoo)}):</b>")
        lines.extend(no_yahoo)

    if bad_yahoo:
        lines.append(f"\n<b>Suspicious Yahoo prices ({len(bad_yahoo)}):</b>")
        lines.extend(bad_yahoo)

    if missing_spread:
        lines.append(f"\n<b>In caches but missing from /prices ({len(missing_spread)}):</b>")
        lines.extend(missing_spread)

    if not no_mexc and not no_yahoo and not bad_yahoo and not missing_spread:
        lines.append("\n✅ All symbols showing in /prices!")

    tg_send(chat_id, "\n".join(lines))


# ---------------------------------------------------------------------------
# Callback handler
# ---------------------------------------------------------------------------


def handle_callback(callback: dict) -> None:
    global muted_until, muteall_active, muteall_exceptions
    callback_id = callback.get("id", "")
    chat_id     = callback.get("from", {}).get("id")
    message_id  = callback.get("message", {}).get("message_id")
    data        = callback.get("data", "")

    if chat_id != ADMIN_ID:
        tg_answer_callback(callback_id, "⛔ Admin only")
        return

    if data == "toggle_muteall":
        if muteall_active:
            muteall_active = False
            muteall_exceptions = set()
            db_save_all_settings()
            tg_answer_callback(callback_id, "✅ MuteAll disabled")
        else:
            tg_answer_callback(callback_id, "Use /muteall GS NKE to set exceptions")
            return
    elif data == "show_mute":
        tg_answer_callback(callback_id,
            f"Muted: {mute_remaining()}" if is_muted() else "Not muted. /mute 30")
        return
    elif data in ("view_spreads", "view_thresholds", "view_subscribers"):
        tg_answer_callback(callback_id, "Sending...")
        {"view_spreads": handle_prices, "view_thresholds": handle_thresholds,
         "view_subscribers": handle_subscribers}[data](chat_id)
        return
    elif data == "show_global":
        tg_answer_callback(callback_id, f"Global: ±{global_threshold}%  /threshold X")
        return
    elif data == "show_step":
        tg_answer_callback(callback_id, f"Step: {alert_step}%  /step X")
        return
    elif data == "refresh_admin":
        tg_answer_callback(callback_id, "✅ Refreshed")

    send_admin_panel(chat_id, message_id=message_id)


# ---------------------------------------------------------------------------
# Command dispatcher
# ---------------------------------------------------------------------------


def dispatch(update: dict) -> None:
    if "callback_query" in update:
        handle_callback(update["callback_query"])
        return

    message = update.get("message", {})
    if not message:
        return

    chat_id = message.get("chat", {}).get("id")
    text    = message.get("text", "").strip()
    user    = message.get("from", {})

    if not chat_id or not text:
        return

    parts = text.split()
    cmd   = parts[0].lower().split("@")[0]
    args  = parts[1:]

    handlers = {
        # ── user ──────────────────────────────────────────────
        "/start":        lambda: handle_start(chat_id, user),
        "/stop":         lambda: handle_stop(chat_id),
        "/prices":       lambda: handle_prices(chat_id),
        "/p":            lambda: handle_prices(chat_id),
        "/check":        lambda: handle_check(chat_id, args),
        "/c":            lambda: handle_check(chat_id, args),
        "/status":       lambda: handle_status(chat_id),
        "/s":            lambda: handle_status(chat_id),
        "/help":         lambda: handle_help(chat_id),
        "/h":            lambda: handle_help(chat_id),
        # ── admin (full) ───────────────────────────────────────
        "/admin":        lambda: handle_admin_cmd(chat_id),
        "/threshold":    lambda: handle_threshold_cmd(chat_id, args),
        "/step":         lambda: handle_step_cmd(chat_id, args),
        "/setthreshold": lambda: handle_setthreshold(chat_id, args),
        "/delthreshold": lambda: handle_delthreshold(chat_id, args),
        "/thresholds":   lambda: handle_thresholds(chat_id),
        "/mute":         lambda: handle_mute(chat_id, args),
        "/unmute":       lambda: handle_unmute(chat_id),
        "/muteall":      lambda: handle_muteall(chat_id, args),
        "/unmuteall":    lambda: handle_unmuteall(chat_id),
        "/subscribers":  lambda: handle_subscribers(chat_id),
        "/test":         lambda: handle_test(chat_id),
        "/missing":      lambda: handle_missing(chat_id),
        # ── admin (short) ─────────────────────────────────────
        "/a":            lambda: handle_admin_cmd(chat_id),
        "/t":            lambda: handle_threshold_cmd(chat_id, args),
        "/st":           lambda: handle_step_cmd(chat_id, args),
        "/sth":          lambda: handle_setthreshold(chat_id, args),
        "/dth":          lambda: handle_delthreshold(chat_id, args),
        "/ths":          lambda: handle_thresholds(chat_id),
        "/m":            lambda: handle_mute(chat_id, args),
        "/um":           lambda: handle_unmute(chat_id),
        "/ma":           lambda: handle_muteall(chat_id, args),
        "/uma":          lambda: handle_unmuteall(chat_id),
        "/subs":         lambda: handle_subscribers(chat_id),
    }
    handlers.get(cmd, lambda: None)()


# ---------------------------------------------------------------------------
# Polling thread
# ---------------------------------------------------------------------------


def polling_thread() -> None:
    offset = 0
    logger.info("Polling started")
    while True:
        try:
            updates = tg_get_updates(offset)
            for update in updates:
                offset = update["update_id"] + 1
                threading.Thread(target=dispatch, args=(update,), daemon=True).start()
        except Exception as exc:
            logger.error("Polling error: %s", exc)
            time.sleep(5)


# ---------------------------------------------------------------------------
# Monitoring loop
# ---------------------------------------------------------------------------


def monitoring_loop() -> None:
    logger.info(
        "Monitor v6.0 started — interval=%ds threshold=%.2f%% step=%.2f%%",
        FETCH_INTERVAL, global_threshold, alert_step,
    )
    while True:
        cycle_start = time.time()
        logger.info("Fetching prices...")

        mexc_data   = fetch_mexc_data()
        yahoo_snaps = fetch_all_yahoo()

        with state_lock:
            mexc_cache.update(mexc_data)
            yahoo_cache.update(yahoo_snaps)

        # Notify if any ticker changed market state
        check_and_notify_state_changes(yahoo_snaps)

        matched = 0

        for mexc_sym, (yahoo_ticker, display) in SYMBOL_MAP.items():
            # Use current cycle data; fall back to cache if this cycle missed it
            mexc  = mexc_data.get(mexc_sym) or mexc_cache.get(mexc_sym)
            yahoo = yahoo_snaps.get(yahoo_ticker) or yahoo_cache.get(yahoo_ticker)
            if not mexc or not yahoo:
                continue

            bid1         = mexc.bid1
            ask1         = mexc.ask1
            yahoo_price  = yahoo.active_price
            market_state = yahoo.active_state
            threshold    = get_threshold(mexc_sym)

            # Sanity check: skip clearly wrong Yahoo prices
            # (yfinance sometimes returns stale/wrong data for some tickers)
            if yahoo_price < 0.5 or yahoo_price > 100_000:
                logger.warning(
                    "Skipping %s — suspicious Yahoo price $%.4f", display, yahoo_price
                )
                continue

            result = calculate_actionable_spread(bid1, ask1, yahoo_price)

            if result is None:
                # Yahoo price is momentarily inside bid/ask — NOT a reason to reset peak.
                # If we reset here, the bot re-alerts every time prices briefly touch.
                # Peak only resets when spread drops below threshold (handled below).
                with state_lock:
                    spread_cache.pop(mexc_sym, None)
                continue

            direction, actionable_price, spread_pct = result
            matched += 1

            rec = SpreadRecord(
                display=display,
                mexc_sym=mexc_sym,
                yahoo_ticker=yahoo_ticker,
                direction=direction,
                actionable_price=actionable_price,
                yahoo_price=yahoo_price,
                spread_pct=spread_pct,
                bid1=bid1,
                ask1=ask1,
                fair=mexc.fair,
                market_state=market_state,
            )

            with state_lock:
                spread_cache[mexc_sym] = rec

            # ── Alert state machine ──────────────────────────────────────
            #
            # peak_spread[sym] = the spread % at which we last sent an alert.
            #
            # Rules:
            #   1. Below threshold       → clear peak, no alert
            #   2. First time above      → alert immediately, store as peak
            #   3. Grew by >= alert_step → alert (GROWING), update peak up
            #   4. Same / small growth   → no alert, no change
            #   5. Dropped significantly → silently lower peak to current
            #      (so next pump from here triggers a fresh alert)
            #
            # "Significant drop" = dropped more than alert_step below peak.
            # This is the dynamic reset that prevents spam on bounces.

            if spread_pct < threshold:
                with state_lock:
                    if peak_spread.get(mexc_sym, 0) > 0:
                        logger.info("PEAK ZERO %s: %.4f%% dropped below threshold %.2f%%",
                                    display, spread_pct, threshold)
                        peak_spread[mexc_sym] = 0.0
                continue

            if is_symbol_muted(mexc_sym):
                continue

            should_alert = False
            is_growing   = False

            with state_lock:
                current_peak = peak_spread.get(mexc_sym, 0.0)

                if current_peak == 0.0:
                    # First time above threshold — alert
                    should_alert          = True
                    is_growing            = False
                    peak_spread[mexc_sym] = spread_pct
                    logger.info("PEAK SET  %s: %.4f%% (first alert)", display, spread_pct)
                elif spread_pct >= current_peak + alert_step:
                    # Spread genuinely grew by at least alert_step — alert
                    should_alert          = True
                    is_growing            = True
                    logger.info("PEAK UP   %s: %.4f%% → %.4f%% (+%.4f%%)",
                                display, current_peak, spread_pct, spread_pct - current_peak)
                    peak_spread[mexc_sym] = spread_pct
                else:
                    # No alert — log every cycle so we can trace what's happening
                    logger.info("PEAK HOLD %s: current=%.4f%% spread=%.4f%% need=%.4f%% (step=%.2f%%)",
                                display, current_peak, spread_pct, current_peak + alert_step, alert_step)

            if should_alert:
                threading.Thread(
                    target=send_spread_alert,
                    args=(rec, is_growing),
                    daemon=True,
                ).start()

        elapsed = time.time() - cycle_start
        total_with_data = sum(
            1 for s in SYMBOL_MAP
            if s in mexc_data and SYMBOL_MAP[s][0] in yahoo_snaps
        )
        logger.info(
            "Cycle %.2fs — mexc=%d yahoo=%d tracked=%d actionable=%d",
            elapsed, len(mexc_data), len(yahoo_snaps), total_with_data, matched,
        )
        time.sleep(max(0, FETCH_INTERVAL - elapsed))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    init_db()
    db_load_all_settings()
    logger.info(
        "Spread bot v7.0 — threshold=%.1f%% step=%.1f%% interval=%ds subs=%d",
        global_threshold, alert_step, FETCH_INTERVAL, len(subscribers),
    )
    threading.Thread(target=polling_thread, daemon=True).start()
    monitoring_loop()
