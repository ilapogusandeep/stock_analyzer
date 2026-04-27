"""Popular US tickers for the autocomplete dropdown.

Not exhaustive — users can still type any symbol not in this list.
Covers ~250 names: S&P 500 leaders, NASDAQ-100, popular retail stocks,
common ETFs. Sorted alphabetically for consistent dropdown ordering.
"""

from __future__ import annotations

POPULAR_TICKERS: dict[str, str] = {
    # --- Mega-cap tech ---
    "AAPL":  "Apple Inc.",
    "MSFT":  "Microsoft Corporation",
    "GOOGL": "Alphabet Inc. (Class A)",
    "GOOG":  "Alphabet Inc. (Class C)",
    "AMZN":  "Amazon.com Inc.",
    "META":  "Meta Platforms Inc.",
    "NVDA":  "NVIDIA Corporation",
    "TSLA":  "Tesla Inc.",
    "AVGO":  "Broadcom Inc.",
    "ORCL":  "Oracle Corporation",

    # --- Semiconductors ---
    "AMD":   "Advanced Micro Devices Inc.",
    "INTC":  "Intel Corporation",
    "QCOM":  "QUALCOMM Incorporated",
    "TXN":   "Texas Instruments Incorporated",
    "MU":    "Micron Technology Inc.",
    "AMAT":  "Applied Materials Inc.",
    "LRCX":  "Lam Research Corporation",
    "KLAC":  "KLA Corporation",
    "ADI":   "Analog Devices Inc.",
    "MRVL":  "Marvell Technology Inc.",
    "NXPI":  "NXP Semiconductors N.V.",
    "ASML":  "ASML Holding N.V.",
    "TSM":   "Taiwan Semiconductor Manufacturing",
    "ARM":   "Arm Holdings plc",
    "SMCI":  "Super Micro Computer Inc.",

    # --- Software / SaaS ---
    "CRM":   "Salesforce Inc.",
    "ADBE":  "Adobe Inc.",
    "NOW":   "ServiceNow Inc.",
    "INTU":  "Intuit Inc.",
    "SNOW":  "Snowflake Inc.",
    "PLTR":  "Palantir Technologies Inc.",
    "MDB":   "MongoDB Inc.",
    "DDOG":  "Datadog Inc.",
    "NET":   "Cloudflare Inc.",
    "CRWD":  "CrowdStrike Holdings Inc.",
    "PANW":  "Palo Alto Networks Inc.",
    "ZS":    "Zscaler Inc.",
    "OKTA":  "Okta Inc.",
    "TEAM":  "Atlassian Corporation",
    "WDAY":  "Workday Inc.",
    "ADSK":  "Autodesk Inc.",
    "ANSS":  "ANSYS Inc.",
    "SHOP":  "Shopify Inc.",
    "SNAP":  "Snap Inc.",
    "PINS":  "Pinterest Inc.",
    "RBLX":  "Roblox Corporation",
    "U":     "Unity Software Inc.",
    "ASAN":  "Asana Inc.",
    "DOCN":  "DigitalOcean Holdings Inc.",
    "ESTC":  "Elastic N.V.",
    "TWLO":  "Twilio Inc.",
    "SQ":    "Block Inc.",
    "PYPL":  "PayPal Holdings Inc.",
    "COIN":  "Coinbase Global Inc.",
    "HOOD":  "Robinhood Markets Inc.",
    "SOFI":  "SoFi Technologies Inc.",
    "AFRM":  "Affirm Holdings Inc.",
    "UBER":  "Uber Technologies Inc.",
    "LYFT":  "Lyft Inc.",
    "DASH":  "DoorDash Inc.",
    "ABNB":  "Airbnb Inc.",
    "BKNG":  "Booking Holdings Inc.",
    "EXPE":  "Expedia Group Inc.",

    # --- Streaming / Media ---
    "NFLX":  "Netflix Inc.",
    "DIS":   "The Walt Disney Company",
    "WBD":   "Warner Bros. Discovery Inc.",
    "PARA":  "Paramount Global",
    "CMCSA": "Comcast Corporation",
    "SPOT":  "Spotify Technology S.A.",
    "RBLX ": "Roblox Corporation",

    # --- E-commerce / Retail ---
    "WMT":   "Walmart Inc.",
    "COST":  "Costco Wholesale Corporation",
    "HD":    "The Home Depot Inc.",
    "LOW":   "Lowe's Companies Inc.",
    "TGT":   "Target Corporation",
    "NKE":   "NIKE Inc.",
    "LULU":  "lululemon athletica inc.",
    "SBUX":  "Starbucks Corporation",
    "MCD":   "McDonald's Corporation",
    "CMG":   "Chipotle Mexican Grill Inc.",
    "EBAY":  "eBay Inc.",
    "ETSY":  "Etsy Inc.",
    "MELI":  "MercadoLibre Inc.",
    "JD":    "JD.com Inc.",
    "BABA":  "Alibaba Group Holding Ltd",
    "PDD":   "PDD Holdings Inc.",
    "CVS":   "CVS Health Corporation",
    "WBA":   "Walgreens Boots Alliance Inc.",

    # --- Financials ---
    "JPM":   "JPMorgan Chase & Co.",
    "BAC":   "Bank of America Corporation",
    "WFC":   "Wells Fargo & Company",
    "C":     "Citigroup Inc.",
    "GS":    "The Goldman Sachs Group Inc.",
    "MS":    "Morgan Stanley",
    "BLK":   "BlackRock Inc.",
    "SCHW":  "The Charles Schwab Corporation",
    "AXP":   "American Express Company",
    "V":     "Visa Inc.",
    "MA":    "Mastercard Incorporated",
    "PYPL ": "PayPal Holdings Inc.",
    "BRK.B": "Berkshire Hathaway Inc.",
    "USB":   "U.S. Bancorp",
    "PNC":   "The PNC Financial Services Group",
    "TFC":   "Truist Financial Corporation",
    "MET":   "MetLife Inc.",
    "AIG":   "American International Group Inc.",
    "PRU":   "Prudential Financial Inc.",
    "COF":   "Capital One Financial Corporation",
    "DFS":   "Discover Financial Services",
    "FI":    "Fiserv Inc.",
    "SPGI":  "S&P Global Inc.",
    "MCO":   "Moody's Corporation",
    "ICE":   "Intercontinental Exchange Inc.",
    "CME":   "CME Group Inc.",

    # --- Healthcare / Pharma ---
    "UNH":   "UnitedHealth Group Incorporated",
    "JNJ":   "Johnson & Johnson",
    "LLY":   "Eli Lilly and Company",
    "PFE":   "Pfizer Inc.",
    "MRK":   "Merck & Co. Inc.",
    "ABBV":  "AbbVie Inc.",
    "ABT":   "Abbott Laboratories",
    "TMO":   "Thermo Fisher Scientific Inc.",
    "DHR":   "Danaher Corporation",
    "BMY":   "Bristol Myers Squibb Company",
    "AMGN":  "Amgen Inc.",
    "GILD":  "Gilead Sciences Inc.",
    "MDT":   "Medtronic plc",
    "ISRG":  "Intuitive Surgical Inc.",
    "REGN":  "Regeneron Pharmaceuticals Inc.",
    "VRTX":  "Vertex Pharmaceuticals Incorporated",
    "BIIB":  "Biogen Inc.",
    "CI":    "The Cigna Group",
    "HUM":   "Humana Inc.",
    "ELV":   "Elevance Health Inc.",
    "MRNA":  "Moderna Inc.",
    "NVAX":  "Novavax Inc.",
    "BNTX":  "BioNTech SE",
    "NVO":   "Novo Nordisk A/S",

    # --- Consumer staples / Beverages ---
    "PG":    "The Procter & Gamble Company",
    "KO":    "The Coca-Cola Company",
    "PEP":   "PepsiCo Inc.",
    "MDLZ":  "Mondelez International Inc.",
    "PM":    "Philip Morris International Inc.",
    "MO":    "Altria Group Inc.",
    "CL":    "Colgate-Palmolive Company",
    "KHC":   "The Kraft Heinz Company",
    "GIS":   "General Mills Inc.",
    "K":     "Kellanova",
    "STZ":   "Constellation Brands Inc.",
    "DEO":   "Diageo plc",

    # --- Energy ---
    "XOM":   "Exxon Mobil Corporation",
    "CVX":   "Chevron Corporation",
    "COP":   "ConocoPhillips",
    "EOG":   "EOG Resources Inc.",
    "SLB":   "Schlumberger N.V.",
    "MPC":   "Marathon Petroleum Corporation",
    "PSX":   "Phillips 66",
    "VLO":   "Valero Energy Corporation",
    "OXY":   "Occidental Petroleum Corporation",
    "PXD":   "Pioneer Natural Resources Company",
    "KMI":   "Kinder Morgan Inc.",
    "WMB":   "The Williams Companies Inc.",

    # --- Industrials / Transport ---
    "CAT":   "Caterpillar Inc.",
    "DE":    "Deere & Company",
    "BA":    "The Boeing Company",
    "GE":    "GE Aerospace",
    "HON":   "Honeywell International Inc.",
    "MMM":   "3M Company",
    "LMT":   "Lockheed Martin Corporation",
    "NOC":   "Northrop Grumman Corporation",
    "RTX":   "RTX Corporation",
    "UPS":   "United Parcel Service Inc.",
    "FDX":   "FedEx Corporation",
    "DAL":   "Delta Air Lines Inc.",
    "UAL":   "United Airlines Holdings Inc.",
    "AAL":   "American Airlines Group Inc.",
    "LUV":   "Southwest Airlines Co.",
    "UNP":   "Union Pacific Corporation",
    "CSX":   "CSX Corporation",
    "NSC":   "Norfolk Southern Corporation",

    # --- Autos / EVs ---
    "F":     "Ford Motor Company",
    "GM":    "General Motors Company",
    "STLA":  "Stellantis N.V.",
    "TM":    "Toyota Motor Corporation",
    "HMC":   "Honda Motor Co. Ltd",
    "RIVN":  "Rivian Automotive Inc.",
    "LCID":  "Lucid Group Inc.",
    "NIO":   "NIO Inc.",
    "XPEV":  "XPeng Inc.",
    "LI":    "Li Auto Inc.",

    # --- Telecom / Networking ---
    "T":     "AT&T Inc.",
    "VZ":    "Verizon Communications Inc.",
    "TMUS":  "T-Mobile US Inc.",
    "CSCO":  "Cisco Systems Inc.",
    "JNPR":  "Juniper Networks Inc.",

    # --- Real estate / REITs ---
    "PLD":   "Prologis Inc.",
    "AMT":   "American Tower Corporation",
    "CCI":   "Crown Castle Inc.",
    "EQIX":  "Equinix Inc.",
    "O":     "Realty Income Corporation",
    "SPG":   "Simon Property Group Inc.",
    "DLR":   "Digital Realty Trust Inc.",

    # --- Retail / meme ---
    "GME":   "GameStop Corp.",
    "AMC":   "AMC Entertainment Holdings Inc.",
    "BB":    "BlackBerry Limited",
    "BBBY":  "Bed Bath & Beyond Inc.",
    "KSS":   "Kohl's Corporation",
    "M":     "Macy's Inc.",

    # --- ETFs ---
    "SPY":   "SPDR S&P 500 ETF Trust",
    "QQQ":   "Invesco QQQ Trust",
    "IWM":   "iShares Russell 2000 ETF",
    "DIA":   "SPDR Dow Jones Industrial Average ETF",
    "VOO":   "Vanguard S&P 500 ETF",
    "VTI":   "Vanguard Total Stock Market ETF",
    "VEA":   "Vanguard FTSE Developed Markets ETF",
    "VWO":   "Vanguard FTSE Emerging Markets ETF",
    "EEM":   "iShares MSCI Emerging Markets ETF",
    "EFA":   "iShares MSCI EAFE ETF",
    "GLD":   "SPDR Gold Shares",
    "SLV":   "iShares Silver Trust",
    "TLT":   "iShares 20+ Year Treasury Bond ETF",
    "HYG":   "iShares iBoxx $ High Yield Corporate Bond ETF",
    "LQD":   "iShares iBoxx $ Investment Grade Corporate Bond ETF",
    "XLK":   "Technology Select Sector SPDR Fund",
    "XLF":   "Financial Select Sector SPDR Fund",
    "XLE":   "Energy Select Sector SPDR Fund",
    "XLV":   "Health Care Select Sector SPDR Fund",
    "XLI":   "Industrial Select Sector SPDR Fund",
    "XLY":   "Consumer Discretionary Select Sector SPDR Fund",
    "XLP":   "Consumer Staples Select Sector SPDR Fund",
    "XLU":   "Utilities Select Sector SPDR Fund",
    "XLB":   "Materials Select Sector SPDR Fund",
    "XLRE":  "Real Estate Select Sector SPDR Fund",
    "ARKK":  "ARK Innovation ETF",
    "SMH":   "VanEck Semiconductor ETF",
    "SOXX":  "iShares Semiconductor ETF",
    "BITO":  "ProShares Bitcoin Strategy ETF",
    "VXX":   "iPath Series B S&P 500 VIX Short-Term Futures ETN",
    "UVXY":  "ProShares Ultra VIX Short-Term Futures ETF",

    # --- Materials / Metals ---
    "FCX":   "Freeport-McMoRan Inc.",
    "NEM":   "Newmont Corporation",
    "LIN":   "Linde plc",
    "APD":   "Air Products and Chemicals Inc.",
    "SHW":   "The Sherwin-Williams Company",
    "DOW":   "Dow Inc.",
    "DD":    "DuPont de Nemours Inc.",

    # --- Other popular ---
    "PLTR ": "Palantir Technologies Inc.",
    "RKLB":  "Rocket Lab USA Inc.",
    "SPCE":  "Virgin Galactic Holdings Inc.",
    "DKNG":  "DraftKings Inc.",
    "PENN":  "PENN Entertainment Inc.",
    "MSTR":  "MicroStrategy Incorporated",
    "ON":    "ON Semiconductor Corporation",
    "ENPH":  "Enphase Energy Inc.",
    "FSLR":  "First Solar Inc.",
    "SEDG":  "SolarEdge Technologies Inc.",
    "PLUG":  "Plug Power Inc.",
    "BE":    "Bloom Energy Corporation",
    "CHWY":  "Chewy Inc.",
    "W":     "Wayfair Inc.",
    "GRPN":  "Groupon Inc.",
    "TRIP":  "Tripadvisor Inc.",
    "YELP":  "Yelp Inc.",
    "RDDT":  "Reddit Inc.",
}

# De-dupe keys that slipped in with trailing spaces (defensive)
POPULAR_TICKERS = {k.strip(): v for k, v in POPULAR_TICKERS.items()}


# ---------------------------------------------------------------------------
# Extended ticker universe — SEC official list + locally remembered searches
# ---------------------------------------------------------------------------
#
# POPULAR_TICKERS is hand-curated and tops out at ~261 names. Two cheap ways
# to widen coverage without a paid data source:
#
# 1. SEC's company_tickers.json (free, official, no API key) lists every US-
#    listed equity with its CIK + company name. ~10K entries, refreshed
#    nightly by the SEC. Cache locally for 7 days.
#
# 2. Anything outside the SEC list (crypto BTC-USD, indices ^GSPC, forex
#    EURUSD=X, foreign ADRs without US listings) — remember per user
#    search. After a successful analyze() call we persist the ticker +
#    its company name to data/searched_tickers.json so it shows up in
#    the dropdown next time.
#
# get_all_tickers() merges the three sources into one dict for the
# selectbox to consume. POPULAR_TICKERS wins when there's a name conflict
# so curated names ("Alphabet Inc. (Class A)") aren't overwritten by SEC's
# "ALPHABET INC.".

import json
import os
from pathlib import Path
from typing import Optional

_SEARCHED_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "searched_tickers.json"
_SEC_URL = "https://www.sec.gov/files/company_tickers.json"


def _fetch_sec_tickers() -> dict[str, str]:
    """Pull the SEC's company_tickers.json. Returns {} on any failure.

    SEC asks every client to identify itself via User-Agent. We send a
    descriptive header per their guidance so they don't rate-limit us.
    Cached for 7 days via st.cache_data when imported under Streamlit;
    a plain function otherwise so unit tests stay simple.
    """
    try:
        import requests
    except ImportError:
        return {}
    try:
        # SEC requires a User-Agent in "Name email@domain" form per
        # https://www.sec.gov/os/accessing-edgar-data. Browser-style or
        # URL-style User-Agents get a 403. Override via env var if you
        # want to identify yourself differently.
        ua = os.environ.get(
            "SEC_USER_AGENT",
            "StockIQ ilapogusandeep stockiq@example.com",
        )
        headers = {"User-Agent": ua, "Accept": "application/json"}
        r = requests.get(_SEC_URL, headers=headers, timeout=10)
        if r.status_code != 200:
            return {}
        raw = r.json() or {}
        out: dict[str, str] = {}
        for entry in raw.values():
            sym = (entry.get("ticker") or "").strip().upper()
            name = (entry.get("title") or "").strip().title()
            if sym and name:
                out[sym] = name
        return out
    except Exception:
        return {}


def _load_searched() -> dict[str, str]:
    """Read previously-searched tickers from disk. Empty dict on failure."""
    try:
        if _SEARCHED_PATH.exists():
            with _SEARCHED_PATH.open() as f:
                data = json.load(f)
            if isinstance(data, dict):
                return {str(k).upper(): str(v) for k, v in data.items() if k and v}
    except Exception:
        pass
    return {}


def remember_ticker(ticker: str, company_name: Optional[str]) -> None:
    """Persist a ticker the user just analyzed so it appears in the
    dropdown next session. Silently no-ops if the filesystem isn't
    writable (Streamlit Cloud may run read-only on some paths)."""
    if not ticker or not company_name:
        return
    ticker = ticker.strip().upper()
    if ticker in POPULAR_TICKERS:
        return  # already in the curated set, nothing to add
    try:
        _SEARCHED_PATH.parent.mkdir(parents=True, exist_ok=True)
        existing = _load_searched()
        if existing.get(ticker) == company_name:
            return
        existing[ticker] = company_name
        with _SEARCHED_PATH.open("w") as f:
            json.dump(existing, f, indent=2, sort_keys=True)
    except Exception:
        pass


def get_all_tickers(include_sec: bool = True) -> dict[str, str]:
    """Merge POPULAR + searched + SEC into one dict for the selectbox.

    Curated POPULAR_TICKERS names win on conflict so manually-set display
    names ("Alphabet Inc. (Class A)") aren't overwritten by SEC's flat
    capitalization. Setting ``include_sec=False`` skips the network call
    — useful for tests or offline runs.
    """
    merged: dict[str, str] = {}
    if include_sec:
        # Try cached fetch via Streamlit if available; fall back to direct.
        sec = {}
        try:
            import streamlit as st
            sec = st.cache_data(ttl=86400 * 7, show_spinner=False)(_fetch_sec_tickers)()
        except Exception:
            sec = _fetch_sec_tickers()
        merged.update(sec)
    merged.update(_load_searched())
    merged.update(POPULAR_TICKERS)  # curated wins on conflict
    return merged
