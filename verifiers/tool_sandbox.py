"""
Docker-based tool execution sandbox for tool_calling verification.

Executes parsed tool calls in a sandboxed environment and compares
results with expected outcomes.

This provides more rigorous verification than just checking JSON structure -
it actually executes the tool calls and validates the results.

Features:
- Deterministic mock implementations for 100+ common tools
- LLM fallback for unknown tools (with caching for determinism)
"""

import json
import logging
import hashlib
import os
from typing import Dict, Any, Optional, List
from pathlib import Path

from .docker_sandbox import DockerSandbox

logger = logging.getLogger(__name__)

# ============================================================================
# LLM Mock Fallback (for unknown tools)
# ============================================================================

# Cache file for LLM-generated mock responses (ensures determinism across runs)
LLM_MOCK_CACHE_FILE = Path(__file__).parent / ".llm_tool_cache.json"

def _load_llm_cache() -> Dict[str, Any]:
    """Load LLM mock cache from disk."""
    if LLM_MOCK_CACHE_FILE.exists():
        try:
            with open(LLM_MOCK_CACHE_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}

def _save_llm_cache(cache: Dict[str, Any]) -> None:
    """Save LLM mock cache to disk."""
    try:
        with open(LLM_MOCK_CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except IOError as e:
        logger.warning(f"Failed to save LLM mock cache: {e}")

def _get_cache_key(tool_name: str, arguments: Dict[str, Any]) -> str:
    """Generate deterministic cache key for tool call."""
    # Sort arguments for consistent hashing
    args_str = json.dumps(arguments, sort_keys=True)
    return hashlib.md5(f"{tool_name}:{args_str}".encode()).hexdigest()

def _llm_mock_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Use LLM to generate plausible mock response for unknown tool.

    Uses caching to ensure deterministic responses across runs.
    """
    # Check cache first
    cache = _load_llm_cache()
    cache_key = _get_cache_key(tool_name, arguments)

    if cache_key in cache:
        logger.debug(f"LLM mock cache hit for {tool_name}")
        return cache[cache_key]

    # Try to call LLM for mock response
    try:
        from openai import OpenAI
        client = OpenAI()  # Uses OPENAI_API_KEY env var

        prompt = f"""Generate a realistic mock JSON response for this API tool call.

Tool name: {tool_name}
Arguments: {json.dumps(arguments, indent=2)}

Requirements:
1. Return ONLY valid JSON (no markdown, no explanation)
2. Generate realistic but fake data appropriate for this tool
3. Keep response concise but complete
4. Use appropriate data types (numbers, strings, arrays, etc.)
5. If it's a search/list tool, return 3-5 items
6. If it's a get/fetch tool, return a single detailed object

Think about what this tool would realistically return based on its name and arguments."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=1000,
            temperature=0.3  # Lower temp for more consistent outputs
        )

        result = json.loads(response.choices[0].message.content)

        # Cache the result
        cache[cache_key] = result
        _save_llm_cache(cache)

        logger.info(f"LLM generated mock for unknown tool: {tool_name}")
        return result

    except ImportError:
        logger.warning("OpenAI not available for LLM mock fallback")
        return {"error": f"Unknown tool: {tool_name}", "fallback": "llm_unavailable"}
    except Exception as e:
        logger.warning(f"LLM mock fallback failed for {tool_name}: {e}")
        return {"error": f"Unknown tool: {tool_name}", "fallback_error": str(e)}


# ============================================================================
# Mock Tool Implementations
# ============================================================================

# These are Python implementations of common tools found in Nemotron dataset
# They return deterministic mock data for reproducible verification

MOCK_TOOL_IMPLEMENTATIONS = '''
import json
import hashlib
from typing import Any, Dict, List, Optional

# ============================================================================
# Mock Data Generators - Deterministic based on input
# ============================================================================

def _hash_seed(s: str) -> int:
    """Generate deterministic seed from string."""
    return int(hashlib.md5(str(s).encode()).hexdigest()[:8], 16)

def _mock_price(ticker: str) -> float:
    """Generate deterministic mock stock price."""
    seed = _hash_seed(ticker)
    return round(50 + (seed % 200) + (seed % 100) / 100, 2)

def _mock_weather(location: str) -> dict:
    """Generate deterministic mock weather."""
    seed = _hash_seed(location)
    temps = [32, 45, 55, 65, 72, 78, 85, 90, 75, 60, 48, 38]
    conditions = ["Sunny", "Cloudy", "Partly Cloudy", "Rainy", "Snowy"]
    return {
        "temperature": temps[seed % 12],
        "condition": conditions[seed % 5],
        "humidity": 30 + (seed % 50),
        "location": location
    }

def _mock_date(seed: int, year_offset: int = 0) -> str:
    """Generate deterministic mock date."""
    year = 2024 + year_offset + (seed % 2)
    month = 1 + (seed % 12)
    day = 1 + (seed % 28)
    return f"{year}-{month:02d}-{day:02d}"

def _mock_company_name(seed: int) -> str:
    """Generate deterministic company name."""
    prefixes = ["Global", "Tech", "Alpha", "Meta", "Neo", "Prime", "Apex", "Delta"]
    suffixes = ["Corp", "Inc", "Ltd", "Holdings", "Industries", "Systems", "Solutions"]
    return f"{prefixes[seed % len(prefixes)]} {suffixes[(seed // 8) % len(suffixes)]}"

def _mock_person_name(seed: int) -> str:
    """Generate deterministic person name."""
    first = ["James", "Maria", "John", "Sarah", "Michael", "Emma", "David", "Lisa"]
    last = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis"]
    return f"{first[seed % len(first)]} {last[(seed // 8) % len(last)]}"

# ============================================================================
# Finance/Stock Tools
# ============================================================================

def get_price(symbols: str = "", **kwargs) -> dict:
    """Get stock prices for comma-separated symbols."""
    symbols = symbols or kwargs.get("symbol", "AAPL")
    tickers = [s.strip().upper() for s in str(symbols).split(",")]
    return {ticker: _mock_price(ticker) for ticker in tickers}

def get_stock_price(symbol: str = "AAPL", **kwargs) -> dict:
    """Get stock price for a single symbol."""
    return {"symbol": str(symbol).upper(), "price": _mock_price(symbol)}

def stock_quote(symbol: str = "AAPL", **kwargs) -> dict:
    """Get detailed stock quote."""
    seed = _hash_seed(symbol)
    price = _mock_price(symbol)
    return {
        "symbol": str(symbol).upper(),
        "price": price,
        "open": round(price * 0.99, 2),
        "high": round(price * 1.02, 2),
        "low": round(price * 0.98, 2),
        "volume": seed % 10000000 + 1000000,
        "change": round(price * 0.01, 2),
        "changePercent": round(1.0 + (seed % 5) / 10, 2)
    }

def quote(symbol: str = "", ticker: str = "", **kwargs) -> dict:
    """Get stock quote (alias)."""
    sym = symbol or ticker or kwargs.get("symbols", "AAPL")
    return stock_quote(sym)

def historical_prices(symbol: str = "AAPL", period: str = "1m", **kwargs) -> dict:
    """Get historical stock prices."""
    seed = _hash_seed(symbol)
    base_price = _mock_price(symbol)
    prices = []
    for i in range(30):
        date = _mock_date(seed + i, -1)
        variance = ((seed + i) % 10 - 5) / 100
        prices.append({
            "date": date,
            "open": round(base_price * (1 + variance), 2),
            "high": round(base_price * (1 + variance + 0.02), 2),
            "low": round(base_price * (1 + variance - 0.02), 2),
            "close": round(base_price * (1 + variance + 0.01), 2),
            "volume": (seed + i) % 5000000 + 500000
        })
    return {"symbol": str(symbol).upper(), "period": period, "prices": prices}

def historical_dividends(symbol: str = "AAPL", **kwargs) -> dict:
    """Get historical dividends."""
    seed = _hash_seed(symbol)
    dividends = []
    for i in range(4):
        dividends.append({
            "date": _mock_date(seed + i * 90),
            "amount": round(0.2 + (seed % 10) / 10, 2),
            "type": "quarterly"
        })
    return {"symbol": str(symbol).upper(), "dividends": dividends}

def financial_income_statement(symbol: str = "AAPL", **kwargs) -> dict:
    """Get income statement."""
    seed = _hash_seed(symbol)
    revenue = (seed % 100 + 10) * 1000000000
    return {
        "symbol": str(symbol).upper(),
        "fiscalYear": 2023,
        "revenue": revenue,
        "grossProfit": int(revenue * 0.4),
        "operatingIncome": int(revenue * 0.25),
        "netIncome": int(revenue * 0.2),
        "eps": round(revenue / 1000000000 * 0.5, 2)
    }

def busca_por_simbolo(simbolo: str = "", symbol: str = "", **kwargs) -> dict:
    """Search by symbol (Portuguese API)."""
    sym = simbolo or symbol or "PETR4"
    seed = _hash_seed(sym)
    return {
        "simbolo": str(sym).upper(),
        "nome": _mock_company_name(seed),
        "preco": _mock_price(sym),
        "variacao": round((seed % 10 - 5) / 10, 2),
        "mercado": "BOVESPA"
    }

def searchsymbol(query: str = "", **kwargs) -> list:
    """Search for stock symbols."""
    seed = _hash_seed(query)
    return [
        {"symbol": f"{query[:3].upper()}{i}", "name": _mock_company_name(seed + i), "exchange": "NYSE"}
        for i in range(5)
    ]

def mostrecentshortvolume(symbol: str = "AAPL", **kwargs) -> dict:
    """Get most recent short volume."""
    seed = _hash_seed(symbol)
    return {
        "symbol": str(symbol).upper(),
        "date": _mock_date(seed),
        "shortVolume": seed % 5000000 + 100000,
        "totalVolume": seed % 20000000 + 1000000,
        "shortRatio": round((seed % 30 + 10) / 100, 2)
    }

def getcompanynames(symbols: str = "", **kwargs) -> dict:
    """Get company names for symbols."""
    syms = [s.strip().upper() for s in str(symbols).split(",")]
    return {sym: _mock_company_name(_hash_seed(sym)) for sym in syms}

def company_info(symbol: str = "AAPL", **kwargs) -> dict:
    """Get company information."""
    seed = _hash_seed(symbol)
    return {
        "symbol": str(symbol).upper(),
        "name": _mock_company_name(seed),
        "sector": ["Technology", "Healthcare", "Finance", "Energy", "Consumer"][seed % 5],
        "industry": ["Software", "Biotech", "Banking", "Oil & Gas", "Retail"][seed % 5],
        "employees": seed % 50000 + 1000,
        "ceo": _mock_person_name(seed)
    }

def free_exchange_rates(base: str = "USD", **kwargs) -> dict:
    """Get exchange rates."""
    seed = _hash_seed(base)
    return {
        "base": base.upper(),
        "date": _mock_date(seed),
        "rates": {
            "EUR": round(0.85 + (seed % 10) / 100, 4),
            "GBP": round(0.75 + (seed % 10) / 100, 4),
            "JPY": round(110 + seed % 20, 2),
            "CAD": round(1.25 + (seed % 10) / 100, 4),
            "AUD": round(1.35 + (seed % 10) / 100, 4)
        }
    }

def currency_exchange_rate_crypto(from_currency: str = "BTC", to_currency: str = "USD", **kwargs) -> dict:
    """Get crypto exchange rate."""
    seed = _hash_seed(from_currency + to_currency)
    rates = {"BTC": 45000, "ETH": 2500, "XRP": 0.5, "LTC": 150, "DOGE": 0.08}
    base_rate = rates.get(from_currency.upper(), 100)
    return {
        "from": from_currency.upper(),
        "to": to_currency.upper(),
        "rate": round(base_rate * (1 + (seed % 10) / 100), 2),
        "timestamp": "2024-01-15T12:00:00Z"
    }

def get_ethereum_price(**kwargs) -> dict:
    """Get Ethereum price."""
    return {"symbol": "ETH", "price": 2547.83, "change_24h": 2.5, "currency": "USD"}

def get_blendedrates(**kwargs) -> dict:
    """Get blended interest rates."""
    return {
        "prime": 8.5,
        "federal_funds": 5.25,
        "libor_1m": 5.45,
        "sofr": 5.31,
        "date": "2024-01-15"
    }

def getactivecurrencylist(**kwargs) -> list:
    """Get list of active currencies."""
    return [
        {"code": "USD", "name": "US Dollar", "symbol": "$"},
        {"code": "EUR", "name": "Euro", "symbol": "€"},
        {"code": "GBP", "name": "British Pound", "symbol": "£"},
        {"code": "JPY", "name": "Japanese Yen", "symbol": "¥"},
        {"code": "CAD", "name": "Canadian Dollar", "symbol": "C$"}
    ]

def commodities(commodity: str = "gold", **kwargs) -> dict:
    """Get commodity prices."""
    seed = _hash_seed(commodity)
    prices = {"gold": 2050, "silver": 23, "oil": 75, "natural_gas": 2.5, "copper": 3.8}
    base = prices.get(commodity.lower(), 100)
    return {
        "commodity": commodity,
        "price": round(base * (1 + (seed % 10) / 100), 2),
        "unit": "USD",
        "change": round((seed % 10 - 5) / 10, 2)
    }

def commodities_today(**kwargs) -> dict:
    """Get today's commodity prices."""
    return {
        "date": "2024-01-15",
        "gold": 2050.50,
        "silver": 23.15,
        "oil_wti": 75.80,
        "oil_brent": 80.25,
        "natural_gas": 2.55
    }

def finance_analytics(symbol: str = "AAPL", **kwargs) -> dict:
    """Get financial analytics."""
    seed = _hash_seed(symbol)
    return {
        "symbol": str(symbol).upper(),
        "pe_ratio": round(15 + seed % 20, 2),
        "pb_ratio": round(2 + seed % 5, 2),
        "debt_to_equity": round(0.5 + (seed % 10) / 10, 2),
        "roe": round(10 + seed % 20, 2),
        "recommendation": ["Strong Buy", "Buy", "Hold", "Sell"][seed % 4]
    }

# ============================================================================
# Weather Tools
# ============================================================================

def get_weather(location: str = "", units: str = "fahrenheit", **kwargs) -> dict:
    """Get weather for a location."""
    loc = location or kwargs.get("city", "New York")
    data = _mock_weather(loc)
    if units.lower() in ["celsius", "c", "metric"]:
        data["temperature"] = round((data["temperature"] - 32) * 5/9, 1)
        data["units"] = "celsius"
    else:
        data["units"] = "fahrenheit"
    return data

def get_current_weather(location: str = "", format: str = "fahrenheit", **kwargs) -> dict:
    """Alias for get_weather."""
    return get_weather(location or kwargs.get("city", "New York"), format)

def weather(location: str = "", city: str = "", **kwargs) -> dict:
    """Get weather (simple alias)."""
    return get_weather(location or city or "New York")

def get_weather_updates(location: str = "", **kwargs) -> dict:
    """Get weather updates."""
    loc = location or kwargs.get("city", "New York")
    base = _mock_weather(loc)
    base["alerts"] = []
    base["updated_at"] = "2024-01-15T12:00:00Z"
    return base

def getweatherforecast(location: str = "", city: str = "", **kwargs) -> dict:
    """Get weather forecast."""
    loc = location or city or "New York"
    seed = _hash_seed(loc)
    return {
        "location": loc,
        "forecast": [
            {"day": f"Day {i+1}", "high": 50 + (seed + i) % 30, "low": 30 + (seed + i) % 20,
             "condition": ["Sunny", "Cloudy", "Rainy"][(seed + i) % 3]}
            for i in range(7)
        ]
    }

def get_wind_speed(location: str = "", **kwargs) -> dict:
    """Get wind speed."""
    loc = location or kwargs.get("city", "New York")
    seed = _hash_seed(loc)
    return {
        "location": loc,
        "wind_speed": seed % 30 + 5,
        "wind_direction": ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][seed % 8],
        "unit": "mph"
    }

def get_5_day_forecast(location: str = "", **kwargs) -> dict:
    """Get 5-day forecast."""
    loc = location or kwargs.get("city", "New York")
    seed = _hash_seed(loc)
    return {
        "location": loc,
        "forecast": [
            {"date": _mock_date(seed + i), "high": 50 + (seed + i) % 30, "low": 30 + (seed + i) % 20}
            for i in range(5)
        ]
    }

def get_weather_tile(lat: float = 0, lon: float = 0, **kwargs) -> dict:
    """Get weather tile data."""
    seed = _hash_seed(f"{lat}{lon}")
    return {"lat": lat, "lon": lon, "temperature": 50 + seed % 40, "tile_id": f"tile_{seed % 1000}"}

# ============================================================================
# Location/Geo Tools
# ============================================================================

def geolocation_from_an_ip_address(ip: str = "8.8.8.8", **kwargs) -> dict:
    """Get geolocation from IP."""
    seed = _hash_seed(ip)
    cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "London", "Tokyo", "Sydney"]
    countries = ["US", "US", "US", "US", "US", "GB", "JP", "AU"]
    return {
        "ip": ip,
        "city": cities[seed % len(cities)],
        "country": countries[seed % len(countries)],
        "latitude": round(30 + (seed % 40), 4),
        "longitude": round(-120 + (seed % 60), 4),
        "timezone": "America/New_York"
    }

def geocode(address: str = "", **kwargs) -> dict:
    """Geocode an address."""
    seed = _hash_seed(address)
    return {
        "address": address,
        "latitude": round(30 + (seed % 40), 6),
        "longitude": round(-120 + (seed % 60), 6),
        "formatted_address": f"{seed % 1000} Main St, {address}"
    }

def v1_geocoding(address: str = "", q: str = "", **kwargs) -> dict:
    """Geocoding API v1."""
    addr = address or q or "New York"
    return geocode(addr)

def get_countries(**kwargs) -> list:
    """Get list of countries."""
    return [
        {"code": "US", "name": "United States"},
        {"code": "GB", "name": "United Kingdom"},
        {"code": "CA", "name": "Canada"},
        {"code": "DE", "name": "Germany"},
        {"code": "FR", "name": "France"},
        {"code": "JP", "name": "Japan"},
        {"code": "AU", "name": "Australia"},
        {"code": "BR", "name": "Brazil"}
    ]

def list_of_countries(**kwargs) -> list:
    """List of countries (alias)."""
    return get_countries()

def countries(**kwargs) -> list:
    """Countries (alias)."""
    return get_countries()

def get_city_by_filter(country: str = "", state: str = "", **kwargs) -> list:
    """Get cities by filter."""
    seed = _hash_seed(country + state)
    cities = ["Springfield", "Franklin", "Clinton", "Madison", "Georgetown", "Salem", "Bristol", "Fairview"]
    return [{"name": cities[(seed + i) % len(cities)], "population": (seed + i) % 500000 + 10000} for i in range(5)]

def get_flag_by_country_code(country_code: str = "", code: str = "", **kwargs) -> dict:
    """Get flag by country code."""
    cc = country_code or code or "US"
    return {
        "country_code": cc.upper(),
        "flag_url": f"https://flagcdn.com/w320/{cc.lower()}.png",
        "emoji": "🇺🇸" if cc.upper() == "US" else "🏳️"
    }

def get_flag_by_country_country_name(country_name: str = "", country: str = "", **kwargs) -> dict:
    """Get flag by country name."""
    name = country_name or country or "United States"
    seed = _hash_seed(name)
    codes = {"united states": "US", "united kingdom": "GB", "canada": "CA", "germany": "DE", "france": "FR"}
    code = codes.get(name.lower(), "XX")
    return {
        "country_name": name,
        "country_code": code,
        "flag_url": f"https://flagcdn.com/w320/{code.lower()}.png"
    }

# ============================================================================
# Amazon/Shopping Tools
# ============================================================================

def get_amazon_search_results(query: str = "", keyword: str = "", **kwargs) -> dict:
    """Search Amazon products."""
    q = query or keyword or "laptop"
    seed = _hash_seed(q)
    return {
        "query": q,
        "results": [
            {
                "asin": f"B0{seed + i:08X}",
                "title": f"{q.title()} Product {i+1}",
                "price": round(20 + (seed + i) % 200, 2),
                "rating": round(3 + (seed + i) % 20 / 10, 1),
                "reviews": (seed + i) % 5000 + 100
            }
            for i in range(10)
        ]
    }

def searching_product_for_amazon(query: str = "", **kwargs) -> dict:
    """Search products on Amazon."""
    return get_amazon_search_results(query)

def get_amazon_product_details(asin: str = "", **kwargs) -> dict:
    """Get Amazon product details."""
    seed = _hash_seed(asin)
    return {
        "asin": asin,
        "title": f"Product {asin}",
        "price": round(20 + seed % 500, 2),
        "rating": round(3 + (seed % 20) / 10, 1),
        "reviews_count": seed % 10000 + 100,
        "in_stock": seed % 10 > 2,
        "brand": _mock_company_name(seed),
        "category": ["Electronics", "Home", "Books", "Clothing", "Sports"][seed % 5]
    }

def get_product_details_on_amazon(asin: str = "", product_id: str = "", **kwargs) -> dict:
    """Get product details on Amazon."""
    return get_amazon_product_details(asin or product_id)

def get_amazon_product_reviews(asin: str = "", **kwargs) -> dict:
    """Get Amazon product reviews."""
    seed = _hash_seed(asin)
    return {
        "asin": asin,
        "reviews": [
            {
                "rating": 3 + (seed + i) % 3,
                "title": f"Review {i+1}",
                "body": f"This is review {i+1} for product {asin}. Great product!",
                "author": _mock_person_name(seed + i),
                "date": _mock_date(seed + i)
            }
            for i in range(5)
        ]
    }

def get_amazon_product_reviews_ratings(asin: str = "", **kwargs) -> dict:
    """Get Amazon product reviews and ratings."""
    seed = _hash_seed(asin)
    return {
        "asin": asin,
        "average_rating": round(3.5 + (seed % 15) / 10, 1),
        "total_reviews": seed % 10000 + 100,
        "rating_breakdown": {
            "5_star": seed % 60 + 20,
            "4_star": seed % 20 + 10,
            "3_star": seed % 10 + 5,
            "2_star": seed % 5 + 2,
            "1_star": seed % 5 + 1
        }
    }

def get_amazon_product_offers(asin: str = "", **kwargs) -> dict:
    """Get Amazon product offers."""
    seed = _hash_seed(asin)
    base_price = 20 + seed % 200
    return {
        "asin": asin,
        "offers": [
            {"seller": "Amazon", "price": round(base_price, 2), "condition": "New", "prime": True},
            {"seller": "ThirdParty1", "price": round(base_price * 0.9, 2), "condition": "New", "prime": False},
            {"seller": "ThirdParty2", "price": round(base_price * 0.8, 2), "condition": "Used", "prime": False}
        ]
    }

def get_offers_of_an_amazon_product(asin: str = "", **kwargs) -> dict:
    """Get offers of an Amazon product."""
    return get_amazon_product_offers(asin)

def get_product_reviews(product_id: str = "", **kwargs) -> dict:
    """Get product reviews (generic)."""
    return get_amazon_product_reviews(product_id)

def get_product_offers(product_id: str = "", **kwargs) -> dict:
    """Get product offers (generic)."""
    return get_amazon_product_offers(product_id)

def get_products(query: str = "", category: str = "", **kwargs) -> dict:
    """Get products."""
    return get_amazon_search_results(query or category)

# ============================================================================
# Sports Tools
# ============================================================================

def premier_league_standings(season: str = "2023-24", **kwargs) -> dict:
    """Get Premier League standings."""
    teams = ["Manchester City", "Arsenal", "Liverpool", "Aston Villa", "Tottenham",
             "Manchester United", "Newcastle", "Brighton", "West Ham", "Chelsea"]
    return {
        "season": season,
        "standings": [
            {"position": i+1, "team": teams[i], "played": 20, "won": 15-i, "drawn": 3, "lost": 2+i,
             "points": (15-i)*3 + 3, "gd": 30 - i*5}
            for i in range(10)
        ]
    }

def match_available(date: str = "", sport: str = "", **kwargs) -> dict:
    """Get available matches."""
    seed = _hash_seed(date + sport)
    return {
        "date": date or _mock_date(seed),
        "sport": sport or "football",
        "matches": [
            {"home": f"Team {i*2+1}", "away": f"Team {i*2+2}", "time": f"{12+i}:00", "venue": f"Stadium {i+1}"}
            for i in range(5)
        ]
    }

def search_team(query: str = "", name: str = "", **kwargs) -> list:
    """Search for teams."""
    q = query or name or "United"
    seed = _hash_seed(q)
    return [
        {"id": seed + i, "name": f"{q} FC {i+1}", "country": "England", "league": "Premier League"}
        for i in range(5)
    ]

def team_information(team_id: str = "", team: str = "", **kwargs) -> dict:
    """Get team information."""
    t = team_id or team or "1"
    seed = _hash_seed(t)
    return {
        "id": t,
        "name": f"Team {t}",
        "founded": 1900 + seed % 100,
        "stadium": f"Stadium {seed % 50}",
        "capacity": 30000 + seed % 50000,
        "manager": _mock_person_name(seed)
    }

def seasonal_statistics_quarter_analysis(team_id: str = "", season: str = "", **kwargs) -> dict:
    """Get seasonal statistics."""
    seed = _hash_seed(team_id + season)
    return {
        "team_id": team_id,
        "season": season or "2023-24",
        "quarters": [
            {"quarter": f"Q{i+1}", "wins": 5 + seed % 5, "losses": seed % 3, "draws": seed % 2}
            for i in range(4)
        ]
    }

def get_players_by_lastname(lastname: str = "", **kwargs) -> list:
    """Get players by last name."""
    seed = _hash_seed(lastname)
    return [
        {"id": seed + i, "name": f"Player {lastname} {i+1}", "team": f"Team {(seed+i) % 20}",
         "position": ["Forward", "Midfielder", "Defender", "Goalkeeper"][i % 4]}
        for i in range(5)
    ]

def get_all_live_score(**kwargs) -> dict:
    """Get all live scores."""
    return {
        "matches": [
            {"home": f"Team {i*2+1}", "away": f"Team {i*2+2}", "score": f"{i}-{i+1}", "status": "Live"}
            for i in range(5)
        ],
        "timestamp": "2024-01-15T15:30:00Z"
    }

def get_all_football_transfer_news(**kwargs) -> dict:
    """Get football transfer news."""
    seed = _hash_seed("transfers")
    return {
        "transfers": [
            {"player": _mock_person_name(seed + i), "from": f"Team {i}", "to": f"Team {i+10}",
             "fee": f"${(seed + i) % 100}M", "date": _mock_date(seed + i)}
            for i in range(5)
        ]
    }

def get_top_upcoming_current_match_list(sport: str = "football", **kwargs) -> dict:
    """Get upcoming matches."""
    seed = _hash_seed(sport)
    return {
        "sport": sport,
        "matches": [
            {"home": f"Team {i*2+1}", "away": f"Team {i*2+2}", "date": _mock_date(seed + i), "time": f"{14+i}:00"}
            for i in range(5)
        ]
    }

def prematch(match_id: str = "", **kwargs) -> dict:
    """Get prematch information."""
    seed = _hash_seed(match_id)
    return {
        "match_id": match_id,
        "home": f"Team Home",
        "away": f"Team Away",
        "date": _mock_date(seed),
        "odds": {"home": 1.5 + seed % 10 / 10, "draw": 3.5, "away": 2.5 + seed % 10 / 10}
    }

def season_schedule(team_id: str = "", season: str = "", **kwargs) -> dict:
    """Get season schedule."""
    seed = _hash_seed(team_id + season)
    return {
        "team_id": team_id,
        "season": season or "2023-24",
        "matches": [
            {"date": _mock_date(seed + i * 7), "opponent": f"Team {(seed+i) % 20}", "home": i % 2 == 0}
            for i in range(10)
        ]
    }

def daily_schedule(date: str = "", sport: str = "", **kwargs) -> dict:
    """Get daily schedule."""
    return match_available(date, sport)

# ============================================================================
# News/Media Tools
# ============================================================================

def get_list_of_news(category: str = "", topic: str = "", **kwargs) -> dict:
    """Get list of news."""
    cat = category or topic or "general"
    seed = _hash_seed(cat)
    return {
        "category": cat,
        "articles": [
            {
                "title": f"News Article {i+1} about {cat}",
                "source": ["CNN", "BBC", "Reuters", "AP", "NYT"][seed % 5],
                "url": f"https://news.example.com/{seed + i}",
                "published": _mock_date(seed + i)
            }
            for i in range(10)
        ]
    }

def article_trending(category: str = "", **kwargs) -> dict:
    """Get trending articles."""
    seed = _hash_seed(category or "trending")
    return {
        "trending": [
            {"title": f"Trending Story {i+1}", "views": (seed + i) % 100000 + 10000,
             "source": ["CNN", "BBC", "Reuters"][i % 3]}
            for i in range(5)
        ]
    }

def article_list(category: str = "", **kwargs) -> dict:
    """Get article list."""
    return get_list_of_news(category)

def get_all_climate_change_news(**kwargs) -> dict:
    """Get climate change news."""
    seed = _hash_seed("climate")
    return {
        "articles": [
            {"title": f"Climate News {i+1}", "source": "Environmental Times",
             "date": _mock_date(seed + i), "topic": "climate change"}
            for i in range(5)
        ]
    }

def media_by_url(url: str = "", **kwargs) -> dict:
    """Get media by URL."""
    seed = _hash_seed(url)
    return {
        "url": url,
        "type": ["image", "video", "audio"][seed % 3],
        "title": f"Media {seed % 1000}",
        "duration": seed % 300 if seed % 3 == 1 else None,
        "thumbnail": f"https://example.com/thumb/{seed}.jpg"
    }

def get_video(video_id: str = "", **kwargs) -> dict:
    """Get video information."""
    seed = _hash_seed(video_id)
    return {
        "id": video_id,
        "title": f"Video {video_id}",
        "duration": seed % 600 + 60,
        "views": seed % 1000000 + 1000,
        "channel": f"Channel {seed % 100}",
        "published": _mock_date(seed)
    }

def video_videoid(videoid: str = "", **kwargs) -> dict:
    """Get video by ID."""
    return get_video(videoid)

def get_youtube_video_image_link(video_id: str = "", **kwargs) -> dict:
    """Get YouTube video thumbnail."""
    return {
        "video_id": video_id,
        "thumbnail_url": f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg",
        "default_url": f"https://img.youtube.com/vi/{video_id}/default.jpg"
    }

# ============================================================================
# Search Tools
# ============================================================================

def search(query: str = "", q: str = "", **kwargs) -> dict:
    """Generic search."""
    term = query or q or "example"
    seed = _hash_seed(term)
    return {
        "query": term,
        "results": [
            {"title": f"Result {i+1} for {term}", "url": f"https://example.com/{seed + i}",
             "snippet": f"This is a result about {term}..."}
            for i in range(10)
        ]
    }

def search_web(query: str = "", num_results: int = 5, **kwargs) -> list:
    """Mock web search."""
    seed = _hash_seed(query)
    return [
        {"title": f"Result {i+1} for {query[:30]}...",
         "url": f"https://example.com/{seed + i}",
         "snippet": f"Mock snippet about {query[:50]}..."}
        for i in range(min(num_results, 10))
    ]

def search_by_query(query: str = "", **kwargs) -> dict:
    """Search by query."""
    return search(query)

def search_results(query: str = "", **kwargs) -> dict:
    """Get search results."""
    return search(query)

def hc360_search(query: str = "", keyword: str = "", **kwargs) -> dict:
    """HC360 search."""
    q = query or keyword or "product"
    return search(q)

def search_advanced(query: str = "", filters: dict = None, **kwargs) -> dict:
    """Advanced search."""
    result = search(query)
    result["filters_applied"] = filters or {}
    return result

# ============================================================================
# Food/Restaurant/Travel/Real Estate Tools
# ============================================================================

def get_restaurant_reviews(restaurant: str = "", restaurant_id: str = "", **kwargs) -> dict:
    """Get restaurant reviews."""
    r = restaurant or restaurant_id or "restaurant"
    seed = _hash_seed(r)
    return {
        "restaurant": r,
        "average_rating": round(3 + (seed % 20) / 10, 1),
        "reviews": [
            {"rating": 3 + (seed + i) % 3, "text": f"Great food! Review {i+1}",
             "author": _mock_person_name(seed + i), "date": _mock_date(seed + i)}
            for i in range(5)
        ]
    }

def find_nearby_cafes(location: str = "", lat: float = None, lon: float = None, **kwargs) -> dict:
    """Find nearby cafes."""
    loc = location or f"{lat},{lon}" or "New York"
    seed = _hash_seed(loc)
    cafes = ["Starbucks", "Blue Bottle", "Peet's", "Dunkin'", "Local Cafe"]
    return {
        "location": loc,
        "cafes": [
            {"name": f"{cafes[i % 5]} #{seed + i}", "distance": f"{0.1 + i * 0.2:.1f} mi",
             "rating": round(3.5 + (seed + i) % 15 / 10, 1)}
            for i in range(5)
        ]
    }

def get_rentals(location: str = "", city: str = "", **kwargs) -> dict:
    """Get rental listings."""
    loc = location or city or "New York"
    seed = _hash_seed(loc)
    return {
        "location": loc,
        "rentals": [
            {
                "id": f"rent_{seed + i}",
                "title": f"{1 + i % 3} BR Apartment",
                "price": 1500 + (seed + i) % 2000,
                "bedrooms": 1 + i % 3,
                "address": f"{(seed + i) % 1000} Main St"
            }
            for i in range(5)
        ]
    }

def airports_nonstop_routes_for_an_airport(airport_code: str = "", **kwargs) -> dict:
    """Get nonstop routes from airport."""
    seed = _hash_seed(airport_code)
    destinations = ["LAX", "JFK", "ORD", "DFW", "DEN", "SFO", "SEA", "MIA", "ATL", "BOS"]
    return {
        "airport": airport_code.upper(),
        "nonstop_routes": [
            {"destination": destinations[(seed + i) % len(destinations)],
             "airline": ["United", "American", "Delta", "Southwest"][i % 4],
             "frequency": f"{2 + i} daily"}
            for i in range(8)
        ]
    }

def search_flights(origin: str = "", destination: str = "", date: str = "", **kwargs) -> dict:
    """Search flights."""
    seed = _hash_seed(origin + destination + date)
    return {
        "origin": origin,
        "destination": destination,
        "date": date or _mock_date(seed),
        "flights": [
            {
                "airline": ["United", "American", "Delta"][i % 3],
                "departure": f"{6 + i * 2}:00",
                "arrival": f"{9 + i * 2}:30",
                "price": 200 + (seed + i) % 300,
                "stops": i % 2
            }
            for i in range(5)
        ]
    }

# ============================================================================
# Entertainment/Events Tools
# ============================================================================

def get_quote_of_the_day(**kwargs) -> dict:
    """Get quote of the day."""
    quotes = [
        ("The only way to do great work is to love what you do.", "Steve Jobs"),
        ("Innovation distinguishes between a leader and a follower.", "Steve Jobs"),
        ("Stay hungry, stay foolish.", "Steve Jobs"),
        ("Be the change you wish to see in the world.", "Mahatma Gandhi"),
        ("In the middle of difficulty lies opportunity.", "Albert Einstein")
    ]
    seed = _hash_seed("today")
    quote, author = quotes[seed % len(quotes)]
    return {"quote": quote, "author": author, "date": "2024-01-15"}

def get_tv_show_schedule(date: str = "", channel: str = "", **kwargs) -> dict:
    """Get TV show schedule."""
    seed = _hash_seed(date + channel)
    return {
        "date": date or _mock_date(seed),
        "channel": channel or "ABC",
        "schedule": [
            {"time": f"{8 + i * 2}:00", "show": f"Show {i+1}", "duration": 60}
            for i in range(8)
        ]
    }

def get_interesting_questions(**kwargs) -> list:
    """Get interesting questions."""
    return [
        {"question": "What is the meaning of life?", "category": "Philosophy"},
        {"question": "How does quantum computing work?", "category": "Technology"},
        {"question": "What causes aurora borealis?", "category": "Science"},
        {"question": "Why is the sky blue?", "category": "Science"},
        {"question": "How do black holes form?", "category": "Astronomy"}
    ]

def get_top_bounty_questions_on_stack_over_flow(tag: str = "", **kwargs) -> dict:
    """Get top bounty questions on Stack Overflow."""
    seed = _hash_seed(tag or "python")
    return {
        "tag": tag or "all",
        "questions": [
            {
                "id": seed + i,
                "title": f"Question about {tag or 'programming'} #{i+1}",
                "bounty": (seed + i) % 500 + 50,
                "votes": (seed + i) % 100,
                "answers": (seed + i) % 10
            }
            for i in range(10)
        ]
    }

def click_to_bet_now(event_id: str = "", **kwargs) -> dict:
    """Get betting information (mock)."""
    seed = _hash_seed(event_id)
    return {
        "event_id": event_id,
        "event": f"Event {event_id}",
        "odds": {"team1": 1.5 + seed % 20 / 10, "team2": 2.0 + seed % 20 / 10, "draw": 3.5},
        "status": "open"
    }

def venue_available(venue_id: str = "", date: str = "", **kwargs) -> dict:
    """Check venue availability."""
    seed = _hash_seed(venue_id + date)
    return {
        "venue_id": venue_id,
        "date": date or _mock_date(seed),
        "available": seed % 3 > 0,
        "capacity": 500 + seed % 2000,
        "price": 1000 + seed % 5000
    }

def get_events_using_get_method(location: str = "", date: str = "", **kwargs) -> dict:
    """Get events."""
    seed = _hash_seed(location + date)
    return {
        "location": location or "New York",
        "date": date or _mock_date(seed),
        "events": [
            {"name": f"Event {i+1}", "venue": f"Venue {(seed + i) % 20}",
             "time": f"{18 + i % 4}:00", "category": ["Music", "Sports", "Theater"][i % 3]}
            for i in range(5)
        ]
    }

def search_for_event_s(query: str = "", location: str = "", date: str = "", **kwargs) -> dict:
    """Search for events."""
    q = query or kwargs.get("keyword", "concert")
    seed = _hash_seed(q + location + date)
    return {
        "query": q,
        "location": location or "Any",
        "date": date or _mock_date(seed),
        "events": [
            {
                "id": f"evt_{seed + i}",
                "name": f"{q.title()} Event {i+1}",
                "venue": f"Venue {(seed + i) % 20}",
                "date": _mock_date(seed + i),
                "time": f"{18 + i % 4}:00",
                "price": 25 + (seed + i) % 100
            }
            for i in range(5)
        ]
    }

# ============================================================================
# Environment Tools
# ============================================================================

def carbon_dioxide_endpoint(location: str = "", year: str = "", **kwargs) -> dict:
    """Get CO2 data."""
    seed = _hash_seed(location + year)
    return {
        "location": location or "Global",
        "year": year or "2024",
        "co2_ppm": 415 + seed % 10,
        "change_from_previous_year": round(2 + (seed % 10) / 10, 2),
        "source": "NOAA"
    }

# ============================================================================
# Utility/Tech Tools
# ============================================================================

def whois(domain: str = "", **kwargs) -> dict:
    """WHOIS lookup."""
    seed = _hash_seed(domain)
    return {
        "domain": domain,
        "registrar": ["GoDaddy", "Namecheap", "Google Domains", "Cloudflare"][seed % 4],
        "created": _mock_date(seed, -10),
        "expires": _mock_date(seed, 1),
        "status": "active"
    }

def get_business_information(domain: str = "", **kwargs) -> dict:
    """Get business information from domain."""
    seed = _hash_seed(domain)
    # Extract business name from domain
    name_part = domain.replace(".com", "").replace(".net", "").replace(".org", "")
    name_part = name_part.replace("-", " ").replace("_", " ").title()

    streets = ["Main St", "Oak Ave", "Commerce Blvd", "Market St", "Business Park Dr"]
    cities = ["Birmingham", "Montgomery", "Huntsville", "Mobile", "Tuscaloosa"]

    return {
        "domain": domain,
        "business_name": f"{name_part}",
        "phone": f"({200 + seed % 800}) {100 + seed % 900}-{1000 + seed % 9000}",
        "email": f"contact@{domain}",
        "address": {
            "street": f"{100 + seed % 900} {streets[seed % len(streets)]}",
            "city": cities[seed % len(cities)],
            "state": "AL",
            "zip": f"{35000 + seed % 1000}"
        },
        "website": f"https://{domain}",
        "hours": "Mon-Fri 9:00 AM - 5:00 PM",
        "description": f"Professional services provided by {name_part}."
    }

def validation_endpoint(data: str = "", **kwargs) -> dict:
    """Validate data."""
    return {"valid": True, "data": data, "message": "Validation passed"}

def user_information(user_id: str = "", **kwargs) -> dict:
    """Get user information."""
    seed = _hash_seed(user_id)
    return {
        "user_id": user_id,
        "name": _mock_person_name(seed),
        "email": f"user{seed % 1000}@example.com",
        "created": _mock_date(seed, -2)
    }

def get_pin_info(pin: str = "", **kwargs) -> dict:
    """Get PIN/postal code info."""
    seed = _hash_seed(pin)
    return {
        "pin": pin,
        "city": ["New York", "Los Angeles", "Chicago", "Houston"][seed % 4],
        "state": ["NY", "CA", "IL", "TX"][seed % 4],
        "country": "US"
    }

def get_data_info(data_id: str = "", **kwargs) -> dict:
    """Get data info."""
    seed = _hash_seed(data_id)
    return {
        "id": data_id,
        "type": ["json", "csv", "xml"][seed % 3],
        "size": seed % 10000 + 100,
        "created": _mock_date(seed)
    }

def get_detect(text: str = "", **kwargs) -> dict:
    """Detect language."""
    return {"text": text[:50], "language": "en", "confidence": 0.95}

def healthz_get(**kwargs) -> dict:
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": "2024-01-15T12:00:00Z"}

def list_models(**kwargs) -> list:
    """List available models."""
    return [
        {"id": "gpt-4", "name": "GPT-4", "provider": "OpenAI"},
        {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo", "provider": "OpenAI"},
        {"id": "claude-3", "name": "Claude 3", "provider": "Anthropic"}
    ]

def v1_caloriesburned(activity: str = "", duration: int = 30, weight: int = 70, **kwargs) -> dict:
    """Calculate calories burned."""
    activities = {"running": 10, "walking": 4, "cycling": 8, "swimming": 9, "yoga": 3}
    rate = activities.get(activity.lower(), 5)
    return {
        "activity": activity,
        "duration_minutes": duration,
        "weight_kg": weight,
        "calories_burned": int(rate * duration * weight / 70)
    }

def v1_tts_languages(**kwargs) -> list:
    """Get TTS languages."""
    return [
        {"code": "en-US", "name": "English (US)"},
        {"code": "en-GB", "name": "English (UK)"},
        {"code": "es-ES", "name": "Spanish"},
        {"code": "fr-FR", "name": "French"},
        {"code": "de-DE", "name": "German"},
        {"code": "ja-JP", "name": "Japanese"}
    ]

def mytemplate(template_id: str = "", **kwargs) -> dict:
    """Get template."""
    seed = _hash_seed(template_id)
    return {
        "id": template_id,
        "name": f"Template {template_id}",
        "content": f"This is template content for {template_id}",
        "created": _mock_date(seed)
    }

def load_post_v2(post_id: str = "", **kwargs) -> dict:
    """Load post."""
    seed = _hash_seed(post_id)
    return {
        "id": post_id,
        "title": f"Post {post_id}",
        "content": f"Content of post {post_id}...",
        "author": _mock_person_name(seed),
        "created": _mock_date(seed),
        "likes": seed % 1000
    }

def documentdownload(document_id: str = "", **kwargs) -> dict:
    """Download document."""
    return {
        "document_id": document_id,
        "download_url": f"https://example.com/docs/{document_id}",
        "filename": f"document_{document_id}.pdf",
        "size_bytes": 1024 * ((_hash_seed(document_id) % 100) + 1)
    }

def create_thumbnail(url: str = "", width: int = 100, height: int = 100, **kwargs) -> dict:
    """Create thumbnail."""
    return {
        "original_url": url,
        "thumbnail_url": f"https://thumbnails.example.com/{_hash_seed(url)}.jpg",
        "width": width,
        "height": height
    }

def v_card_qr_code(name: str = "", email: str = "", phone: str = "", **kwargs) -> dict:
    """Generate vCard QR code."""
    return {
        "name": name,
        "email": email,
        "phone": phone,
        "qr_url": f"https://qr.example.com/{_hash_seed(name + email)}.png"
    }

def instant_webpage_pdf_api(url: str = "", **kwargs) -> dict:
    """Convert webpage to PDF."""
    return {
        "url": url,
        "pdf_url": f"https://pdf.example.com/{_hash_seed(url)}.pdf",
        "pages": 1 + _hash_seed(url) % 10,
        "size_bytes": 50000 + _hash_seed(url) % 100000
    }

def make_todo_list(items: list = None, **kwargs) -> dict:
    """Create todo list."""
    items = items or ["Task 1", "Task 2", "Task 3"]
    return {
        "id": f"todo_{_hash_seed(str(items))}",
        "items": [{"task": item, "completed": False} for item in items],
        "created": "2024-01-15T12:00:00Z"
    }

def create_new_account(email: str = "", username: str = "", **kwargs) -> dict:
    """Create new account (mock)."""
    return {
        "success": True,
        "user_id": f"user_{_hash_seed(email + username)}",
        "email": email,
        "username": username,
        "message": "Account created successfully"
    }

# ============================================================================
# Misc Tools
# ============================================================================

def tema(topic: str = "", **kwargs) -> dict:
    """Get topic information (Portuguese)."""
    seed = _hash_seed(topic)
    return {
        "tema": topic,
        "descricao": f"Information about {topic}",
        "artigos": seed % 100 + 10
    }

def islamiblockchain_islamicoin_main_cs_txt(**kwargs) -> dict:
    """Islamic blockchain info."""
    return {
        "name": "IslamiCoin",
        "symbol": "ISLAMI",
        "blockchain": "Ethereum",
        "sharia_compliant": True,
        "price_usd": 0.05
    }

def indicator_route(indicator: str = "", **kwargs) -> dict:
    """Get indicator data."""
    seed = _hash_seed(indicator)
    return {
        "indicator": indicator,
        "value": round(50 + (seed % 50), 2),
        "change": round((seed % 10 - 5) / 10, 2),
        "date": _mock_date(seed)
    }

def boxes(box_id: str = "", **kwargs) -> dict:
    """Get box information."""
    seed = _hash_seed(box_id)
    return {
        "id": box_id,
        "dimensions": {"length": 10 + seed % 20, "width": 10 + seed % 15, "height": 5 + seed % 10},
        "weight": round(1 + (seed % 50) / 10, 1)
    }

def api(endpoint: str = "", **kwargs) -> dict:
    """Generic API call."""
    return {"endpoint": endpoint, "status": "success", "data": {"message": f"Response from {endpoint}"}}

def analyze_get(data: str = "", **kwargs) -> dict:
    """Analyze data."""
    return {"input": data[:100] if data else "", "analysis": "Analysis complete", "score": 0.85}

def job(job_id: str = "", query: str = "", **kwargs) -> dict:
    """Job search/info."""
    q = job_id or query or "engineer"
    seed = _hash_seed(q)
    return {
        "query": q,
        "jobs": [
            {
                "id": f"job_{seed + i}",
                "title": f"Software Engineer {i+1}",
                "company": _mock_company_name(seed + i),
                "location": ["New York", "San Francisco", "Remote"][i % 3],
                "salary": f"${80 + (seed + i) % 80}K - ${120 + (seed + i) % 80}K"
            }
            for i in range(5)
        ]
    }

# ============================================================================
# Additional Tools (from j5 experiment log analysis)
# ============================================================================

def searchstores(query: str = "", location: str = "", **kwargs) -> dict:
    """Search for stores."""
    q = query or kwargs.get("keyword", "store")
    seed = _hash_seed(q + location)
    return {
        "query": q,
        "location": location or "nearby",
        "stores": [
            {"name": f"Store {i+1}", "address": f"{100 + seed + i} Main St",
             "distance": f"{0.5 + i * 0.3:.1f} mi", "rating": round(3.5 + (seed + i) % 15 / 10, 1)}
            for i in range(7)
        ]
    }

def search_video(query: str = "", q: str = "", **kwargs) -> dict:
    """Search for videos."""
    term = query or q or kwargs.get("keyword", "video")
    seed = _hash_seed(term)
    return {
        "query": term,
        "videos": [
            {"id": f"vid_{seed + i}", "title": f"{term} Video {i+1}",
             "channel": f"Channel {(seed + i) % 100}", "views": (seed + i) % 1000000 + 1000,
             "duration": f"{(seed + i) % 10 + 1}:{(seed + i) % 60:02d}"}
            for i in range(10)
        ]
    }

def cheapest_tickets_grouped_by_month(origin: str = "", destination: str = "", **kwargs) -> dict:
    """Get cheapest flight tickets by month."""
    seed = _hash_seed(origin + destination)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return {
        "origin": origin or "NYC",
        "destination": destination or "LAX",
        "currency": "USD",
        "tickets_by_month": [
            {"month": months[i], "price": 150 + (seed + i * 17) % 300, "departure_date": f"2024-{i+1:02d}-15"}
            for i in range(12)
        ]
    }

def insights(query: str = "", topic: str = "", **kwargs) -> dict:
    """Get insights/analytics data."""
    q = query or topic or "general"
    seed = _hash_seed(q)
    return {
        "topic": q,
        "insights": [
            {"title": f"Insight {i+1}", "value": (seed + i) % 100, "trend": ["up", "down", "stable"][(seed + i) % 3],
             "change_percent": round((seed + i) % 30 - 15, 1)}
            for i in range(5)
        ],
        "generated_at": "2024-01-15T12:00:00Z"
    }

def geocoding(address: str = "", q: str = "", **kwargs) -> dict:
    """Geocoding - convert address to coordinates."""
    addr = address or q or kwargs.get("query", "New York")
    seed = _hash_seed(addr)
    return {
        "address": addr,
        "latitude": round(30 + (seed % 40) + (seed % 100) / 100, 6),
        "longitude": round(-120 + (seed % 60) + (seed % 100) / 100, 6),
        "formatted_address": f"{seed % 1000} Main St, {addr}",
        "place_id": f"place_{seed}"
    }

def list_orders_received(user_id: str = "", status: str = "", **kwargs) -> dict:
    """List orders received."""
    seed = _hash_seed(user_id + status)
    statuses = ["pending", "processing", "shipped", "delivered"]
    return {
        "user_id": user_id or "default_user",
        "orders": [
            {"order_id": f"ORD-{seed + i:06d}", "status": statuses[(seed + i) % 4],
             "total": round(20 + (seed + i) % 200, 2), "date": _mock_date(seed + i),
             "items": (seed + i) % 5 + 1}
            for i in range(5)
        ]
    }

def prices(symbol: str = "", symbols: str = "", **kwargs) -> dict:
    """Get prices (generic)."""
    syms = symbol or symbols or kwargs.get("ticker", "AAPL")
    tickers = [s.strip().upper() for s in str(syms).split(",")]
    return {ticker: {"price": _mock_price(ticker), "change": round(((_hash_seed(ticker) % 10) - 5) / 10, 2)} for ticker in tickers}

def divisa_tipo_divisa(divisa: str = "", **kwargs) -> dict:
    """Get currency exchange info (Spanish API)."""
    currency = divisa or kwargs.get("currency", "USD")
    seed = _hash_seed(currency)
    return {
        "divisa": currency.upper(),
        "nombre": {"USD": "Dólar estadounidense", "EUR": "Euro", "GBP": "Libra esterlina"}.get(currency.upper(), currency),
        "tipo_cambio": round(1 + (seed % 100) / 100, 4),
        "fecha": _mock_date(seed)
    }

def search_products(query: str = "", keyword: str = "", category: str = "", **kwargs) -> dict:
    """Search products."""
    q = query or keyword or category or "product"
    seed = _hash_seed(q)
    return {
        "query": q,
        "products": [
            {"id": f"prod_{seed + i}", "name": f"{q.title()} Item {i+1}",
             "price": round(10 + (seed + i) % 200, 2), "rating": round(3 + (seed + i) % 20 / 10, 1),
             "in_stock": (seed + i) % 10 > 2}
            for i in range(10)
        ]
    }

def location(ip: str = "", address: str = "", **kwargs) -> dict:
    """Get location info."""
    query = ip or address or kwargs.get("query", "")
    seed = _hash_seed(query)
    cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"]
    return {
        "query": query,
        "city": cities[seed % len(cities)],
        "country": "US",
        "latitude": round(30 + (seed % 40), 4),
        "longitude": round(-120 + (seed % 60), 4)
    }

def convert_currency_with_amount(amount: float = 1, from_currency: str = "USD", to_currency: str = "EUR", **kwargs) -> dict:
    """Convert currency with specific amount."""
    seed = _hash_seed(from_currency + to_currency)
    rate = round(0.5 + (seed % 150) / 100, 4)
    return {
        "from": from_currency.upper(),
        "to": to_currency.upper(),
        "amount": float(amount),
        "rate": rate,
        "converted": round(float(amount) * rate, 2)
    }

def sanctions_and_watch_lists_screening_name_check(name: str = "", **kwargs) -> dict:
    """Screen name against sanctions/watch lists."""
    seed = _hash_seed(name)
    return {
        "name": name,
        "screened": True,
        "matches": [],
        "risk_level": "low",
        "screening_date": _mock_date(seed),
        "lists_checked": ["OFAC", "UN", "EU", "UK"]
    }

def the_prices_for_the_airline_tickets(origin: str = "", destination: str = "", date: str = "", **kwargs) -> dict:
    """Get airline ticket prices."""
    seed = _hash_seed(origin + destination + date)
    airlines = ["United", "American", "Delta", "Southwest", "JetBlue"]
    return {
        "origin": origin or "JFK",
        "destination": destination or "LAX",
        "date": date or _mock_date(seed),
        "prices": [
            {"airline": airlines[i % len(airlines)], "price": 150 + (seed + i * 23) % 400,
             "departure": f"{6 + i * 2}:00", "arrival": f"{9 + i * 2}:30", "stops": i % 2}
            for i in range(5)
        ]
    }

def para(query: str = "", **kwargs) -> dict:
    """Generic 'para' endpoint (Portuguese 'for')."""
    seed = _hash_seed(query)
    return {
        "query": query,
        "results": [{"id": seed + i, "value": f"Result {i+1}"} for i in range(5)],
        "total": 5
    }

def panchang(date: str = "", location: str = "", **kwargs) -> dict:
    """Get Panchang (Hindu calendar) details."""
    seed = _hash_seed(date + location)
    tithis = ["Pratipada", "Dwitiya", "Tritiya", "Chaturthi", "Panchami", "Shashthi", "Saptami"]
    nakshatras = ["Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashira", "Ardra", "Punarvasu"]
    return {
        "date": date or _mock_date(seed),
        "location": location or "Delhi",
        "tithi": tithis[seed % len(tithis)],
        "nakshatra": nakshatras[seed % len(nakshatras)],
        "yoga": f"Yoga {seed % 27 + 1}",
        "karana": f"Karana {seed % 11 + 1}",
        "sunrise": "06:30",
        "sunset": "18:15"
    }

def getforecastweather(location: str = "", city: str = "", **kwargs) -> dict:
    """Get weather forecast (alias for getweatherforecast)."""
    return getweatherforecast(location or city, **kwargs)

# ============================================================================
# Original Tools (kept for backwards compatibility)
# Note: All functions include **kwargs to handle unexpected arguments gracefully
# ============================================================================

def check_valid_vin(vin: str = "", **kwargs) -> dict:
    """Check if VIN is valid."""
    vin = str(vin).upper().strip()
    if len(vin) != 17:
        return {"valid": False, "error": "VIN must be 17 characters"}
    if any(c in vin for c in "IOQ"):
        return {"valid": False, "error": "VIN cannot contain I, O, or Q"}
    seed = _hash_seed(vin)
    makes = ["Ford", "Toyota", "Honda", "Chevrolet", "BMW", "Tesla"]
    models = ["F-150", "Camry", "Civic", "Silverado", "3 Series", "Model 3"]
    return {
        "valid": True,
        "vin_details": {
            "make": makes[seed % len(makes)],
            "model": models[seed % len(models)],
            "year": 2015 + (seed % 10),
            "vehicle_type": "Truck" if seed % 3 == 0 else "Sedan",
        }
    }

def check_valid_registration(reg: str = "", state: str = "", **kwargs) -> dict:
    """Check vehicle registration."""
    seed = _hash_seed(f"{reg}{state}")
    return {
        "valid": True,
        "registration": str(reg).upper(),
        "state": str(state).upper(),
        "status": "Active" if seed % 10 > 2 else "Expired",
        "expiry": f"202{5 + seed % 3}-{1 + seed % 12:02d}-{1 + seed % 28:02d}"
    }

def ppsr_lookup_by_vin(vin: str = "", **kwargs) -> dict:
    """PPSR lookup by VIN."""
    seed = _hash_seed(vin)
    return {
        "vin": str(vin).upper(),
        "security_interests": [] if seed % 5 > 0 else [{"type": "Lien", "holder": "Bank"}],
        "encumbrances": False if seed % 5 > 0 else True
    }

def calculate(expression: str = "", **kwargs) -> dict:
    """Safe math evaluation."""
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in str(expression)):
            return {"error": "Invalid characters in expression"}
        result = eval(str(expression))
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": str(e)}

def get_time(timezone: str = "UTC", **kwargs) -> dict:
    """Get current time (mock)."""
    return {"timezone": timezone, "time": "2024-01-15T12:00:00Z", "note": "Mock time"}

def send_email(to: str = "", subject: str = "", body: str = "", **kwargs) -> dict:
    """Mock send email."""
    return {"status": "sent", "to": to, "subject": subject, "message_id": f"mock-{_hash_seed(str(to) + str(subject))}"}

# Note: Removed duplicate get_price - use the main implementation above

# ============================================================================
# Tool Registry - All Available Tools
# ============================================================================

TOOL_REGISTRY = {
    # Finance/Stock
    "get_price": get_price,
    "get_stock_price": get_stock_price,
    "stock_quote": stock_quote,
    "quote": quote,
    "historical_prices": historical_prices,
    "historical_dividends": historical_dividends,
    "financial_income_statement": financial_income_statement,
    "busca_por_simbolo": busca_por_simbolo,
    "searchsymbol": searchsymbol,
    "mostrecentshortvolume": mostrecentshortvolume,
    "getcompanynames": getcompanynames,
    "company_info": company_info,
    "free_exchange_rates": free_exchange_rates,
    "currency_exchange_rate_crypto": currency_exchange_rate_crypto,
    "get_ethereum_price": get_ethereum_price,
    "get_blendedrates": get_blendedrates,
    "getactivecurrencylist": getactivecurrencylist,
    "commodities": commodities,
    "commodities_today": commodities_today,
    "finance_analytics": finance_analytics,

    # Weather
    "get_weather": get_weather,
    "get_current_weather": get_current_weather,
    "weather": weather,
    "get_weather_updates": get_weather_updates,
    "getweatherforecast": getweatherforecast,
    "get_wind_speed": get_wind_speed,
    "get_5_day_forecast": get_5_day_forecast,
    "get_weather_tile": get_weather_tile,

    # Location/Geo
    "geolocation_from_an_ip_address": geolocation_from_an_ip_address,
    "geocode": geocode,
    "v1_geocoding": v1_geocoding,
    "get_countries": get_countries,
    "list_of_countries": list_of_countries,
    "countries": countries,
    "get_city_by_filter": get_city_by_filter,
    "get_flag_by_country_code": get_flag_by_country_code,
    "get_flag_by_country_country_name": get_flag_by_country_country_name,

    # Amazon/Shopping
    "get_amazon_search_results": get_amazon_search_results,
    "searching_product_for_amazon": searching_product_for_amazon,
    "get_amazon_product_details": get_amazon_product_details,
    "get_product_details_on_amazon": get_product_details_on_amazon,
    "get_amazon_product_reviews": get_amazon_product_reviews,
    "get_amazon_product_reviews_ratings": get_amazon_product_reviews_ratings,
    "get_amazon_product_offers": get_amazon_product_offers,
    "get_offers_of_an_amazon_product": get_offers_of_an_amazon_product,
    "get_product_reviews": get_product_reviews,
    "get_product_offers": get_product_offers,
    "get_products": get_products,

    # Sports
    "premier_league_standings": premier_league_standings,
    "match_available": match_available,
    "search_team": search_team,
    "team_information": team_information,
    "seasonal_statistics_quarter_analysis": seasonal_statistics_quarter_analysis,
    "get_players_by_lastname": get_players_by_lastname,
    "get_all_live_score": get_all_live_score,
    "get_all_football_transfer_news": get_all_football_transfer_news,
    "get_top_upcoming_current_match_list": get_top_upcoming_current_match_list,
    "prematch": prematch,
    "season_schedule": season_schedule,
    "daily_schedule": daily_schedule,

    # News/Media
    "get_list_of_news": get_list_of_news,
    "article_trending": article_trending,
    "article_list": article_list,
    "get_all_climate_change_news": get_all_climate_change_news,
    "media_by_url": media_by_url,
    "get_video": get_video,
    "video_videoid": video_videoid,
    "get_youtube_video_image_link": get_youtube_video_image_link,

    # Search
    "search": search,
    "search_web": search_web,
    "search_by_query": search_by_query,
    "search_results": search_results,
    "hc360_search": hc360_search,
    "search_advanced": search_advanced,

    # Food/Restaurant/Travel/Real Estate
    "get_restaurant_reviews": get_restaurant_reviews,
    "find_nearby_cafes": find_nearby_cafes,
    "get_rentals": get_rentals,
    "airports_nonstop_routes_for_an_airport": airports_nonstop_routes_for_an_airport,
    "search_flights": search_flights,

    # Entertainment/Events
    "get_quote_of_the_day": get_quote_of_the_day,
    "get_tv_show_schedule": get_tv_show_schedule,
    "get_interesting_questions": get_interesting_questions,
    "get_top_bounty_questions_on_stack_over_flow": get_top_bounty_questions_on_stack_over_flow,
    "click_to_bet_now": click_to_bet_now,
    "venue_available": venue_available,
    "get_events_using_get_method": get_events_using_get_method,
    "search_for_event_s": search_for_event_s,

    # Environment
    "carbon_dioxide_endpoint": carbon_dioxide_endpoint,

    # Utility/Tech
    "whois": whois,
    "get_business_information": get_business_information,
    "validation_endpoint": validation_endpoint,
    "user_information": user_information,
    "get_pin_info": get_pin_info,
    "get_data_info": get_data_info,
    "get_detect": get_detect,
    "healthz_get": healthz_get,
    "list_models": list_models,
    "v1_caloriesburned": v1_caloriesburned,
    "v1_tts_languages": v1_tts_languages,
    "mytemplate": mytemplate,
    "load_post_v2": load_post_v2,
    "documentdownload": documentdownload,
    "create_thumbnail": create_thumbnail,
    "v_card_qr_code": v_card_qr_code,
    "instant_webpage_pdf_api": instant_webpage_pdf_api,
    "make_todo_list": make_todo_list,
    "create_new_account": create_new_account,

    # Misc
    "tema": tema,
    "islamiblockchain_islamicoin_main_cs_txt": islamiblockchain_islamicoin_main_cs_txt,
    "indicator_route": indicator_route,
    "boxes": boxes,
    "api": api,
    "analyze_get": analyze_get,
    "job": job,

    # Additional tools (from j5 experiment log analysis)
    "searchstores": searchstores,
    "search_video": search_video,
    "cheapest_tickets_grouped_by_month": cheapest_tickets_grouped_by_month,
    "insights": insights,
    "geocoding": geocoding,
    "list_orders_received": list_orders_received,
    "prices": prices,
    "divisa_tipo_divisa": divisa_tipo_divisa,
    "search_products": search_products,
    "location": location,
    "convert_currency_with_amount": convert_currency_with_amount,
    "sanctions_and_watch_lists_screening_name_check": sanctions_and_watch_lists_screening_name_check,
    "the_prices_for_the_airline_tickets": the_prices_for_the_airline_tickets,
    "para": para,
    "panchang": panchang,
    "getforecastweather": getforecastweather,

    # Original/Legacy
    "check_valid_vin": check_valid_vin,
    "check_valid_registration": check_valid_registration,
    "ppsr_lookup_by_vin": ppsr_lookup_by_vin,
    "calculate": calculate,
    "get_time": get_time,
    "send_email": send_email,
}

def execute_tool_call(tool_name: str, arguments: dict) -> dict:
    """Execute a tool call and return result."""
    if tool_name not in TOOL_REGISTRY:
        return {"error": f"Unknown tool: {tool_name}", "available": list(TOOL_REGISTRY.keys())[:20]}

    try:
        func = TOOL_REGISTRY[tool_name]
        result = func(**arguments)
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}
'''


class ToolSandbox:
    """
    Docker-based sandbox for executing tool calls.

    Uses deterministic mock implementations to enable reproducible
    verification of tool calling behavior.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        # Build sandbox config dict from provided config with defaults
        sandbox_config = {
            'timeout': self.config.get('timeout', 10),  # Tool calls should be fast
            'memory_limit': self.config.get('memory_limit', '128m'),
            'network_disabled': self.config.get('network_disabled', True),  # No network for safety
            'container_pool_size': self.config.get('container_pool_size', 0),  # Disable pooling by default
        }
        self.sandbox = DockerSandbox(sandbox_config)

    def execute_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute a tool call in the Docker sandbox.

        Args:
            tool_name: Name of the tool/function to call
            arguments: Arguments to pass to the tool

        Returns:
            Dict with 'success', 'result' or 'error'
        """
        # Build Python code to execute
        # Note: We pass arguments as a JSON string and parse inside to avoid
        # JSON boolean (false/true) vs Python boolean (False/True) issues
        args_json = json.dumps(arguments)
        code = f'''
{MOCK_TOOL_IMPLEMENTATIONS}

import json
args = json.loads({repr(args_json)})
result = execute_tool_call({repr(tool_name)}, args)
print(json.dumps(result))
'''

        # Execute in sandbox
        result = self.sandbox.execute(code, language="python")

        if not result.succeeded:
            return {
                "success": False,
                "error": result.stderr or "Sandbox execution failed",
            }

        # Parse output
        try:
            output = json.loads(result.stdout.strip())

            # Check if sandbox returned "Unknown tool" - use LLM fallback
            if "error" in output and "Unknown tool" in str(output.get("error", "")):
                logger.info(f"Unknown tool '{tool_name}' - trying LLM fallback")
                llm_result = _llm_mock_tool(tool_name, arguments)

                # If LLM fallback succeeded (no error key or has actual data)
                if "error" not in llm_result or len(llm_result) > 2:
                    return {"success": True, "result": llm_result, "source": "llm_fallback"}
                else:
                    # LLM fallback also failed, return original error
                    return output

            return output
        except json.JSONDecodeError:
            return {
                "success": False,
                "error": f"Invalid JSON output: {result.stdout[:200]}",
            }

    def execute_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple tool calls.

        Args:
            tool_calls: List of tool calls in OpenAI format:
                [{"function": {"name": "...", "arguments": "..."}}]

        Returns:
            List of results
        """
        results = []
        for call in tool_calls:
            func = call.get("function", {})
            name = func.get("name", "")
            args_str = func.get("arguments", "{}")

            try:
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
            except json.JSONDecodeError:
                results.append({
                    "success": False,
                    "error": f"Invalid arguments JSON: {args_str[:100]}",
                })
                continue

            result = self.execute_tool_call(name, args)
            results.append(result)

        return results

    def verify_tool_call(
        self,
        tool_call: Dict[str, Any],
        expected_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Verify a tool call produces expected result.

        Args:
            tool_call: Tool call to execute
            expected_result: Expected result

        Returns:
            Dict with 'is_correct', 'actual', 'expected', 'explanation'
        """
        func = tool_call.get("function", {})
        name = func.get("name", "")
        args_str = func.get("arguments", "{}")

        try:
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
        except json.JSONDecodeError:
            return {
                "is_correct": False,
                "explanation": f"Invalid arguments JSON",
            }

        actual = self.execute_tool_call(name, args)

        if not actual.get("success"):
            return {
                "is_correct": False,
                "actual": actual,
                "expected": expected_result,
                "explanation": f"Tool execution failed: {actual.get('error')}",
            }

        actual_result = actual.get("result", {})

        # Compare results (with some flexibility)
        is_correct = self._compare_results(actual_result, expected_result)

        return {
            "is_correct": is_correct,
            "actual": actual_result,
            "expected": expected_result,
            "explanation": "Results match" if is_correct else "Results differ",
        }

    def _compare_results(self, actual: Any, expected: Any) -> bool:
        """Compare results with flexibility for mock vs real data."""
        # Exact match
        if actual == expected:
            return True

        # For dicts, check key overlap and type matching
        if isinstance(actual, dict) and isinstance(expected, dict):
            # Check that actual has at least the expected keys
            for key in expected:
                if key not in actual:
                    return False
                # Recursive comparison for nested dicts
                if isinstance(expected[key], dict):
                    if not self._compare_results(actual[key], expected[key]):
                        return False
            return True

        # For numbers, allow small differences
        if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
            return abs(actual - expected) < 0.01 * max(abs(actual), abs(expected), 1)

        return False


# ============================================================================
# Convenience Functions
# ============================================================================

def execute_tool_call_sandboxed(
    tool_name: str,
    arguments: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Execute a single tool call in sandbox.

    Example:
        >>> result = execute_tool_call_sandboxed("get_price", {"symbols": "AAPL,GOOG"})
        >>> result["success"]
        True
        >>> "AAPL" in result["result"]
        True
    """
    sandbox = ToolSandbox()
    return sandbox.execute_tool_call(tool_name, arguments)


def verify_tool_calls(
    candidate_calls: List[Dict[str, Any]],
    expected_calls: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Verify candidate tool calls match expected calls.

    Checks:
    1. Same number of calls
    2. Same tool names
    3. Compatible arguments
    4. (Optional) Same results when executed

    Returns:
        Dict with 'is_correct', 'score', 'details'
    """
    if len(candidate_calls) != len(expected_calls):
        return {
            "is_correct": False,
            "score": 0.0,
            "details": f"Call count mismatch: {len(candidate_calls)} vs {len(expected_calls)}",
        }

    sandbox = ToolSandbox()
    matches = 0
    details = []

    for i, (cand, exp) in enumerate(zip(candidate_calls, expected_calls)):
        cand_name = cand.get("function", {}).get("name", "")
        exp_name = exp.get("function", {}).get("name", "")

        if cand_name != exp_name:
            details.append(f"Call {i}: tool name mismatch ({cand_name} vs {exp_name})")
            continue

        # Execute both and compare
        cand_result = sandbox.execute_tool_call(
            cand_name,
            json.loads(cand.get("function", {}).get("arguments", "{}"))
        )
        exp_result = sandbox.execute_tool_call(
            exp_name,
            json.loads(exp.get("function", {}).get("arguments", "{}"))
        )

        if cand_result == exp_result:
            matches += 1
            details.append(f"Call {i}: match ({cand_name})")
        else:
            details.append(f"Call {i}: result mismatch")

    score = matches / len(expected_calls) if expected_calls else 1.0

    return {
        "is_correct": score >= 0.8,
        "score": score,
        "details": "; ".join(details),
    }
