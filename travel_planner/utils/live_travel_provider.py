"""Low-call, no-retry adapters for SerpApi (Google Flights/Hotels) and RailRadar.

Each function performs at most one HTTP request. It never logs API keys and doesn't
retry failed requests. Google Flights/Hotels prices are search-result snapshots, not
booking guarantees. RailRadar's free sandbox route endpoint is used only for train
schedules; this module does not claim free-tier seat inventory or fare access.
"""
from __future__ import annotations

import json
import os
import re
from datetime import date
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

SERPAPI_ENDPOINT = "https://serpapi.com/search.json"
RAILRADAR_BASE = "https://api.railradar.in/v1"


class ProviderError(RuntimeError):
    """Safe, displayable provider error; never contains credential-bearing URLs."""


def _clean_error(message: str, secret: str | None = None) -> str:
    message = str(message or "Unknown provider error").replace("\n", " ").strip()
    if secret:
        message = message.replace(secret, "[redacted]")
    return message[:300]


def _get_json_once(provider: str, url: str, *, headers: dict[str, str] | None = None,
                   secret: str | None = None) -> dict[str, Any]:
    """Issue exactly one GET attempt; deliberately no retry/backoff logic."""
    req = Request(url, headers={"User-Agent": "AITravelPlanner/0.1", **(headers or {})}, method="GET")
    try:
        with urlopen(req, timeout=20) as response:  # exactly one request attempt
            body = response.read().decode("utf-8", errors="replace")
        payload = json.loads(body)
        if not isinstance(payload, dict):
            raise ProviderError(f"{provider}: unexpected response format")
        if payload.get("error"):
            raise ProviderError(f"{provider}: {_clean_error(payload['error'], secret)}")
        return payload
    except HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", errors="replace")
            parsed = json.loads(body) if body else {}
            message = parsed.get("error", parsed.get("message", "")) if isinstance(parsed, dict) else ""
        except Exception:
            message = ""
        raise ProviderError(f"{provider}: HTTP {exc.code}" + (f" — {_clean_error(message, secret)}" if message else "")) from None
    except URLError as exc:
        raise ProviderError(f"{provider}: network error ({_clean_error(getattr(exc, 'reason', exc), secret)})") from None
    except TimeoutError:
        raise ProviderError(f"{provider}: request timed out; no retry was attempted") from None
    except json.JSONDecodeError:
        raise ProviderError(f"{provider}: response was not valid JSON") from None


def _iso_date(value: str | date, field: str) -> str:
    result = value.isoformat() if isinstance(value, date) else str(value)
    try:
        date.fromisoformat(result)
    except (ValueError, TypeError):
        raise ValueError(f"{field} must be a date in YYYY-MM-DD format") from None
    return result


def _serpapi_once(params: dict[str, Any], api_key: str | None = None) -> dict[str, Any]:
    key = (api_key or os.getenv("SERPAPI_API_KEY", "")).strip()
    if not key:
        raise ProviderError("SerpApi key missing. Set SERPAPI_API_KEY in .env or the hosting secret settings.")
    safe_params = {k: str(v) for k, v in params.items() if v is not None}
    safe_params["api_key"] = key
    url = f"{SERPAPI_ENDPOINT}?{urlencode(safe_params)}"
    return _get_json_once("SerpApi", url, secret=key)


def search_google_flights_once(
    departure_iata: str, arrival_iata: str, outbound_date: str | date,
    return_date: str | date | None, adults: int = 1, *, api_key: str | None = None,
) -> dict[str, Any]:
    """One Google Flights query. Requires real IATA airport codes; no autocomplete call."""
    dep, arr = departure_iata.strip().upper(), arrival_iata.strip().upper()
    if not re.fullmatch(r"[A-Z]{3}", dep) or not re.fullmatch(r"[A-Z]{3}", arr):
        raise ValueError("Enter valid 3-letter IATA airport codes for both airports")
    if dep == arr:
        raise ValueError("Departure and arrival airport codes must be different")
    if not isinstance(adults, int) or not 1 <= adults <= 9:
        raise ValueError("Adults must be between 1 and 9")
    depart = _iso_date(outbound_date, "outbound_date")
    params: dict[str, Any] = {
        "engine": "google_flights", "departure_id": dep, "arrival_id": arr,
        "outbound_date": depart, "type": "2", "adults": adults,
        "currency": "INR", "gl": "in", "hl": "en", "api_key": None,
    }
    if return_date:
        ret = _iso_date(return_date, "return_date")
        if ret <= depart:
            raise ValueError("Return date must be after the departure date")
        params["type"] = "1"
        params["return_date"] = ret
    return _serpapi_once(params, api_key)


def search_google_hotels_once(
    destination: str, check_in_date: str | date, check_out_date: str | date,
    adults: int = 1, *, api_key: str | None = None,
) -> dict[str, Any]:
    """One Google Hotels query for the destination and exact stay dates."""
    destination = destination.strip()
    if not destination:
        raise ValueError("Destination is required for hotel search")
    checkin = _iso_date(check_in_date, "check_in_date")
    checkout = _iso_date(check_out_date, "check_out_date")
    if checkout <= checkin:
        raise ValueError("Hotel check-out date must be after check-in date")
    if not isinstance(adults, int) or not 1 <= adults <= 20:
        raise ValueError("Adults must be between 1 and 20")
    params = {
        "engine": "google_hotels", "q": f"Hotels in {destination}",
        "check_in_date": checkin, "check_out_date": checkout,
        "adults": adults, "currency": "INR", "gl": "in", "hl": "en",
        "sort_by": "3", "api_key": None,
    }
    return _serpapi_once(params, api_key)


def search_trains_between_once(
    from_station: str, to_station: str, journey_date: str | date,
    *, api_key: str | None = None,
) -> dict[str, Any]:
    """One RailRadar trains-between query. Returns timetable only, not fares/seats."""
    key = (api_key or os.getenv("RAILRADAR_API_KEY", "")).strip()
    if not key:
        raise ProviderError("RailRadar key missing. Train seat/fare lookup was not attempted.")
    source, destination = from_station.strip().upper(), to_station.strip().upper()
    if not re.fullmatch(r"[A-Z0-9]{2,6}", source) or not re.fullmatch(r"[A-Z0-9]{2,6}", destination):
        raise ValueError("Enter valid source and destination station codes")
    if source == destination:
        raise ValueError("Source and destination station codes must differ")
    journey = _iso_date(journey_date, "journey_date")
    url = f"{RAILRADAR_BASE}/trains/between/{source}/{destination}?{urlencode({'date': journey, 'live': 'false'})}"
    return _get_json_once("RailRadar", url, headers={"Authorization": f"Bearer {key}"}, secret=key)


def parse_flight_results(payload: dict[str, Any], limit: int = 6) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for section in ("best_flights", "other_flights"):
        for item in payload.get(section, []) or []:
            segments = item.get("flights", []) or []
            airlines = list(dict.fromkeys(str(s.get("airline", "Unknown airline")) for s in segments))
            first = segments[0] if segments else {}
            last = segments[-1] if segments else {}
            departure = first.get("departure_airport", {}) or {}
            arrival = last.get("arrival_airport", {}) or {}
            layovers = item.get("layovers", []) or []
            duration = item.get("total_duration")
            price = item.get("price")
            rows.append({
                "airlines": ", ".join(airlines) if airlines else "Airline not listed",
                "departure": departure.get("time", ""),
                "arrival": arrival.get("time", ""),
                "departure_airport": departure.get("name", departure.get("id", "")),
                "arrival_airport": arrival.get("name", arrival.get("id", "")),
                "duration_minutes": duration if isinstance(duration, (int, float)) else None,
                "stops": max(0, len(segments) - 1) if segments else len(layovers),
                "price_inr": price if isinstance(price, (int, float)) else None,
                "price_text": str(price) if price is not None else "Price not provided",
                "kind": item.get("type", section.replace("_", " ")),
            })
            if len(rows) >= limit:
                return rows
    return rows


def _price_fields(rate: Any) -> tuple[str | None, float | None]:
    if not isinstance(rate, dict):
        return (str(rate) if rate else None, None)
    text = rate.get("lowest") or rate.get("total") or rate.get("extracted_lowest") or rate.get("extracted_total")
    amount = rate.get("extracted_lowest")
    if amount is None:
        amount = rate.get("extracted_total")
    try:
        numeric = float(amount) if amount is not None else None
    except (TypeError, ValueError):
        numeric = None
    return (str(text) if text is not None else None, numeric)


def parse_hotel_results(payload: dict[str, Any], limit: int = 8) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in payload.get("properties", []) or []:
        nightly_text, nightly_numeric = _price_fields(item.get("rate_per_night"))
        total_text, total_numeric = _price_fields(item.get("total_rate"))
        rating = item.get("overall_rating", item.get("rating"))
        rows.append({
            "name": str(item.get("name", "Hotel result")),
            "rating": rating,
            "reviews": item.get("reviews"),
            "nightly_text": nightly_text,
            "nightly_inr": nightly_numeric,
            "total_text": total_text,
            "total_inr": total_numeric,
            "link": item.get("link"),
        })
        if len(rows) >= limit:
            break
    return rows


def parse_train_results(payload: dict[str, Any], limit: int = 20) -> list[dict[str, Any]]:
    envelope = payload.get("data", payload)
    trains = envelope.get("trains", []) if isinstance(envelope, dict) else []
    rows: list[dict[str, Any]] = []
    for item in trains or []:
        train = item.get("train", {}) or {}
        source = item.get("from", {}) or {}
        target = item.get("to", {}) or {}
        live = item.get("live", {}) or {}
        rows.append({
            "number": str(train.get("number", "")),
            "name": str(train.get("name", "Train")),
            "departure": str(source.get("departure", "")),
            "arrival": str(target.get("arrival", "")),
            "duration_minutes": item.get("duration"),
            "delay_minutes": live.get("delayMinutes"),
        })
        if len(rows) >= limit:
            break
    return rows
