"""Offline tests for live-provider adapters. These tests never call the network."""
import json
from unittest.mock import patch

import pytest

from travel_planner.utils.live_travel_provider import (
    ProviderError,
    parse_flight_results,
    parse_hotel_results,
    parse_train_results,
    search_google_flights_once,
    search_google_hotels_once,
    search_trains_between_once,
)


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_flight_search_makes_one_request_and_uses_parameters():
    payload = {"best_flights": [{"price": 4321, "total_duration": 155, "flights": [
        {"airline": "Example Air", "departure_airport": {"id": "MAA", "time": "08:00"},
         "arrival_airport": {"id": "BLR", "time": "10:35"}}
    ]}]}
    with patch("travel_planner.utils.live_travel_provider.urlopen", return_value=FakeResponse(payload)) as mocked:
        result = search_google_flights_once("MAA", "BLR", "2026-11-01", "2026-11-03", 1, api_key="test-secret")
    assert mocked.call_count == 1
    url = mocked.call_args.args[0].full_url
    assert "engine=google_flights" in url
    assert "departure_id=MAA" in url and "arrival_id=BLR" in url
    assert "return_date=2026-11-03" in url
    assert result["best_flights"][0]["price"] == 4321
    assert "test-secret" not in repr(result)


def test_hotels_search_makes_one_request():
    payload = {"properties": [{"name": "Example Hotel", "overall_rating": 4.2,
                                "rate_per_night": {"lowest": "₹2,000", "extracted_lowest": 2000},
                                "total_rate": {"lowest": "₹4,000", "extracted_lowest": 4000}}]}
    with patch("travel_planner.utils.live_travel_provider.urlopen", return_value=FakeResponse(payload)) as mocked:
        result = search_google_hotels_once("Chennai", "2026-11-01", "2026-11-03", 1, api_key="test-secret")
    assert mocked.call_count == 1
    url = mocked.call_args.args[0].full_url
    assert "engine=google_hotels" in url
    assert "check_in_date=2026-11-01" in url
    assert "check_out_date=2026-11-03" in url
    parsed = parse_hotel_results(result)
    assert parsed[0]["name"] == "Example Hotel"
    assert parsed[0]["total_inr"] == 4000


def test_railradar_schedule_makes_one_request_and_authenticates():
    payload = {"success": True, "data": {"trains": [{
        "train": {"number": "12007", "name": "Shatabdi"},
        "from": {"departure": "06:00"}, "to": {"arrival": "08:15"}, "duration": 135,
    }]}}
    with patch("travel_planner.utils.live_travel_provider.urlopen", return_value=FakeResponse(payload)) as mocked:
        result = search_trains_between_once("KPD", "MAS", "2026-11-01", api_key="rail-test-key")
    assert mocked.call_count == 1
    request = mocked.call_args.args[0]
    assert request.full_url.endswith("date=2026-11-01&live=false")
    assert request.get_header("Authorization") == "Bearer rail-test-key"
    assert parse_train_results(result)[0]["number"] == "12007"


def test_http_failure_is_not_retried():
    from urllib.error import URLError
    with patch("travel_planner.utils.live_travel_provider.urlopen", side_effect=URLError("offline")) as mocked:
        with pytest.raises(ProviderError, match="network error"):
            search_google_hotels_once("Chennai", "2026-11-01", "2026-11-03", api_key="test-key")
    assert mocked.call_count == 1


def test_missing_key_does_not_make_request():
    with patch.dict("os.environ", {}, clear=True), patch("travel_planner.utils.live_travel_provider.urlopen") as mocked:
        with pytest.raises(ProviderError, match="SERPAPI_API_KEY"):
            search_google_hotels_once("Chennai", "2026-11-01", "2026-11-03")
    mocked.assert_not_called()


def test_flight_and_hotel_validation_prevents_calls():
    with patch("travel_planner.utils.live_travel_provider.urlopen") as mocked:
        with pytest.raises(ValueError, match="IATA"):
            search_google_flights_once("Vellore", "MAA", "2026-11-01", "2026-11-03", api_key="k")
        with pytest.raises(ValueError, match="after check-in"):
            search_google_hotels_once("Chennai", "2026-11-03", "2026-11-03", api_key="k")
    mocked.assert_not_called()
