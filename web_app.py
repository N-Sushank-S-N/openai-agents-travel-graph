"""Streamlit UI for AI itinerary generation and optional live-price snapshots.

Live searches are provider-backed, user-triggered, single-attempt calls. Results are
search snapshots, not bookings or a guarantee that price/availability will persist.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import quote_plus

import streamlit as st
from dotenv import load_dotenv
from travel_planner.utils.live_travel_provider import (
    parse_flight_results,
    parse_hotel_results,
    parse_train_results,
    search_google_flights_once,
    search_google_hotels_once,
    search_trains_between_once,
)

ROOT = Path(__file__).resolve().parent
# Load local development secrets when running from the repository root. Render uses dashboard env vars.
load_dotenv(ROOT / ".env")

st.set_page_config(page_title="AI Travel Planner", page_icon="🧭", layout="wide")
st.title("🧭 AI Travel Planner")
st.caption("AI itinerary planning with optional low-quota flight/hotel price snapshots and train timetable lookup.")
st.info(
    "The planner generates itinerary guidance from AI and web research. When configured, the comparison "
    "workspace can fetch one Google Flights snapshot and one Google Hotels snapshot per exact trip search "
    "through SerpApi, plus an optional RailRadar timetable lookup. Results are not reservations; confirm fares, "
    "room prices, and availability on the linked provider before booking. This app never books tickets."
)

plan_tab, compare_tab = st.tabs(["🗺️ Plan itinerary", "🔎 Compare live prices"])

with plan_tab:
    st.subheader("Plan your trip")
    with st.form("travel_request"):
        left, right = st.columns(2)
        with left:
            origin = st.text_input("Starting point", value="Vellore", key="plan_origin")
            days = st.number_input("Trip duration (days)", min_value=1, max_value=30, value=2)
            travelers = st.number_input("Travelers", min_value=1, max_value=20, value=1)
        with right:
            destination = st.text_input("Destination", value="Chennai", key="plan_destination")
            min_budget = st.number_input("Minimum budget (INR)", min_value=0, value=5000, step=500)
            max_budget = st.number_input("Maximum budget (INR)", min_value=0, value=8000, step=500)
        submitted = st.form_submit_button("Generate itinerary", type="primary", use_container_width=True)

    if submitted:
        origin, destination = origin.strip(), destination.strip()
        if not origin or not destination:
            st.error("Enter both a starting point and a destination.")
            st.stop()
        if max_budget < min_budget:
            st.error("Maximum budget must be greater than or equal to minimum budget.")
            st.stop()

        # The current CLI configuration treats Supabase credentials as required.
        missing = []
        if not (os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")):
            missing.append("GEMINI_API_KEY")
        if not os.getenv("SUPABASE_URL"):
            missing.append("SUPABASE_URL")
        if not os.getenv("SUPABASE_KEY"):
            missing.append("SUPABASE_KEY")
        if missing:
            st.error("Missing required deployment environment variables: " + ", ".join(missing))
            st.stop()

        query = (
            f"Plan a {int(days)}-day trip for {int(travelers)} traveler(s) from {origin} to {destination}. "
            f"Budget: INR {int(min_budget)}-{int(max_budget)}. Use web research for destination-specific guidance. "
            "Clearly label estimated costs and do not invent bookable offers or live availability. "
            "Do not report flight, hotel, train, or bus prices as verified unless a provider explicitly supplies them."
        )
        command = [
            sys.executable, "-m", "travel_planner.main", "--query", query,
            "--origin", origin, "--destination", destination,
            "--travelers", str(int(travelers)), "--budget", f"{int(min_budget)}-{int(max_budget)}",
        ]
        with st.spinner("Planning your trip. This can take a couple of minutes…"):
            try:
                child_env = os.environ.copy()
                child_env["PYTHONIOENCODING"] = "utf-8"
                result = subprocess.run(
                    command, cwd=ROOT, env=child_env, capture_output=True,
                    text=True, encoding="utf-8", errors="replace", timeout=300, check=False,
                )
            except subprocess.TimeoutExpired:
                st.error("Planning timed out after 5 minutes. Check service logs before retrying.")
                st.stop()
            except Exception as exc:
                st.error(f"Could not start the travel planner: {type(exc).__name__}: {exc}")
                st.stop()

        output = (result.stdout or "").strip()
        error_output = (result.stderr or "").strip()
        if output:
            st.subheader("Your itinerary")
            st.markdown(output)
        if result.returncode != 0:
            st.error(f"The planner exited with code {result.returncode}.")
            if error_output:
                with st.expander("Execution log (last 40 lines)"):
                    st.code("\n".join(error_output.splitlines()[-40:]))
        elif error_output:
            with st.expander("Execution log"):
                st.code("\n".join(error_output.splitlines()[-40:]))

with compare_tab:
    st.subheader("Fetch live price snapshots")
    st.write(
        "SerpApi can return structured Google Flights and Google Hotels search results. "
        "Each provider is called only after you press the button, at most once for this exact trip search "
        "in this browser session, and never automatically retried. If no SerpApi key is configured, no request is made."
    )
    with st.form("price_search_form"):
        c1, c2 = st.columns(2)
        with c1:
            compare_origin = st.text_input("From city", value="Vellore", key="compare_origin")
            depart_date = st.date_input("Departure / hotel check-in", value=date.today() + timedelta(days=7), key="depart_date")
            flight_from = st.text_input(
                "Departure airport IATA code (flights only)", value="",
                help="Enter a real 3-letter airport code, e.g. MAA. Vellore does not have a mapped commercial-airport code, so flight search is skipped unless you choose an actual departure airport.",
                key="flight_from_code",
            )
            station_from = st.text_input(
                "Train source station code", value="KPD",
                help="Suggested default for Vellore is Katpadi Junction (KPD). Confirm station code before searching.",
                key="station_from_code",
            )
        with c2:
            compare_destination = st.text_input("To city", value="Chennai", key="compare_destination")
            return_date = st.date_input("Return / hotel check-out", value=date.today() + timedelta(days=9), key="return_date")
            travelers_compare = st.number_input("Adults / travelers", min_value=1, max_value=9, value=1, key="compare_travelers")
            flight_to = st.text_input(
                "Arrival airport IATA code (flights only)", value="MAA",
                help="For Chennai, MAA is the airport code. Enter the airport you actually plan to fly to.",
                key="flight_to_code",
            )
            station_to = st.text_input(
                "Train destination station code", value="MAS",
                help="Suggested default for Chennai Central is MAS. Confirm station code before searching.",
                key="station_to_code",
            )
        compare_budget = st.number_input("Total budget (INR)", min_value=0, value=8000, step=500, key="compare_budget")
        make_search = st.form_submit_button("Set journey details", type="primary", use_container_width=True)

    if make_search:
        clean_origin = compare_origin.strip()
        clean_destination = compare_destination.strip()
        if not clean_origin or not clean_destination:
            st.error("Enter both origin and destination.")
        elif return_date <= depart_date:
            st.error("Return / check-out date must be after departure / check-in date.")
        else:
            new_details = {
                "origin": clean_origin,
                "destination": clean_destination,
                "depart": depart_date.isoformat(),
                "return": return_date.isoformat(),
                "people": int(travelers_compare),
                "budget": int(compare_budget),
                "flight_from": flight_from.strip().upper(),
                "flight_to": flight_to.strip().upper(),
                "station_from": station_from.strip().upper(),
                "station_to": station_to.strip().upper(),
            }
            old_details = st.session_state.get("live_search_details")
            st.session_state["live_search_details"] = new_details
            if old_details != new_details:
                st.session_state.setdefault("live_search_results", {})
                st.session_state.setdefault("live_search_attempts", set())
            st.success("Journey details saved. Provider calls will happen only when you press Fetch results below.")

    details = st.session_state.get("live_search_details")
    if details:
        from urllib.parse import quote_plus, urlencode
        signature = json.dumps(details, sort_keys=True, separators=(",", ":"))
        flight_key = f"{signature}|flights"
        hotel_key = f"{signature}|hotels"
        train_key = f"{signature}|trains"
        results_cache = st.session_state.setdefault("live_search_results", {})
        attempts = st.session_state.setdefault("live_search_attempts", set())
        hotel_search_url = (
            "https://www.google.com/travel/hotels?" + urlencode({
                "q": f"Hotels in {details['destination']}",
                "checkin": details["depart"], "checkout": details["return"],
                "adults": details["people"],
            })
        )
        flight_query = (
            f"Flights from {details['origin']} to {details['destination']} on {details['depart']} "
            f"returning {details['return']} for {details['people']} adult(s)"
        )
        flight_search_url = "https://www.google.com/travel/flights?q=" + quote_plus(flight_query)
        # Open the actual redBus route page instead of a brittle Google site: query.
        # Vellore/Chennai IDs are grounded in redBus's own route URL; other routes
        # use the standard route slug and let redBus resolve the requested cities.
        bus_origin = details["origin"].strip()
        bus_destination = details["destination"].strip()
        slug_origin = re.sub(r"[^a-z0-9]+", "-", bus_origin.casefold()).strip("-")
        slug_destination = re.sub(r"[^a-z0-9]+", "-", bus_destination.casefold()).strip("-")
        depart_day = date.fromisoformat(details["depart"])
        month_name = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")[depart_day.month - 1]
        redbus_date = f"{depart_day.day:02d}-{month_name}-{depart_day.year}"
        bus_search_url = (
            f"https://www.redbus.in/bus-tickets/{slug_origin}-to-{slug_destination}?"
            + urlencode({"onward": redbus_date})
        )
        known_redbus_routes = {
            ("vellore", "chennai"): ("960", "123"),
            ("chennai", "vellore"): ("123", "960"),
        }
        route_ids = known_redbus_routes.get((bus_origin.casefold(), bus_destination.casefold()))
        if route_ids:
            from_id, to_id = route_ids
            bus_search_url = (
                f"https://www.redbus.in/bus-tickets/{slug_origin}-to-{slug_destination}?"
                + urlencode({
                    "fromCityId": from_id,
                    "fromCityName": bus_origin,
                    "onward": redbus_date,
                    "toCityId": to_id,
                    "toCityName": bus_destination,
                })
            )

        st.markdown("#### Search details")
        st.caption(
            f"**{details['origin']} → {details['destination']}** · {details['depart']} to {details['return']} · "
            f"{details['people']} adult(s) · budget ₹{details['budget']:,}"
        )
        st.caption(
            "Free-tier safety: one request per provider for these exact inputs in this Streamlit session. "
            "Failed calls are cached as failures to prevent accidental retry. Changing route or dates is a new search."
        )

        serp_key = os.getenv("SERPAPI_API_KEY", "").strip()
        rail_key = os.getenv("RAILRADAR_API_KEY", "").strip()
        if not serp_key:
            st.warning("SERPAPI_API_KEY is not set. Automatic flight/hotel searches are disabled and no SerpApi calls will be made.")
            st.link_button("Get SerpApi free-plan details", "https://serpapi.com/pricing")
        if not rail_key:
            st.info("RAILRADAR_API_KEY is not set. Train timetable API is disabled. IRCTC seat/fare inventory is not queried by this app's free mode.")

        if st.button("Fetch provider results once", type="primary", use_container_width=True, key="fetch_live_provider_results"):
            if serp_key:
                if re.fullmatch(r"[A-Z]{3}", details["flight_from"]) and re.fullmatch(r"[A-Z]{3}", details["flight_to"]) and details["flight_from"] != details["flight_to"]:
                    if flight_key not in attempts:
                        attempts.add(flight_key)
                        try:
                            payload = search_google_flights_once(
                                details["flight_from"], details["flight_to"], details["depart"],
                                details["return"], details["people"], api_key=serp_key,
                            )
                            results_cache[flight_key] = {"ok": True, "payload": payload, "fetched_at": datetime.now().astimezone().isoformat(timespec="seconds")}
                        except Exception as exc:
                            results_cache[flight_key] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                else:
                    st.info("Flight search skipped: enter two different valid 3-letter airport IATA codes. No flight request was made.")

                if hotel_key not in attempts:
                    attempts.add(hotel_key)
                    try:
                        payload = search_google_hotels_once(
                            details["destination"], details["depart"], details["return"],
                            details["people"], api_key=serp_key,
                        )
                        results_cache[hotel_key] = {"ok": True, "payload": payload, "fetched_at": datetime.now().astimezone().isoformat(timespec="seconds")}
                    except Exception as exc:
                        results_cache[hotel_key] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            if rail_key:
                if re.fullmatch(r"[A-Z0-9]{2,6}", details["station_from"]) and re.fullmatch(r"[A-Z0-9]{2,6}", details["station_to"]) and details["station_from"] != details["station_to"]:
                    if train_key not in attempts:
                        attempts.add(train_key)
                        try:
                            payload = search_trains_between_once(
                                details["station_from"], details["station_to"], details["depart"], api_key=rail_key,
                            )
                            results_cache[train_key] = {"ok": True, "payload": payload, "fetched_at": datetime.now().astimezone().isoformat(timespec="seconds")}
                        except Exception as exc:
                            results_cache[train_key] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                else:
                    st.info("Train timetable search skipped: enter two valid different station codes. No train request was made.")
            if not serp_key and not rail_key:
                st.warning("No provider keys are configured. No API calls were made.")

        st.markdown("#### ✈️ Flight price snapshots")
        flight_result = results_cache.get(flight_key)
        if flight_result:
            if not flight_result.get("ok"):
                st.error("Flight search failed and will not be retried automatically for these exact inputs: " + flight_result.get("error", "Unknown error"))
            else:
                offers = parse_flight_results(flight_result.get("payload", {}))
                st.caption(f"Provider: SerpApi / Google Flights · fetched {flight_result.get('fetched_at')} · displayed prices may change.")
                st.caption("One-call mode conserves credits: SerpApi may require a follow-up request to expand return-leg flight details. This app does not make that second request automatically; confirm the complete itinerary at Google Flights.")
                if offers:
                    for idx, offer in enumerate(offers, start=1):
                        price = offer.get("price_inr")
                        price_label = f"₹{price:,.0f}" if price is not None else offer.get("price_text", "Price not provided")
                        stops_label = "nonstop" if offer.get("stops", 0) == 0 else f"{offer['stops']} stop(s)"
                        duration = offer.get("duration_minutes")
                        duration_label = f"{int(duration)//60}h {int(duration)%60}m" if duration is not None else "duration n/a"
                        st.markdown(f"**{idx}. {offer['airlines']} — {price_label}** · {stops_label} · {duration_label}")
                        st.caption(f"{offer.get('departure_airport','')} {offer.get('departure','')} → {offer.get('arrival_airport','')} {offer.get('arrival','')}")
                        st.link_button("Confirm current fare on Google Flights", flight_search_url, key=f"confirm_flight_{idx}_{hash(signature)}")
                    st.caption("These are fetched search-result prices, not guaranteed final fares. Reconfirm details, baggage, and availability at the seller before purchase.")
                else:
                    st.info("SerpApi returned no structured flight offers for these airports and dates.")
                    st.link_button("Open Google Flights search", flight_search_url, key="flight_fallback_link")
        else:
            st.caption("No flight search has been made for this trip yet. Enter real airport codes and press Fetch provider results once.")
            st.link_button("Open Google Flights manually", flight_search_url, key="flight_manual_link")

        st.markdown("#### 🏨 Hotel price snapshots")
        hotel_result = results_cache.get(hotel_key)
        if hotel_result:
            if not hotel_result.get("ok"):
                st.error("Hotel search failed and will not be retried automatically for these exact inputs: " + hotel_result.get("error", "Unknown error"))
            else:
                offers = parse_hotel_results(hotel_result.get("payload", {}))
                st.caption(f"Provider: SerpApi / Google Hotels · fetched {hotel_result.get('fetched_at')} · prices can change.")
                if offers:
                    for idx, offer in enumerate(offers, start=1):
                        labels = []
                        if offer.get("total_text"):
                            labels.append(f"total stay: {offer['total_text']}")
                        if offer.get("nightly_text"):
                            labels.append(f"per night: {offer['nightly_text']}")
                        if offer.get("rating") is not None:
                            labels.append(f"rating: {offer['rating']}")
                        st.markdown(f"**{idx}. {offer['name']}**")
                        st.write(" · ".join(labels) if labels else "Price not returned for this property.")
                        st.link_button("Check hotel price and room availability", offer.get("link") or hotel_search_url, key=f"confirm_hotel_{idx}_{hash(signature)}")
                    st.caption("Google Hotels' returned price is a search snapshot. Room availability, taxes, fees, and final price must be confirmed at the booking provider.")
                else:
                    st.info("No structured hotel properties were returned for those dates.")
                    st.link_button("Open Google Hotels search", hotel_search_url, key="hotel_fallback_link")
        else:
            st.caption("Hotel prices have not been fetched yet.")
            st.link_button("Open Google Hotels manually", hotel_search_url, key="hotel_manual_link")

        st.markdown("#### 🚆 Train timetables")
        train_result = results_cache.get(train_key)
        if train_result:
            if not train_result.get("ok"):
                st.warning("RailRadar timetable lookup failed; no retry was attempted: " + train_result.get("error", "Unknown error"))
            else:
                trains = parse_train_results(train_result.get("payload", {}))
                st.caption(f"RailRadar timetable snapshot · fetched {train_result.get('fetched_at')}.")
                if trains:
                    try:
                        import pandas as pd
                        st.dataframe(pd.DataFrame(trains), use_container_width=True, hide_index=True)
                    except Exception:
                        for train in trains:
                            st.write(f"{train['number']} {train['name']} · {train['departure']} → {train['arrival']}")
                    st.warning("Free-mode timetable data does not mean a seat is available. This result has no verified live fare or seat status; confirm on IRCTC.")
                else:
                    st.info("No trains were returned for these station codes/date.")
        else:
            st.caption("Train timetable API is optional and only queried if RAILRADAR_API_KEY is configured.")
        st.link_button("Check official IRCTC fares and seats", "https://www.irctc.co.in/nget/train-search", key="irctc_check_link")

        st.markdown("#### 🚌 Bus search")
        st.write("No authorised, unrestricted free Indian intercity bus API has been wired into this MVP. We will not invent bus fares or seats.")
        st.link_button("Search route and date on redBus", bus_search_url, key="redbus_search_link")

        st.markdown("#### Quota and trust notes")
        st.caption(
            "SerpApi free plan currently advertises 250 searches/month; each successful flight/hotel search generally consumes one search. "
            "This app avoids repeat calls for identical inputs within the same browser session. SerpApi cached/failed searches may be handled differently by its quota rules. "
            "RailRadar is optional; its free sandbox has different endpoint entitlements and does not mean all seat/fare APIs are free. "
            "If the provider signup asks for billing details, do not enter them; leave that provider unconfigured."
        )

st.divider()
st.caption("Automatic offers are shown only when a configured provider returns them. All dynamic offers are price snapshots and require confirmation at the seller; no tickets or reservations are made here.")
