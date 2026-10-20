# Live travel inventory status

Current verified capability:
- AI itinerary generation through the existing CLI.
- One shared Tavily search and at most one Firecrawl scrape for a trip session.

Not implemented as a provider-backed inventory integration:
- Live flight prices/availability.
- Live hotel prices/availability.
- Live rail and intercity bus schedule/seat inventory.
- Booking or payment.

Tavily search and Firecrawl extraction are sources for research, not booking inventory APIs. Do not turn a search snippet or an LLM estimate into a purported live quote.

Possible future provider: Amadeus Self-Service offers a free monthly request quota in its test environment, but its test environment uses limited/cached data. Its docs say production data is real-time, and over-quota production use is billed. This should be labelled/tested accurately before using it for live-flight claims. For Vellore–Chennai specifically, rail/bus research is more relevant than a flight search; a suitable official/authorized rail or bus data source would be needed for live timetable/inventory.
