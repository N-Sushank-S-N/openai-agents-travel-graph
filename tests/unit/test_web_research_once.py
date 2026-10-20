from __future__ import annotations

import asyncio
import os
import unittest
import uuid
from unittest.mock import patch

from travel_planner.utils import web_research_once as research


class SharedWebResearchTests(unittest.TestCase):
    def setUp(self):
        self.session_key = "offline-test-" + uuid.uuid4().hex
        with research._SESSION_LOCK:
            research._SESSION_CACHE.pop(self.session_key, None)
            research._SESSION_STARTED.discard(self.session_key)
        self.env_patch = patch.dict(os.environ, {
            "TAVILY_API_KEY": "unit-test-tavily-key",
            "FIRECRAWL_API_KEY": "unit-test-firecrawl-key",
        })
        self.env_patch.start()

    def tearDown(self):
        self.env_patch.stop()
        with research._SESSION_LOCK:
            research._SESSION_CACHE.pop(self.session_key, None)
            research._SESSION_STARTED.discard(self.session_key)

    def test_one_search_one_scrape_and_session_cache(self):
        responses = [
            (200, {"results": [{
                "title": "Official city tourism",
                "url": "https://example.gov.in/tourism",
                "content": "Visitor information and attractions.",
                "score": 0.99,
            }]}, None),
            (200, {"success": True, "data": {
                "markdown": "Official visitor information and attraction details.",
                "metadata": {"title": "Official city tourism"},
            }}, None),
        ]
        with patch.object(research, "_post_json_once", side_effect=responses) as mocked:
            first = asyncio.run(research.collect_destination_research(
                "Plan a weekend trip", "Chennai", self.session_key
            ))
            again = asyncio.run(research.collect_destination_research(
                "Plan a weekend trip", "Chennai", self.session_key
            ))
        self.assertIn("Official city tourism", first)
        self.assertIn("Official visitor information", first)
        self.assertEqual(first, again)
        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(mocked.call_args_list[0].args[0], research.TAVILY_SEARCH_URL)
        self.assertEqual(mocked.call_args_list[1].args[0], research.FIRECRAWL_SCRAPE_URL)

    def test_missing_firecrawl_key_only_searches_once(self):
        os.environ.pop("FIRECRAWL_API_KEY", None)
        with patch.object(research, "load_dotenv", None), patch.object(research, "_post_json_once", return_value=(200, {
            "results": [{
                "title": "Official city tourism",
                "url": "https://example.gov.in/tourism",
                "content": "Visitor information.",
            }]
        }, None)) as mocked:
            result = asyncio.run(research.collect_destination_research(
                "Plan a trip", "Chennai", self.session_key
            ))
        self.assertIn("TAVILY SEARCH RESULTS", result)
        self.assertEqual(mocked.call_count, 1)

    def test_failed_tavily_does_not_retry_or_scrape(self):
        with patch.object(research, "_post_json_once", return_value=(429, None, "HTTP 429")) as mocked:
            result = asyncio.run(research.collect_destination_research(
                "Plan a trip", "Chennai", self.session_key
            ))
        self.assertEqual(result, "")
        self.assertEqual(mocked.call_count, 1)


if __name__ == "__main__":
    unittest.main()
