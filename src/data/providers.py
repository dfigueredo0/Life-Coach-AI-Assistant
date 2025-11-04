from __future__ import annotations
import aiohttp
from typing import Any, Dict, List

GOOGLE_EVENTS_URL = "https://www.googleapis.com/calendar/v3/calendars/primary/events"
OUTLOOK_EVENTS_URL = "https://graph.microsoft.com/v1.0/me/events"

class GoogleCalendarProvider:
    async def fetch_events(self, token: str) -> Dict[str, Any]:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                GOOGLE_EVENTS_URL,
                headers={"Authorization": f"Bearer {token}"},
                timeout=aiohttp.ClientTimeout(total=20),
            ) as res:
                res.raise_for_status()
                return await res.json()

class OutlookProvider:
    async def fetch_events(self, token: str) -> Dict[str, Any]:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                OUTLOOK_EVENTS_URL,
                headers={"Authorization": f"Bearer {token}"},
                timeout=aiohttp.ClientTimeout(total=20),
            ) as res:
                res.raise_for_status()
                return await res.json()