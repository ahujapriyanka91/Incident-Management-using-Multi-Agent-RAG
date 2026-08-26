"""Extracts ERROR-level events from a pipeline log file, grouped into incidents."""
import re
from datetime import datetime, timedelta
from typing import List, Dict

LOG_LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+(?P<level>\w+)\s+\[(?P<source>[^\]]+)\]\s+(?P<message>.*)$"
)

TS_FORMAT = "%Y-%m-%d %H:%M:%S"

# Same-source ERROR lines within this many seconds of each other (even with
# other log lines in between, e.g. WARN retry messages) are treated as one
# ongoing incident rather than separate events.
DEFAULT_GROUPING_WINDOW_SECONDS = 120


def parse_log_lines(path: str) -> List[Dict]:
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            match = LOG_LINE_RE.match(line.strip())
            if match:
                entries.append(match.groupdict())
    return entries


def extract_error_events(
    path: str, grouping_window_seconds: int = DEFAULT_GROUPING_WINDOW_SECONDS
) -> List[Dict]:
    """Groups ERROR lines from the same source into one incident event if they
    fall within `grouping_window_seconds` of the most recent ERROR from that
    source, regardless of non-ERROR lines (INFO/WARN) appearing in between."""
    entries = parse_log_lines(path)
    window = timedelta(seconds=grouping_window_seconds)

    # open_events: source -> event dict, still eligible to receive more lines
    open_events: Dict[str, Dict] = {}
    events: List[Dict] = []

    for entry in entries:
        if entry["level"] != "ERROR":
            continue

        source = entry["source"]
        ts = datetime.strptime(entry["ts"], TS_FORMAT)

        existing = open_events.get(source)
        if existing and ts - datetime.strptime(existing["last_ts"], TS_FORMAT) <= window:
            existing["messages"].append(entry["message"])
            existing["last_ts"] = entry["ts"]
        else:
            if existing:
                events.append(existing)
            open_events[source] = {
                "source": source,
                "first_ts": entry["ts"],
                "last_ts": entry["ts"],
                "messages": [entry["message"]],
            }

    events.extend(open_events.values())
    events.sort(key=lambda e: e["first_ts"])
    return events
