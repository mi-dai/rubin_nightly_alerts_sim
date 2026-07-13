"""
Build a single multi-night visit table from Rubin Observatory schedview
"nightsum" reports.

What this does
---------------
1. Downloads the report index page (schedview_reports_toc.html), which
   contains one big HTML table listing every simulated night, per
   instrument, with a link to that night's "nightsum" report and some
   summary stats (including the number of science visits).
2. Keeps only the rows where instrument == "lsstcam" and
   science_visits > 0 -- those are the only nightsum pages that embed a
   downloadable per-visit JSON table (this was confirmed by hand: pages
   with 0 science visits, or latiss pages, don't have that download link).
3. For each qualifying night, downloads that night's (large!) nightsum
   HTML page and pulls out the visit-level JSON. That JSON isn't hosted
   as a separate file on the server -- it's embedded directly in the
   page as a base64-encoded data: URI behind a "Downloading them as json
   here" link (this is what a JS-rendered browser lets you click to
   download `visits_<night>_lsstcam.json`).
4. Decodes each night's JSON into a pandas DataFrame, tags it with the
   night, and concatenates everything into one long multi-night table.
5. Saves the result to CSV (and, if you want, individual per-night JSON
   files as a cache so re-runs don't re-download everything).

Install dependencies:
    pip install requests beautifulsoup4 pandas lxml

Usage:
    python build_schedview_visit_table.py
"""

import base64
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://s3df.slac.stanford.edu/data/rubin/sim-data/schedview/reports/"
TOC_URL = BASE_URL + "schedview_reports_toc.html"

# Where to write outputs
OUTPUT_DIR = Path(".")
CACHE_DIR = OUTPUT_DIR / "nightsum_json_cache"   # raw per-night JSON saved here
COMBINED_CSV = OUTPUT_DIR / "lsstcam_visits_all_nights.csv"

# Set to an integer (e.g. 3) to only process a handful of nights while
# testing; set to None to process everything.
LIMIT_NIGHTS = None

# Be polite to the server between requests (seconds).
REQUEST_DELAY = 0.5


def get_qualifying_nights():
    """
    Parse the report index table and return a list of dicts:
        {"night": "2026-06-30", "instrument": "lsstcam",
         "science_visits": 700, "nightsum_url": "https://..."}
    for every row where instrument is lsstcam and science_visits > 0.
    """
    resp = requests.get(TOC_URL, timeout=60)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")
    table = soup.find("table", class_="dataframe")
    tbody = table.find("tbody")

    # Column order in the body (after the two header rows), matching the
    # header labels: night | instrument | prenight | multiprenight |
    # nightsum | compareprenight | Total | science | night_hours |
    # median FWHM | total eff_time/total exp_time
    #
    # The "night" cell is a <th rowspan="2"> that's only present on the
    # FIRST row of each night (pandas' to_html() merges the two
    # instrument rows under one night cell). So we track the most
    # recently seen night and reuse it whenever a row doesn't have its
    # own night cell.
    current_night = None
    rows_out = []

    for tr in tbody.find_all("tr"):
        cells = tr.find_all(["th", "td"])

        # A row that starts a new night has 11 cells: [night_th,
        # instrument_th, 9 data cells]. A row that continues the same
        # night has 10 cells: [instrument_th, 9 data cells].
        if len(cells) == 11:
            current_night = cells[0].get_text(strip=True)
            rest = cells[1:]
        else:
            rest = cells

        instrument = rest[0].get_text(strip=True)
        # rest[1] = prenight, rest[2] = multiprenight, rest[3] = nightsum,
        # rest[4] = compareprenight, rest[5] = Total, rest[6] = science, ...
        nightsum_cell = rest[3]
        science_text = rest[6].get_text(strip=True)

        nightsum_link = nightsum_cell.find("a")
        nightsum_url = nightsum_link["href"] if nightsum_link else None

        try:
            science_visits = int(science_text)
        except ValueError:
            science_visits = None  # blank / not reported

        if (
            instrument == "lsstcam"
            and science_visits is not None
            and science_visits > 0
            and nightsum_url is not None
        ):
            rows_out.append(
                {
                    "night": current_night,
                    "instrument": instrument,
                    "science_visits": science_visits,
                    "nightsum_url": nightsum_url,
                }
            )

    return rows_out


def extract_embedded_json(html_text):
    """
    Given the raw HTML of a nightsum page, find the embedded
        <a download="visits_<night>_lsstcam.json"
           href="data:application/json;base64,....">here</a>
    link and return (filename, decoded_json_bytes).

    Returns (None, None) if no such link is present (e.g. the page
    doesn't have a downloadable visit table).

    Note: we deliberately avoid parsing the whole (often 20+ MB) page
    with BeautifulSoup -- that's slow and unnecessary. A plain substring
    search for the data: URI marker, followed by a small regex on the
    text just before it to grab the filename, is much faster.
    """
    marker = 'href="data:application/json;base64,'
    idx = html_text.find(marker)
    if idx == -1:
        return None, None

    start = idx + len(marker)
    end = html_text.find('"', start)
    b64_data = html_text[start:end]

    # The download="...json" attribute is usually just before the href
    # attribute on the same <a> tag -- search a small window behind it.
    window = html_text[max(0, idx - 300):idx]
    m = re.search(r'download="([^"]+\.json)"', window)
    filename = m.group(1) if m else None

    return filename, base64.b64decode(b64_data)


def main():
    CACHE_DIR.mkdir(exist_ok=True)

    print("Fetching report index...")
    nights = get_qualifying_nights()
    print(f"Found {len(nights)} lsstcam nights with science_visits > 0")

    if LIMIT_NIGHTS:
        nights = nights[:LIMIT_NIGHTS]

    all_frames = []
    failures = []

    for i, row in enumerate(nights, start=1):
        night = row["night"]
        url = row["nightsum_url"]
        cache_path = CACHE_DIR / f"visits_{night}_lsstcam.json"

        print(f"[{i}/{len(nights)}] {night} ({row['science_visits']} science visits)")

        try:
            if cache_path.exists():
                # Already downloaded on a previous run -- reuse it.
                json_bytes = cache_path.read_bytes()
            else:
                resp = requests.get(url, timeout=120)
                resp.raise_for_status()
                filename, json_bytes = extract_embedded_json(resp.text)

                if json_bytes is None:
                    print(f"    no embedded JSON found on this page, skipping")
                    failures.append(night)
                    continue

                cache_path.write_bytes(json_bytes)
                time.sleep(REQUEST_DELAY)

            visits = json.loads(json_bytes)
            df = pd.DataFrame(visits)
            df.insert(0, "source_night", night)
            df.insert(1, "source_instrument", "lsstcam")
            all_frames.append(df)

        except Exception as exc:
            print(f"    FAILED: {exc}")
            failures.append(night)

    if not all_frames:
        print("No data collected -- nothing to save.")
        return

    combined = pd.concat(all_frames, ignore_index=True)
    combined.to_csv(COMBINED_CSV, index=False)

    print()
    print(f"Combined table: {len(combined)} visits across {len(all_frames)} nights")
    print(f"Saved to: {COMBINED_CSV.resolve()}")
    if failures:
        print(f"Nights that failed or had no JSON ({len(failures)}): {failures}")


if __name__ == "__main__":
    main()