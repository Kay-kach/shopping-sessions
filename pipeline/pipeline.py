#!/usr/bin/env python3
"""
Shopping Sessions: Polars implementation.

Reads $DATA_DIR, writes $OUTPUT_DIR/user_daily/ and campaign_daily/ as Parquet.
Implements the full rules spec: custom CSV dialect, timestamp forms F1/F2/F3,
linked-ID union-find, event dedup-before-validation, sessionization, and
click attribution with assists.
"""
import glob
import json
import os
import re
import bisect
from datetime import datetime, timezone, timedelta

NCPU = os.environ.get("NCPU")
if NCPU:
    os.environ.setdefault("POLARS_MAX_THREADS", NCPU)

import polars as pl  # noqa: E402

DATA = os.environ.get("DATA_DIR", "/data")
OUT = os.environ.get("OUTPUT_DIR", "/output")

MS_PER_DAY = 86_400_000
MS_PER_7_DAYS = 7 * MS_PER_DAY
SESSION_GAP_MS = 1_800_000

RE_F1 = re.compile(r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})Z$")
RE_F2 = re.compile(r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})([+-])(\d{2}):(\d{2})$")
RE_DIGITS = re.compile(r"^\d+$")


def is_leap(y):
    return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)


def _valid_calendar(y, mo, d, h, mi, sec):
    if mo < 1 or mo > 12 or d < 1 or h > 23 or mi > 59 or sec > 59:
        return False
    if mo in (4, 6, 9, 11):
        return d <= 30
    if mo == 2:
        return d <= (29 if is_leap(y) else 28)
    return d <= 31


def parse_ts(s):
    if not s:
        return None
    m = RE_F1.match(s)
    if m:
        y, mo, d, h, mi, sec = (int(g) for g in m.groups())
        if not _valid_calendar(y, mo, d, h, mi, sec):
            return None
        return int(datetime(y, mo, d, h, mi, sec, tzinfo=timezone.utc).timestamp() * 1000)
    m = RE_F2.match(s)
    if m:
        y, mo, d, h, mi, sec, sign, oh, om = m.groups()
        y, mo, d, h, mi, sec = int(y), int(mo), int(d), int(h), int(mi), int(sec)
        oh, om = int(oh), int(om)
        if oh > 23 or om > 59:
            return None
        if not _valid_calendar(y, mo, d, h, mi, sec):
            return None
        offset = timedelta(hours=oh, minutes=om)
        if sign == "-":
            offset = -offset
        dt = datetime(y, mo, d, h, mi, sec, tzinfo=timezone.utc) - offset
        return int(dt.timestamp() * 1000)
    if RE_DIGITS.match(s):
        try:
            return int(s)
        except ValueError:
            return None
    return None


def parse_csv_line(line, n_fields):
    if line == "":
        return None
    fields = []
    i, n = 0, len(line)
    while i < n:
        if line[i] == '"':
            if i > 0 and line[i - 1] != ",":
                return None
            j = i + 1
            while j < n and line[j] != '"':
                j += 1
            if j >= n:
                return None
            fields.append(line[i + 1:j])
            i = j + 1
            if i < n:
                if line[i] == ",":
                    i += 1
                else:
                    return None
        else:
            j = i
            while j < n and line[j] != "," and line[j] != '"':
                j += 1
            if j < n and line[j] == '"':
                return None
            fields.append(line[i:j])
            i = j + 1
    if len(fields) != n_fields:
        return None
    return fields


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            return x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        self.parent[self.find(x)] = self.find(y)


def main():
    with open(os.path.join(DATA, "manifest.json")) as f:
        manifest = json.load(f)
    window_start = parse_ts(manifest["window_start"])
    window_end = parse_ts(manifest["window_end"])
    click_window_start = window_start - MS_PER_7_DAYS

    products = {}
    with open(os.path.join(DATA, "products.csv"), "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            fields = parse_csv_line(line.rstrip("\n"), 2)
            if fields:
                products[fields[0]] = fields[1]

    campaigns = {}
    with open(os.path.join(DATA, "campaigns.csv"), "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            fields = parse_csv_line(line.rstrip("\n"), 2)
            if fields:
                campaigns[fields[0]] = fields[1]

    uf = UnionFind()
    for fpath in sorted(glob.glob(os.path.join(DATA, "links", "*.csv"))):
        with open(fpath, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                fields = parse_csv_line(line.rstrip("\n"), 2)
                if not fields:
                    continue
                a, b = fields
                if a == "" or b == "":
                    continue
                uf.union(a, b)

    group_min = {}
    for id_ in uf.parent:
        root = uf.find(id_)
        if root not in group_min or id_ < group_min[root]:
            group_min[root] = id_
    person_of = {id_: group_min[uf.find(id_)] for id_ in uf.parent}

    def get_person(uid):
        return person_of.get(uid, uid)

    raw_events = []
    for fpath in sorted(glob.glob(os.path.join(DATA, "events", "*.csv"))):
        with open(fpath, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                fields = parse_csv_line(line.rstrip("\n"), 7)
                if not fields:
                    continue
                eid, uid, ts_str, its_str, etype, pid, price = fields
                if eid == "":
                    continue
                ts = parse_ts(ts_str)
                its = parse_ts(its_str)
                if ts is None or its is None:
                    continue
                raw_events.append((eid, uid, ts, its, etype, pid, price))

    best = {}
    for eid, uid, ts, its, etype, pid, price in raw_events:
        cur = best.get(eid)
        if cur is None or its < cur[2]:
            best[eid] = (uid, ts, its, etype, pid, price)

    valid_events = []
    for uid, ts, its, etype, pid, price in best.values():
        if uid == "":
            continue
        if ts < window_start or ts >= window_end:
            continue
        if etype not in ("view", "cart", "purchase"):
            continue
        if pid == "":
            continue
        if etype == "purchase":
            if not price or not RE_DIGITS.match(price):
                continue
            try:
                pv = int(price)
            except ValueError:
                continue
            if pv < 1 or pv > 10_000_000:
                continue
        if its - ts > MS_PER_DAY:
            continue
        valid_events.append((uid, ts, etype, pid, price))

    person_events = {}
    for uid, ts, etype, pid, price in valid_events:
        p = get_person(uid)
        person_events.setdefault(p, []).append((ts, etype, pid, price))

    sessions = []
    for p, evts in person_events.items():
        evts.sort(key=lambda e: e[0])
        session_start = None
        n_ev = 0
        revenue = 0
        purchases = {}
        prev_ts = None
        for ts, etype, pid, price in evts:
            if prev_ts is None or ts - prev_ts > SESSION_GAP_MS:
                if session_start is not None:
                    sessions.append((p, session_start, n_ev, revenue, purchases))
                session_start = ts
                n_ev = 0
                revenue = 0
                purchases = {}
            n_ev += 1
            prev_ts = ts
            if etype == "purchase":
                pv = int(price)
                revenue += pv
                cat = products.get(pid, "unknown")
                purchases[cat] = purchases.get(cat, 0) + pv
        if session_start is not None:
            sessions.append((p, session_start, n_ev, revenue, purchases))

    raw_clicks = []
    for fpath in sorted(glob.glob(os.path.join(DATA, "clicks", "*.csv"))):
        with open(fpath, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                fields = parse_csv_line(line.rstrip("\n"), 4)
                if not fields:
                    continue
                cid, uid, camp_id, ts_str = fields
                if cid == "":
                    continue
                ts = parse_ts(ts_str)
                if ts is None:
                    continue
                raw_clicks.append((cid, uid, ts, camp_id))

    click_map = {}
    for cid, uid, ts, camp_id in raw_clicks:
        click_map[cid] = (uid, ts, camp_id)

    person_clicks = {}
    for cid, (uid, ts, camp_id) in click_map.items():
        if uid == "":
            continue
        if ts < click_window_start or ts >= window_end:
            continue
        if camp_id not in campaigns:
            continue
        p = get_person(uid)
        person_clicks.setdefault(p, []).append((ts, camp_id, cid))

    for p in person_clicks:
        person_clicks[p].sort(key=lambda c: (c[0], c[2]))

    MAX_CHAR = chr(0x10FFFF)
    attr = []
    for p, s_start, n_ev, revenue, purchases in sessions:
        clicks = person_clicks.get(p, [])
        lo = s_start - MS_PER_7_DAYS
        hi = s_start
        left = bisect.bisect_left(clicks, (lo, "", ""))
        right = bisect.bisect_right(clicks, (hi, MAX_CHAR, MAX_CHAR))
        best_click = None
        assisting = set()
        for i in range(left, right):
            cts, ccamp, ccid = clicks[i]
            assisting.add(ccamp)
            if best_click is None or cts > best_click[0] or (cts == best_click[0] and ccid < best_click[2]):
                best_click = (cts, ccamp, ccid)
        credited = best_click[1] if best_click else "organic"
        is_purchasing = revenue > 0
        day = datetime.fromtimestamp(s_start / 1000, tz=timezone.utc).date()
        attr.append((p, day, credited, n_ev, revenue, purchases, assisting, is_purchasing))

    user_agg = {}
    for p, day, credited, n_ev, revenue, purchases, assisting, is_purchasing in attr:
        key = (p, day)
        if key not in user_agg:
            user_agg[key] = [0, 0, 0, {}]
        agg = user_agg[key]
        agg[0] += 1
        agg[1] += n_ev
        agg[2] += revenue
        for cat, amt in purchases.items():
            agg[3][cat] = agg[3].get(cat, 0) + amt

    user_rows = []
    for (p, day), (sessions_n, events_n, revenue, cat_sums) in user_agg.items():
        top_cat = None
        if cat_sums:
            best_cat = None
            best_amt = -1
            for cat, amt in cat_sums.items():
                if amt > best_amt or (amt == best_amt and cat < best_cat):
                    best_cat = cat
                    best_amt = amt
            top_cat = best_cat
        user_rows.append((p, day, sessions_n, events_n, revenue, top_cat))

    camp_agg = {}
    for p, day, credited, n_ev, revenue, purchases, assisting, is_purchasing in attr:
        key = (credited, day)
        if key not in camp_agg:
            camp_agg[key] = [0, 0, 0, 0]
        agg = camp_agg[key]
        agg[0] += 1
        if is_purchasing:
            agg[1] += 1
        agg[2] += revenue
        if credited != "organic":
            for camp in assisting:
                if camp == "organic":
                    continue
                akey = (camp, day)
                if akey not in camp_agg:
                    camp_agg[akey] = [0, 0, 0, 0]
                camp_agg[akey][3] += 1

    camp_rows = []
    for (camp, day), (sessions_n, purchasing, revenue, assisted) in camp_agg.items():
        channel = "organic" if camp == "organic" else campaigns.get(camp, "organic")
        camp_rows.append((camp, channel, day, sessions_n, purchasing, revenue, assisted))

    os.makedirs(os.path.join(OUT, "user_daily"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "campaign_daily"), exist_ok=True)

    if user_rows:
        df = pl.DataFrame({
            "user_id": [r[0] for r in user_rows],
            "day": [r[1] for r in user_rows],
            "sessions": [r[2] for r in user_rows],
            "events": [r[3] for r in user_rows],
            "revenue_cents": [r[4] for r in user_rows],
            "top_category": [r[5] for r in user_rows],
        })
    else:
        df = pl.DataFrame(schema={"user_id": pl.String, "day": pl.Date, "sessions": pl.Int64,
                                   "events": pl.Int64, "revenue_cents": pl.Int64, "top_category": pl.String})
    df.write_parquet(os.path.join(OUT, "user_daily", "part-0.parquet"))

    if camp_rows:
        df = pl.DataFrame({
            "campaign_id": [r[0] for r in camp_rows],
            "channel": [r[1] for r in camp_rows],
            "day": [r[2] for r in camp_rows],
            "sessions": [r[3] for r in camp_rows],
            "purchasing_sessions": [r[4] for r in camp_rows],
            "revenue_cents": [r[5] for r in camp_rows],
            "assisted_sessions": [r[6] for r in camp_rows],
        })
    else:
        df = pl.DataFrame(schema={"campaign_id": pl.String, "channel": pl.String, "day": pl.Date,
                                   "sessions": pl.Int64, "purchasing_sessions": pl.Int64,
                                   "revenue_cents": pl.Int64, "assisted_sessions": pl.Int64})
    df.write_parquet(os.path.join(OUT, "campaign_daily", "part-0.parquet"))


if __name__ == "__main__":
    main()
