#!/usr/bin/env python3
"""Shopping Sessions pipeline.

Raw event, click and link files are parsed with the challenge's strict CSV
rules. DuckDB performs the large relational operations and may spill to SCRATCH.
"""
import csv
import glob
import json
import os
import shutil
from pathlib import Path

import duckdb

DATA = os.environ.get("DATA_DIR", "/data")
OUT = os.environ.get("OUTPUT_DIR", "/output")
SCRATCH = os.environ.get("SCRATCH_DIR", "/scratch")
NCPU = max(1, int(os.environ.get("NCPU", os.cpu_count() or 1)))
MEM_GB = max(1, int(os.environ.get("MEM_GB", "24")))


def strict_fields(line):
    """Parse one line of the specified CSV dialect; return None if malformed.

    The caller removes LF only. CR is ordinary field data and is retained.
    Quotes may surround a complete field only. Empty fields become None.
    """
    if line == "":
        return None
    out = []
    i = 0
    n = len(line)
    while True:
        if i < n and line[i] == '"':
            i += 1
            j = line.find('"', i)
            if j < 0:
                return None
            value = line[i:j]
            i = j + 1
            if i < n and line[i] != ",":
                return None
        else:
            j = line.find(",", i)
            if j < 0:
                value = line[i:]
                i = n
            else:
                value = line[i:j]
                if '"' in value:
                    return None
                i = j
        out.append(value if value != "" else None)
        if i >= n:
            break
        if line[i] != ",":
            return None
        i += 1
        if i == n:
            out.append(None)
            break
    return out


def parse_raw_files(data_dir, scratch_dir, folder, prefix, expected_fields):
    """Strictly parse all CSV parts and write normalized CSV parts to scratch."""
    source_paths = sorted(glob.glob(os.path.join(data_dir, folder, "*.csv")))
    if not source_paths:
        raise FileNotFoundError(f"No CSV files found in {data_dir}/{folder}/")
    target_dir = os.path.join(scratch_dir, f"parsed_{prefix}")
    os.makedirs(target_dir, exist_ok=True)
    total = 0
    for file_no, source in enumerate(source_paths):
        target = os.path.join(target_dir, f"{prefix}-{file_no:05d}.csv")
        with open(source, "r", encoding="utf-8", newline="\n") as src, \
             open(target, "w", encoding="utf-8", newline="") as dst:
            reader = iter(src)
            next(reader, None)  # every input file has its own header
            writer = csv.writer(dst, lineterminator="\n")
            writer.writerow(expected_fields)
            for raw in reader:
                # Remove LF, never strip CR or whitespace.
                line = raw[:-1] if raw.endswith("\n") else raw
                row = strict_fields(line)
                if row is None or len(row) != len(expected_fields):
                    continue
                writer.writerow(row)
                total += 1
    return os.path.join(target_dir, "*.csv"), total


class DSU:
    """Union-find for transitive linked user IDs."""
    __slots__ = ("parent", "rank")

    def __init__(self):
        self.parent = {}
        self.rank = {}

    def add(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x):
        p = self.parent[x]
        while p != self.parent[p]:
            self.parent[p] = self.parent[self.parent[p]]
            p = self.parent[p]
        while x != p:
            nxt = self.parent[x]
            self.parent[x] = p
            x = nxt
        return p

    def union(self, a, b):
        self.add(a)
        self.add(b)
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def load_links(data_dir, scratch_dir):
    dsu = DSU()
    paths = sorted(glob.glob(os.path.join(data_dir, "links", "*.csv")))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {data_dir}/links/")
    for path in paths:
        with open(path, "r", encoding="utf-8", newline="\n") as f:
            next(f, None)
            for raw in f:
                line = raw[:-1] if raw.endswith("\n") else raw
                row = strict_fields(line)
                if row is None or len(row) != 2 or not row[0] or not row[1]:
                    continue
                dsu.union(row[0], row[1])

    # Canonical names are byte-wise smallest, including linked IDs never used
    # by an event or click.
    members = {}
    for value in dsu.parent:
        members.setdefault(dsu.find(value), []).append(value)
    mapping_path = os.path.join(scratch_dir, "person_map.csv")
    with open(mapping_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["user_id", "person_id"])
        for values in members.values():
            canonical = min(values, key=lambda value: value.encode("utf-8"))
            for value in values:
                writer.writerow([value, canonical])
    return mapping_path, len(dsu.parent)


def qpath(path):
    return "'" + path.replace("'", "''") + "'"


def read_clean_table(con, table_name, pattern, columns):
    """Load normalized CSV parts; all source fields remain VARCHAR."""
    cols = ",".join(f"'{name}':'VARCHAR'" for name in columns)
    con.execute(
        f"CREATE TEMP TABLE {table_name} AS "
        f"SELECT * FROM read_csv({qpath(pattern)}, header=true, "
        f"columns={{{cols}}}, delim=',', quote='\"', escape='\"', "
        f"new_line='\\n', nullstr='', union_by_name=true, "
        f"strict_mode=true, ignore_errors=false)"
    )


def main():
    os.makedirs(SCRATCH, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    temp_dir = os.path.join(SCRATCH, "duckdb_tmp")
    os.makedirs(temp_dir, exist_ok=True)

    with open(os.path.join(DATA, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    start_text = manifest["window_start"]
    end_text = manifest["window_end"]
    # The manifest timestamps are guaranteed UTC F1.
    from datetime import datetime, timezone
    start_dt = datetime.strptime(start_text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    start_ms = int((start_dt - epoch).total_seconds() * 1000)
    end_ms = int((end_dt - epoch).total_seconds() * 1000)

    event_pattern, event_rows = parse_raw_files(
        DATA, SCRATCH, "events", "events",
        ["event_id", "user_id", "ts", "ingest_ts", "event_type", "product_id", "price_cents"],
    )
    click_pattern, click_rows = parse_raw_files(
        DATA, SCRATCH, "clicks", "clicks",
        ["click_id", "user_id", "campaign_id", "ts"],
    )
    map_path, link_ids = load_links(DATA, SCRATCH)
    print(
        f"window {start_text} .. {end_text}; parsed event rows={event_rows}; "
        f"click rows={click_rows}; linked IDs={link_ids}", flush=True
    )

    con = duckdb.connect()
    con.execute(f"SET threads={NCPU}")
    con.execute(f"SET memory_limit='{max(1, int(MEM_GB * 0.80))}GB'")
    con.execute(f"SET temp_directory={qpath(temp_dir)}")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET TimeZone='UTC'")

    read_clean_table(con, "event_raw", event_pattern,
                     ["event_id", "user_id", "ts", "ingest_ts", "event_type", "product_id", "price_cents"])
    read_clean_table(con, "click_raw", click_pattern,
                     ["click_id", "user_id", "campaign_id", "ts"])
    con.execute(
        f"CREATE TEMP TABLE people AS SELECT * FROM read_csv({qpath(map_path)}, "
        "header=true, all_varchar=true, delim=',', quote='\"', escape='\"', nullstr='')"
    )
    con.execute(
        f"CREATE TEMP TABLE products AS SELECT product_id, category FROM "
        f"read_csv({qpath(os.path.join(DATA, 'products.csv'))}, header=true, "
        "all_varchar=true, delim=',', quote='\"', escape='\"', nullstr='')"
    )
    con.execute(
        f"CREATE TEMP TABLE campaigns AS SELECT campaign_id, channel FROM "
        f"read_csv({qpath(os.path.join(DATA, 'campaigns.csv'))}, header=true, "
        "all_varchar=true, delim=',', quote='\"', escape='\"', nullstr='')"
    )

    # Strict timestamp grammar plus calendar-aware parsing. F3 overflow is
    # represented as NULL, which makes it fail the relevant time-window test.
    con.execute(r"""
      CREATE MACRO ts_ms(s) AS (
        CASE
          WHEN regexp_full_match(s, '[0-9]+') THEN try_cast(s AS BIGINT)
          WHEN regexp_full_match(s, '[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z')
            THEN epoch_ms(try_strptime(s, '%Y-%m-%dT%H:%M:%SZ'))
          WHEN regexp_full_match(s, '[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}[+-][0-9]{2}:[0-9]{2}')
            AND try_cast(regexp_extract(s, '[+-]([0-9]{2}):([0-9]{2})$', 1) AS INTEGER) BETWEEN 0 AND 23
            AND try_cast(regexp_extract(s, '[+-]([0-9]{2}):([0-9]{2})$', 2) AS INTEGER) BETWEEN 0 AND 59
            THEN epoch_ms(try_cast(s AS TIMESTAMPTZ))
          ELSE NULL
        END
      );
      CREATE MACRO f1_ms(s) AS (
        CASE WHEN regexp_full_match(s, '[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z')
          THEN epoch_ms(try_strptime(s, '%Y-%m-%dT%H:%M:%SZ')) ELSE NULL END
      );
    """)

    # Event dedup happens after strict line parsing but before event validation.
    # ingest_ts is guaranteed F1 by the rules.
    con.execute("""
      CREATE TEMP TABLE ev_dedup AS
      SELECT *, f1_ms(ingest_ts) AS ingest_ms, ts_ms(ts) AS event_ms
      FROM event_raw
      WHERE event_id IS NOT NULL AND event_id <> ''
      QUALIFY row_number() OVER (PARTITION BY event_id ORDER BY f1_ms(ingest_ts)) = 1
    """)
    con.execute(f"""
      CREATE TEMP TABLE events AS
      SELECT coalesce(p.person_id, e.user_id) AS person_id,
             e.event_id, e.event_ms AS ts_ms, e.event_type, e.product_id,
             CASE WHEN e.event_type='purchase' THEN try_cast(e.price_cents AS BIGINT) ELSE 0 END AS price_cents,
             pr.category
      FROM ev_dedup e
      LEFT JOIN people p ON p.user_id=e.user_id
      LEFT JOIN products pr ON pr.product_id=e.product_id
      WHERE e.user_id IS NOT NULL AND e.user_id <> ''
        AND e.event_ms >= {start_ms} AND e.event_ms < {end_ms}
        AND e.event_type IN ('view','cart','purchase')
        AND e.product_id IS NOT NULL AND e.product_id <> ''
        AND (e.event_type <> 'purchase' OR
             (regexp_full_match(e.price_cents, '[0-9]+')
              AND try_cast(e.price_cents AS BIGINT) BETWEEN 1 AND 10000000))
        AND CAST(e.ingest_ms AS HUGEINT) - CAST(e.event_ms AS HUGEINT) <= 86400000
    """)

    # Sessionize distinct timestamps, so all events at the same millisecond
    # remain together and a gap of exactly 30 minutes does not split a session.
    con.execute("""
      CREATE TEMP TABLE distinct_times AS
      SELECT DISTINCT person_id, ts_ms FROM events
    """)
    con.execute("""
      CREATE TEMP TABLE time_sessions AS
      WITH boundaries AS (
        SELECT person_id, ts_ms,
               CASE WHEN lag(ts_ms) OVER (PARTITION BY person_id ORDER BY ts_ms) IS NULL
                         OR ts_ms - lag(ts_ms) OVER (PARTITION BY person_id ORDER BY ts_ms) > 1800000
                    THEN 1 ELSE 0 END AS new_session
        FROM distinct_times
      )
      SELECT person_id, ts_ms,
             sum(new_session) OVER (PARTITION BY person_id ORDER BY ts_ms
               ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS session_no
      FROM boundaries
    """)
    con.execute("""
      CREATE TEMP TABLE ev_numbered AS
      SELECT e.*, t.session_no
      FROM events e JOIN time_sessions t USING(person_id, ts_ms)
    """)
    con.execute("""
      CREATE TEMP TABLE session_base AS
      SELECT person_id, session_no, min(ts_ms) AS start_ms,
             count(*)::BIGINT AS event_count,
             sum(CASE WHEN event_type='purchase' THEN price_cents ELSE 0 END)::BIGINT AS revenue_cents,
             bool_or(event_type='purchase') AS has_purchase
      FROM ev_numbered GROUP BY person_id, session_no
    """)
    con.execute("""
      CREATE TEMP TABLE sessions AS
      SELECT person_id, session_no, start_ms,
             CAST(epoch_ms(start_ms) AS DATE) AS day,
             event_count, revenue_cents, has_purchase
      FROM session_base
    """)

    # Top category is calculated over all purchases for the person/day,
    # not independently per session.
    con.execute("""
      CREATE TEMP TABLE user_cat_day AS
      SELECT n.person_id, s.day, coalesce(n.category, 'unknown') AS category,
             sum(n.price_cents)::BIGINT AS amount
      FROM ev_numbered n JOIN sessions s USING(person_id, session_no)
      WHERE n.event_type='purchase'
      GROUP BY n.person_id, s.day, coalesce(n.category, 'unknown')
    """)
    con.execute("""
      CREATE TEMP TABLE user_daily AS
      WITH totals AS (
        SELECT person_id AS user_id, day, count(*)::BIGINT AS sessions,
               sum(event_count)::BIGINT AS events,
               sum(revenue_cents)::BIGINT AS revenue_cents
        FROM sessions GROUP BY person_id, day
      ), cats AS (
        SELECT person_id, day, category,
               row_number() OVER (PARTITION BY person_id,day
                                  ORDER BY amount DESC, category ASC) AS rn
        FROM user_cat_day
      )
      SELECT t.user_id, t.day, t.sessions, t.events, t.revenue_cents,
             c.category AS top_category
      FROM totals t LEFT JOIN cats c
        ON c.person_id=t.user_id AND c.day=t.day AND c.rn=1
    """)

    # Click IDs are deduplicated before validation. Duplicate copies are
    # guaranteed identical by the challenge.
    con.execute("""
      CREATE TEMP TABLE click_dedup AS
      SELECT *, ts_ms(ts) AS click_ms
      FROM click_raw
      WHERE click_id IS NOT NULL AND click_id <> ''
      QUALIFY row_number() OVER (PARTITION BY click_id ORDER BY click_id)=1
    """)
    con.execute(f"""
      CREATE TEMP TABLE clicks AS
      SELECT coalesce(p.person_id,c.user_id) AS person_id,
             c.click_id, c.campaign_id, c.click_ms
      FROM click_dedup c
      LEFT JOIN people p ON p.user_id=c.user_id
      JOIN campaigns ca ON ca.campaign_id=c.campaign_id
      WHERE c.user_id IS NOT NULL AND c.user_id <> ''
        AND c.click_ms >= {start_ms - 604800000} AND c.click_ms < {end_ms}
    """)

    # Qualifying clicks power both last-click credit and campaign assists.
    con.execute("""
      CREATE TEMP TABLE qualifying AS
      SELECT s.person_id, s.session_no, s.day, s.start_ms, s.has_purchase,
             s.revenue_cents, c.click_id, c.campaign_id, c.click_ms,
             row_number() OVER (PARTITION BY s.person_id,s.session_no
                                ORDER BY c.click_ms DESC, c.click_id ASC) AS rn
      FROM sessions s JOIN clicks c ON c.person_id=s.person_id
        AND c.click_ms <= s.start_ms AND c.click_ms >= s.start_ms - 604800000
    """)
    con.execute("""
      CREATE TEMP TABLE credited AS
      SELECT s.person_id, s.session_no, s.day, s.has_purchase, s.revenue_cents,
             coalesce(q.campaign_id,'organic') AS campaign_id
      FROM sessions s LEFT JOIN qualifying q
        ON q.person_id=s.person_id AND q.session_no=s.session_no AND q.rn=1
    """)
    con.execute("""
      CREATE TEMP TABLE assists AS
      SELECT DISTINCT person_id, session_no, day, campaign_id
      FROM qualifying
    """)
    con.execute("""
      CREATE TEMP TABLE camp_credit AS
      SELECT campaign_id, day, count(*)::BIGINT AS sessions,
             sum(CASE WHEN has_purchase THEN 1 ELSE 0 END)::BIGINT AS purchasing_sessions,
             sum(revenue_cents)::BIGINT AS revenue_cents
      FROM credited GROUP BY campaign_id, day
    """)
    con.execute("""
      CREATE TEMP TABLE camp_assist AS
      SELECT campaign_id, day, count(*)::BIGINT AS assisted_sessions
      FROM assists GROUP BY campaign_id, day
    """)
    con.execute("""
      CREATE TEMP TABLE campaign_daily AS
      SELECT coalesce(cc.campaign_id, aa.campaign_id) AS campaign_id,
             CASE WHEN coalesce(cc.campaign_id,aa.campaign_id)='organic' THEN 'organic'
                  ELSE ca.channel END AS channel,
             coalesce(cc.day,aa.day) AS day,
             coalesce(cc.sessions,0)::BIGINT AS sessions,
             coalesce(cc.purchasing_sessions,0)::BIGINT AS purchasing_sessions,
             coalesce(cc.revenue_cents,0)::BIGINT AS revenue_cents,
             coalesce(aa.assisted_sessions,0)::BIGINT AS assisted_sessions
      FROM camp_credit cc FULL OUTER JOIN camp_assist aa
        ON cc.campaign_id=aa.campaign_id AND cc.day=aa.day
      LEFT JOIN campaigns ca ON ca.campaign_id=coalesce(cc.campaign_id,aa.campaign_id)
    """)

    checks = con.execute("""
      SELECT (SELECT coalesce(sum(sessions),0) FROM user_daily),
             (SELECT coalesce(sum(sessions),0) FROM campaign_daily),
             (SELECT coalesce(sum(revenue_cents),0) FROM user_daily),
             (SELECT coalesce(sum(revenue_cents),0) FROM campaign_daily)
    """).fetchone()
    if checks[0] != checks[1] or checks[2] != checks[3]:
        raise RuntimeError(f"aggregation invariant failed: {checks}")

    for table in ("user_daily", "campaign_daily"):
        folder = os.path.join(OUT, table)
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)
        output_file = os.path.join(folder, "part-0.parquet")
        con.execute(
            f"COPY (SELECT * FROM {table}) TO {qpath(output_file)} "
            "(FORMAT PARQUET, COMPRESSION ZSTD)"
        )
    print(f"wrote outputs; totals={checks}", flush=True)
    con.close()


if __name__ == "__main__":
    main()
