#!/usr/bin/env python3
"""Shopping Sessions reference pipeline (DuckDB implementation).

The large event/click tables are processed in DuckDB so sorts and joins can spill
under the scorer's memory limit. Links are comparatively small and are resolved
with a streaming, strict CSV reader plus union-find before the SQL pipeline.
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
MEM_GB = max(1, int(os.environ.get("MEM_GB", "8")))


def strict_fields(line):
    """Parse this challenge's deliberately small CSV dialect.

    Newline has already been removed, but a preceding CR is retained as data.
    Quotes are legal only around a complete field; embedded quotes/commas are
    not supported by the input format. Return None for malformed records.
    """
    if line == "":
        return None
    out, i, n = [], 0, len(line)
    while True:
        if i < n and line[i] == '"':
            i += 1
            j = line.find('"', i)
            if j < 0:
                return None
            value = line[i:j]
            i = j + 1
            if i < n and line[i] != ',':
                return None
        else:
            j = line.find(',', i)
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
        # The only permitted character here is the delimiter.
        if line[i] != ',':
            return None
        i += 1
        if i == n:  # trailing comma: final empty field
            out.append(None)
            break
    return out


class DSU:
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
            q = self.parent[x]
            self.parent[x] = p
            x = q
        return p
    def union(self, a, b):
        self.add(a); self.add(b)
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
    for path in paths:
        with open(path, "r", encoding="utf-8", newline="") as f:
            first = True
            for raw in f:
                # Remove LF only. CR is intentionally part of the last field.
                line = raw[:-1] if raw.endswith("\n") else raw
                if first:
                    first = False
                    continue
                row = strict_fields(line)
                if row is None or len(row) != 2 or not row[0] or not row[1]:
                    continue
                dsu.union(row[0], row[1])
    # The challenge's canonical person key is the byte-wise smallest ID.
    members = {}
    for value in dsu.parent:
        root = dsu.find(value)
        members.setdefault(root, []).append(value)
    mapping_path = os.path.join(scratch_dir, "person_map.csv")
    with open(mapping_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["user_id", "person_id"])
        for values in members.values():
            canonical = min(values, key=lambda x: x.encode("utf-8"))
            for value in values:
                w.writerow([value, canonical])
    return mapping_path, sum(map(len, members.values()))


def qpath(path):
    return "'" + path.replace("'", "''") + "'"


def main():
    os.makedirs(SCRATCH, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(DATA, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)

    start_text, end_text = manifest["window_start"], manifest["window_end"]
    start_ms = int(__import__("datetime").datetime.fromisoformat(start_text.replace("Z", "+00:00")).timestamp() * 1000)
    end_ms = int(__import__("datetime").datetime.fromisoformat(end_text.replace("Z", "+00:00")).timestamp() * 1000)
    map_path, link_ids = load_links(DATA, SCRATCH)
    print(f"window {start_text} .. {end_text}; linked IDs: {link_ids}", flush=True)

    con = duckdb.connect()
    con.execute(f"SET threads={NCPU}")
    con.execute(f"SET memory_limit='{max(1, int(MEM_GB * 0.80))}GB'")
    con.execute(f"SET temp_directory={qpath(os.path.join(SCRATCH, 'duckdb_tmp'))}")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET TimeZone='UTC'")

    # Strict shape checks are applied by DuckDB's CSV scanner; all columns are
    # read as text so that validation below, not type inference, decides validity.
    event_glob = os.path.join(DATA, "events", "*.csv")
    click_glob = os.path.join(DATA, "clicks", "*.csv")
    con.execute(f"""
      CREATE TEMP TABLE event_raw AS
      SELECT * FROM read_csv({qpath(event_glob)}, header=true,
        columns={{'event_id':'VARCHAR','user_id':'VARCHAR','ts':'VARCHAR',
                 'ingest_ts':'VARCHAR','event_type':'VARCHAR','product_id':'VARCHAR',
                 'price_cents':'VARCHAR'}}, delim=',', quote='"', escape='"', new_line='\\n',
        nullstr='', ignore_errors=true, strict_mode=true, union_by_name=true)
    """)
    con.execute(f"""
      CREATE TEMP TABLE click_raw AS
      SELECT * FROM read_csv({qpath(click_glob)}, header=true,
        columns={{'click_id':'VARCHAR','user_id':'VARCHAR','campaign_id':'VARCHAR','ts':'VARCHAR'}},
        delim=',', quote='"', escape='"', new_line='\\n', nullstr='', ignore_errors=true,
        strict_mode=true, union_by_name=true)
    """)
    con.execute(f"""
      CREATE TEMP TABLE people AS
      SELECT * FROM read_csv({qpath(map_path)}, header=true, all_varchar=true,
        delim=',', quote='"', escape='"', nullstr='')
    """)
    con.execute(f"""
      CREATE TEMP TABLE products AS
      SELECT product_id, category FROM read_csv({qpath(os.path.join(DATA,'products.csv'))},
        header=true, all_varchar=true, delim=',', quote='"', escape='"', nullstr='')
    """)
    con.execute(f"""
      CREATE TEMP TABLE campaigns AS
      SELECT campaign_id, channel FROM read_csv({qpath(os.path.join(DATA,'campaigns.csv'))},
        header=true, all_varchar=true, delim=',', quote='"', escape='"', nullstr='')
    """)

    # Parse only the three timestamp forms allowed by rules.md. The regexes
    # prevent permissive casts from accepting fractions, lowercase separators,
    # missing zones, or offsets outside the specified range.
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

    con.execute(f"""
      CREATE TEMP TABLE ev_dedup AS
      SELECT *, f1_ms(ingest_ts) AS ingest_ms, ts_ms(ts) AS event_ms
      FROM event_raw
      WHERE event_id IS NOT NULL AND event_id <> ''
      QUALIFY row_number() OVER (PARTITION BY event_id ORDER BY f1_ms(ingest_ts)) = 1
    """)
    con.execute(f"""
      CREATE TEMP TABLE events AS
      SELECT coalesce(p.person_id, e.user_id) AS person_id,
             e.event_id, e.event_ms AS ts_ms, e.event_type,
             e.product_id,
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
    # Session boundaries are based on distinct ordered timestamps. Equal-ts
    # rows naturally have zero gap and remain together.
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
    # Join the session number back by timestamp so all events sharing the same
    # millisecond are guaranteed to remain in the same session.
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
      CREATE TEMP TABLE session_category AS
      SELECT person_id, session_no, coalesce(category, 'unknown') AS category,
             sum(price_cents)::BIGINT AS category_revenue
      FROM ev_numbered WHERE event_type='purchase'
      GROUP BY person_id, session_no, coalesce(category, 'unknown')
    """)
    con.execute("""
      CREATE TEMP TABLE sessions AS
      SELECT s.*, CAST(epoch_ms(s.start_ms) AS DATE) AS day,
             coalesce(c.category, NULL) AS top_category
      FROM session_base s
      LEFT JOIN (
        SELECT person_id, session_no, category,
               row_number() OVER (PARTITION BY person_id, session_no
                                  ORDER BY category_revenue DESC, category ASC) AS rn
        FROM session_category
      ) c ON c.person_id=s.person_id AND c.session_no=s.session_no AND c.rn=1
    """)

    # User/day output: category winner is selected after summing purchase value
    # across every session belonging to the person on that day.
    con.execute("""
      CREATE TEMP TABLE user_cat_day AS
      SELECT n.person_id, s.day, coalesce(n.category,'unknown') AS category,
             sum(n.price_cents)::BIGINT AS amount
      FROM ev_numbered n JOIN sessions s USING(person_id, session_no)
      WHERE n.event_type='purchase'
      GROUP BY n.person_id, s.day, coalesce(n.category,'unknown')
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
               row_number() OVER (PARTITION BY person_id,day ORDER BY amount DESC, category ASC) rn
        FROM user_cat_day
      )
      SELECT t.user_id, t.day, t.sessions, t.events, t.revenue_cents,
             c.category AS top_category
      FROM totals t LEFT JOIN cats c ON c.person_id=t.user_id AND c.day=t.day AND c.rn=1
    """)

    # Dedup click IDs before validating, then resolve IDs to the same person.
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
    # Rank all qualifying clicks. The chosen click is one row per session;
    # assists are distinct campaign/session pairs from the full qualifying set.
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

    # Cheap invariants catch accidental query regressions before writing output.
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
        con.execute(f"COPY (SELECT * FROM {table}) TO {qpath(os.path.join(folder,'part-0.parquet'))} (FORMAT PARQUET, COMPRESSION ZSTD)")
    print(f"wrote outputs; totals={checks}", flush=True)
    con.close()


if __name__ == "__main__":
    main()
