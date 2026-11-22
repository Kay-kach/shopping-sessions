
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
