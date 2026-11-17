#!/usr/bin/env python3
"""Plenarsuche: daily totals from the web server's access log, without keeping any address.

Reads the Caddy access log (JSON lines, including the rotated and gzipped files), counts for each COMPLETE
day (Europe/Berlin) what happened on the Plenarsuche host and appends one row per day to a CSV:

    date, page_loads, visitors, searches, answers

  page_loads  GET / answered 200 or 304
  visitors    number of different addresses with a page load, a search or an answer (a number only)
  searches    POST /api/search answered 200
  answers     POST /api/answer answered 200

Addresses listed in the exclude file (the developer's own machines) are left out. A day counts as complete
when the log has no silence longer than two hours inside it; the log of the host is busy around the clock,
so a gap means missing data. Rows that already exist are never changed. With --cleanup, rolled log files
older than six days are deleted afterwards, so the retention stated for the access log holds for them too.
Standard library only.
"""
import argparse, csv, datetime as dt, glob, gzip, json, os, sys, time, zoneinfo
from pathlib import Path

TZ = zoneinfo.ZoneInfo("Europe/Berlin")
HOST = "germaparl.geolab.soz.uni-bielefeld.de"
FIELDS = ["date", "page_loads", "visitors", "searches", "answers"]


def read_log(pattern):
    """Yield (ts, host, ip, method, path, status) for every line of every matching file."""
    for f in sorted(glob.glob(pattern)):
        op = gzip.open if f.endswith(".gz") else open
        try:
            with op(f, "rt", errors="replace") as fh:
                for line in fh:
                    try:
                        d = json.loads(line)
                        r = d["request"]
                        yield d["ts"], r.get("host", ""), r.get("remote_ip", ""), r.get("method", ""), \
                            r.get("uri", "").split("?")[0], d.get("status")
                    except Exception:
                        continue
        except Exception as e:
            print(f"skipped {os.path.basename(f)}: {type(e).__name__}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-glob", default="/var/log/caddy/access*")
    ap.add_argument("--csv", default="/var/lib/plenarsuche-stats/daily.csv")
    ap.add_argument("--exclude-file", default="/etc/plenarsuche-stats.exclude")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cleanup", action="store_true")
    a = ap.parse_args()

    exclude = {"127.0.0.1", "::1"}
    if os.path.exists(a.exclude_file):
        exclude |= {l.split("#")[0].strip() for l in open(a.exclude_file) if l.split("#")[0].strip()}

    stamps, mine = [], []
    for ts, host, ip, m, path, st in read_log(a.log_glob):
        stamps.append(ts)
        if host == HOST and ip not in exclude:
            mine.append((ts, ip, m, path, st))
    if not stamps:
        sys.exit("no log lines found")
    stamps = sorted(set(stamps))
    day_of = lambda ts: dt.datetime.fromtimestamp(ts, TZ).date()
    today = dt.datetime.now(TZ).date()

    by_day = {}
    for t in stamps:
        by_day.setdefault(day_of(t), []).append(t)
    complete = []
    for d, v in by_day.items():
        if d >= today:
            continue
        start = dt.datetime.combine(d, dt.time(0), TZ).timestamp()
        pts = [start] + v + [start + 86400]
        if max(b - x for x, b in zip(pts, pts[1:])) <= 7200:
            complete.append(d)

    have = set()
    if os.path.exists(a.csv):
        with open(a.csv, newline="") as fh:
            have = {row["date"] for row in csv.DictReader(fh)}

    rows = []
    for d in sorted(complete):
        if d.isoformat() in have:
            continue
        page = search = answer = 0
        ips = set()
        for ts, ip, m, path, st in mine:
            if day_of(ts) != d:
                continue
            if m == "GET" and path == "/" and st in (200, 304):
                page += 1; ips.add(ip)
            elif m == "POST" and path == "/api/search" and st == 200:
                search += 1; ips.add(ip)
            elif m == "POST" and path == "/api/answer" and st == 200:
                answer += 1; ips.add(ip)
        rows.append({"date": d.isoformat(), "page_loads": page, "visitors": len(ips), "searches": search, "answers": answer})

    print(f"log lines {len(stamps)}, complete days {len(complete)}, new rows {len(rows)}")
    for r in rows:
        print(" ", *(r[k] for k in FIELDS))
    if a.dry_run:
        print("dry run, nothing written")
    elif rows:
        Path(a.csv).parent.mkdir(parents=True, exist_ok=True)
        old = []
        if os.path.exists(a.csv):
            with open(a.csv, newline="") as fh:
                old = list(csv.DictReader(fh))
        tmp = a.csv + ".tmp"
        with open(tmp, "w", newline="") as fh:
            w = csv.DictWriter(fh, FIELDS); w.writeheader()
            w.writerows(sorted(old + rows, key=lambda r: r["date"]))
        os.replace(tmp, a.csv)
        os.chmod(a.csv, 0o644)

    if a.cleanup or a.dry_run:
        cutoff = time.time() - 6 * 86400
        stale = [f for f in glob.glob(os.path.join(os.path.dirname(a.log_glob.rstrip("*")) or ".", "access-*.log.gz"))
                 if os.path.getmtime(f) < cutoff]
        for f in stale:
            print(("would delete " if not a.cleanup else "deleting ") + os.path.basename(f))
            if a.cleanup and not a.dry_run:
                os.remove(f)


if __name__ == "__main__":
    main()
