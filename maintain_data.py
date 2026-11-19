"""Preview or apply local review and drift retention cutoffs."""
import argparse
from datetime import datetime, timedelta, timezone
import json

from scamshift import store


def days(value):
    number = int(value)
    if not 1 <= number <= 3650:
        raise argparse.ArgumentTypeError("Enter a number of days from 1 to 3650.")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-days", type=days)
    parser.add_argument("--drift-days", type=days)
    parser.add_argument("--apply", action="store_true", help="Delete matching records. Default is a preview.")
    args = parser.parse_args()
    if args.review_days is None and args.drift_days is None:
        parser.error("Specify --review-days, --drift-days, or both.")
    if not (store.STATE / "scamshift.db").is_file():
        parser.error("Run this against an existing ScamShift database.")
    now = datetime.now(timezone.utc)
    cutoffs = {
        table: (field, (now - timedelta(days=age)).isoformat(timespec="seconds"))
        for table, field, age in (("reviews", "updated", args.review_days),
                                  ("scan_features", "created", args.drift_days),
                                  ("drift_reports", "created", args.drift_days)) if age is not None
    }
    totals = {}
    with store.db() as conn:
        if args.apply:
            conn.execute("BEGIN IMMEDIATE")
        for table, (field, cutoff) in cutoffs.items():
            totals[table] = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {field} < ?", (cutoff,)).fetchone()[0]
            if args.apply:
                if table == "reviews":
                    conn.execute("DELETE FROM review_events WHERE review_id IN "
                                 "(SELECT id FROM reviews WHERE updated < ?)", (cutoff,))
                conn.execute(f"DELETE FROM {table} WHERE {field} < ?", (cutoff,))
    print(json.dumps({"applied": args.apply, "matching_rows": totals}, indent=2))


if __name__ == "__main__":
    main()
