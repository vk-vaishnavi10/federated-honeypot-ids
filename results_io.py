"""Shared helper: append experiment results to results/<name>.csv."""
import os, csv
os.makedirs("results", exist_ok=True)
def save_result(name, rows, header):
    """rows: list of dicts. header: list of column names."""
    path = f"results/{name}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    print(f"saved {len(rows)} rows -> {path}")
