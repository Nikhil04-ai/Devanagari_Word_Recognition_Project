"""Download the IIIT-HW-Dev (Version 1) archive with several parallel connections.

The CVIT server limits every single connection to about 100-200 KB/s, so one connection needs
hours for 1.8 GB. Several connections at once are limited only by your own internet speed.
The download can be resumed: run the same command again after an interruption.

Usage (run from the repository root; keep the file OUTSIDE OneDrive):
    python scripts/download_iiit.py --out C:/dl_data/IIIT-HW-Hindi_v1.tar.gz
"""
import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

URL = "https://cvit.iiit.ac.in/images/Projects/wordlevel-Indicscripts/IIIT-HW-Hindi_v1.tar.gz"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=URL)
    ap.add_argument("--out", default="IIIT-HW-Hindi_v1.tar.gz")
    ap.add_argument("--workers", type=int, default=8, help="parallel connections")
    ap.add_argument("--chunk_mb", type=int, default=16)
    args = ap.parse_args()

    head = requests.head(args.url, headers=HEADERS, allow_redirects=True, timeout=30)
    head.raise_for_status()
    total = int(head.headers["Content-Length"])
    chunk = args.chunk_mb * 1024 * 1024
    n_chunks = (total + chunk - 1) // chunk
    state_path = args.out + ".parts.json"

    # which chunks are already finished
    if os.path.exists(state_path):
        done = set(json.load(open(state_path)))
    elif os.path.exists(args.out) and os.path.getsize(args.out) < total:
        existing = os.path.getsize(args.out)  # a plain partial download: its start is complete
        done = {i for i in range(n_chunks) if (i + 1) * chunk <= existing}
    else:
        done = set()
    if os.path.exists(args.out) and os.path.getsize(args.out) == total and not done:
        print("File already has the full size; nothing to do.")
        return
    mode = "r+b" if os.path.exists(args.out) else "wb"
    with open(args.out, mode) as f:
        f.truncate(total)  # reserve the full size

    lock = threading.Lock()
    received = [0]

    def save_state():
        with open(state_path, "w") as f:
            json.dump(sorted(done), f)

    def fetch(i):
        start = i * chunk
        end = min(start + chunk, total) - 1
        for attempt in range(40):
            try:
                with requests.get(args.url, headers={**HEADERS, "Range": f"bytes={start}-{end}"},
                                  stream=True, timeout=(15, 60)) as r:
                    if r.status_code != 206:
                        raise RuntimeError(f"HTTP {r.status_code} (expected 206)")
                    pos = start
                    with open(args.out, "r+b") as f:
                        f.seek(start)
                        for block in r.iter_content(256 * 1024):
                            f.write(block)
                            pos += len(block)
                            with lock:
                                received[0] += len(block)
                if pos != end + 1:
                    raise RuntimeError("short read")
                with lock:
                    done.add(i)
                    save_state()
                return
            except Exception as e:  # network hiccup: wait and retry this chunk
                time.sleep(min(30, 2 * attempt + 1))
        raise RuntimeError(f"chunk {i} failed after many retries")

    todo = [i for i in range(n_chunks) if i not in done]
    print(f"{total / 1e6:.0f} MB, {n_chunks} chunks, {len(done)} already done, {len(todo)} to go, "
          f"{args.workers} connections", flush=True)
    t0 = time.time()
    stop = threading.Event()

    def report():
        while not stop.wait(15):
            with lock:
                d, rec = len(done), received[0]
            speed = rec / max(time.time() - t0, 1) / 1024
            left = (n_chunks - d) * chunk / max(speed * 1024, 1) / 60
            print(f"  {d}/{n_chunks} chunks  {speed:.0f} KB/s  about {left:.0f} min left", flush=True)

    threading.Thread(target=report, daemon=True).start()
    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(fetch, todo))
    stop.set()
    assert len(done) == n_chunks and os.path.getsize(args.out) == total
    print(f"DONE {args.out} ({total} bytes) in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
