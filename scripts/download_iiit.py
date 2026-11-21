"""Download the IIIT-HW-Dev (Version 1) archive with several parallel connections.

The CVIT server limits every single connection to about 100-200 KB/s, so one connection needs
hours for 1.8 GB. Several connections at once are limited only by your own internet speed.
The download can be resumed: run the same command again after an interruption.

Usage (run from the repository root; keep the file OUTSIDE OneDrive):
    python scripts/download_iiit.py --out C:/dl_data/IIIT-HW-Hindi_v1.tar.gz

Where this file fits in the pipeline:
    download_iiit.py (get the .tar.gz) -> extract it into data/raw/iiit_hw_dev
    -> prepare_iiit.py (make CSVs and vocab.txt) -> train.py.

How it works:
    1. Ask the server for the total file size (HEAD request).
    2. Cut the file into chunks of 16 MB. Each chunk is downloaded with an HTTP "Range" request,
       so many chunks can be downloaded at the same time, each by its own thread.
    3. Each finished chunk is written at its own place in the output file, and its number is saved
       in a small ".parts.json" file. After a crash, only the missing chunks are downloaded again.
"""
# Standard library: arguments, JSON state file, file paths, threads, timing.
import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

# requests: HTTP downloads.
import requests

URL = "https://cvit.iiit.ac.in/images/Projects/wordlevel-Indicscripts/IIIT-HW-Hindi_v1.tar.gz"
HEADERS = {"User-Agent": "Mozilla/5.0"}  # a browser-like User-Agent; some servers refuse the default one


def main():
    """Download the archive. Safe to run again: it resumes and skips finished chunks."""
    # description=__doc__ shows the text above in --help. RawDescription keeps its line breaks.
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=URL)
    ap.add_argument("--out", default="IIIT-HW-Hindi_v1.tar.gz")
    ap.add_argument("--workers", type=int, default=8, help="parallel connections")
    ap.add_argument("--chunk_mb", type=int, default=16)
    args = ap.parse_args()

    # HEAD asks only for the headers (no body), so we learn the file size without downloading it.
    head = requests.head(args.url, headers=HEADERS, allow_redirects=True, timeout=30)
    head.raise_for_status()  # stop with an error if the server says 404, 500 and so on
    total = int(head.headers["Content-Length"])  # file size in bytes
    chunk = args.chunk_mb * 1024 * 1024  # chunk size in bytes
    n_chunks = (total + chunk - 1) // chunk  # integer division rounded up (the last chunk may be smaller)
    state_path = args.out + ".parts.json"  # list of finished chunk numbers

    # which chunks are already finished
    if os.path.exists(state_path):
        done = set(json.load(open(state_path)))
    elif os.path.exists(args.out) and os.path.getsize(args.out) < total:
        existing = os.path.getsize(args.out)  # a plain partial download: its start is complete
        # A chunk counts as done only if the file already reaches the end of that chunk.
        done = {i for i in range(n_chunks) if (i + 1) * chunk <= existing}
    else:
        done = set()
    if os.path.exists(args.out) and os.path.getsize(args.out) == total and not done:
        print("File already has the full size; nothing to do.")
        return
    # "r+b" opens the existing file without erasing it (keeps finished chunks). "wb" creates a new one.
    mode = "r+b" if os.path.exists(args.out) else "wb"
    with open(args.out, mode) as f:
        # Make the file its final size now, so every thread can write at any position (seek) at once.
        f.truncate(total)  # reserve the full size

    lock = threading.Lock()  # protects the data shared by threads: `done`, `received` and the state file
    received = [0]  # bytes received so far; a list so the inner functions can change it (no "nonlocal")

    def save_state():
        """Write the sorted list of finished chunk numbers to the state file. Call it while holding the lock."""
        with open(state_path, "w") as f:
            json.dump(sorted(done), f)

    def fetch(i):
        """Download chunk number i and write it into the output file. Retries up to 40 times."""
        start = i * chunk  # first byte of this chunk
        end = min(start + chunk, total) - 1  # last byte (inclusive; HTTP ranges include both ends)
        for attempt in range(40):
            try:
                # Range header asks the server for only these bytes. stream=True reads the body piece by piece.
                # timeout=(connect seconds, read seconds).
                with requests.get(args.url, headers={**HEADERS, "Range": f"bytes={start}-{end}"},
                                  stream=True, timeout=(15, 60)) as r:
                    # 206 = "Partial Content". Any other code (such as 200) means the server ignored the Range.
                    if r.status_code != 206:
                        raise RuntimeError(f"HTTP {r.status_code} (expected 206)")
                    pos = start  # next byte position to write
                    with open(args.out, "r+b") as f:
                        f.seek(start)
                        for block in r.iter_content(256 * 1024):  # read 256 KB at a time
                            f.write(block)
                            pos += len(block)
                            with lock:
                                received[0] += len(block)
                # If we got fewer bytes than the chunk should have, the connection broke: try again.
                if pos != end + 1:
                    raise RuntimeError("short read")
                # Mark the chunk done only after all its bytes were written, and save it at once.
                with lock:
                    done.add(i)
                    save_state()
                return
            except Exception as e:  # network hiccup: wait and retry this chunk
                time.sleep(min(30, 2 * attempt + 1))  # wait longer after each failure, at most 30 s
        raise RuntimeError(f"chunk {i} failed after many retries")

    todo = [i for i in range(n_chunks) if i not in done]  # chunks still to download
    print(f"{total / 1e6:.0f} MB, {n_chunks} chunks, {len(done)} already done, {len(todo)} to go, "
          f"{args.workers} connections", flush=True)
    t0 = time.time()
    stop = threading.Event()  # set when the download ends, to stop the progress thread

    def report():
        """Print progress every 15 seconds (runs in its own thread)."""
        while not stop.wait(15):  # wait() returns True as soon as stop is set; False after 15 s
            with lock:
                d, rec = len(done), received[0]
            speed = rec / max(time.time() - t0, 1) / 1024  # average KB/s since the start
            # Time left (min) = bytes of unfinished chunks / speed in bytes per second / 60.
            left = (n_chunks - d) * chunk / max(speed * 1024, 1) / 60
            print(f"  {d}/{n_chunks} chunks  {speed:.0f} KB/s  about {left:.0f} min left", flush=True)

    # daemon=True: this thread never keeps the program alive after the main work ends.
    threading.Thread(target=report, daemon=True).start()
    with ThreadPoolExecutor(args.workers) as ex:
        # list() makes us wait for all chunks and re-raises the first error from any thread.
        list(ex.map(fetch, todo))
    stop.set()
    # Final check: every chunk done and the file has exactly the expected size.
    assert len(done) == n_chunks and os.path.getsize(args.out) == total
    print(f"DONE {args.out} ({total} bytes) in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
