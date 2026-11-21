"""Merge the report sections into one PDF (Monu).

Usage (from the repository root):
    python report/tools/build_report.py            # draft -> report/build/report_draft.pdf (not in Git)
    python report/tools/build_report.py --final    # -> report/P09_report.pdf (do not add it to Git),
                                                   #    refused while any placeholder is left

What it does:
1. Joins report/sections/01..08 in order, then the numbered reference list (from
   report/references.md), then the contribution statement (09), under a title block.
2. Drops any "*Owner: ...*" helper lines (notes to the writers, not report text).
3. Converts Markdown to HTML (python-markdown, tables) and prints it to PDF with Microsoft Edge
   (headless), which shapes Devanagari correctly with the Windows "Nirmala UI" font.
4. Prints the page count and file size (limit 10 MB, target 6 to 8 pages) and lists every line that
   still contains a placeholder, "(check)" or an italic writer's note, with its file and line number.

Needs: pip install markdown pypdf (in requirements/analysis.txt), and Microsoft Edge or Google Chrome.
"""
import argparse
import glob
import os
import re
import shutil
import subprocess
import sys

import markdown

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REPORT = os.path.join(ROOT, "report")
BUILD = os.path.join(REPORT, "build")
FINAL_PDF = os.path.join(REPORT, "P09_report.pdf")
MAX_MB = 10.0

TITLE = "Handwritten Devanagari Word Recognition: CRNN-CTC vs Transformer OCR (P09)"
TEAM = "Nikhil, Prabhat, Ravi, Monu Rajpoot"

BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "msedge", "google-chrome", "chromium", "chromium-browser",
]

CSS = """
@page { size: A4; margin: 12mm 13mm 12mm 13mm; }
body { font-family: "Segoe UI", "Nirmala UI", "Noto Sans Devanagari", Arial, sans-serif;
       font-size: 9pt; line-height: 1.2; color: #111; text-align: justify; }
h1.title { font-size: 17pt; margin: 0 0 2px 0; text-align: center; }
p.team { text-align: center; margin: 0 0 14px 0; color: #333; }
h1 { font-size: 12.5pt; margin: 10px 0 3px 0; border-bottom: 1px solid #999; padding-bottom: 2px; }
h2 { font-size: 11pt; margin: 10px 0 3px 0; }
p, ul, ol { margin: 3px 0 6px 0; }
li { margin: 1px 0; }
table { border-collapse: collapse; margin: 6px 0 8px 0; font-size: 8.5pt; }
tr { page-break-inside: avoid; }
th, td { border: 1px solid #aaa; padding: 2px 6px; vertical-align: top; }
th { background: #eee; }
img { max-width: 62%; max-height: 4.4cm; display: block; margin: 6px auto 2px auto; page-break-inside: avoid; }
code { font-family: Consolas, monospace; font-size: 8.8pt; }
ol.refs { padding-left: 22px; font-size: 8.3pt; }
ol.refs li { margin: 1px 0; }
"""

# The placeholder word is assembled from two halves so that a text search of report/ finds only
# real placeholders in the report text, not this checker.
PLACEHOLDER = "TO" + "DO"
NOTE_RE = re.compile(PLACEHOLDER + r"|\(check\)|\*Owner:|^\*[A-Z][^*]*(add|fill|check|delete|confirm)", re.I)


def section_files():
    return sorted(glob.glob(os.path.join(REPORT, "sections", "[0-9][0-9]_*.md")))


def main_sections():
    """Sections 01 to 08, printed before the references."""
    return [p for p in section_files() if os.path.basename(p)[:2] <= "08"]


def after_references():
    """Sections after 08 (the contribution statement), printed after the references."""
    return [p for p in section_files() if os.path.basename(p)[:2] > "08"]


def strip_helper_notes(text):
    """Remove whole-line italic notes that start with "*Owner:" (instructions to the writers)."""
    return "\n".join(line for line in text.splitlines() if not line.strip().startswith("*Owner:"))


def references_html():
    """Turn the table in references.md into a numbered list."""
    path = os.path.join(REPORT, "references.md")
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = re.match(r"\|\s*\[(\d+)\]\s*\|\s*(.+?)\s*\|\s*$", line)
            if m:
                items.append((int(m.group(1)), m.group(2)))
    lis = "".join(f'<li value="{n}">{markdown.markdown(t)[3:-4]}</li>' for n, t in sorted(items))
    return f'<h1>References</h1>\n<ol class="refs">{lis}</ol>'


def find_open_items():
    """Every line that still needs work before the final PDF."""
    hits = []
    for path in section_files() + [os.path.join(REPORT, "references.md")]:
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                s = line.strip()
                if s.startswith("*Owner:"):
                    continue
                if NOTE_RE.search(s) or "**Verify**" in s:
                    hits.append(f"{os.path.relpath(path, ROOT)}:{i}: {s[:110]}")
    return hits


def build_html():
    parts = [f'<h1 class="title">{TITLE}</h1>', f'<p class="team">{TEAM}</p>']
    md = markdown.Markdown(extensions=["tables", "sane_lists"])
    def add(path):
        with open(path, encoding="utf-8") as f:
            parts.append(md.reset().convert(strip_helper_notes(f.read())))

    for path in main_sections():
        add(path)
    parts.append(references_html())
    for path in after_references():
        add(path)
    body = "\n".join(parts)
    # Section files reference figures as ../figures/x.png; the HTML lives in report/build/,
    # so the same relative path still points to report/figures/.
    return f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{TITLE}</title>' \
           f"<style>{CSS}</style></head><body>{body}</body></html>"


def find_browser():
    for b in BROWSERS:
        p = b if os.path.isabs(b) else shutil.which(b)
        if p and os.path.exists(p):
            return p
    return None


def print_pdf(html_path, pdf_path):
    browser = find_browser()
    if not browser:
        raise SystemExit("No Edge/Chrome found. Open the HTML in a browser and use Print -> Save as PDF.")
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
    url = "file:///" + html_path.replace("\\", "/")
    cmd = [browser, "--headless", "--disable-gpu", "--no-pdf-header-footer", "--print-to-pdf-no-header",
           f"--print-to-pdf={pdf_path}", url]
    subprocess.run(cmd, check=False, capture_output=True, timeout=180)
    if not os.path.exists(pdf_path):
        raise SystemExit(f"PDF was not created. Command: {' '.join(cmd)}")


def page_count(pdf_path):
    try:
        from pypdf import PdfReader
        return len(PdfReader(pdf_path).pages)
    except ImportError:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final", action="store_true", help="write the submission PDF (refused while placeholders remain)")
    a = ap.parse_args()

    open_items = find_open_items()
    if a.final and open_items:
        print("Refusing --final: these lines still need work:\n  " + "\n  ".join(open_items))
        sys.exit(1)

    os.makedirs(BUILD, exist_ok=True)
    html_path = os.path.join(BUILD, "report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(build_html())
    pdf_path = FINAL_PDF if a.final else os.path.join(BUILD, "report_draft.pdf")
    print_pdf(html_path, pdf_path)

    mb = os.path.getsize(pdf_path) / 1e6
    pages = page_count(pdf_path)
    print(f"PDF: {os.path.relpath(pdf_path, ROOT)}")
    print(f"Pages: {pages if pages is not None else '? (pip install pypdf)'} (target 6 to 8)")
    print(f"Size: {mb:.2f} MB (limit {MAX_MB:.0f} MB){'  <-- TOO LARGE' if mb > MAX_MB else ''}")
    if open_items:
        print(f"\n{len(open_items)} open item(s) before the final version:")
        for h in open_items:
            print("  " + h)


if __name__ == "__main__":
    main()
