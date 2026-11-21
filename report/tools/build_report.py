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

How it works, in plain words:
- The report text is written by hand as Markdown files. This script only joins them. It does not
  train or test any model.
- It does not draw PDF pages itself. It builds one HTML page and asks a headless browser
  (Edge or Chrome, no window) to print that page to PDF.
- --final is a safety gate. It refuses to write the submission PDF while any placeholder,
  "(check)" or writer's note is still in the text.

Main steps in the code: main() -> find_open_items() -> build_html() -> print_pdf() -> page_count().

"""
# Standard library: argparse = command-line flags, glob = find files by pattern, os = file paths,
# re = regular expressions, shutil = find a program on PATH, subprocess = run the browser,
# sys = exit codes.
import argparse
import glob
import os
import re
import shutil
import subprocess
import sys

# python-markdown: turns Markdown text into HTML.
import markdown

# Repository root. This file is in report/tools/, so go up two folders. abspath makes the result
# the same whatever folder the script is started from.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# report/ folder: holds sections/, references.md and figures/.
REPORT = os.path.join(ROOT, "report")
# Folder for the draft HTML and draft PDF (these are not added to Git).
BUILD = os.path.join(REPORT, "build")
# Where the submission PDF goes when --final is used.
FINAL_PDF = os.path.join(REPORT, "P09_report.pdf")
# Size limit of the PDF in megabytes (printed as a warning if the file is bigger).
MAX_MB = 10.0

# Title block printed at the top of the PDF (also used as the HTML page title).
TITLE = "Handwritten Devanagari Word Recognition: CRNN-CTC vs Transformer OCR (P09)"
# Team member names printed under the title.
TEAM = "Nikhil Kumar Mehta, Prabhat Patidar, Monu Rajpoot, Ravi Rajpoot"

# Browsers to try, in this order. Full Windows paths first, then plain program names that are
# looked up on PATH. find_browser() uses the first one that exists.
BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "msedge", "google-chrome", "chromium", "chromium-browser",
]

# Style sheet for the printed page. It is placed inside a <style> tag in build_html().
# Notes are here, not inside the string, so the CSS text stays exactly as it is.
# - @page: A4 paper with small margins.
# - body: font list (Nirmala UI is the Windows font that draws Devanagari), 9 pt text, justified.
# - h1.title and p.team: the title block. h1 and h2: section headings.
# - table, th, td: thin borders; td.nw and th.nw: no line wrap (class added in build_html).
# - span.dv: Devanagari font for a lone vowel sign (see the regex in build_html).
# - img: figures are limited in width and height and are centred.
# - code and ol.refs: font sizes for code text and for the reference list.
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
td.nw, th.nw { white-space: nowrap; }
span.dv { font-family: "Nirmala UI", "Noto Sans Devanagari", sans-serif; display: inline-block; white-space: nowrap; }
img { max-width: 62%; max-height: 4.4cm; display: block; margin: 6px auto 2px auto; page-break-inside: avoid; }
code { font-family: Consolas, monospace; font-size: 8.8pt; }
ol.refs { padding-left: 22px; font-size: 8.3pt; }
ol.refs li { margin: 1px 0; }
"""

# The placeholder word is assembled from two halves so that a text search of report/ finds only
# real placeholders in the report text, not this checker.
PLACEHOLDER = "TO" + "DO"
# Regex that finds lines still needing work (case-insensitive). It matches any one of:
# the placeholder word, the text (check), the text *Owner:, or a line that starts with * and a
# capital letter and later contains add / fill / check / delete / confirm (an italic writer's note).
NOTE_RE = re.compile(PLACEHOLDER + r"|\(check\)|\*Owner:|^\*[A-Z][^*]*(add|fill|check|delete|confirm)", re.I)


def section_files():
    """Return all section files in report/sections/ as a sorted list of paths.
    A file name must be two digits, an underscore, anything, then .md (for example 01_intro.md).
    Sorting by name puts the sections in order 01, 02, 03 ...
    """
    return sorted(glob.glob(os.path.join(REPORT, "sections", "[0-9][0-9]_*.md")))


def main_sections():
    """Sections 01 to 08, printed before the references.

    Returns a list of paths. The two-digit prefix of the file name is compared as text, which is
    correct for two-digit numbers.
    """
    # Keep only files whose first two characters of the name are '08' or lower.
    return [p for p in section_files() if os.path.basename(p)[:2] <= "08"]


def after_references():
    """Sections after 08 (the contribution statement), printed after the references.

    Returns a list of paths (section 09 and above).
    """
    return [p for p in section_files() if os.path.basename(p)[:2] > "08"]


def strip_helper_notes(text):
    """Remove whole-line italic notes that start with "*Owner:" (instructions to the writers).

    Input: text of one Markdown file. Returns the same text without those lines. These lines are notes
    for the writers and must not appear in the PDF.
    """
    # Keep every line except the ones that start (after spaces) with *Owner:, then join with newlines.
    return "\n".join(line for line in text.splitlines() if not line.strip().startswith("*Owner:"))


def references_html():
    """Turn the table in references.md into a numbered list.

    Returns an HTML string: a References heading and a numbered list. The table is Markdown, one row
    per reference, like: | [1] | text |. Numbers are kept as written in the file.
    """
    path = os.path.join(REPORT, "references.md")
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            # Match a table row of the form | [number] | text |. Group 1 = the number, group 2 = the text
            # (lazy match, so it stops before the closing | at the end of the line).
            m = re.match(r"\|\s*\[(\d+)\]\s*\|\s*(.+?)\s*\|\s*$", line)
            if m:
                items.append((int(m.group(1)), m.group(2)))
    # Build one <li> per reference, sorted by number. value= keeps the original number in the list.
    # markdown.markdown() wraps text in <p>...</p>; [3:-4] cuts off the 3-character <p> and the
    # 4-character </p> so only the inline HTML (links, italics) stays.
    lis = "".join(f'<li value="{n}">{markdown.markdown(t)[3:-4]}</li>' for n, t in sorted(items))
    # Return a heading plus the ordered list (class refs, styled in the CSS).
    return f'<h1>References</h1>\n<ol class="refs">{lis}</ol>'


def find_open_items():
    """Every line that still needs work before the final PDF.

    Returns a list of strings like "path:line: text". Used by --final to refuse, and by every run to
    print a to-do list. A line counts if it matches NOTE_RE or has the text **Verify**.
    """
    hits = []
    # Check all section files and the reference list.
    for path in section_files() + [os.path.join(REPORT, "references.md")]:
        with open(path, encoding="utf-8") as f:
            # enumerate(..., 1) numbers the lines from 1, as in an editor.
            for i, line in enumerate(f, 1):
                s = line.strip()
                # Skip *Owner: lines. They are removed from the PDF anyway, so they are not open items.
                if s.startswith("*Owner:"):
                    continue
                # Record the file (relative to the repository root), the line number and the first 110 characters.
                if NOTE_RE.search(s) or "**Verify**" in s:
                    hits.append(f"{os.path.relpath(path, ROOT)}:{i}: {s[:110]}")
    return hits


def build_html():
    """Build the full HTML page for the report and return it as one string.
    Order: title block, sections 01 to 08, references, then the later sections (09).
    """
    parts = [f'<h1 class="title">{TITLE}</h1>', f'<p class="team">{TEAM}</p>']
    # Markdown converter. 'tables' lets | a | b | tables work; 'sane_lists' makes lists behave as expected.
    md = markdown.Markdown(extensions=["tables", "sane_lists"])
    # Small helper: read one section file, drop the writer notes, convert it to HTML, keep the result.
    def add(path):
        with open(path, encoding="utf-8") as f:
            # reset() clears the converter state, so one file does not affect the next.
            parts.append(md.reset().convert(strip_helper_notes(f.read())))

    for path in main_sections():
        add(path)
    parts.append(references_html())
    for path in after_references():
        add(path)
    # Join all HTML parts into one body.
    body = "\n".join(parts)
    # Short table cells (numbers such as "0.30 %", model names such as "C: TrOCR-style") must not
    # wrap; longer text cells still wrap normally.
    # Find table cells (td or th) whose text is 1 to 14 characters with no tag inside, and add
    # class="nw" (no wrap). \1 repeats the tag name, \2 keeps other attributes, \3 is the text.
    body = re.sub(r"<(td|th)( [^>]*)?>([^<]{1,14})</\1>", r'<\1 class="nw"\2>\3</\1>', body)
    # A vowel sign shown alone is written on a dotted circle (U+25CC). Both must come from the same
    # font, or the browser shapes the sign alone and draws a second circle.
    # Find a dotted circle followed by one or more Devanagari vowel signs or marks and wrap the
    # group in <span class="dv"> so that it is drawn with the Devanagari font.
    body = re.sub("(◌[ऀ-ःऺ-ॏॕ-ॗॢॣ]+)", r'<span class="dv">\1</span>', body)
    # Section files reference figures as ../figures/x.png; the HTML lives in report/build/,
    # so the same relative path still points to report/figures/.
    # Return the complete page. The CSS is placed inside the <style> tag.
    return f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{TITLE}</title>' \
           f"<style>{CSS}</style></head><body>{body}</body></html>"


def find_browser():
    """Find an installed browser. Returns its path, or None if none of BROWSERS exists."""
    for b in BROWSERS:
        # A full path is used as it is. A plain name is looked up on PATH with shutil.which.
        p = b if os.path.isabs(b) else shutil.which(b)
        # Return the first browser that really exists.
        if p and os.path.exists(p):
            return p
    return None


def print_pdf(html_path, pdf_path):
    """Print an HTML file to a PDF file with a headless browser.
    Inputs: html_path = the HTML file, pdf_path = where the PDF should be written.
    Stops the program with a message if there is no browser or the PDF was not created.
    """
    browser = find_browser()
    if not browser:
        raise SystemExit("No Edge/Chrome found. Open the HTML in a browser and use Print -> Save as PDF.")
    # Delete an old PDF first. Then the existence check below proves a new PDF was made.
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
    # Browsers open local files with a file:/// URL, so change the Windows backslashes to slashes.
    url = "file:///" + html_path.replace("\\", "/")
    # Browser command: --headless = no window; --disable-gpu = no GPU use; the two --no-pdf-header-footer /
    # --print-to-pdf-no-header flags switch off the page header and footer; --print-to-pdf=<file> = output.
    cmd = [browser, "--headless", "--disable-gpu", "--no-pdf-header-footer", "--print-to-pdf-no-header",
           f"--print-to-pdf={pdf_path}", url]
    # Run the browser. check=False: do not raise on a non-zero exit code (the next lines check the file
    # instead). capture_output=True: hide the browser's console text. timeout=180: give up after 180 seconds.
    subprocess.run(cmd, check=False, capture_output=True, timeout=180)
    # If no PDF file exists, stop with a message that shows the command used.
    if not os.path.exists(pdf_path):
        raise SystemExit(f"PDF was not created. Command: {' '.join(cmd)}")


def page_count(pdf_path):
    """Return the number of pages in the PDF, or None if the pypdf package is not installed."""
    try:
        # Imported here, not at the top, because pypdf is optional (the script works without it).
        from pypdf import PdfReader
        return len(PdfReader(pdf_path).pages)
    except ImportError:
        return None


def main():
    """Command-line entry point: check open items, build HTML, print the PDF, report page count and size."""
    # Command-line parser. The module docstring is used as the help text; RawDescriptionHelpFormatter
    # keeps its line breaks.
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # --final is a yes/no flag (False when not given).
    ap.add_argument("--final", action="store_true", help="write the submission PDF (refused while placeholders remain)")
    a = ap.parse_args()

    # Collect the lines that still need work.
    open_items = find_open_items()
    # Gate: with --final and any open item, print the list and exit with code 1 (failure).
    if a.final and open_items:
        print("Refusing --final: these lines still need work:\n  " + "\n  ".join(open_items))
        sys.exit(1)

    # Create report/build/ if it is missing (exist_ok: no error if it already exists).
    os.makedirs(BUILD, exist_ok=True)
    # Write the HTML next to the draft PDF.
    html_path = os.path.join(BUILD, "report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(build_html())
    # Final PDF for --final, otherwise the draft PDF in report/build/.
    pdf_path = FINAL_PDF if a.final else os.path.join(BUILD, "report_draft.pdf")
    print_pdf(html_path, pdf_path)

    # File size in megabytes (1 MB = 1,000,000 bytes here).
    mb = os.path.getsize(pdf_path) / 1e6
    pages = page_count(pdf_path)
    # Print the result: output path, page count (target 6 to 8) and size against the limit.
    print(f"PDF: {os.path.relpath(pdf_path, ROOT)}")
    print(f"Pages: {pages if pages is not None else '? (pip install pypdf)'} (target 6 to 8)")
    print(f"Size: {mb:.2f} MB (limit {MAX_MB:.0f} MB){'  <-- TOO LARGE' if mb > MAX_MB else ''}")
    # Show the remaining open items so they can be fixed before the final version.
    if open_items:
        print(f"\n{len(open_items)} open item(s) before the final version:")
        for h in open_items:
            print("  " + h)


# Run main() only when the file is started as a script, not when it is imported.
if __name__ == "__main__":
    main()
