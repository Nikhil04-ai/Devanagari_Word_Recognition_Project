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
4. Lays the page out in the format of the instructor's ELCVIA template (ELCVIA_Template.pdf):
   A4, 2 cm top and side margins, 17 x 25 cm text area, Times (title 16 pt bold, sections 14 pt,
   subsections 12 pt, body 11 pt, abstract 10 pt, captions 11 pt), centred title and authors,
   an "Abstract" block with "Key Words" between two rules, sections numbered "1 Introduction",
   numbered captions below tables and figures ("Table 1: ..."), references as "[1] ...", and a
   running header with page numbers on the outer side. Journal-only items of the template
   (journal name, received/accepted dates, ISSN footnote) are left out on purpose.
5. Prints the page count and file size (limit 10 MB; the template asks for 8 to 12 pages) and lists
   every line that still contains a placeholder, "(check)" or an italic writer's note, with its file
   and line number.

Captions are written in the section files as a line "Table: text" after a table, or "*Figure: text*"
after an image; this script numbers them in order.

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
# Authors printed under the title, each with the roll number below the name.
AUTHORS = [
    ("Nikhil Kumar Mehta", "202410101200085"),
    ("Prabhat Patidar", "202410101200093"),
    ("Monu Rajpoot", "202410101200080"),
    ("Ravi Rajpoot", "202410101200112"),
]
# Running header on every page after the first (the template uses "Name1 et al. / <journal>";
# we use the first author and a short title instead of a journal name).
RUNNING_HEAD = "Mehta et al. / Handwritten Devanagari Word Recognition"

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
# - @page: A4, margins 20 mm top and sides and 27 mm bottom, which gives the template's
#   17 cm x 25 cm text area. The margin boxes hold the running header (centre) and the page number
#   (outer side: left on even pages, right on odd pages); the first page has neither, as in the
#   template. %RUNNING_HEAD% is replaced in build_html().
# - body: Times New Roman 11 pt (Nirmala UI draws the Devanagari), justified; a paragraph that
#   follows another paragraph starts with an indent, as in the template.
# - .titleblock: title 16 pt bold, authors with roll numbers in italics, all centred.
# - hr.rule and .abstract: the two rules and the 10 pt abstract with "Key Words".
# - h1, h2, h3: section 14 pt, subsection 12 pt, subsubsection 11 pt, all bold; .num is the number.
# - table, th, td: centred tables with thin black borders; td.nw and th.nw: no line wrap.
# - p.caption: numbered table and figure captions, 11 pt, centred, below the table or figure.
# - span.dv: Devanagari font for a lone vowel sign (see the regex in build_html).
# - img: figures are limited in width and height and are centred.
# - .refs: reference list as "[n] text" with a hanging indent.
CSS = """
@page { size: A4; margin: 20mm 20mm 27mm 20mm;
        @top-center { content: "%RUNNING_HEAD%"; font-family: "Times New Roman", Times, serif;
                      font-style: italic; font-size: 10pt; vertical-align: bottom; padding-bottom: 5mm; } }
@page :left { @top-left { content: counter(page); font-family: "Times New Roman", Times, serif;
                          font-size: 10pt; vertical-align: bottom; padding-bottom: 5mm; } }
@page :right { @top-right { content: counter(page); font-family: "Times New Roman", Times, serif;
                            font-size: 10pt; vertical-align: bottom; padding-bottom: 5mm; } }
@page :first { @top-center { content: none; } @top-left { content: none; } @top-right { content: none; } }
body { font-family: "Times New Roman", Times, "Nirmala UI", "Noto Sans Devanagari", serif;
       font-size: 11pt; line-height: 1.2; color: #000; text-align: justify; }
.titleblock { text-align: center; margin: 0 0 4pt 0; }
h1.title { font-size: 16pt; font-weight: bold; margin: 0 0 12pt 0; }
.authors { display: flex; flex-wrap: wrap; justify-content: center; gap: 4pt 22pt; }
.author { font-size: 11pt; }
.roll { display: block; font-style: italic; font-size: 10pt; }
hr.rule { border: 0; border-top: 0.6pt solid #000; margin: 12pt 0; }
.abstract { font-size: 10pt; margin: 0 2.4em; }
.abstract .abs-title { font-weight: bold; text-align: center; margin: 0 0 5pt 0; }
.abstract p { text-indent: 1.5em; }
.abstract p.kw { text-indent: 0; margin-top: 7pt; }
h1 { font-size: 14pt; font-weight: bold; margin: 16pt 0 6pt 0; page-break-after: avoid; }
h2 { font-size: 12pt; font-weight: bold; margin: 11pt 0 4pt 0; page-break-after: avoid; }
h3 { font-size: 11pt; font-weight: bold; margin: 9pt 0 3pt 0; page-break-after: avoid; }
.num { margin-right: 0.9em; }
p { margin: 0; }
p + p { text-indent: 1.5em; }
ul, ol { margin: 5pt 0; padding-left: 2.2em; }
li { margin: 2pt 0; }
table { border-collapse: collapse; margin: 7pt auto 3pt auto; font-size: 10pt; text-align: left; }
tr { page-break-inside: avoid; }
th, td { border: 0.6pt solid #000; padding: 1.5pt 5pt; vertical-align: top; }
th { font-weight: bold; }
td.nw, th.nw { white-space: nowrap; }
p.caption { text-align: center; text-indent: 0; font-size: 11pt; margin: 3pt 0 10pt 0; }
span.dv { font-family: "Nirmala UI", "Noto Sans Devanagari", sans-serif; display: inline-block; white-space: nowrap; }
img { max-width: 65%; max-height: 5cm; display: block; margin: 7pt auto 2pt auto; page-break-inside: avoid; }
code { font-family: Consolas, monospace; font-size: 9.5pt; }
.refs p { padding-left: 2.2em; text-indent: -2.2em; margin: 2pt 0; }
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
    # Build one paragraph per reference, sorted by number, written "[n] text" as in the template.
    # markdown.markdown() wraps text in <p>...</p>; [3:-4] cuts off the 3-character <p> and the
    # 4-character </p> so only the inline HTML (links, italics) stays.
    ps = "".join(f"<p>[{n}] {markdown.markdown(t)[3:-4]}</p>" for n, t in sorted(items))
    # Return a heading plus the list (class refs, styled in the CSS).
    return f'<h1>References</h1>\n<div class="refs">{ps}</div>'


def number_captions(body):
    """Turn caption lines into numbered captions, in document order.

    "<p>Table: text</p>" becomes "Table 1: Text", "<p><em>Figure: text</em></p>" becomes
    "Figure 1: Text"; tables and figures are counted separately, as in the template.
    """
    counters = {"Table": 0, "Figure": 0}

    def repl(m):
        kind = m.group(1) or m.group(3)
        text = (m.group(2) or m.group(4)).strip()
        counters[kind] += 1
        return f'<p class="caption">{kind} {counters[kind]}: {text[:1].upper()}{text[1:]}</p>'

    return re.sub(r"<p>(Table): (.+?)</p>|<p><em>(Figure): (.+?)</em></p>", repl, body)


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
    authors = "".join(f'<span class="author">{name}<span class="roll">{roll}</span></span>'
                      for name, roll in AUTHORS)
    parts = [f'<div class="titleblock"><h1 class="title">{TITLE}</h1><div class="authors">{authors}</div></div>']
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
    # Abstract block of the template: the "**Abstract.** ..." paragraph and the "*Key Words*: ..."
    # line at the top of 01_intro.md go between two rules, under a centred "Abstract" title.
    body = re.sub(r"<p><strong>Abstract\.</strong>\s*(.+?)</p>\s*<p><em>Key Words</em>:\s*(.+?)</p>",
                  r'<hr class="rule"><div class="abstract"><div class="abs-title">Abstract</div><p>\1</p>'
                  r'<p class="kw"><em>Key Words</em>: \2</p></div><hr class="rule">', body, count=1, flags=re.S)
    # Section numbers as in the template: "1 Introduction" (no dot after the number), "3.1 ...".
    body = re.sub(r"<(h[123])>(\d+(?:\.\d+)*)\.?\s+", r'<\1><span class="num">\2</span>', body)
    # Numbered table and figure captions.
    body = number_captions(body)
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
           f"<style>{CSS.replace('%RUNNING_HEAD%', RUNNING_HEAD)}</style></head><body>{body}</body></html>"


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
    # Print the result: output path, page count (the ELCVIA template asks for 8 to 12) and size.
    print(f"PDF: {os.path.relpath(pdf_path, ROOT)}")
    print(f"Pages: {pages if pages is not None else '? (pip install pypdf)'} (template: 8 to 12)")
    print(f"Size: {mb:.2f} MB (limit {MAX_MB:.0f} MB){'  <-- TOO LARGE' if mb > MAX_MB else ''}")
    # Show the remaining open items so they can be fixed before the final version.
    if open_items:
        print(f"\n{len(open_items)} open item(s) before the final version:")
        for h in open_items:
            print("  " + h)


# Run main() only when the file is started as a script, not when it is imported.
if __name__ == "__main__":
    main()
