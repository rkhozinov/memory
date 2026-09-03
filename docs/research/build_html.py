#!/usr/bin/env python3
"""Render the research markdown to self-contained HTML.

Deliberately dependency-free. The alternative was adding `markdown` to the
project just to build two documents; this handles the subset those documents
actually use — headings, tables, fenced and inline code, bold, italic, links,
lists, blockquotes, rules — in less code than the dependency's import line costs
to justify. If a report ever needs more than this, reach for the library then.

Usage:
  uv run python docs/research/build_html.py                 # build all *.md here
  uv run python docs/research/build_html.py foo.md bar.md
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

CSS = """
:root { color-scheme: light dark; }
body { max-width: 52rem; margin: 3rem auto; padding: 0 1.5rem;
       font: 16px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       color: #1a1a1a; background: #fff; }
@media (prefers-color-scheme: dark) {
  body { color: #ddd; background: #16181c; }
  code, pre { background: #23262c !important; }
  th { background: #23262c !important; }
  td, th { border-color: #333 !important; }
  blockquote { border-color: #444 !important; color: #aaa !important; }
  a { color: #7aa2f7 !important; }
  h1, h2 { border-color: #2a2d33 !important; }
}
h1 { font-size: 1.9rem; border-bottom: 2px solid #eee; padding-bottom: .4rem; }
h2 { font-size: 1.4rem; margin-top: 2.5rem; border-bottom: 1px solid #eee;
     padding-bottom: .3rem; }
h3 { font-size: 1.15rem; margin-top: 1.8rem; }
code { background: #f3f4f6; padding: .12em .35em; border-radius: 3px;
       font-size: .88em; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
pre { background: #f3f4f6; padding: 1rem; border-radius: 6px; overflow-x: auto; }
pre code { background: none; padding: 0; font-size: .85em; }
table { border-collapse: collapse; width: 100%; margin: 1.2rem 0; font-size: .92em; }
th, td { border: 1px solid #ddd; padding: .45rem .7rem; text-align: left;
         vertical-align: top; }
th { background: #f6f7f9; font-weight: 600; }
blockquote { border-left: 3px solid #ddd; margin: 1rem 0; padding: .1rem 1rem;
             color: #666; }
hr { border: none; border-top: 1px solid #ddd; margin: 2.5rem 0; }
a { color: #0b57d0; }
li { margin: .25rem 0; }
"""

_INLINE = (
    (re.compile(r"`([^`]+)`"), lambda m: f"<code>{html.escape(m.group(1))}</code>"),
    (re.compile(r"\*\*([^*]+)\*\*"), r"<strong>\1</strong>"),
    (re.compile(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])"), r"<em>\1</em>"),
    (re.compile(r"\[([^\]]+)\]\(([^)]+)\)"), r'<a href="\2">\1</a>'),
)


def _inline(text: str) -> str:
    # Escape first, then re-introduce markup, so document text can never inject
    # tags. Code spans are handled inside the escape-aware lambda above.
    out = html.escape(text)
    for pattern, repl in _INLINE:
        out = pattern.sub(repl, out)
    return out


def _table(rows: list[str]) -> str:
    def cells(line: str) -> list[str]:
        return [c.strip() for c in line.strip().strip("|").split("|")]

    head = cells(rows[0])
    body = [cells(r) for r in rows[2:]]  # rows[1] is the |---|---| separator
    out = ["<table>", "<thead><tr>"]
    out += [f"<th>{_inline(c)}</th>" for c in head]
    out.append("</tr></thead><tbody>")
    for r in body:
        out.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def to_html(md: str) -> str:
    lines = md.split("\n")
    out: list[str] = []
    i = 0
    para: list[str] = []
    list_open: str | None = None

    def flush_para():
        if para:
            out.append(f"<p>{_inline(' '.join(para))}</p>")
            para.clear()

    def close_list():
        nonlocal list_open
        if list_open:
            out.append(f"</{list_open}>")
            list_open = None

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            flush_para()
            close_list()
            i += 1
            block = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            out.append(f"<pre><code>{html.escape(chr(10).join(block))}</code></pre>")
            i += 1
            continue

        # A table is a pipe row followed by a |---| separator row.
        if stripped.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|$", lines[i + 1].strip()):
            flush_para()
            close_list()
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            out.append(_table(block))
            continue

        if not stripped:
            flush_para()
            close_list()
            i += 1
            continue

        if m := re.match(r"^(#{1,6})\s+(.*)$", stripped):
            flush_para()
            close_list()
            lvl = len(m.group(1))
            out.append(f"<h{lvl}>{_inline(m.group(2))}</h{lvl}>")
            i += 1
            continue

        if re.match(r"^(---|\*\*\*|___)$", stripped):
            flush_para()
            close_list()
            out.append("<hr>")
            i += 1
            continue

        if stripped.startswith("> "):
            flush_para()
            close_list()
            out.append(f"<blockquote>{_inline(stripped[2:])}</blockquote>")
            i += 1
            continue

        if m := re.match(r"^(\d+)\.\s+(.*)$", stripped):
            flush_para()
            if list_open != "ol":
                close_list()
                out.append("<ol>")
                list_open = "ol"
            out.append(f"<li>{_inline(m.group(2))}</li>")
            i += 1
            continue

        if m := re.match(r"^[-*]\s+(.*)$", stripped):
            flush_para()
            if list_open != "ul":
                close_list()
                out.append("<ul>")
                list_open = "ul"
            out.append(f"<li>{_inline(m.group(1))}</li>")
            i += 1
            continue

        # Continuation of a list item wrapped across lines.
        if list_open and line.startswith(("  ", "\t")):
            if out and out[-1].endswith("</li>"):
                out[-1] = out[-1][: -len("</li>")] + " " + _inline(stripped) + "</li>"
            i += 1
            continue

        para.append(stripped)
        i += 1

    flush_para()
    close_list()
    return "\n".join(out)


def build(md_path: Path) -> Path:
    md = md_path.read_text()
    title = next((ln.lstrip("# ").strip() for ln in md.split("\n") if ln.startswith("# ")), md_path.stem)
    out_path = md_path.with_suffix(".html")
    out_path.write_text(
        "<!doctype html>\n<html lang=\"en\">\n<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{html.escape(title)}</title>\n"
        f"<style>{CSS}</style>\n</head>\n<body>\n{to_html(md)}\n</body>\n</html>\n"
    )
    return out_path


def main() -> int:
    here = Path(__file__).resolve().parent
    targets = [Path(a) for a in sys.argv[1:]] or sorted(here.glob("*.md"))
    for t in targets:
        print(f"  {build(t)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
