"""Regenerate every figure and assemble them in one static page,
app/plot/report.html (figures in app/plot/figures/; both are not versioned).

Run:  python3 app/plot/report.py

To add a study: write a module here with a make() that returns a list of
(png name, caption) and add it to SECTIONS.
"""
import datetime
import html
import subprocess
import traceback

import bench_green_cost
import green_irlm
import green_siam
import quench_vs_excitation
from common import HERE, REPO

SECTIONS = [
    ("IRLM Green function: separate vs shared frames", green_irlm),
    ("SIAM Green function: FBR vs star vs chain", green_siam),
    ("Green function cost: few-body vs full star", bench_green_cost),
    ("Cost of one evolution, by protocol", quench_vs_excitation),
]

STYLE = """
body { font: 15px/1.5 system-ui, sans-serif; max-width: 1100px; margin: 2rem auto;
       padding: 0 1rem; color: #1f1f1e; background: #fcfcfb; }
h1 { font-size: 1.5rem; margin-bottom: .2rem; }
h2 { font-size: 1.15rem; margin-top: 2.5rem; border-bottom: 1px solid #ddd; }
.meta, figcaption { color: #52514e; font-size: .9rem; }
figure { margin: 1.2rem 0; }
img { max-width: 100%; border: 1px solid #eee; }
code { background: #f1f0ec; padding: 0 .25rem; border-radius: 3px; }
pre.error { color: #a02020; white-space: pre-wrap; }
"""


def git_describe():
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "describe", "--always", "--dirty"],
                                       text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main():
    parts = [f"<title>impurityMPS app report</title><style>{STYLE}</style>",
             "<h1>impurityMPS numerical experiments</h1>",
             f"<p class='meta'>Generated {datetime.datetime.now():%Y-%m-%d %H:%M} at "
             f"<code>{git_describe()}</code> from <code>app/output</code> and "
             f"<code>test/ref/output</code> by <code>app/plot/report.py</code>.</p>"]
    for title, module in SECTIONS:
        parts.append(f"<h2>{html.escape(title)}</h2>")
        parts.append(f"<p class='meta'><code>app/plot/{module.__name__}.py</code></p>")
        try:
            figs = module.make()
        except Exception:
            parts.append(f"<pre class='error'>{html.escape(traceback.format_exc())}</pre>")
            continue
        if not figs:
            parts.append("<p class='meta'>No data found.</p>")
        for name, caption in figs:
            parts.append(f"<figure><img src='figures/{name}' alt='{html.escape(caption)}'>"
                         f"<figcaption>{html.escape(caption)}</figcaption></figure>")
            print(f"  {name}")
    out = HERE / "report.html"
    out.write_text("\n".join(parts))
    print(f"wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
