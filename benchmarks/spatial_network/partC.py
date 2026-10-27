import json, re, sys, urllib.request

sys.path.insert(0, "/tmp/gs_bench")
from common import log, table


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "gs-bench"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


PY = ["cp310", "cp311", "cp312", "cp313", "cp314"]
PLAT = {"macos-arm64": r"macosx_.*arm64|universal2", "linux-x86_64": r"(manylinux|musllinux).*x86_64", "win-amd64": r"win_amd64"}

log("\n## Part C — packaging facts (PyPI JSON / GitHub API)\n")
rows = []
for pkg in ["pandana", "osmnx", "igraph", "pyrosm", "quackosm", "osmium"]:
    d = get(f"https://pypi.org/pypi/{pkg}/json")
    info = d["info"]
    ver = info["version"]
    files = d["urls"]
    up = min((f["upload_time"] for f in files), default="?")[:10]
    wheels = [f["filename"] for f in files if f["filename"].endswith(".whl")]
    abi3 = [w for w in wheels if "abi3" in w or "py3-none" in w]
    cov = []
    for p in PY:
        plats = [k for k, rx in PLAT.items() if any((p in w or (w in abi3)) and re.search(rx, w) for w in wheels)]
        if any("py3-none-any" in w for w in wheels):
            plats = ["any"]
        cov.append(f"{p}: {','.join(plats) or '-'}")
    sdist = any(f["packagetype"] == "sdist" for f in files)
    req = info.get("requires_dist") or []
    rows.append([pkg, ver, up, info.get("requires_python") or "", "<br>".join(cov), "yes" if sdist else "no", "; ".join(req)[:400]])
    print(rows[-1], flush=True)
table(["package", "latest", "uploaded", "requires_python", "wheels (cp → platforms)", "sdist", "requires_dist"], rows)

r = get("https://api.github.com/repos/UDST/pandana")
c = get("https://api.github.com/repos/UDST/pandana/commits?per_page=5")
rel = get("https://api.github.com/repos/UDST/pandana/releases?per_page=3")
log(f"- UDST/pandana: pushed_at {r['pushed_at']}, updated_at {r['updated_at']}, archived={r['archived']}, open issues {r['open_issues_count']}, stars {r['stargazers_count']}")
log("- last commits: " + "; ".join(f"{x['commit']['committer']['date'][:10]} {x['commit']['message'].splitlines()[0][:60]}" for x in c))
log("- last releases: " + "; ".join(f"{x['tag_name']} ({x['published_at'][:10]})" for x in rel))
