#!/usr/bin/env python3
"""Download a Roboflow Universe dataset (YOLOv8 format) into datasets/raw/<project>.

  export ROBOFLOW_API_KEY=...        # your own key: Roboflow -> Settings -> API Keys (never commit it)
  python ai/download_roboflow.py --url https://universe.roboflow.com/<workspace>/<project>/dataset/<version>
  python ai/download_roboflow.py --workspace <ws> --project <proj> --version 3

Then merge it into the AVM 6-class dataset (see docs/roboflow_guide.md):
  python ai/prepare_dataset.py --out datasets/avm6 --append \
        --yolo datasets/raw/<project> --yolo-map "<their class>:<our class>"

Needs the `roboflow` package (pip install roboflow), in its own virtualenv if you like.
The key is read from the environment (or asked without echo) - it is never written to disk.
"""
import argparse
import datetime
import getpass
import os
import re
import sys
from pathlib import Path

URL_RE = re.compile(r"universe\.roboflow\.com/([^/]+)/([^/]+)(?:/dataset/(\d+))?")


def parse_url(url):
    m = URL_RE.search(url)
    if not m:
        sys.exit(f"not a Roboflow Universe dataset URL: {url}\n"
                 "expected https://universe.roboflow.com/<workspace>/<project>/dataset/<version>")
    ws, proj, ver = m.groups()
    return ws, proj, int(ver) if ver else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url")
    ap.add_argument("--workspace")
    ap.add_argument("--project")
    ap.add_argument("--version", type=int)
    ap.add_argument("--format", default="yolov8", help="export format (yolov8 = what prepare_dataset.py reads)")
    ap.add_argument("--out", default="datasets/raw")
    a = ap.parse_args()

    if a.url:
        ws, proj, ver = parse_url(a.url)
        ws, proj, ver = a.workspace or ws, a.project or proj, a.version or ver
    else:
        ws, proj, ver = a.workspace, a.project, a.version
    if not (ws and proj and ver):
        sys.exit("give --url (with /dataset/<version>) or --workspace, --project and --version")

    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        if not sys.stdin.isatty():
            sys.exit("set ROBOFLOW_API_KEY in the environment")
        key = getpass.getpass("Roboflow API key (input hidden): ")

    from roboflow import Roboflow                                       # imported late: --help works without it

    dest = Path(a.out) / proj
    dest.mkdir(parents=True, exist_ok=True)
    rf = Roboflow(api_key=key)
    version = rf.workspace(ws).project(proj).version(ver)
    version.download(a.format, location=str(dest))

    (dest / "SOURCE.txt").write_text(
        f"workspace: {ws}\nproject: {proj}\nversion: {ver}\n"
        f"url: https://universe.roboflow.com/{ws}/{proj}/dataset/{ver}\n"
        f"downloaded: {datetime.date.today()}\n"
        "licence: <FILL IN from the dataset page before using / delivering this data>\n")
    print(f"\ndownloaded to {dest}")
    print("class names / licence: open data.yaml and the dataset page; record the licence in SOURCE.txt")


if __name__ == "__main__":
    main()
