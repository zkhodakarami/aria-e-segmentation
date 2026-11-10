#!/usr/bin/env python3
"""Download T1 NIfTI files from Flywheel for the (file_id, date) pairs in a CSV.

Uses the Flywheel REST API directly via `requests`, so no flywheel-sdk install
is needed.

Sessions are found with the dataexplorer search API, one query per subject
(the site-wide /api/subjects and /api/projects listings can time out), and are
matched to the CSV by the 8-digit date in the session label or timestamp.
The defacing gear emits "<acq>_..._<modality>_deFaced.nii.gz"; the matched
CSV's FLAIR paths are exactly those "_FLAIR_deFaced.nii.gz" files, so by
default we take the corresponding "_T1_deFaced.nii.gz".

Host: pass --host, set FW_HOST, or use an API key of the form "host:token".
API key: pass --api-key, or set FW_API_KEY, or put it in ~/.flywheel_api_key.

Typical use:
    python scripts/fw_download_t1.py --list-acquisitions   # what is on Flywheel
    python scripts/fw_download_t1.py --dry-run             # what would be picked
    python scripts/fw_download_t1.py --out-dir /path/to/cohort-T1
"""

import argparse
import csv
import os
import re
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import requests

DEFAULT_CSV = Path(__file__).resolve().parent.parent / "aria_e_matched.csv"
DEFAULT_HOST = os.environ.get("FW_HOST", "")

# The defacing gear tags modality into the filename; this is the reliable selector.
DEFACED_T1 = re.compile(r"_T1_deFaced\.nii(\.gz)?$", re.I)
# Which acquisitions are T1 structurals at all.
T1_LABEL = re.compile(r"(mprage|bravo|\bt1\b|t1w|3d.*t1|t1.*3d)", re.I)
# Acquisitions that mention T1 but are not the pre-contrast structural we want.
# "vNav_setter" is the navigator calibration series, not an image.
T1_EXCLUDE = re.compile(
    r"(t1[ _-]?map|t1rho|scout|localizer|calibration|phoenix|report|perfusion|dynamic"
    r"|setter|post|\+ ?c\b|gad|contrast)", re.I
)
# Other modalities' defaced output, which we must not mistake for a T1.
OTHER_DEFACED = re.compile(r"_(FLAIR|T2ST|T2|ADC|TRACEW|SWI)_deFaced", re.I)
# Siemens phase images ("_ph"), never the structural we want.
PHASE = re.compile(r"_ph\.nii(\.gz)?$", re.I)
# Per-echo output of a multi-echo acquisition; the RMS combination is preferred.
ECHO = re.compile(r"_e\d+(_ph)?\.nii(\.gz)?$", re.I)
# "ND" = Siemens "No Distortion correction"; the untagged series is corrected.
ND = re.compile(r"(^|[ _])ND([ _]|$)")
# Derived reformats of the original 3D acquisition.
REFORMAT = re.compile(r"reformat", re.I)


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def load_api_key(cli_key):
    if cli_key:
        return cli_key.strip()
    if os.environ.get("FW_API_KEY"):
        return os.environ["FW_API_KEY"].strip()
    keyfile = Path.home() / ".flywheel_api_key"
    if keyfile.is_file() and keyfile.read_text().strip():
        return keyfile.read_text().strip()
    sys.exit(
        "No API key. Generate one on your Flywheel instance (Profile -> Flywheel Access "
        "-> Add New API Key), then write it to ~/.flywheel_api_key (chmod 600), "
        "or set FW_API_KEY, or pass --api-key."
    )


class FW:
    def __init__(self, api_key, host=DEFAULT_HOST):
        # Keys are often "host:token"; the host part tells us where to talk to.
        if ":" in api_key:
            key_host = api_key.partition(":")[0]
            if "." in key_host:
                host = key_host
        if not host:
            sys.exit("No Flywheel host. Pass --host, set FW_HOST, or use an "
                     "API key of the form host:token.")
        self.base = f"https://{host}/api"
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"scitran-user {api_key}"})

    def get(self, path, timeout=120, **kw):
        r = self.s.get(f"{self.base}{path}", timeout=timeout, **kw)
        r.raise_for_status()
        return r.json()

    def whoami(self):
        return self.get("/users/self")

    def search_sessions(self, subject_label, limit=200):
        """All sessions for a subject label. Site-wide /subjects listing is too slow."""
        body = {
            "structured_query": f"subject.label = {subject_label}",
            "return_type": "session",
            "limit": limit,
        }
        r = self.s.post(f"{self.base}/dataexplorer/search", json=body, timeout=120)
        r.raise_for_status()
        payload = r.json()
        out = []
        for hit in payload.get("results", payload):
            src = hit.get("_source", hit)
            sess = src.get("session") or {}
            if sess.get("_id"):
                out.append({
                    "_id": sess["_id"],
                    "label": sess.get("label"),
                    "timestamp": sess.get("timestamp"),
                    "project": (src.get("project") or {}).get("label"),
                    "subject": (src.get("subject") or {}).get("label"),
                })
        return out

    def acquisitions(self, session_id):
        return self.get(f"/sessions/{session_id}/acquisitions")

    def download_file(self, acq_id, filename, dest):
        url = f"{self.base}/acquisitions/{acq_id}/files/{quote(filename, safe='')}"
        with self.s.get(url, stream=True, timeout=900) as r:
            r.raise_for_status()
            tmp = dest.with_name(dest.name + ".part")
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
            tmp.rename(dest)
        return dest.stat().st_size


def read_pairs(csv_path):
    """Ordered unique (file_id, date) pairs from the matched CSV."""
    pairs = OrderedDict()
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            pairs.setdefault((row["file_id"].strip(), row["date"].strip()), None)
    return list(pairs.keys())


def session_dates(sess):
    """Dates this session could be, from its label and its timestamp."""
    found = set()
    for m in re.finditer(r"(20\d{6})", sess.get("label") or ""):
        found.add(m.group(1))
    ts = sess.get("timestamp")
    if ts:
        try:
            found.add(datetime.fromisoformat(ts.replace("Z", "+00:00")).strftime("%Y%m%d"))
        except ValueError:
            pass
    return found


def match_session(sessions, want_date):
    hits = [s for s in sessions if want_date in session_dates(s)]
    return hits


def nifti_files(acq):
    out = []
    for f in acq.get("files") or []:
        name = f.get("name", "")
        if f.get("type") == "nifti" or name.endswith((".nii", ".nii.gz")):
            out.append(f)
    return out


def pick_t1(acquisitions, prefer_defaced=True, prefer_nd=False):
    """Return (acquisition, file, how) for the best T1, or (None, None, reason).

    Ranked preferences rather than "largest file wins", because the biggest NIfTI in
    a session is often the wrong one: Siemens phase maps (~39 MB) outweigh the real
    ABCD RMS structural (~26 MB), and an axial MPR reformat (~25 MB) outweighs the
    original sagittal MPRAGE (~10 MB).
    """
    candidates = []
    for acq in acquisitions:
        label = acq.get("label") or ""
        if T1_EXCLUDE.search(label) or not T1_LABEL.search(label):
            continue
        for f in nifti_files(acq):
            name = f["name"]
            if PHASE.search(name) or OTHER_DEFACED.search(name):
                continue
            defaced = bool(DEFACED_T1.search(name))
            is_nd = bool(ND.search(label))
            rank = (
                0 if defaced == prefer_defaced else 1,   # match the FLAIRs' defacing
                1 if ECHO.search(name) else 0,           # prefer the RMS combination
                0 if is_nd == prefer_nd else 1,          # prefer distortion-corrected
                1 if REFORMAT.search(label) else 0,      # prefer original geometry
                -(f.get("size") or 0),                   # tie-break: larger
            )
            candidates.append((rank, acq, f, defaced))

    if not candidates:
        return None, None, "no T1 acquisition"
    candidates.sort(key=lambda t: t[0])
    _, acq, f, defaced = candidates[0]
    return acq, f, "defaced" if defaced else "NOT DEFACED"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--api-key")
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--out-dir", type=Path, default=Path("t1_downloads"))
    ap.add_argument("--raw", action="store_true",
                    help="prefer the non-defaced T1 (default is the deFaced one, matching the FLAIRs)")
    ap.add_argument("--prefer-nd", action="store_true",
                    help="take the Siemens 'ND' (distortion-uncorrected) series instead of the corrected one")
    ap.add_argument("--list-acquisitions", action="store_true",
                    help="discovery mode: print every acquisition + nifti file, download nothing")
    ap.add_argument("--dry-run", action="store_true", help="resolve and report picks, download nothing")
    ap.add_argument("--limit", type=int, help="only process the first N pairs")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--manifest", type=Path, help="write a CSV of what was resolved/downloaded")
    args = ap.parse_args()

    fw = FW(load_api_key(args.api_key), host=args.host)
    try:
        me = fw.whoami()
    except requests.HTTPError as e:
        sys.exit(f"Auth failed ({e}). The API key is invalid or expired -- generate a new one.")
    log(f"Authenticated as {me.get('email') or me.get('_id')}")

    pairs = read_pairs(args.csv)
    if args.limit:
        pairs = pairs[:args.limit]
    log(f"{len(pairs)} unique (file_id, date) pairs from {args.csv}")

    if not (args.list_acquisitions or args.dry_run):
        args.out_dir.mkdir(parents=True, exist_ok=True)

    rows, missing = [], []
    sess_cache = {}

    for fid, date in pairs:
        tag = f"{fid} {date}"
        if fid not in sess_cache:
            sess_cache[fid] = fw.search_sessions(fid)
        sessions = sess_cache[fid]
        if not sessions:
            log(f"[MISS] {tag}: subject not found on Flywheel")
            missing.append((fid, date, "subject not found"))
            continue

        hits = match_session(sessions, date)
        if not hits:
            avail = sorted({d for s in sessions for d in session_dates(s)})
            log(f"[MISS] {tag}: no session on that date. available={avail}")
            missing.append((fid, date, "no session on date"))
            continue
        if len(hits) > 1:
            log(f"[warn] {tag}: {len(hits)} sessions on this date "
                f"({[s['label'] for s in hits]}); using the first")
        sess = hits[0]

        acqs = fw.acquisitions(sess["_id"])

        if args.list_acquisitions:
            print(f"\n=== {fid} / {date} / {sess['label']} (project {sess['project']}) ===")
            for a in acqs:
                print(f"  {a.get('label')!r}  nifti={[f['name'] for f in nifti_files(a)]}")
            continue

        acq, f, how = pick_t1(acqs, prefer_defaced=not args.raw, prefer_nd=args.prefer_nd)
        if acq is None:
            log(f"[MISS] {tag}: {how}. labels={[a.get('label') for a in acqs]}")
            missing.append((fid, date, how))
            continue

        ext = ".nii.gz" if f["name"].endswith(".nii.gz") else Path(f["name"]).suffix
        dest = args.out_dir / fid / f"{fid}_{date}_T1{ext}"
        mb = (f.get("size") or 0) / 1e6
        log(f"[T1  ] {tag}: {acq.get('label')!r} -> {f['name']} ({mb:.1f} MB, via {how})")

        if not args.dry_run:
            if dest.exists() and not args.overwrite:
                log(f"         exists, skipping {dest}")
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                size = fw.download_file(acq["_id"], f["name"], dest)
                log(f"         wrote {dest} ({size/1e6:.1f} MB)")
        rows.append({
            "file_id": fid, "date": date, "session_label": sess["label"],
            "acquisition": acq.get("label"), "fw_filename": f["name"],
            "t1_path": str(dest), "selected_via": how,
        })

    if args.list_acquisitions:
        return

    if args.manifest and rows:
        with open(args.manifest, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        log(f"manifest -> {args.manifest}")

    log(f"\nDone: {len(rows)} resolved, {len(missing)} missing (of {len(pairs)})")
    for fid, date, why in missing:
        log(f"  MISSING {fid} {date}: {why}")


if __name__ == "__main__":
    main()
