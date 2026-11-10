#!/usr/bin/env python3
"""Match each segmentation to its same-date FLAIR under a cohort case root.

Case folders are named "<file_id>" (sometimes with a trailing checkmark). Inside,
segmentations are "<YYYYMMDD>_seg.nii.gz" and the defacing gear's FLAIRs carry a
14-digit acquisition datetime, e.g.
    8_AXIAL_FLAIR_BRAIN_AXIAL_FLAIR_BRAIN_20250502114200_1008_FLAIR_deFaced.nii.gz

Emits the matched-CSV schema (file_id, date, flair_path, seg_path) that
fw_download_t1.py consumes.

    python scripts/make_matched_csv.py /path/to/cohort -o aria_e_matched.csv
"""

import argparse
import csv
import re
import sys
from pathlib import Path

SEG = re.compile(r"^(20\d{6})_seg(?:\s*\(\d+\))?\.nii(\.gz)?$", re.I)
FLAIR = re.compile(r"_(20\d{6})\d{6}_.*_FLAIR_deFaced\.nii(\.gz)?$", re.I)
# Case dirs are "101234" or "101234 ✅"; take the leading digits as the id.
CASE_ID = re.compile(r"^(\d+)")


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="directory holding the case folders")
    ap.add_argument("-o", "--out", type=Path, required=True)
    args = ap.parse_args()

    rows, unmatched = [], []
    for case in sorted(p for p in args.root.iterdir() if p.is_dir()):
        m = CASE_ID.match(case.name)
        if not m:
            log(f"[skip] {case.name}: not a case folder")
            continue
        fid = m.group(1)

        segs, flairs = {}, {}
        for f in sorted(case.iterdir()):
            if (s := SEG.match(f.name)):
                # Keep the first seg per date; "(1)" duplicates sort after the plain name.
                segs.setdefault(s.group(1), f)
            elif (l := FLAIR.search(f.name)):
                flairs.setdefault(l.group(1), f)

        for date, seg in sorted(segs.items()):
            flair = flairs.get(date)
            if flair is None:
                log(f"[MISS] {fid} {date}: seg has no same-date FLAIR "
                    f"(FLAIR dates: {sorted(flairs)})")
                unmatched.append((fid, date))
                continue
            rows.append({
                "file_id": fid, "date": date,
                "flair_path": str(flair), "seg_path": str(seg),
            })
        for date in sorted(set(flairs) - set(segs)):
            log(f"[note] {fid} {date}: FLAIR with no segmentation, not included")

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file_id", "date", "flair_path", "seg_path"])
        w.writeheader()
        w.writerows(rows)

    log(f"\n{len(rows)} matched pairs across "
        f"{len({r['file_id'] for r in rows})} cases -> {args.out}")
    if unmatched:
        log(f"{len(unmatched)} segmentations left unmatched")


if __name__ == "__main__":
    main()
