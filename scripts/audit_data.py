import os, re, json, glob
import numpy as np, nibabel as nib
from collections import defaultdict

ROOT = "/path/to/cohort"   # one folder per subject
rows = []
subjects = sorted([d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))])
print(f"subjects: {len(subjects)}")

def grid_key(img):
    return (tuple(img.shape[:3]), tuple(np.round(img.affine.flatten(), 3)))

summary = {}
for s in subjects:
    sd = os.path.join(ROOT, s)
    files = sorted(os.listdir(sd))
    segs = [f for f in files if "_seg" in f and f.endswith(".nii.gz")]
    imgs = [f for f in files if f.endswith(".nii.gz") and "_seg" not in f]
    # date of image = 14-digit or 8-digit datetime in filename
    img_dates = {}
    for f in imgs:
        m = re.findall(r"(\d{8})\d{6}", f)
        img_dates[f] = m[0] if m else None
    seg_dates = {}
    for f in segs:
        m = re.match(r"(\d{8})_seg", f)
        seg_dates[f] = m.group(1) if m else None
    summary[s] = dict(n_img=len(imgs), n_seg=len(segs),
                      img_dates=img_dates, seg_dates=seg_dates)

    # geometry per image
    grids = defaultdict(list)
    for f in imgs:
        try:
            im = nib.load(os.path.join(sd, f))
            grids[grid_key(im)].append(f)
        except Exception as e:
            print("ERR load", s, f, e)
    summary[s]["n_distinct_grids"] = len(grids)

    for sf, sdate in seg_dates.items():
        try:
            sg = nib.load(os.path.join(sd, sf))
        except Exception as e:
            rows.append(dict(subj=s, seg=sf, status=f"seg_load_error:{e}")); continue
        lab = np.asanyarray(sg.dataobj)
        vals = np.unique(lab)
        # candidate images with same date
        cands = [f for f, d in img_dates.items() if d == sdate]
        match = [f for f in cands if grid_key(nib.load(os.path.join(sd, f))) == grid_key(sg)]
        # any image (any date) matching seg grid
        any_match = [f for f in imgs if grid_key(nib.load(os.path.join(sd, f))) == grid_key(sg)]
        rows.append(dict(subj=s, seg=sf, seg_date=sdate,
                         seg_shape=tuple(sg.shape), seg_zooms=tuple(np.round(sg.header.get_zooms()[:3],3)),
                         labels=[float(v) for v in vals], n_pos=int((lab>0).sum()),
                         n_date_cands=len(cands), date_grid_match=match,
                         any_grid_match=any_match, n_any_match=len(any_match)))

os.makedirs("audit", exist_ok=True)
json.dump(dict(rows=rows, summary={k:{kk:vv for kk,vv in v.items()} for k,v in summary.items()}),
          open("audit/audit_raw.json","w"), indent=1, default=str)

# ---- report ----
print("\n=== SEG -> IMAGE PAIRING ===")
ok = bad_date = bad_grid = 0
for r in rows:
    if r.get("status"): print("  !!", r); continue
    tag = "OK" if r["date_grid_match"] else ("GRID-MISMATCH" if r["n_date_cands"] else "NO-DATE-IMG")
    if r["date_grid_match"]: ok += 1
    elif r["n_date_cands"]: bad_grid += 1
    else: bad_date += 1
    if tag != "OK":
        print(f"  [{tag}] {r['subj']}/{r['seg']} date={r['seg_date']} shape={r['seg_shape']} zooms={r['seg_zooms']} "
              f"date_cands={r['n_date_cands']} any_grid_match={r['n_any_match']}")
print(f"\nsegs total={len(rows)} | date+grid OK={ok} | grid mismatch={bad_grid} | no same-date image={bad_date}")

print("\n=== LABELS ===")
allv = sorted({v for r in rows if not r.get('status') for v in r['labels']})
print("label values across segs:", allv)
empt = [f"{r['subj']}/{r['seg']}" for r in rows if not r.get('status') and r['n_pos']==0]
print("empty segs:", empt)
vox = [r['n_pos'] for r in rows if not r.get('status')]
print("lesion voxels: min", min(vox), "median", int(np.median(vox)), "max", max(vox))

print("\n=== GRIDS PER SUBJECT (are images co-registered to one grid?) ===")
for s, v in summary.items():
    print(f"  {s:12s} imgs={v['n_img']:2d} segs={v['n_seg']:2d} distinct_grids={v['n_distinct_grids']}")
