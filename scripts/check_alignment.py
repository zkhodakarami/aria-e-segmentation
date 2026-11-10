"""Test whether each seg is voxel-aligned to the FLAIR of the SAME date.
FLAIR lesions are hyperintense: a correctly-aligned seg must sit on bright tissue.
z = (mean intensity in seg - brain mean) / brain std.  z>>0 => aligned."""
import os, re, numpy as np, nibabel as nib
ROOT="/path/to/cohort"   # one folder per subject
def zscore(img_path, mask):
    im = nib.load(img_path); a = np.asanyarray(im.dataobj).astype(np.float32)
    if a.shape != mask.shape: return None
    brain = a > np.percentile(a, 60)          # rough brain/tissue
    mu, sd = a[brain].mean(), a[brain].std()
    if sd == 0: return None
    return float((a[mask].mean() - mu) / sd)

def aff_close(a, b): return np.allclose(a, b, atol=1e-3)

print(f"{'subject':14s} {'seg':22s} {'samedate_z':>10s} {'best_other_z':>12s}  verdict")
bad=[]
for s in sorted(os.listdir(ROOT)):
    sd=os.path.join(ROOT,s)
    if not os.path.isdir(sd): continue
    files=sorted(os.listdir(sd))
    imgs=[f for f in files if f.endswith(".nii.gz") and "_seg" not in f]
    segs=[f for f in files if f.endswith(".nii.gz") and "_seg" in f]
    for sf in segs:
        m=re.match(r"(\d{8})_seg", sf); sdate=m.group(1) if m else None
        sg=nib.load(os.path.join(sd,sf)); mask=np.asanyarray(sg.dataobj)>0
        if mask.sum()==0: continue
        same=[f for f in imgs if sdate and re.findall(r"(\d{8})\d{6}",f) and re.findall(r"(\d{8})\d{6}",f)[0]==sdate]
        zs = None; sfile=None
        for f in same:
            im=nib.load(os.path.join(sd,f))
            if im.shape==sg.shape and aff_close(im.affine, sg.affine):
                zs=zscore(os.path.join(sd,f), mask); sfile=f; break
        others=[]
        for f in imgs:
            if f==sfile: continue
            im=nib.load(os.path.join(sd,f))
            if im.shape==sg.shape:
                z=zscore(os.path.join(sd,f), mask)
                if z is not None: others.append((z,f))
        bo=max(others)[0] if others else float('nan')
        verdict = "ALIGNED" if (zs is not None and zs>0.8) else ("SUSPECT" if zs is not None else "NO-SAMEDATE-MATCH")
        if verdict!="ALIGNED": bad.append((s,sf,zs))
        print(f"{s:14s} {sf:22s} {str(round(zs,2)) if zs is not None else 'n/a':>10s} {bo:12.2f}  {verdict}")
print("\nnon-aligned:", len(bad)); [print("  ",b) for b in bad]
