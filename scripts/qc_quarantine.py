"""QC figure: the 6 quarantined label/image pairs, one row each, vs a good control."""
import os, csv, re, numpy as np, nibabel as nib
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT="/path/to/cohort"; OUT="audit"
rows=[r for r in csv.DictReader(open(f"{OUT}/quarantined.csv")) if r["image"]]
ctrl=list(csv.DictReader(open(f"{OUT}/manifest.csv")))[0]
rows.append({"subject":ctrl["subject"],"seg":ctrl["seg"],"image":ctrl["image"],
             "reason":f"GOOD control (z={ctrl['z']})","z":ctrl["z"]})
fig,axes=plt.subplots(len(rows),5,figsize=(17,3.1*len(rows)))
for i,r in enumerate(rows):
    d=[x for x in os.listdir(ROOT) if x.startswith(r["subject"])][0]
    sg=nib.load(os.path.join(ROOT,d,r["seg"])); m=np.asanyarray(sg.dataobj)>0
    im=nib.load(os.path.join(ROOT,d,r["image"])); a=np.asanyarray(im.dataobj).astype(np.float32)
    zs=np.where(m.any(axis=(0,1)))[0]; pick=zs[np.linspace(0,len(zs)-1,5).astype(int)]
    vmin,vmax=np.percentile(a,1),np.percentile(a,99.5)
    for j,z in enumerate(pick):
        ax=axes[i,j]; sl=np.rot90(a[:,:,z]); mm=np.rot90(m[:,:,z])
        ax.imshow(sl,cmap="gray",vmin=vmin,vmax=vmax)
        ov=np.zeros((*mm.shape,4)); ov[mm]=[1,0,0,0.40]; ax.imshow(ov)
        if mm.any(): ax.contour(mm,levels=[0.5],colors="red",linewidths=1.0)
        ax.axis("off")
        if j==0: ax.set_ylabel(r["subject"])
    good = "GOOD" in r["reason"]
    axes[i,0].text(-0.04,0.5,f"{r['subject']}\n{r['seg']}\n{'OK' if good else 'QUARANTINED'} z={r.get('z','')}",
                   transform=axes[i,0].transAxes, rotation=90, va="center", ha="right",
                   fontsize=8, color="green" if good else "red")
fig.suptitle("Quarantined label/image pairs: mask (red) does NOT sit on FLAIR hyperintensity\n(bottom row = correctly aligned control)",fontsize=12)
fig.tight_layout(); fig.savefig(f"{OUT}/quarantined_qc.png",dpi=80,bbox_inches="tight")
print("wrote", f"{OUT}/quarantined_qc.png")
