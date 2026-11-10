#!/usr/bin/env python
"""ARIA-E FLAIR lesion segmentation - MONAI nnUNetV2Runner pipeline.

Stages: install verified raw dataset -> plan/preprocess -> train 5 folds in
parallel (one GPU per fold) -> find best configuration.

Dataset is pre-built by scripts/build_dataset.py, which pairs each *_seg.nii.gz
with the FLAIR of the same acquisition date and QUARANTINES pairs whose label
does not sit on hyperintense tissue (misregistered labels). Folds are grouped by
subject so no patient appears in both train and val.
"""
import argparse, os, shutil, sys, time

WORK    = "/path/to/nnunet/work_dir"
SRC_RAW = "/path/to/nnunet/nnUNet_raw/Dataset501_ARIAE"
SPLITS  = "/path/to/nnunet/nnUNet_preprocessed/Dataset501_ARIAE/splits_final.json"
CFG     = "configs/input.yaml"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="3d_fullres")
    ap.add_argument("--trainer", default="nnUNetTrainer_500epochs")
    ap.add_argument("--gpus", default="0,1,2,3,4")
    ap.add_argument("--skip-prep", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    a = ap.parse_args()
    gpus = [int(g) for g in a.gpus.split(",") if g != ""]

    from monai.apps.nnunet import nnUNetV2Runner
    runner = nnUNetV2Runner(input_config=CFG, trainer_class_name=a.trainer, work_dir=WORK)
    print("[paths] raw          :", runner.nnunet_raw, flush=True)
    print("[paths] preprocessed :", runner.nnunet_preprocessed, flush=True)
    print("[paths] results      :", runner.nnunet_results, flush=True)

    if not a.skip_prep:
        dst = os.path.join(runner.nnunet_raw, "Dataset501_ARIAE")
        if not os.path.isdir(dst):
            print(f"[prep] copying verified raw dataset -> {dst}", flush=True)
            shutil.copytree(SRC_RAW, dst)
        n_img = len(os.listdir(f"{dst}/imagesTr")); n_lab = len(os.listdir(f"{dst}/labelsTr"))
        print(f"[prep] imagesTr={n_img} labelsTr={n_lab}", flush=True)
        assert n_img == n_lab == 49, f"unexpected case count: {n_img}/{n_lab}"

        t0 = time.time()
        print(f"[prep] plan_and_process (config={a.config}) ...", flush=True)
        runner.plan_and_process(verify_dataset_integrity=True, c=(a.config,), n_proc=(8,))
        print(f"[prep] done in {time.time()-t0:.0f}s", flush=True)

        pp = os.path.join(runner.nnunet_preprocessed, "Dataset501_ARIAE")
        shutil.copyfile(SPLITS, os.path.join(pp, "splits_final.json"))
        print("[prep] installed subject-grouped splits_final.json", flush=True)

    if not a.skip_train:
        print(f"[train] {a.config} | trainer={a.trainer} | 5 folds on GPUs {gpus}", flush=True)
        t0 = time.time()
        runner.train_parallel(configs=a.config, gpu_id_for_all=gpus)
        print(f"[train] finished in {(time.time()-t0)/3600:.2f} h", flush=True)

    print("[done]", flush=True)

if __name__ == "__main__":
    main()
