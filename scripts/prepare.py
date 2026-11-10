"""MONAI nnUNetV2Runner: install verified raw dataset, then fingerprint/plan/preprocess.
We bypass runner.convert_dataset() because our dataset is already in nnU-Net raw
format with alignment-verified label/image pairs (see scripts/build_dataset.py)."""
import os, shutil
from monai.apps.nnunet import nnUNetV2Runner

WORK = "/path/to/nnunet/work_dir"
SRC_RAW = "/path/to/nnunet/nnUNet_raw/Dataset501_ARIAE"
SPLITS  = "/path/to/nnunet/nnUNet_preprocessed/Dataset501_ARIAE/splits_final.json"

runner = nnUNetV2Runner(input_config="configs/input.yaml",
                        trainer_class_name="nnUNetTrainer_500epochs",
                        work_dir=WORK)
dst_raw = os.path.join(runner.nnunet_raw, "Dataset501_ARIAE")
if not os.path.isdir(dst_raw):
    shutil.copytree(SRC_RAW, dst_raw)
print("raw dataset  :", dst_raw)
print("  imagesTr   :", len(os.listdir(f"{dst_raw}/imagesTr")), "  labelsTr:", len(os.listdir(f"{dst_raw}/labelsTr")))

runner.plan_and_process(verify_dataset_integrity=True, c=("3d_fullres",), n_proc=(8,))

pp_dir = os.path.join(runner.nnunet_preprocessed, "Dataset501_ARIAE")
shutil.copyfile(SPLITS, os.path.join(pp_dir, "splits_final.json"))
print("\ninstalled subject-grouped splits ->", os.path.join(pp_dir, "splits_final.json"))
