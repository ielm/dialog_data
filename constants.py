from pathlib import Path
import os

ROOT_DIR = Path(f"{os.path.abspath(__file__)}").parent

SRC_DATA_DIR = f"{ROOT_DIR}/data/src"
ANTISCAM_SRC_PATH = f"{SRC_DATA_DIR}AntiScam_annotated.txt"
PERSUASION_SRC_PATH = f"{SRC_DATA_DIR}PersuasionForGood_dialogs.csv"

DIALOG_DATA_DIR = f"{ROOT_DIR}/data/dialogs"
RAW_DIALOGS = f"{DIALOG_DATA_DIR}/raw"
ENHANCED_DIALOGS = f"{DIALOG_DATA_DIR}/enhanced"
