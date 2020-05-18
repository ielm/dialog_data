from pathlib import Path
from collections import OrderedDict
from nltk.corpus import stopwords
from pprint import pprint
import json
import csv
import uuid
import os

from omicron.nlp import get_tokens, get_topics

ROOT_DIR = Path(f"{os.path.abspath(__file__)}").parent

SRC_DATA_DIR = f"{ROOT_DIR}/data/src"
ANTISCAM_SRC_PATH = f"{SRC_DATA_DIR}/AntiScam_annotated.txt"
PERSUASION_SRC_PATH = f"{SRC_DATA_DIR}/PersuasionForGood_dialogs.csv"

DIALOG_DATA_DIR = f"{ROOT_DIR}/data/dialogs"
ANTISCAM_RAW_DIALOGS = f"{DIALOG_DATA_DIR}/antiscam/raw"
ANTISCAM_ENHANCED_DIALOGS = f"{DIALOG_DATA_DIR}/antiscam/enhanced"
PERSUASION_RAW_DIALOGS = f"{DIALOG_DATA_DIR}/persuasion/raw"
PERSUASION_ENHANCED_DIALOGS = f"{DIALOG_DATA_DIR}/persuasion/enhanced"


def build_antiscam_data(input_file_dir: str = ANTISCAM_SRC_PATH, verbose: bool = False):
    header = ['turn', 'agent', 'text', 'tokens', 'intent', 'semantic_slot', 'topics']

    def _process_dialog(_data: list):

        def _process_row(_index, _row):
            print(".")
            _row = [el.strip() for el in _row.split('\t') if el != '']
            _row.insert(0, _index)
            _row.insert(3, get_tokens(_row[2], False))
            _row.insert(6, get_topics(_row[3]))
            return _row

        _dialog = []
        _turn_index = 0
        for _turn in _data[:-1]:
            if _turn:
                _row = _process_row(_turn_index, _turn)
                _dialog.append(OrderedDict(zip(header, _row)))
                _turn_index += 1
        return _dialog

    dialog_index = 0
    temp_dialog = []
    raw_dialogs = []
    dialogs = []
    with open(input_file_dir, 'r', encoding="ISO-8859-1") as input_file:
        for line in input_file:
            line = line.strip()
            if not line:
                if temp_dialog:
                    print(f"CONV{dialog_index}")
                    dialog_index += 1
                    raw_dialogs.append(temp_dialog)
                    processed_dialog = _process_dialog(temp_dialog)
                    dialogs.append(processed_dialog)
                    temp_dialog = []
            temp_dialog.append(line)

    # -- UNCOMMENT THESE TO WRITE FILES -- #
    # write_raw_files(raw_dialogs)
    # write_enhanced_files(dialogs)


def build_persuasion_data(input_file_dir: str = PERSUASION_SRC_PATH, verbose: bool = False):
    # header = ['index', 'text', 'turn', 'role', 'dialog_id']

    def _process_dialog(_data: tuple):
        header = ['index', 'text', 'tokens', 'topics', 'turn', 'role', 'dialog_id']

        def _process_turn(_turn):
            print(".")
            _turn.insert(2, get_tokens(_turn[1], False))
            _turn.insert(3, get_topics(_turn[2]))
            return _turn

        print(f"\n\nProcessing Dialog: {_data[0]}")

        _dialog = []
        for _turn in _data[1]:
            _row = _process_turn(_turn)
            _dialog.append(OrderedDict(zip(header, _row)))
        return _dialog

    def _process_raw_dialog(_data: tuple):
        raw_header = ['index', 'text', 'turn', 'role', 'dialog_id']
        _dialog = []
        for _turn in _data[1]:
            _dialog.append(OrderedDict(zip(raw_header, _turn)))
        return _dialog

    raw_dialogs = OrderedDict()
    dialogs = OrderedDict()
    dialog_ids = []
    with open(input_file_dir, 'r', encoding="ISO-8859-1") as input_file:
        for line in input_file:
            if line != ",Unit,Turn,B4,B2\n":
                line = line.strip()
                reader = csv.reader([line], skipinitialspace=True)
                for r in reader:
                    if r[4] in dialog_ids:
                        dialogs[r[4]].append(r)
                    else:
                        dialog_ids.append(r[4])
                        dialogs[r[4]] = [r]
        for dialog in dialogs.items():
            # raw_dialogs.append(dialog[1])
            raw_dialogs[dialog[0]] = _process_raw_dialog(dialog)
        # dialogs[dialog[0]] = _process_dialog(dialog)

        pprint(raw_dialogs)

    # -- UNCOMMENT THESE TO WRITE FILES -- #
    # write_raw_persuasion_files(dialogs=[d[1] for d in raw_dialogs.items()],
    #                            directory=PERSUASION_RAW_DIALOGS,
    #                            verbose=True)
    # write_enhanced_files(dialogs=[d[1] for d in dialogs.items()],
    #                          directory=PERSUASION_ENHANCED_DIALOGS,
    #                          verbose=True)


def write_enhanced_files(dialogs: list, directory: str = ANTISCAM_ENHANCED_DIALOGS, verbose: bool = False):
    print(f"\n=== WRITING ENHANCED DATA FILES ===\n")
    for dialog in dialogs:
        filename = str(uuid.uuid4())[:8]
        print(f"writing to -> {filename}.json")
        with open(f"{directory}/{filename}.json", 'w') as json_file:
            json.dump(dialog, json_file, indent=2)


def write_raw_antiscam_files(dialogs: list, directory: str = ANTISCAM_RAW_DIALOGS, verbose: bool = False):
    print(f"\n=== WRITING RAW ANTISCAM DATA FILES ===\n")
    for dialog in dialogs:
        filename = str(uuid.uuid4())[:8]
        print(f"writing to -> {filename}.json")
        with open(f"{directory}/{filename}.txt", 'w') as file:
            for _r in dialog:
                file.write(f"{_r}\n\n")


def write_raw_persuasion_files(dialogs: list, directory: str = ANTISCAM_RAW_DIALOGS, verbose: bool = False):
    print(f"\n=== WRITING RAW PERSUASION DATA FILES ===\n")
    for dialog in dialogs:
        filename = str(uuid.uuid4())[:8]
        print(f"writing to -> {filename}.json")
        with open(f"{directory}/{filename}.json", 'w') as json_file:
            json.dump(dialog, json_file, indent=2)


if __name__ == '__main__':
    build_persuasion_data()
