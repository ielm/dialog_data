from pathlib import Path
from collections import OrderedDict
from nltk.corpus import stopwords
import json
import stanza
import uuid
import os


ROOT_DIR = Path(f"{os.path.abspath(__file__)}").parent
SRC_PATH = f"{ROOT_DIR}/data/src/AntiScam_annotated.txt"
RAW_DIALOGS = f"{ROOT_DIR}/data/dialogs/raw"
ENHANCED_DIALOGS = f"{ROOT_DIR}/data/dialogs/enhanced"

stanza_nlp = stanza.Pipeline('en')


def build_data(input_filedir: str = SRC_PATH, _pprint: bool = False):
    header = ['turn', 'agent', 'text', 'tokens', 'intent', 'semantic_slot', 'topics']
    
    def _process_dialog(_data: list):

        def _process_row(_index, _row):
            print(".")
            _row = [el.strip() for el in _row.split('\t') if el != '']
            _row.insert(0, _index)
            _row.insert(3, tokens(_row[2], False))
            _row.insert(6, topics(_row))
            return _row

        _dialog = []
        _turn_index = 0
        for _turn in _data[:-1]:
            if _turn:
                _row = _process_row(_turn_index, _turn)
                _dialog.append(OrderedDict(zip(header, _row)))
                _turn_index += 1
        return _dialog

    with open(input_filedir, 'r', encoding="ISO-8859-1") as input_file:
        dialog_index = 0 
        temp_dialog = []
        raw_dialogs = []
        dialogs = []
        for line in input_file:
            line = line.strip()
            if not line:
                if temp_dialog != []:
                    print(f"CONV{dialog_index}")
                    dialog_index += 1
                    raw_dialogs.append(temp_dialog)
                    processed_dialog = _process_dialog(temp_dialog) 
                    dialogs.append(processed_dialog)
                    temp_dialog = []
            temp_dialog.append(line)
        # write_raw_files(raw_dialogs)  # UNCOMMENT THESE TO WRITE FILES
        # write_enhanced_files(dialogs)  # UNCOMMENT THESE TO WRITE FILES


def write_enhanced_files(dialogs: list, directory: str = ENHANCED_DIALOGS, _pprint: bool = False):
    print(f"\n=== WRITING ENHANCED DATA FILES ===\n")
    for dialog in dialogs:
        filename = str(uuid.uuid4())[:8]
        print(f"writing to -> {filename}.json")
        with open(f"{directory}/{filename}.json", 'w') as jsonfile:
            json.dump(dialog, jsonfile, indent=2)


def write_raw_files(dialogs: list, directory: str = RAW_DIALOGS, _pprint: bool = False):
    print(f"\n=== WRITING RAW DATA FILES ===\n")
    for dialog in dialogs:
        filename = str(uuid.uuid4())[:8]
        print(f"writing to -> {filename}.json")
        with open(f"{directory}/{filename}.txt", 'w') as file:
            for _r in dialog:
                file.write(f"{_r}\n\n")


def tokens(text: str, _pprint: bool = False):
    text = text.strip('"')
    _tokens = {}
    for s in stanza_nlp(f"{text}").sentences:
        if _pprint:
            print(f"\nSentence: {s.text}")
        for w in s.words:
            _tokens[f"{w.lemma}"] = w.to_dict()
            if _pprint:
                print(f"\t{w.lemma}:")
                for item in _tokens[w.lemma]:
                    print(f"\t\t{item}:\t{_tokens[w.lemma][item]}")
    return _tokens


def topics(turn: list):
    swords = set(stopwords.words("english"))
    postags = ["NN", "NNP", "NNS", "NNPS", "PRP", "VB", "VBG", "VBD", "VBN", "VBP"]
    tkns = turn[3]
    tpcs = [w for w in tkns if tkns[w]['text'] not in swords and tkns[w]['xpos'] in postags]
    return tpcs


if __name__ == '__main__':
    build_data()