"""One tokeniser for training and sampling. sample.py mapped a character outside
the frozen vocab to id 0 (a newline) while prepare.py mapped it to '?', so a
prompt with a curly apostrophe was sampled from a line break (2026-10-18)."""
import pickle, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare import encode

STOI = {"\n": 0, " ": 1, "?": 2, "I": 3, "s": 4}


def test_unknown_character_is_question_mark_not_newline():
    assert encode("I’s", STOI) == [3, 2, 4]


def test_known_text_round_trips():
    assert encode("Is ", STOI) == [3, 4, 1]


def test_vocab_without_question_mark_falls_back_to_zero():
    assert encode("x", {"a": 5}) == [0]


def test_shipped_vocab_maps_unknown_to_question_mark():
    meta_path = Path(__file__).resolve().parents[1] / "data" / "meta.pkl"
    if not meta_path.exists():
        return  # data/ is gitignored; the frozen vocab only exists on the training machine
    stoi = pickle.loads(meta_path.read_bytes())["stoi"]
    assert encode("\u4e2d", stoi) == [stoi["?"]]  # a CJK character is outside every English vocab
    assert stoi["?"] != 0
