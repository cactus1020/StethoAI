from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW_ICBHI = DATA / "raw" / "icbhi"
PROCESSED = DATA / "processed"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
FIELD = DATA / "field"

SAMPLE_RATE = 16000
LABELS = ["normal", "crackle", "wheeze", "both"]
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}
ID_TO_LABEL = {i: name for name, i in LABEL_TO_ID.items()}

for _d in (PROCESSED, MODELS, RESULTS, FIELD):
    try:
        _d.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def cycle_label(crackle: int, wheeze: int) -> str:
    if crackle and wheeze:
        return "both"
    if crackle:
        return "crackle"
    if wheeze:
        return "wheeze"
    return "normal"
