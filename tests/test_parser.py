from src.parser import Parser
from pathlib import Path

MAIN_DIR = Path(__file__).parent.parent

def test_wczytaj_opis():
    data_file = MAIN_DIR / "data" / "example" / "opis.json"

    parser = Parser(data_file)
    opis = parser.wczytaj_opis()

    assert opis is not None
    assert opis.meta['description'] == "Przykład: model student z tutoriala pgmpy"