from pathlib import Path
from models import OpisZestawu
import json

class Parser: 
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.variables: list[str] = []
        self.states: dict[str, list[str]] = {}
        self.probabilities: dict[str, float] = {}

    def wczytaj_opis(self) -> OpisZestawu:
        """
        Wczytuje opis zestawu z pliku JSON i zwraca obiekt OpisZestawu.
        """
        with open(self.file_path, 'r', encoding='utf-8') as file:
            json_data: dict = json.load(file)
        
        opis_zestawu = OpisZestawu(json_data)

        self.waliduj_opis(opis_zestawu)

        return opis_zestawu

    def waliduj_opis(self, opis: OpisZestawu) -> None:
        """
        Waliduje opis zestawu.
        """
        niepoprawne_pola = False 
        istniejace_pola = {"meta", "variables", "expert_edges", "required_edges", "forbidden_edges", "tiers"}

        niepoprawne_pola = [pole for pole in opis.__dict__.keys() if pole not in istniejace_pola]

        if niepoprawne_pola:
            raise ValueError(f"Niepoprawne pola w opisie zestawu: {niepoprawne_pola}")

