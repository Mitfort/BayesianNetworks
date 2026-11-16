from dataclasses import dataclass, field
from typing import Literal
import numpy as np
import pandas as pd

class OpisZestawu:
    def __init__(self, json_data: dict):
        self.meta: dict =json_data.get("meta", {})
        self.variables: dict[str, dict] = json_data.get("variables", {})

        self.expert_edges: list[tuple[str,str]] = [(u,v) for u,v in json_data.get("expert_edges", {})]
        self.required_edges: list[tuple[str,str]] = [(u,v) for u,v in json_data.get("required_edges", {})]
        self.forbidden_edges: list[tuple[str,str]] = [(u,v) for u,v in json_data.get("forbidden_edges", {})]
        self.tiers:list[list[str]] = [list(tiers) for tiers in json_data.get("tiers", {})]

        @property
        def states(self)->dict[str,list[str]]:
            return {nazwa: list(spec[states])
                    for nazwa, spec in self.variable.items()
                   }

@dataclass(frozen=True)
class Zapytanie:
    typ:Literal["P", "MAP"]
    cele: list[str]
    warunki: dict[str,str] = field(default_factory=dict)
    numer_linii = int | None = None
    
    def __str__(self) -> str:
        cele_tekst = ", ".join(self.cele)
        if not self.warunki:
            return f"{self.typ}({cele_tekst})"
        warunki_tekst = ", ".join(f"{k}=={v}" for k, v in self.warunki.items())
        return f"{self.typ}({cele_tekst} | {warunki_tekst})"

class ZbiorDanych:
    def __init__(self, df: pd.DataFrame, nazwa_pliku: str, czy_testowy: bool = False):
        self.df = df
        self.nazwa_pliku = nazwa_pliku
        self.czy_testowy = czy_testowy

    @property
    def liczba_wierszy(self) ->int:
        return len(self.df)

    @property
    def liczba_pelnych_wierszy(self) ->int:
        return int(self.df.notna().all(axis=1).sum())

    def statystyki_brakow(self) ->dict[str, tuple[int, float]]:
        n=self.liczba_wierszy
        wynik: dict[str, tuple[int, float]] = {}
        for kolumna in self.df.columns:
            braki = int(self.df[kolumna].isna().sum())
            procent = (braki / n *100.0) if n>0 else 0.0
        wynik[kolumna]= (braki, procent)
        return wynik


@dataclass
class ZestawDanych:
    opis: OpisZestawu
    dane: ZbiorDanych
    test: ZbiorDanych
    zapytania: list[Zapytanie]