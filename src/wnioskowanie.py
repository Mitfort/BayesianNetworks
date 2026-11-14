from dataclasses import dataclass
from itertools import product

import numpy as np
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

from src.models import OpisZestawu, Zapytanie

@dataclass
class WynikZapytaniaP:
    cele: list[str]
    warunki: dict[str, str]
    stany_celow: dict[str, list[str]]
    tablica = np.ndarray

    def rozklad_slownik(self) ->dict[tuple[str, ...], float]:
        kombinacje = product(*(self.stany_celow[v] for v  in self.cele))
        return{ kombinacja: float(prob) for kombinacja, prob in zip(kombinacje, self.tablica.flat)}

    def __str__(self) ->str:
        wpisy:list[str] = []
        if len(self.cele) == 1:
            zmienna = self.cele[0]
            for stan, prob in zip(self.stany_celow[zmienna], self.tablica.flat):
                wpisy.append(f"{stan} {prob:.3f}")
        else:
            for kombinacja, prob in self.rozklad_slownik().items():
                etykieta = ", ".join(f"{z} = {s}" for z, s in zip(self.cele, kombinacja))
                wpisy.append(f"({etykieta}) {prob:.3f}")
        return", ".join(wpisy)

@dataclass
class WynikZapytaniaMAP:
    cele: list[str]
    warunki: dict[str, str]
    konfiguracja: dict [str, str]
    prawdopodobienstwo = float

    def __str__(self) -> str:
        przypisania = ", ".join(
            f"{zmienna}={stan}" for zmienna, stan in self.konfiguracja.items()
        )
        return f"{przypisania} (P={self.prawdopodobienstwo:.3f})"


class SilnikWnioskowania:
    def __init__(self, model: DiscreteBayesianNetwork, opis: OpisZestawu):
        self.model = model
        self.opis = opis
        self.infer = VariableElimination(model)

    def wykonaj_p(self ,cele: list[str], warunki: dict[str, str] | None = None) -> WynikZapytaniaP:
        dowody =  warunki if warunki else None
        factor = self.infer.query(variables=cele, evidence=dowody, show_progress= False)

        permutacja = [factor.variables.index(v) for v in cele]
        tablica = np.transpose(factor.values, permutacja)

        for os_idx, zmienna in enumerate(cele):
            docelowe_stany = self.opis.states[zmienna]
            aktualne_stany = factor.state_names[zmienna]
            if aktualne_stany != docelowe_stany:
                indeksy_stanow =[aktualne_stany.index(s) for s in docelowe_stany]
                tablica = np.take(tablica, indeksy_stanow, axis = os_idx)

        stany_celow = {v: self.opis.states[v] for v in cele}
        return WynikZapytaniaP(cele = list(cele), warunki=dict(warunki or {}), stany_celow=stany_celow, tablica=np.ndarray(tablica, dtype=float))


    def wykonaj_map(self, cele: list [str], warunki: dict[str,str] | None = None) -> WynikZapytaniaMAP:
        dowody = warunki if warunki else None
        surowy_wynik = self.infer.map_query(variables = cele, evidence= dowody, show_progress= False)
        konfiguracja = {zmienna: surowy_wynik[zmienna] for zmienna in cele}

        rozklad=self.wykonaj_p(cele, warunki)
        indeksy = tuple(self.opis.states[zmienna].index(konfiguracja[zmienna]) for zmienna in cele)
        prob = float(rozklad.tablica[indeksy])

        return WynikZapytaniaMAP( cele=list(cele), warunki=dict(warunki or {}),konfiguracja=konfiguracja,prawdopodobienstwo=prob)

    def odpowiedz(self, zapytanie: Zapytanie) -> WynikZapytaniaP | WynikZapytaniaMAP:
        if zapytanie.typ == "P":
            return self.wykonaj_p(zapytanie.cele, zapytanie.warunki)
        if zapytanie.typ == "MAP":
            return self.wykonaj_map(zapytanie.cele, zapytanie.warunki)
        raise ValueError(f"Nieobsługiwany typ zapytania: {zapytanie.typ}")

    def odpowiedz_na_wszystkie(self, zapytania: list[Zapytanie]) -> list[tuple[Zapytanie, WynikZapytaniaP, WynikZapytaniaMAP]]:
        return [(z, self.odpowiedz(z)) for z in zapytania]

    

