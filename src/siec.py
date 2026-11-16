from dataclasses import dataclass
from itertools import product

import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.parameter_estimator import DiscreteMLE

from src.models import OpisZestawu, ZbiorDanych


@dataclass
class WynikAnalizyCPD:
    zmienna: str
    rodzice: list[str]
    ksztalt_tablicy: tuple[int, ...]
    liczba_parametrow: int
    liczba_konfiguracji_rodzicow: int
    liczba_nieobserwowanych_konfiguracji: int
    nieobserwowane_przyklady: list[tuple[dict[str, str], np.ndarray]]


def zbuduj_siec_mle(
    opis: OpisZestawu,
    dane: ZbiorDanych,
    krawedzie: list[tuple[str, str]] | None = None,
) -> DiscreteBayesianNetwork:
    
    wybrane_krawedzie = opis.expert_edges if krawedzie is None else krawedzie

    model = DiscreteBayesianNetwork()
    model.add_nodes_from(opis.states.keys())
    model.add_edges_from(wybrane_krawedzie)

    estymator = DiscreteMLE(state_names=opis.states)
    model.fit(dane.df, estimator=estymator)
    return model


def analizuj_cpd(
    cpd: TabularCPD, dane: ZbiorDanych, max_przykladow: int = 5
) -> WynikAnalizyCPD:
    zmienna = cpd.variable
    rodzice = list(cpd.variables[1:])
    ksztalt = tuple(int(dim) for dim in cpd.values.shape)
    liczba_parametrow = int(cpd.values.size)

    rodzina = [zmienna] + rodzice
    pelne_wiersze_rodziny = dane.df[rodzina].dropna()

    if not rodzice:
        liczba_konfiguracji = 1
        nieobserwowane: list[tuple[dict[str, str], np.ndarray]] = []
        if len(pelne_wiersze_rodziny) == 0:
            nieobserwowane.append(({}, cpd.values.copy()))
        return WynikAnalizyCPD(
            zmienna=zmienna,
            rodzice=rodzice,
            ksztalt_tablicy=ksztalt,
            liczba_parametrow=liczba_parametrow,
            liczba_konfiguracji_rodzicow=liczba_konfiguracji,
            liczba_nieobserwowanych_konfiguracji=len(nieobserwowane),
            nieobserwowane_przyklady=nieobserwowane[:max_przykladow],
        )

    stany_rodzicow = [cpd.state_names[rodzic] for rodzic in rodzice]
    wszystkie_konfiguracje = list(product(*stany_rodzicow))

    zaobserwowane_konfiguracje = set(
        pelne_wiersze_rodziny[rodzice].itertuples(index=False, name=None)
    )

    nieobserwowane = []
    for konfig in wszystkie_konfiguracje:
        if konfig not in zaobserwowane_konfiguracje:
            indeksy = tuple(
                [slice(None)]
                + [
                    cpd.state_names[rodzic].index(stan)
                    for rodzic, stan in zip(rodzice, konfig)
                ]
            )
            wpis_cpd = cpd.values[indeksy].copy()
            slownik_konfig = dict(zip(rodzice, konfig))
            nieobserwowane.append((slownik_konfig, wpis_cpd))

    return WynikAnalizyCPD(
        zmienna=zmienna,
        rodzice=rodzice,
        ksztalt_tablicy=ksztalt,
        liczba_parametrow=liczba_parametrow,
        liczba_konfiguracji_rodzicow=len(wszystkie_konfiguracje),
        liczba_nieobserwowanych_konfiguracji=len(nieobserwowane),
        nieobserwowane_przyklady=nieobserwowane[:max_przykladow],
    )


def znajdz_najwieksze_cpd(
    model: DiscreteBayesianNetwork, dane: ZbiorDanych
) -> list[WynikAnalizyCPD]:
    analizy = [analizuj_cpd(cpd, dane) for cpd in model.get_cpds()]
    max_rozmiar = max(a.liczba_parametrow for a in analizy)
    return [a for a in analizy if a.liczba_parametrow == max_rozmiar]