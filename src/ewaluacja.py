from dataclasses import dataclass

import numpy as np
from pgmpy.models import DiscreteBayesianNetwork

from src.models import OpisZestawu, ZbiorDanych
from src.siec import zbuduj_siec_mle


@dataclass
class WynikEwaluacji:
    nazwa_modelu: str
    sredni_log_p: float | None
    liczba_zerowych_p: int | None
    trafnosc_g: float
    liczba_wierszy_testowych: int


def oblicz_prawdopodobienstwa_wierszy(
    model: DiscreteBayesianNetwork, dane_testowe: ZbiorDanych
) -> np.ndarray:
    df = dane_testowe.df
    n = len(df)
    p_laczne = np.ones(n, dtype=float)

    for cpd in model.get_cpds():
        zmienne_rodziny = list(cpd.variables)
        indeksy_osi = []
        for zmienna in zmienne_rodziny:
            mapa_stanow = {
                stan: idx for idx, stan in enumerate(cpd.state_names[zmienna])
            }
            indeksy = df[zmienna].map(mapa_stanow).to_numpy(dtype=int)
            indeksy_osi.append(indeksy)

        p_rodziny = cpd.values[tuple(indeksy_osi)]
        p_laczne *= p_rodziny

    return p_laczne


def przewiduj_g(
    model: DiscreteBayesianNetwork,
    opis: OpisZestawu,
    dane_testowe: ZbiorDanych,
    zmienna_celu: str = "G",
) -> np.ndarray:
    stany_g = opis.states[zmienna_celu]
    n = len(dane_testowe.df)
    p_dla_stanow = np.zeros((len(stany_g), n), dtype=float)

    df_kopia = dane_testowe.df.copy()
    for idx, stan in enumerate(stany_g):
        df_kopia[zmienna_celu] = stan
        tymczasowy_zbior = ZbiorDanych(
            df=df_kopia,
            nazwa_pliku=dane_testowe.nazwa_pliku,
            czy_testowy=True,
        )
        p_dla_stanow[idx] = oblicz_prawdopodobienstwa_wierszy(model, tymczasowy_zbior)

    najlepsze_indeksy = np.argmax(p_dla_stanow, axis=0)
    return np.array([stany_g[i] for i in najlepsze_indeksy], dtype=object)


def ocen_siec_bayesowska(
    nazwa_modelu: str,
    model: DiscreteBayesianNetwork,
    opis: OpisZestawu,
    dane_testowe: ZbiorDanych,
    zmienna_celu: str = "G",
) -> WynikEwaluacji:
    p_wierszy = oblicz_prawdopodobienstwa_wierszy(model, dane_testowe)
    liczba_zer = int(np.sum(p_wierszy == 0.0))

    p_obciete = np.maximum(p_wierszy, 1e-10)
    sredni_log_p = float(np.mean(np.log(p_obciete)))

    predykcje_g = przewiduj_g(model, opis, dane_testowe, zmienna_celu=zmienna_celu)
    rzeczywiste_g = dane_testowe.df[zmienna_celu].to_numpy(dtype=object)
    trafnosc = float(np.mean(predykcje_g == rzeczywiste_g))

    return WynikEwaluacji(
        nazwa_modelu=nazwa_modelu,
        sredni_log_p=sredni_log_p,
        liczba_zerowych_p=liczba_zer,
        trafnosc_g=trafnosc,
        liczba_wierszy_testowych=len(dane_testowe.df),
    )


def ocen_klasyfikator_wiekszosciowy(
    opis: OpisZestawu,
    dane_uczace: ZbiorDanych,
    dane_testowe: ZbiorDanych,
    zmienna_celu: str = "G",
) -> tuple[WynikEwaluacji, str]:
    obserwowane_g = dane_uczace.df[zmienna_celu].dropna()
    zliczenia = obserwowane_g.value_counts()

    stany_g = opis.states[zmienna_celu]
    najczestszy_stan = max(
        stany_g,
        key=lambda s: (int(zliczenia.get(s, 0)), -stany_g.index(s)),
    )

    rzeczywiste_g = dane_testowe.df[zmienna_celu].to_numpy(dtype=object)
    trafnosc = float(np.mean(rzeczywiste_g == najczestszy_stan))

    wynik = WynikEwaluacji(
        nazwa_modelu=f"Klasyfikator większościowy (G={najczestszy_stan})",
        sredni_log_p=None,
        liczba_zerowych_p=None,
        trafnosc_g=trafnosc,
        liczba_wierszy_testowych=len(dane_testowe.df),
    )
    return wynik, najczestszy_stan


def porownaj_modele_na_tescie(
    model_ekspercki: DiscreteBayesianNetwork,
    opis: OpisZestawu,
    dane_uczace: ZbiorDanych,
    dane_testowe: ZbiorDanych,
) -> tuple[list[WynikEwaluacji], str]:
    wynik_ekspercki = ocen_siec_bayesowska(
        "Graf ekspercki (MLE)", model_ekspercki, opis, dane_testowe
    )

    model_pusty = zbuduj_siec_mle(opis, dane_uczace, krawedzie=[])
    wynik_pusty = ocen_siec_bayesowska(
        "Model bez krawędzi (MLE)", model_pusty, opis, dane_testowe
    )

    wynik_wiekszosciowy, najczestsze_g = ocen_klasyfikator_wiekszosciowy(
        opis, dane_uczace, dane_testowe
    )

    return [wynik_ekspercki, wynik_pusty, wynik_wiekszosciowy], najczestsze_g