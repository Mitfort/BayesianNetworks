from pathlib import Path
from src.models import OpisZestawu, ZbiorDanych, Zapytanie, ZestawDanych
import json
import keyword
import re
import numpy as np
import pandas as pd

class BladWalidacjiDanych:
    def __init__(self, lokalizajca:str, komunikat: str):
        self.lokalizacja = lokalizajca
        self.komunikat = komunikat
        super().__init__(f"[{lokalizajca}] {komunikat}")




class Parser: 
    DOZWOLONE_POLA = {"meta", "variables","expert_edges","required_edges","forbidden_edges", "tiers"}
    SLOWA_ZAREZERWOWANE = {"P", "MAP", "atleast", "atmost"}
    WZORZEC_NAZWY = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


    def __init__(self, sciezka: Path | str):
        sciezka = Path(sciezka)

        if sciezka.is_file():
            self.katalog = sciezka.parent
            self.sciezka_opisu = sciezka
        else:
            self.katalog = sciezka
            self.sciezka_opisu = sciezka / "opis.json"



    def wczytaj_opis(self) -> OpisZestawu:
        """
        Wczytuje opis zestawu z pliku JSON i zwraca obiekt OpisZestawu.
        """
        if not self.sciezka_opisu.exists():
            raise FileNotFoundError(f"Nie znaleziono pliku: {self.sciezka_opisu}")
        try:
            with open(self.sciezka_opisu, "r", encoding="utf-8") as plik:
                json_data =json.load(plik)
        except json.JSONDecodeError as e:
            raise BladWalidacjiDanych(f"{self.sciezka_opisu.name}:{e.lineno}:{e.colno}",
                f"Niepoprawny format JSON: {e.msg}",
            ) from e

        self.waliduj_opis(json_data)
        return OpisZestawu(json_data)

    def waliduj_opis(self, json_data: object) -> None:
        """
        Waliduje opis zestawu.
        """
        if not isinstance(json_data, dict):
            raise BladWalidacjiDanych("$", "Korzeń dokumentu JSON musi być obiektem.")
        nieznane_pola = [klucz for klucz in json_data.keys() if klucz not in self.DOZWOLONE_POLA]
        if nieznane_pola:
            raise BladWalidacjiDanych("$", f"Niedozwolone pola na najwyższym poziomie: {nieznane_pola}")
        if "meta" not in json_data and not isinstance(json_data["meta"], dict):
            raise BladWalidacjiDanych("meta", "Pole 'meta' musi być obiektem JSON.")
        if "variables" not in json_data and not isinstance(json_data["variables"], dict):
            raise BladWalidacjiDanych("variables", "Brak wymaganego pola 'variables' lub nie jest ono obiektem.")

        variables = json_data["variables"]
        if not variables:
            raise BladWalidacjiDanych("variables", "Słownik 'variables' nie może być pusty.")

        for nazwa_zmiennej, spec in variables.items():
            sciezka_zmiennej = f"variables.{nazwa_zmiennej}"
            self._waliduj_identyfikator(nazwa_zmiennej, sciezka_zmiennej, "zmienniej")

            if not isinstance(spec, dict):
                raise BladWalidacjiDanych(sciezka_zmiennej, "Definicja zmiennej musi być obiektem JSON.")

            nieznane_pola_zmiennej = set(spec.keys()) - {"descriptiob", "states"}
            if nieznane_pola_zmiennej:
                raise BladWalidacjiDanych(sciezka_zmiennej,f"Nieznane pola w definicji zmiennej: {sorted(nieznane_pola_zmiennej)}")
            if "states" not in spec and not isinstance(spec["states"], list):
                raise BladWalidacjiDanych(f"{sciezka_zmiennej}.states", "Zmienna musi posiadać listę stanów 'states'.")
            stany = spec["states"]
            if len(stany) < 2:
                raise BladWalidacjiDanych(f"{sciezka_zmiennej}.states", f"Zmienna musi mieć co najmniej 2 stany (podano: {len(stany)}).")

            widziane_stany: set[str] = set()
            for idx, stan in enumerate(stany):
                sciezka_stanu = f"{sciezka_zmiennej}.states[{idx}]"
                if not isinstance(stan, str):
                    raise BladWalidacjiDanych(sciezka_stanu, "Nazwa stanu musi być napisem.")
                self._waliduj_identyfikator(stan, sciezka_stanu, "stanu")
                if stan in widziane_stany:
                    raise BladWalidacjiDanych(sciezka_stanu, f"Powtórzony stan '{stan}'.")
                widziane_stany.add(stan)

            dozwolone_zmienne = set(variables.keys())

            for pole_krawedzi in ("expert_edges", "required_edges", "forbidden_edges"):
                if pole_krawedzi in json_data:
                    self._waliduj_krawedzie(json_data[pole_krawedzi], pole_krawedzi, dozwolone_zmienne)

            if "expert_edges" in json_data:
                self._sprawdz_brak_cykli(json_data["expert_edges"], dozwolone_zmienne)
            if "tiers" in json_data:
                self._waliduj_warstwy(json_data["tiers"], dozwolone_zmienne)

    def _waliduj_identyfikator(self, nazwa: str, sciezka: str, rola: str) ->None:
        if not self.WZORZEC_NAZWY.match(nazwa):
            raise BladWalidacjiDanych(sciezka, f"Nazwa {rola} '{nazwa}' jest słowem kluczowym Pythona lub słowem zarezerwowanym.")

    def _waliduj_krawedzie(self, krawiedzie: object, pole: str, dozwolone_zmienne: set[str]) ->None:
        if not isinstance(krawiedzie, list):
            raise BladWalidacjiDanych(pole, f"Pole '{pole}' musi być listą krawędzi.")
        for idx, krawedz in enumerate(krawiedzie):
            sciezka= f"{pole}[{idx}]"
            if not isinstance(krawedz, list) or len(krawedz) != 2:
                raise BladWalidacjiDanych(sciezka, "Krawędź musi być dwuelementową listą [źródło, cel].")

            for pos, wierzcholek in enumerate(krawedz):
                if not isinstance(wierzcholek, str) or wierzcholek not in dozwolone_zmienne:
                    raise BladWalidacjiDanych(f"{sciezka}[{pos}]", f"Nieznana zmienna '{wierzcholek}' w definicji krawędzi.")

    def _sprawdz_brak_cykli(self, krawedzie: list[list[str]], dozwolone_zmienne: set[str]) -> None:
        graf: dict[str, list[str]] = {v: [] for v in dozwolone_zmienne}
        for u, v in krawedzie:
            graf[u].append(v)

        stan_odwiedzin: dict[str, list[str]] = {v : [] for v in dozwolone_zmienne}
        stos: list[str] = []

        def dfs(u:str) ->None:
            stan_odwiedzin[u] = 1
            stos.append(u)
            for v in graf[u]:
                if stan_odwiedzin[v] == 1:
                    poczatek_cyklu = stos.index(v)
                    cykl = " -> ".join(stos[poczatek_cyklu:] + [v])
                    raise BladWalidacjiDanych("expert_edges", f"Wykryto cykl w grafie eksperckim: {cykl}")
                if stan_odwiedzin[v] == 0:
                    dfs(v)
                stos.pop()
                stan_odwiedzin[u] = 2 # 0 = nieodwiedzony, 1 = w trakcie odwiedzania (na stosie), 2 = zakończony

        for zmienna in dozwolone_zmienne:
            if stan_odwiedzin[zmienna] == 0:
                dfs(zmienna)

    def _waliduj_warstwy(self, warstwy: object, dozwolone_zmienne: set[str]) -> None:
        if not isinstance(warstwy, list):
            raise BladWalidacjiDanych("tiers", "Pole 'tiers' musi być listą warstw.")

        uzyte_zmienne = set[str] = set() 
        for t_idx, warstwa in enumerate(warstwy):
            sciezka_warstwy = f"tiers[{t_idx}]"
            if not isinstance(warstwa, list):
                raise BladWalidacjiDanych(sciezka_warstwy, "Każda warstwa w 'tiers' musi być listą.")

            for v_idx, zmienna in enumerate(warstwa):
                sciezka_zmiennej = f"{sciezka_warstwy}[{"v_idx"}]"
                if not isinstance(zmienna, str) or zmienna not in dozwolone_zmienne:
                    raise BladWalidacjiDanych (sciezka_zmiennej, f"Nieznana zmienna '{zmienna}' w definicji warstw.")
                if zmienna in uzyte_zmienne:
                    raise BladWalidacjiDanych(sciezka_zmiennej,f"Zmienna '{zmienna}' występuje w 'tiers' więcej niż jeden raz.")
                uzyte_zmienne.add(zmienna)


    def wczytaj_dane(self, opis: OpisZestawu) -> ZbiorDanych:
        """Wczytuje i waliduje plik uczący dane.csv (dopuszcza braki wartości)."""
        return self._wczytaj_csv(self.katalog / "dane.csv", opis, czy_testowy=False)

    def wczytaj_test(self, opis: OpisZestawu) -> ZbiorDanych:
        """Wczytuje i waliduje plik testowy test.csv (brak wartości jest błędem)."""
        return self._wczytaj_csv(self.katalog / "test.csv", opis, czy_testowy=True)

    def _wczytaj_csv(
        self, sciezka: Path, opis: OpisZestawu, czy_testowy: bool
    ) -> ZbiorDanych:
        if not sciezka.exists():
            raise FileNotFoundError(f"Nie znaleziono pliku: {sciezka}")

        nazwa_pliku = sciezka.name
        dozwolone_stany = {
            zmienna: set(stany) for zmienna, stany in opis.states.items()
        }
        oczekiwane_zmienne = list(opis.variables.keys())

        with open(sciezka, "r", encoding="utf-8", newline="") as plik:
            reader = pd.csv.reader(plik)
            naglowek = next(reader, None)

            if naglowek is None:
                raise BladWalidacjiDanych(
                    f"{nazwa_pliku}:1", "Plik CSV jest pusty (brak wiersza nagłówka)."
                )

            kolumny = [k.strip() for k in naglowek]

            if len(set(kolumny)) != len(kolumny):
                raise BladWalidacjiDanych(
                    f"{nazwa_pliku}:1",
                    f"Zduplikowane nazwy kolumn w nagłówku: {kolumny}",
                )

            nieznane = [k for k in kolumny if k not in dozwolone_stany]
            if nieznane:
                raise BladWalidacjiDanych(
                    f"{nazwa_pliku}:1",
                    f"Nieznane kolumny w nagłówku CSV: {nieznane}",
                )

            brakujace = [k for k in oczekiwane_zmienne if k not in kolumny]
            if brakujace:
                raise BladWalidacjiDanych(
                    f"{nazwa_pliku}:1",
                    f"Brakujące kolumny w nagłówku CSV: {brakujace}",
                )

            liczba_kolumn = len(kolumny)
            rekordy: list[dict[str, object]] = []

            for nr_linii, wiersz in enumerate(reader, start=2):
                if len(wiersz) != liczba_kolumn:
                    raise BladWalidacjiDanych(
                        f"{nazwa_pliku}:{nr_linii}",
                        f"Zła liczba pól w wierszu: oczekiwano {liczba_kolumn}, otrzymano {len(wiersz)}.",
                    )

                rekord: dict[str, object] = {}
                for kolumna, surowa_wartosc in zip(kolumny, wiersz):
                    wartosc = surowa_wartosc.strip()
                    if wartosc == "":
                        if czy_testowy:
                            raise BladWalidacjiDanych(
                                f"{nazwa_pliku}:{nr_linii}",
                                f"Brak wartości dla zmiennej '{kolumna}' (w pliku testowym braki są niedozwolone).",
                            )
                        rekord[kolumna] = np.nan
                    else:
                        if wartosc not in dozwolone_stany[kolumna]:
                            raise BladWalidacjiDanych(
                                f"{nazwa_pliku}:{nr_linii}",
                                f"Nieznany stan '{wartosc}' dla zmiennej '{kolumna}' "
                                f"(dozwolone: {opis.states[kolumna]}).",
                            )
                        rekord[kolumna] = wartosc

                rekordy.append(rekord)

        if not rekordy:
            raise BladWalidacjiDanych(
                nazwa_pliku, "Plik CSV nie zawiera żadnych wierszy z danymi."
            )

        df = pd.DataFrame(rekordy, columns=oczekiwane_zmienne)
        return ZbiorDanych(df=df, nazwa_pliku=nazwa_pliku, czy_testowy=czy_testowy)

    WZORZEC_ZAPYTANIA = re.compile(r"^(P|MAP)\((.*)\)$")

    def wczytaj_zapytania(self, opis: OpisZestawu) -> list[Zapytanie]:
        sciezka = self.katalog / "zapytania.txt"
        if not sciezka.exists():
            raise FileNotFoundError(f"Nie znaleziono pliku: {sciezka}")

        nazwa_pliku = sciezka.name
        dozwolone_stany = {
            zmienna: set(stany) for zmienna, stany in opis.states.items()
        }
        zapytania: list[Zapytanie] = []

        with open(sciezka, "r", encoding="utf-8") as plik:
            for nr_linii, surowa_linia in enumerate(plik, start=1):
                bez_komentarza = surowa_linia.split("#", 1)[0].strip()
                if not bez_komentarza:
                    continue

                zapytanie = self._parsuj_linie_zapytania(
                    bez_komentarza, nr_linii, nazwa_pliku, dozwolone_stany, opis.states
                )
                zapytania.append(zapytanie)

        if not zapytania:
            raise BladWalidacjiDanych(
                nazwa_pliku, "Plik zapytania.txt nie zawiera żadnych zapytań."
            )

        return zapytania

    
    def _parsuj_linie_zapytania(
        self,
        linia: str,
        nr_linii: int,
        nazwa_pliku: str,
        dozwolone_stany: dict[str, set[str]],
        kolejnosc_stanow: dict[str, list[str]],
    ) -> Zapytanie:
        lokalizacja = f"{nazwa_pliku}:{nr_linii}"
        dopasowanie = self.WZORZEC_ZAPYTANIA.fullmatch(linia)
        if not dopasowanie:
            raise BladWalidacjiDanych(
                lokalizacja,
                f"Niepoprawna składnia zapytania '{linia}' (oczekiwano P(...) lub MAP(...)).",
            )

        typ, wnetrze = dopasowanie.groups()
        if wnetrze.count("|") > 1:
            raise BladWalidacjiDanych(
                lokalizacja, "Zapytanie może zawierać co najwyżej jeden znak '|'."
            )

        if "|" in wnetrze:
            czesc_celu, czesc_warunku = wnetrze.split("|", 1)
        else:
            czesc_celu, czesc_warunku = wnetrze, None

        surowe_cele = [c.strip() for c in czesc_celu.split(",")]
        if not surowe_cele or any(c == "" for c in surowe_cele):
            raise BladWalidacjiDanych(
                lokalizacja, "Brak lub pusta nazwa zmiennej celu w zapytaniu."
            )

        cele: list[str] = []
        for zmienna in surowe_cele:
            if zmienna not in dozwolone_stany:
                raise BladWalidacjiDanych(
                    lokalizacja, f"Zapytanie o nieznaną zmienną '{zmienna}'."
                )
            if zmienna in cele:
                raise BladWalidacjiDanych(
                    lokalizacja,
                    f"Zmienna '{zmienna}' występuje wielokrotnie jako cel zapytania.",
                )
            cele.append(zmienna)

        warunki: dict[str, str] = {}
        if czesc_warunku is not None:
            if czesc_warunku.strip() == "":
                raise BladWalidacjiDanych(
                    lokalizacja, "Pusty warunek po znaku '|'."
                )
            for fragment in czesc_warunku.split(","):
                fragment = fragment.strip()
                if "==" not in fragment:
                    raise BladWalidacjiDanych(
                        lokalizacja,
                        f"Niepoprawny warunek '{fragment}' (oczekiwano zapisu zmienna==stan).",
                    )
                czesci = fragment.split("==")
                if len(czesci) != 2:
                    raise BladWalidacjiDanych(
                        lokalizacja, f"Niepoprawna składnia warunku '{fragment}'."
                    )
                zmienna, stan = czesci[0].strip(), czesci[1].strip()
                if zmienna not in dozwolone_stany:
                    raise BladWalidacjiDanych(
                        lokalizacja,
                        f"Nieznana zmienna '{zmienna}' w warunku zapytania.",
                    )
                if zmienna in cele:
                    raise BladWalidacjiDanych(
                        lokalizacja,
                        f"Zmienna '{zmienna}' nie może być jednocześnie celem i warunkiem.",
                    )
                if zmienna in warunki:
                    raise BladWalidacjiDanych(
                        lokalizacja,
                        f"Zmienna '{zmienna}' występuje wielokrotnie w warunku.",
                    )
                if stan not in dozwolone_stany[zmienna]:
                    raise BladWalidacjiDanych(
                        lokalizacja,
                        f"Nieznany stan '{stan}' dla zmiennej '{zmienna}' "
                        f"(dozwolone: {kolejnosc_stanow[zmienna]}).",
                    )
                warunki[zmienna] = stan

        return Zapytanie(
            typ=typ,  
            cele=cele,
            warunki=warunki,
            numer_linii=nr_linii,
        )

    def wczytaj_zestaw(self) -> ZestawDanych:
        opis = self.wczytaj_opis()
        dane = self.wczytaj_dane(opis)
        test = self.wczytaj_test(opis)
        zapytania = self.wczytaj_zapytania(opis)
        return ZestawDanych(opis=opis, dane=dane, test=test, zapytania=zapytania)