import sys
from pathlib import Path

from src.analiza import pokaz_analize_d_separacji, pokaz_przyklady_wnioskowania
from src.ewaluacja import porownaj_modele_na_tescie
from src.parser import BladWalidacjiDanych, Parser
from src.siec import zbuduj_siec_mle, znajdz_najwieksze_cpd
from src.wnioskowanie import SilnikWnioskowania


def uruchom_prog_4_punkty(sciezka_zestawu: Path) -> None:
    print(f"=== ZESTAW: {sciezka_zestawu} ===")

    # --- ZADANIE 4.1 ---
    parser = Parser(sciezka_zestawu)
    zestaw = parser.wczytaj_zestaw()

    print("\n--- ZADANIE 4.1: Statystyki danych uczących (dane.csv) ---")
    print(f"Liczba wszystkich wierszy: {zestaw.dane.liczba_wierszy}")
    print(f"Liczba wierszy w pełni obserwowanych: {zestaw.dane.liczba_pelnych_wierszy}")
    print("Odsetek braków dla każdej zmiennej:")
    for zmienna, (braki, procent) in zestaw.dane.statystyki_brakow().items():
        print(f"  {zmienna}: {braki}/{zestaw.dane.liczba_wierszy} ({procent:.2f}%)")

    # --- ZADANIE 4.2 ---
    print("\n--- ZADANIE 4.2: Sieć ekspercka MLE i największe CPD ---")
    model = zbuduj_siec_mle(zestaw.opis, zestaw.dane)
    najwieksze = znajdz_najwieksze_cpd(model, zestaw.dane)
    for analiza in najwieksze:
        print(
            f"Największa tablica CPD: zmienna={analiza.zmienna}, rodzice={analiza.rodzice}, "
            f"kształt={analiza.ksztalt_tablicy}, liczba parametrów={analiza.liczba_parametrow}"
        )
        print(
            f"Nieobserwowane konfiguracje rodziców: "
            f"{analiza.liczba_nieobserwowanych_konfiguracji}/{analiza.liczba_konfiguracji_rodzicow}"
        )
        for konfig, rozklad in analiza.nieobserwowane_przyklady:
            print(f"  Przykład pustej konfiguracji {konfig} -> wpis w CPD: {rozklad}")

    # --- ZADANIE 4.3 ---
    print("\n--- ZADANIE 4.3: Odpowiedzi na zapytania (zapytania.txt) ---")
    silnik = SilnikWnioskowania(model, zestaw.opis)
    for zapytanie, wynik in silnik.odpowiedz_na_wszystkie(zestaw.zapytania):
        print(f"  {zapytanie}  =>  {wynik}")

    # --- ZADANIE 4.4 ---
    print("\n--- ZADANIE 4.4: Ewaluacja na test.csv i modele bazowe ---")
    wyniki_ewaluacji, _ = porownaj_modele_na_tescie(
        model, zestaw.opis, zestaw.dane, zestaw.test
    )
    for w in wyniki_ewaluacji:
        log_p_str = f"{w.sredni_log_p:.4f}" if w.sredni_log_p is not None else "nd."
        zera_str = str(w.liczba_zerowych_p) if w.liczba_zerowych_p is not None else "nd."
        print(
            f"  {w.nazwa_modelu:<42} | średni log P = {log_p_str:>8} | "
            f"wiersze P=0: {zera_str:>4} | trafność G = {w.trafnosc_g:.3f}"
        )

    pokaz_przyklady_wnioskowania(model, zestaw.opis, silnik)
    pokaz_analize_d_separacji(model, zestaw.opis, silnik)


if __name__ == "__main__":
    katalog = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("przyklad")
    try:
        uruchom_prog_4_punkty(katalog)
    except (BladWalidacjiDanych, FileNotFoundError) as blad:
        print(f"BŁĄD DANYCH: {blad}", file=sys.stderr)
        sys.exit(1)