from itertools import combinations, permutations

import numpy as np
from pgmpy.models import DiscreteBayesianNetwork

from src.models import OpisZestawu
from src.wnioskowanie import SilnikWnioskowania


def _tv(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(p - q)))


def pokaz_przyklady_wnioskowania(
    model: DiscreteBayesianNetwork, opis: OpisZestawu, silnik: SilnikWnioskowania
) -> None:
    krawedzie = list(model.edges())
    if not krawedzie:
        print("Brak krawędzi w grafie — nie można zilustrować rozumowania.")
        return

    najlepsza_para = None
    najlepszy_wynik = -1.0

    for przyczyna, skutek in krawedzie:
        prior_skutek = silnik.wykonaj_p([skutek]).tablica
        prior_przyczyna = silnik.wykonaj_p([przyczyna]).tablica

        for s_p in opis.states[przyczyna]:
            post_skutek = silnik.wykonaj_p([skutek], {przyczyna: s_p}).tablica
            for s_s in opis.states[skutek]:
                post_przyczyna = silnik.wykonaj_p([przyczyna], {skutek: s_s}).tablica
                sila = _tv(prior_skutek, post_skutek) + _tv(
                    prior_przyczyna, post_przyczyna
                )
                if sila > najlepszy_wynik:
                    najlepszy_wynik = sila
                    najlepsza_para = (przyczyna, skutek, s_p, s_s)

    if najlepsza_para is not None:
        przyczyna, skutek, s_p, s_s = najlepsza_para
        print(f"\n[1] Rozumowanie przyczynowe ({przyczyna} -> {skutek}):")
        print(f"  Bez obserwacji:      P({skutek}) = {silnik.wykonaj_p([skutek])}")
        print(
            f"  Po obserwacji {przyczyna}={s_p}: P({skutek} | {przyczyna}=={s_p}) = "
            f"{silnik.wykonaj_p([skutek], {przyczyna: s_p})}"
        )

        print(f"\n[2] Rozumowanie diagnostyczne ({przyczyna} <- {skutek}):")
        print(
            f"  Bez obserwacji:      P({przyczyna}) = {silnik.wykonaj_p([przyczyna])}"
        )
        print(
            f"  Po obserwacji {skutek}={s_s}: P({przyczyna} | {skutek}=={s_s}) = "
            f"{silnik.wykonaj_p([przyczyna], {skutek: s_s})}"
        )

    zbior_krawedzi = set(krawedzie)
    najlepsze_ea = None
    najlepszy_skok_ea = -1.0

    for z in model.nodes():
        rodzice = list(model.predecessors(z))
        for a, b in combinations(rodzice, 2):
            if (a, b) in zbior_krawedzi or (b, a) in zbior_krawedzi:
                continue
            prior_a = silnik.wykonaj_p([a]).tablica
            for s_z in opis.states[z]:
                p_a_po_z = silnik.wykonaj_p([a], {z: s_z}).tablica
                for s_b in opis.states[b]:
                    p_a_po_zb = silnik.wykonaj_p([a], {z: s_z, b: s_b}).tablica
                    skok = _tv(prior_a, p_a_po_z) + _tv(p_a_po_z, p_a_po_zb)
                    if skok > najlepszy_skok_ea and _tv(p_a_po_z, p_a_po_zb) > 1e-4:
                        najlepszy_skok_ea = skok
                        najlepsze_ea = (a, b, z, s_z, s_b)

    if najlepsze_ea is not None:
        a, b, z, s_z, s_b = najlepsze_ea
        print(f"\n[3] Explaining away (v-struktura: {a} -> {z} <- {b}):")
        print(f"  Krok 1 (przed obserwacją):   P({a}) = {silnik.wykonaj_p([a])}")
        print(
            f"  Krok 2 (skutek {z}={s_z}):      P({a} | {z}=={s_z}) = "
            f"{silnik.wykonaj_p([a], {z: s_z})}"
        )
        print(
            f"  Krok 3 (+ przyczyna {b}={s_b}): P({a} | {z}=={s_z}, {b}=={s_b}) = "
            f"{silnik.wykonaj_p([a], {z: s_z, b: s_b})}"
        )


def pokaz_analize_d_separacji(
    model: DiscreteBayesianNetwork, opis: OpisZestawu, silnik: SilnikWnioskowania
) -> None:
    zmienne = list(opis.states.keys())
    trojki_oddzielone: list[tuple[str, str, list[str]]] = []
    trojki_polaczone: list[tuple[str, str, list[str]]] = []

    for x, y, z in permutations(zmienne, 3):
        if x > y:
            continue
        if not model.is_dconnected(x, y, observed=[z]):
            trojki_oddzielone.append((x, y, [z]))
        else:
            trojki_polaczone.append((x, y, [z]))

    print("\nAnaliza d-separacji (is_dconnected vs rozkłady liczbowe):")
    wybrane = trojki_oddzielone[:3] + trojki_polaczone[:2]

    for x, y, warunek in wybrane:
        z = warunek[0]
        polaczone = model.is_dconnected(x, y, observed=warunek)
        stan_z = opis.states[z][0]
        stan_x1 = opis.states[x][0]
        stan_x2 = opis.states[x][1]

        r1 = silnik.wykonaj_p([y], {z: stan_z, x: stan_x1})
        r2 = silnik.wykonaj_p([y], {z: stan_z, x: stan_x2})
        max_roznica = float(np.max(np.abs(r1.tablica - r2.tablica)))

        status = "d-POŁĄCZONE" if polaczone else "d-ODDZIELONE"
        print(
            f"  Trójka ({x}, {y} | {z}): is_dconnected={polaczone} [{status}] "
            f"| max|P({y}|{z}={stan_z},{x}={stan_x1}) - P({y}|{z}={stan_z},{x}={stan_x2})| = {max_roznica:.6f}"
        )
        print(f"    P({y} | {z}=={stan_z}, {x}=={stan_x1}) -> {r1}")
        print(f"    P({y} | {z}=={stan_z}, {x}=={stan_x2}) -> {r2}")