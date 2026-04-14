"""
Pagan-Sossounov (2003) Bull/Bear-Market-Labeling.

Referenz
--------
Pagan, A. R. & Sossounov, K. A. (2003). "A Simple Framework for Analysing
Bull and Bear Markets." Journal of Applied Econometrics, 18(1), 23-46.
DOI: 10.1002/jae.664

Algorithmus (Adaption des Bry-Boschan-Verfahrens für Aktienmärkte)
-----------------------------------------------------------------
1. Lokale Extrema in rollierendem Fenster ±`window_months` identifizieren.
2. Alternierung erzwingen (Peak -> Trough -> Peak ...).
3. Dauer-Filter: jede Phase muss >= `min_phase_months` dauern.
4. Zyklus-Filter: Peak->Peak- bzw. Trough->Trough-Abstand >= `min_cycle_months`.
5. Amplitude-Filter: |log-Return| zwischen benachbarten Extrema >= `amplitude_threshold`.
6. Bear = 1 zwischen Peak und nachfolgendem Trough, sonst Bull = 0.

Determinismus
-------------
Gleicher Preis-Input -> gleiches Label. Keine Random-Seeds, keine globalen States.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Handelstage pro Monat (Näherung, US-Markt)
TRADING_DAYS_PER_MONTH = 21


def label_pagan_sossounov(
    prices: pd.Series,
    window_months: int = 8,
    min_phase_months: int = 4,
    min_cycle_months: int = 16,
    amplitude_threshold: float = 0.20,
) -> pd.Series:
    """
    Erzeugt binäres Regime-Label (0 = Bull, 1 = Bear) nach Pagan & Sossounov (2003).

    Parameter
    ---------
    prices : pd.Series
        Close-Preise mit monoton steigendem DatetimeIndex. Keine NaN zulässig.
    window_months : int, default 8
        Fenster-Halbbreite (in Monaten) zur Identifikation lokaler Extrema.
    min_phase_months : int, default 4
        Minimale Dauer einer Bull- oder Bear-Phase (in Monaten).
    min_cycle_months : int, default 16
        Minimale Dauer eines Gesamtzyklus Peak->Peak bzw. Trough->Trough.
    amplitude_threshold : float, default 0.20
        Minimaler |log-Return| zwischen benachbarten Extrema (z.B. 0.20 = 20%).

    Rückgabe
    --------
    pd.Series
        int8-Serie (0/1) mit identischem DatetimeIndex wie `prices`.
        Name: "PagSoss_Signal".
    """
    if not isinstance(prices, pd.Series):
        raise TypeError("prices muss pd.Series sein.")
    if prices.isna().any():
        raise ValueError("prices enthält NaN-Werte.")
    if not prices.index.is_monotonic_increasing:
        raise ValueError("prices.index muss monoton steigend sein.")
    if len(prices) < 2 * window_months * TRADING_DAYS_PER_MONTH:
        raise ValueError(
            f"Zu wenige Preisbeobachtungen ({len(prices)}) für window_months={window_months}."
        )

    window_days = window_months * TRADING_DAYS_PER_MONTH
    min_phase_days = min_phase_months * TRADING_DAYS_PER_MONTH
    min_cycle_days = min_cycle_months * TRADING_DAYS_PER_MONTH

    # Schritt 1: lokale Extrema in ±window_days finden
    extrema = _find_local_extrema(prices, window_days)

    # Schritt 2: Alternierung erzwingen
    extrema = _enforce_alternation(extrema)

    # Schritt 3: Dauer-Filter (jede Phase >= min_phase_days)
    extrema = _apply_phase_filter(extrema, min_phase_days)
    extrema = _enforce_alternation(extrema)

    # Schritt 4: Zyklus-Filter (Peak-Peak bzw. Trough-Trough >= min_cycle_days)
    extrema = _apply_cycle_filter(extrema, min_cycle_days)
    extrema = _enforce_alternation(extrema)

    # Schritt 5: Amplitude-Filter (|log-Return| >= amplitude_threshold)
    extrema = _apply_amplitude_filter(extrema, amplitude_threshold)
    extrema = _enforce_alternation(extrema)

    # Schritt 6: binäre Label-Serie aus alternierenden Extrema bauen
    labels = _build_label_series(prices.index, extrema)

    return labels.rename("PagSoss_Signal").astype("int8")


# ---------------------------------------------------------------------------
# Interne Helper
# ---------------------------------------------------------------------------

def _find_local_extrema(prices: pd.Series, window_days: int) -> list[tuple]:
    """
    Identifiziert lokale Maxima und Minima in einem Fenster von ±window_days.

    Ein Punkt t ist lokales Maximum, wenn prices[t] = max(prices[t-w : t+w+1]).
    Analog für Minima.

    Rückgabe
    --------
    list[tuple[pd.Timestamp, float, str]]
        Sortierte Liste von (timestamp, price, type) mit type in {"P", "T"}.
    """
    values = prices.values
    index = prices.index
    n = len(values)
    extrema = []

    # Rolling Max/Min per zentriertem Fenster
    # (pandas' min_periods=1 stellt sicher, dass Randbereiche abgedeckt sind)
    series = prices
    roll_max = series.rolling(window=2 * window_days + 1, center=True, min_periods=1).max()
    roll_min = series.rolling(window=2 * window_days + 1, center=True, min_periods=1).min()

    for i in range(n):
        # Ausschluss der äußeren window_days/2 kann erwogen werden —
        # hier nehmen wir Randextrema mit (für Fold-Labels wichtig, da
        # aktuelle Marktphase oft am Rand liegt).
        if values[i] == roll_max.iloc[i]:
            extrema.append((index[i], values[i], "P"))
        elif values[i] == roll_min.iloc[i]:
            extrema.append((index[i], values[i], "T"))

    # Nach Zeit sortieren (dedupliziert, falls Preis == Max == Min in flachem Fenster)
    # Bei Ties "P" vor "T" bevorzugen — Alternierung regelt den Rest.
    extrema.sort(key=lambda x: (x[0], 0 if x[2] == "P" else 1))
    return extrema


def _enforce_alternation(extrema: list[tuple]) -> list[tuple]:
    """
    Erzwingt abwechselnde Peaks und Troughs.

    Bei zwei aufeinanderfolgenden Peaks: den niedrigeren entfernen.
    Bei zwei aufeinanderfolgenden Troughs: den höheren entfernen.
    """
    if len(extrema) < 2:
        return extrema

    cleaned: list[tuple] = []
    for ext in extrema:
        if not cleaned:
            cleaned.append(ext)
            continue

        prev = cleaned[-1]
        if ext[2] == prev[2]:
            # Gleicher Typ -> extremeren Wert behalten
            if ext[2] == "P":
                if ext[1] >= prev[1]:
                    cleaned[-1] = ext
            else:  # "T"
                if ext[1] <= prev[1]:
                    cleaned[-1] = ext
        else:
            cleaned.append(ext)

    return cleaned


def _apply_phase_filter(extrema: list[tuple], min_phase_days: int) -> list[tuple]:
    """
    Entfernt Extrema, die zu einer zu kurzen Phase (< min_phase_days) gehören.

    Strategie: Iteriere über benachbarte Extrema; liegt ihr Abstand unter
    dem Schwellwert, entferne den "weniger extremen" der beiden.
    """
    if len(extrema) < 2:
        return extrema

    changed = True
    while changed:
        changed = False
        for i in range(len(extrema) - 1):
            t0, _, _ = extrema[i]
            t1, _, _ = extrema[i + 1]
            duration = (t1 - t0).days
            if duration < min_phase_days:
                # Entferne den mittleren, weniger ausgeprägten Kandidaten
                # Heuristik: entferne das schwächere Extremum
                e0, e1 = extrema[i], extrema[i + 1]
                if e0[2] == "P" and e1[2] == "T":
                    # Peak gefolgt von Trough: beide entfernen wenn beide schwach,
                    # sonst den schwächeren. Pragmatisch: beide entfernen.
                    del extrema[i:i + 2]
                elif e0[2] == "T" and e1[2] == "P":
                    del extrema[i:i + 2]
                else:
                    # Gleicher Typ (sollte nach Alternierung nicht passieren)
                    del extrema[i + 1]
                changed = True
                break
    return extrema


def _apply_cycle_filter(extrema: list[tuple], min_cycle_days: int) -> list[tuple]:
    """
    Entfernt Extrema, die zu einem zu kurzen Zyklus führen.

    Ein Zyklus ist Peak->Peak oder Trough->Trough (Abstand 2 Indizes).
    """
    if len(extrema) < 3:
        return extrema

    changed = True
    while changed:
        changed = False
        for i in range(len(extrema) - 2):
            t0, _, type0 = extrema[i]
            t2, _, type2 = extrema[i + 2]
            if type0 != type2:
                continue  # nach Alternierung eigentlich nicht möglich
            cycle_days = (t2 - t0).days
            if cycle_days < min_cycle_days:
                # Entferne mittleres Extremum + eins der beiden Endpunkte
                # (das schwächere der gleichtypigen Endpunkte).
                _, p0, _ = extrema[i]
                _, p2, _ = extrema[i + 2]
                if type0 == "P":
                    weaker_idx = i if p0 <= p2 else i + 2
                else:  # "T"
                    weaker_idx = i if p0 >= p2 else i + 2
                middle_idx = i + 1
                for idx in sorted([weaker_idx, middle_idx], reverse=True):
                    del extrema[idx]
                changed = True
                break
    return extrema


def _apply_amplitude_filter(
    extrema: list[tuple],
    amplitude_threshold: float,
) -> list[tuple]:
    """
    Entfernt Extrema-Paare, deren |log-Return| unter dem Schwellwert liegt.
    """
    if len(extrema) < 2:
        return extrema

    changed = True
    while changed:
        changed = False
        for i in range(len(extrema) - 1):
            _, p0, _ = extrema[i]
            _, p1, _ = extrema[i + 1]
            amplitude = abs(np.log(p1 / p0))
            if amplitude < amplitude_threshold:
                # Beide Extrema entfernen -> umgebende Phase wird zusammengeführt
                del extrema[i:i + 2]
                changed = True
                break
    return extrema


def _build_label_series(
    index: pd.DatetimeIndex,
    extrema: list[tuple],
) -> pd.Series:
    """
    Konstruiert 0/1-Serie aus alternierenden Peaks/Troughs.

    - Vor dem ersten Extremum: initialisiere mit dem *gegensätzlichen* Zustand
      (vor einem Peak ist die Phase Bull, vor einem Trough Bear).
    - Zwischen Peak_t und folgendem Trough: 1 (Bear).
    - Zwischen Trough_t und folgendem Peak: 0 (Bull).
    - Nach dem letzten Extremum: letzter Zustand bleibt bestehen.
    """
    labels = pd.Series(0, index=index, dtype="int8")

    if not extrema:
        return labels

    # Vor erstem Extremum
    first_ts, _, first_type = extrema[0]
    if first_type == "P":
        labels.loc[:first_ts] = 0  # Phase vor Peak = Bull
    else:
        labels.loc[:first_ts] = 1  # Phase vor Trough = Bear

    # Zwischen Extrema
    for i in range(len(extrema) - 1):
        t0, _, type0 = extrema[i]
        t1, _, _ = extrema[i + 1]
        segment = (index > t0) & (index <= t1)
        labels.loc[segment] = 1 if type0 == "P" else 0

    # Nach letztem Extremum
    last_ts, _, last_type = extrema[-1]
    labels.loc[index > last_ts] = 1 if last_type == "P" else 0

    return labels