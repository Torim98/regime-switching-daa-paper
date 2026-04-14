"""Unit-Tests fuer Label-Methoden. Aufruf: pytest src/data/labels/test_labels.py"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.labels.pagan_sossounov import label_pagan_sossounov
from src.data.labels.peak_to_trough import label_peak_to_trough
from src.data.labels.lunde_timmermann import label_lunde_timmermann


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sp500_prices() -> pd.Series:
    """Laedt S&P-500-Close-Preise 1995-2024 via yfinance (wird gecached)."""
    import yfinance as yf
    data = yf.download("^GSPC", start="1995-01-01", end="2024-12-31",
                       auto_adjust=True, progress=False)
    prices = data["Close"].squeeze()
    prices.name = "Close"
    return prices


# ---------------------------------------------------------------------------
# Pagan-Sossounov
# ---------------------------------------------------------------------------

class TestPaganSossounov:
    def test_no_nan(self, sp500_prices):
        labels = label_pagan_sossounov(sp500_prices)
        assert labels.notna().all()

    def test_is_binary(self, sp500_prices):
        labels = label_pagan_sossounov(sp500_prices)
        assert set(labels.unique()).issubset({0, 1})

    def test_identifies_known_bear_markets(self, sp500_prices):
        labels = label_pagan_sossounov(sp500_prices)
        # Bekannte Baerenmaerkte: mind. ein markanter Tag pro Phase muss Bear sein
        known_bear_dates = ["2001-09-17", "2008-11-20", "2020-03-23", "2022-06-16"]
        for d in known_bear_dates:
            ts = pd.Timestamp(d)
            nearest = labels.index[labels.index.get_indexer([ts], method="nearest")[0]]
            assert labels.loc[nearest] == 1, f"{d} (naechster: {nearest}) nicht als Bear erkannt"

    def test_deterministic(self, sp500_prices):
        a = label_pagan_sossounov(sp500_prices)
        b = label_pagan_sossounov(sp500_prices)
        pd.testing.assert_series_equal(a, b)

    def test_rejects_nan_input(self):
        s = pd.Series([100.0, np.nan, 101.0],
                      index=pd.date_range("2020-01-01", periods=3))
        with pytest.raises(ValueError):
            label_pagan_sossounov(s)


# ---------------------------------------------------------------------------
# Peak-to-Trough
# ---------------------------------------------------------------------------

class TestPeakToTrough:
    def test_no_nan(self, sp500_prices):
        labels = label_peak_to_trough(sp500_prices)
        assert labels.notna().all()

    def test_covid_crash_detected(self, sp500_prices):
        labels = label_peak_to_trough(sp500_prices, threshold=0.20)
        mar_2020 = pd.Timestamp("2020-03-23")
        nearest = labels.index[labels.index.get_indexer([mar_2020], method="nearest")[0]]
        assert labels.loc[nearest] == 1

    def test_deterministic(self, sp500_prices):
        a = label_peak_to_trough(sp500_prices)
        b = label_peak_to_trough(sp500_prices)
        pd.testing.assert_series_equal(a, b)


# ---------------------------------------------------------------------------
# Lunde-Timmermann
# ---------------------------------------------------------------------------

class TestLundeTimmermann:
    def test_no_nan(self, sp500_prices):
        labels = label_lunde_timmermann(sp500_prices)
        assert labels.notna().all()

    def test_gfc_detected(self, sp500_prices):
        labels = label_lunde_timmermann(sp500_prices)
        gfc = pd.Timestamp("2008-11-20")
        nearest = labels.index[labels.index.get_indexer([gfc], method="nearest")[0]]
        assert labels.loc[nearest] == 1

    def test_deterministic(self, sp500_prices):
        a = label_lunde_timmermann(sp500_prices)
        b = label_lunde_timmermann(sp500_prices)
        pd.testing.assert_series_equal(a, b)