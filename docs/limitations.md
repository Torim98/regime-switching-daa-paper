# Limitations & Scope Boundaries

This document describes deliberate scope boundaries and design decisions of the implementation that should be considered when interpreting the results.

---

## 1. Tax Effects (Tax Modeling)

**Status:** Deliberately excluded (thesis Table 1: out of scope)

The backtesting simulation accounts for **transaction costs** (0.1% per rebalancing event) but does **not model tax effects** on realized capital gains.

### Rationale

1. **The relative comparison remains valid:** All four models (MSM, HMM, LSTM, Transformer) and the buy-and-hold benchmark operate under identical tax conditions. A capital gains tax would reduce the absolute returns of all strategies equally without changing the relative ranking of the risk-adjusted metrics (Sharpe ratio, Sortino ratio, Calmar ratio).

2. **Jurisdiction neutrality:** Tax regulations differ substantially between jurisdictions (e.g., the German flat-rate withholding tax of 26.375% incl. solidarity surcharge vs. US capital gains tax with short-/long-term differentiation). Modeling taxes would tie the results to a specific tax system and limit generalizability.

### Limitation: Differential Tax Burden Due to Signal Frequency

Models with **higher signal frequency** (more regime switches) generate more taxable rebalancing events than models with stable signals:

| Model | Signal stability | Tax exposure |
|--------|-----------------|--------------------------|
| HMM | High frequency | Elevated |
| HMM_Uni | High frequency | Elevated |
| MSM | Moderate stability | Moderate |
| LSTM | Tends toward higher frequency | Elevated |
| Transformer | Variable | Variable |

If a capital gains tax were modeled, models with many short regime switches would be taxed relatively more heavily than stable models. This could further amplify the transaction cost disadvantages of frequent switchers already visible in the evaluation.

### Possible Extensions

- German flat-rate withholding tax (26.375% incl. solidarity surcharge) on realized gains per rebalancing
- US short-term vs. long-term capital gains differentiation (holding period < / > 1 year)
- Tax-optimized rebalancing strategies (tax-loss harvesting)

---

## 2. Two-Regime Assumption

**Status:** Deliberate design decision

All models operate with exactly **two regimes** (bull/bear):
- MSM: `k_regimes: 2`
- HMM: `n_components: 2`
- HMM_Uni: `n_components: 2`
- LSTM / Transformer: binary classification (sigmoid output)

### Rationale

The two-regime assumption follows the dominant literature on Markov-switching models and enables a clear, interpretable trading rule (risk-on vs. risk-off). It forms the basis for the binary allocation strategy (100% equity vs. 100% safe haven).

### Limitation

Financial markets can exhibit more than two states (e.g., bull, bear, sideways/recovery). A higher number of regimes (k=3+) could enable more granular allocation levels (e.g., 100% / 60% / 0% equity) but would increase model complexity and the risk of overfitting. In addition, direct comparability between econometric and DL models would be more difficult, since the label assignment for k>2 is no longer trivial.

---

## 3. Data Source: Yahoo Finance

**Status:** Deliberate choice with known trade-offs

All market data is obtained via the `yfinance` library (Yahoo Finance API).

### Rationale

Yahoo Finance provides free access to adjusted historical price data with sufficient coverage of the study period (from 1990). For an academic study focused on method comparison (not live trading), the data quality is sufficient.

### Limitation

- **No guaranteed API stability:** Yahoo Finance offers no official API; `yfinance` uses unofficial endpoints that may change.
- **Survivorship bias:** The indices used (S&P 500, VUSTX) are subject to survivorship bias, since only surviving companies/funds are included.
- **Adjusted prices:** Yahoo's adjustment for splits and dividends is not always transparently documented.
- **No intraday data:** The pipeline works with daily closing prices. Intraday regime switches are not captured.

For production applications, institutional data providers (Bloomberg, Refinitiv) would be preferable.

### Automated Quality Assurance

Since Issue #2, the ingestion generates a data quality report on every run
(`assets/data_quality_report.md`): coverage against expected trading days, missing-value counts
on the raw data, adjustment plausibility (daily jumps against known crisis days), and the
row loss from cleaning. This report is intended to expose potential weaknesses in the data basis.

---

## 4. Asset Universe

**Status:** Deliberately restricted

The portfolio consists of two US asset classes:
- **Risk asset:** 60% S&P 500 (`^GSPC`) + 40% US long-term bonds (`VUSTX`)
- **Safe haven:** 3-month Treasury bill rate (`^IRX`) as cash proxy

### Rationale

The classic 60/40 portfolio serves as an established benchmark in the finance literature and enables a focused analysis of the regime-switching effect without interference from multi-asset allocation decisions.

### Limitation

- **US market only:** Results are not directly transferable to other markets (Europe, emerging markets), since regime dynamics can vary regionally.
- **Only 2 asset classes:** Diversification effects from commodities, real estate, international bonds, or cryptocurrencies are not represented.
- **Correlation assumption:** The 2022 period (simultaneous decline of equities and bonds) shows that the historically negative correlation need not hold permanently.

---

## 5. Walk-Forward Configuration

**Status:** Deliberate parameter choice with trade-offs

The walk-forward validation uses the following configuration:
- **Mode:** rolling window (not expanding)
- **Training window:** 10 years (`train_window_years: 10`)
- **Test window:** 12 months (`test_window_months: 12`)
- **Step size:** 12 months (`step_months: 12`, non-overlapping)

### Rationale

The 10y/12m/12m configuration balances sufficient training data for stable model estimation against enough OOS folds for a robust evaluation. Non-overlapping folds avoid autocorrelation between test periods.

### Limitation

- **Rolling vs. expanding:** An expanding window would give the later folds in particular more training data, but could overweight older, less relevant market regimes.
- **DL models with 10 years of training:** LSTM and Transformer typically benefit from larger datasets. Longer training windows could improve DL performance but would reduce the number of available folds.
- **12-month folds:** Shorter folds (e.g., 6 months) would provide more data points for the evaluation but increase computation time and the risk of unstable estimates for the econometric models.

### OOS Bear Coverage (Issue #8)

To quantify how much crisis exposure the OOS test windows actually carry, the diagnostic in `src/backtest/bear_coverage.py` labels the full price history with the Pagan-Sossounov scheme and measures, per fold, the bear-day share, the number of overlapping bear phases, and whether a complete bear phase (peak and trough) falls inside the window. The full table is in [`assets/bear_coverage.md`](../assets/bear_coverage.md) and is reproducible via `python -m src.backtest.bear_coverage`.

The result confirms the concern about fold granularity: across the 26 walk-forward folds, only 2 OOS test windows contain a complete Pagan-Sossounov bear phase (the 2022 rate shock and the partial final fold), 6 folds overlap at least one bear phase, and 20 folds carry no bear day at all. The major crises (Dotcom, GFC) span multiple consecutive 12-month folds and therefore enter the OOS windows as partial, window-truncated phases rather than as fully contained episodes. Every 10-year training window, by contrast, spans several complete bear phases, so model estimation is not starved of crisis data; the sparsity is specific to the OOS side.

This concentration of crisis signal in a few folds was the original motivation for a fold-wise aggregation critique. That critique is mitigated by the pooled-OOS objective introduced in Issue #5: the hyperparameter search no longer optimizes a fold-median (which the many bullish folds would dominate) but a metric computed on the OOS return series pooled across all folds. Crisis periods therefore enter the optimization signal in proportion to their length instead of being averaged away as one fold among many.

An expanding-window re-run remains open as a robustness variant (future work). It would give the later folds more training history and is a natural complement to this diagnostic, but it is not expected to change the OOS bear-coverage picture, which is governed by the 12-month fold length rather than by the training scheme.
