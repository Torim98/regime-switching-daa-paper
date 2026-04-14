import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_msm_regimes(df, model_name: str, color: str, save_path: str):
    """MSM: Bärenmarkt-Wahrscheinlichkeiten + Handelssignale."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

    ax1.plot(df.index, df[f'{model_name}_Prob'], label='MSM', alpha=0.7, color=color)
    ax1.set_title("MS Bärenmarkt-Wahrscheinlichkeiten (Out-of-Sample)")
    ax1.legend()

    ax2.fill_between(df.index, 0, df[f'{model_name}_Signal'], alpha=0.3, label='Signal', color=color)
    ax2.set_title("MS Handelssignale (Out-of-Sample, 1 = Bear/Cash, 0 = Bull/Investiert)")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_hmm_regimes(df, model_name: str, color: str, save_path: str):
    """HMM: Regime-Übersicht."""
    fig, ax = plt.subplots(figsize=(15, 3))
    ax.fill_between(df.index, 0, 1, where=(df[f'{model_name}_Signal'] == 1),
                    color=color, alpha=0.3, label='Bear Market (HMM)')
    ax.plot(df.index, df[f'{model_name}_Prob'], color='black', alpha=0.2, label='Bear Prob')
    ax.set_title(f"{model_name} Regimes (Out-of-Sample)")
    ax.legend()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

# Mapping: label_source -> Anzeigename (Modul-Ebene, einmal definieren)
_LABEL_DISPLAY_NAMES = {
    "pagan_sossounov":  "Pagan-Sossounov",
    "peak_to_trough":   "Peak-to-Trough (20%)",
    "lunde_timmermann": "Lunde-Timmermann",
    "nber":             "NBER",
    "hmm":              "HMM",
}

def plot_dl_model(
    test_df,
    model_name: str,
    color: str,
    save_path: str,
    cfg=None,
    ground_truth_col: str = "Supervised_Label",
):
    """LSTM / Transformer: Wahrscheinlichkeit + Signal vs. Ground Truth.

    Args:
        test_df: DataFrame mit {model_name}_Prob, {model_name}_Signal und
                 der Ground-Truth-Spalte (default: 'Supervised_Label').
        model_name: 'LSTM' oder 'Transformer'.
        color: Farbe für Prob/Signal-Kurven.
        save_path: Ziel-PNG-Pfad.
        cfg: PipelineConfig — für dynamischen Legendentext. Optional.
        ground_truth_col: Spaltenname der Ground-Truth-Reihe.
    """
    # Legendentext für Ground Truth dynamisch
    if cfg is not None:
        source = getattr(cfg.labels, "supervised_label_source", "pagan_sossounov")
        gt_name = _LABEL_DISPLAY_NAMES.get(source, source)
    else:
        gt_name = "Supervised"
    gt_legend = f"{gt_name} Ground Truth"

    # Fallback: falls ground_truth_col fehlt, alte Spalte versuchen
    if ground_truth_col not in test_df.columns:
        for alt in ("Supervised_Label", "MSM_Signal", "HMM_Signal"):
            if alt in test_df.columns:
                ground_truth_col = alt
                break

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

    ax1.plot(
        test_df.index, test_df[f"{model_name}_Prob"],
        color=color, label=f"{model_name} Bear Probability",
    )
    ax1.axhline(y=0.5, color="red", linestyle="--", label="Threshold 0.5")
    ax1.set_title(f"{model_name}: Wahrscheinlichkeit für Bärenmarkt (Out-of-Sample)")
    ax1.legend()

    ax2.plot(
        test_df.index, test_df[ground_truth_col],
        label=gt_legend, alpha=0.3, color="gray",
    )
    ax2.step(
        test_df.index, test_df[f"{model_name}_Signal"], where="post",
        label=f"{model_name} Signal", color=color,
    )
    ax2.set_title(f"{model_name} Handelssignale (Out-of-Sample, 0=Bull, 1=Bear)")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_regime_comparison(test_df, color_map: dict, save_path: str):
    """Dynamischer Vergleich aller Modelle (Probs + Signale)."""
    model_names = [col.rsplit('_', 1)[0] for col in test_df.columns if col.endswith('_Signal')]
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    for i, model in enumerate(model_names):
        color = color_map.get(model, default_colors[i % len(default_colors)])

        prob_col = f"{model}_Prob"
        if prob_col in test_df.columns:
            ax1.plot(test_df.index, test_df[prob_col],
                     label=f"{model.replace('_', ' ')} Prob",
                     color=color, alpha=0.5, linewidth=1.5)

        sig_col = f"{model}_Signal"
        ax2.step(test_df.index, test_df[sig_col], where='post',
                 label=f"{model.replace('_', ' ')} Signal",
                 color=color, alpha=0.5, linewidth=1.5)

    ax1.axhline(y=0.5, color='black', linestyle=':', alpha=0.5, label='Schwelle 0.5')
    ax1.set_title("Dynamischer Vergleich der Regime-Wahrscheinlichkeiten (Test-Set)")
    ax1.set_ylabel("Wahrscheinlichkeit (Bear)")
    ax1.legend(loc='upper left', fontsize='small', ncol=2)
    ax1.grid(alpha=0.2)
    ax1.set_ylim(-0.05, 1.05)

    ax2.set_title("Dynamischer Vergleich der Handelssignale (0 = Bull, 1 = Bear)")
    ax2.set_ylabel("Signal-Status")
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Bulle (0)', 'Bär (1)'])
    ax2.legend(loc='upper left', fontsize='small', ncol=2)
    ax2.grid(alpha=0.2)
    ax2.set_ylim(-0.1, 1.1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)