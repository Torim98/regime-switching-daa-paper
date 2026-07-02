"""Parallele Fold-Ausfuehrung fuer CPU-bound Modelle (MSM, HMM)."""
from joblib import Parallel, delayed
from services.warnings_config import configure_warnings
configure_warnings()

def _run_msm_fold(df, train_idx, test_idx, msm_cfg):
    from src.models.msm import train_msm_fold
    df_train = df.loc[train_idx]
    df_test  = df.loc[test_idx]
    try:
        probs, signal, signal_train = train_msm_fold(
            returns_train=df_train["Returns"],
            returns_test=df_test["Returns"],
            k_regimes=msm_cfg.k_regimes,
            switching_variance=msm_cfg.switching_variance,
            threshold=msm_cfg.threshold,
        )
        return {"ok": True, "test_idx": test_idx, "train_idx": train_idx,
                "probs": probs, "signal": signal, "signal_train": signal_train}
    except Exception as e:
        return {"ok": False, "test_idx": test_idx, "error": repr(e)}


def _run_hmm_fold(df, train_idx, test_idx, hmm_cfg):
    from src.models.hmm import train_hmm_fold
    df_train = df.loc[train_idx]
    df_test  = df.loc[test_idx]
    try:
        probs, signal, signal_train = train_hmm_fold(
            features_df_train=df_train[hmm_cfg.features],
            features_df_test=df_test[hmm_cfg.features],
            returns_train=df_train["Returns"],
            n_components=hmm_cfg.n_components,
            covariance_type=hmm_cfg.covariance_type,
            n_iter=hmm_cfg.n_iter,
            random_state=hmm_cfg.random_state,
            threshold=hmm_cfg.threshold,
        )
        return {"ok": True, "test_idx": test_idx, "train_idx": train_idx,
                "probs": probs, "signal": signal, "signal_train": signal_train}
    except Exception as e:
        return {"ok": False, "test_idx": test_idx, "error": repr(e)}


def run_folds_parallel(df, splits, msm_cfg=None, hmm_cfg=None, hmm_uni_cfg=None, n_jobs=-1):
    """MSM und HMM parallel ueber alle Folds (loky-Backend, windows-safe)."""
    results = {}
    if msm_cfg is not None:
        results["MSM"] = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
            delayed(_run_msm_fold)(df, tr, te, msm_cfg) for (tr, te) in splits
        )
    if hmm_cfg is not None:
        results["HMM"] = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
            delayed(_run_hmm_fold)(df, tr, te, hmm_cfg) for (tr, te) in splits
        )
    if hmm_uni_cfg is not None:
        results["HMM_Uni"] = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
            delayed(_run_hmm_fold)(df, tr, te, hmm_uni_cfg) for (tr, te) in splits
        )
    return results