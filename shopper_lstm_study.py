"""
Optimizing Online Shopper Purchasing Prediction with LSTM and Hyperparameter Tuning on Imbalanced Data
Dataset : UCI Online Shoppers Purchasing Intention (12,330 sessions, target = Revenue)

Usage (run one step at a time, share the console output of each step):
    python shopper_lstm_study.py --step 1
    ...
    python shopper_lstm_study.py --step 9
Optional:
    --data path/to/online_shoppers_intention.csv
    --n_trials 50      number of Optuna trials (step 6)
    --force            ignore cached runs of this step and recompute them
    --quick            tiny smoke test; writes to outputs_quick/ so real results are never touched

Resumability / traceability (every step can be interrupted and re-run with the SAME command):
    * every unit of work (baseline model, LSTM run, Optuna trial) is cached on disk the moment it finishes
      and is skipped on re-run; cache files are tied to config + seed + data signature, so a changed
      configuration is recomputed instead of silently reusing stale results
    * outputs/run_log.jsonl      one line per run/trial/step: config, seed, seconds, metrics, output files
    * outputs/environment.json   python/package versions, GPU, data hash, command line
    * outputs/console/           copy of the console output of every step
    * outputs/backups/           zip snapshot of all outputs after every completed step

STEP 1  Data audit (shape, dtypes, missing, duplicates, class balance)      -> Fig 1, Table 1
STEP 2  Exploratory & statistical analysis (feature vs target)             -> Fig 2, Fig 3, Table 2
STEP 3  Preprocessing + stratified train/val/test split (no leakage)       -> processed data
STEP 4  Baseline ML models (LR, RF, XGBoost) with CV tuning                -> predictions
STEP 5  Imbalance-handling comparison with a default LSTM                  -> Fig 4, Table 3
STEP 6  Hyperparameter tuning of LSTM (Optuna TPE + pruning)               -> Fig 5, best params
STEP 7  Final training: default vs tuned LSTM, 5 random seeds              -> Fig 6, predictions
STEP 8  Model comparison, ROC/PR curves, confusion matrix, paired bootstrap-> Fig 7, Fig 8, Table 4, Table 5
STEP 9  Permutation feature importance of tuned LSTM (all seeds)           -> Fig 9, Table 6
STEP 10 Sensitivity analysis: all models re-trained without PageValues -> Fig 10, Table 7

Requirements: python>=3.9, torch (CUDA build for RTX 3060), numpy, pandas, scikit-learn>=1.2,
              imbalanced-learn, optuna, xgboost, scipy, matplotlib, joblib
"""
import os, sys, io, json, glob, time, copy, shutil, hashlib, platform, argparse, warnings, zipfile
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------------------------
# GLOBAL SETTINGS
# ----------------------------------------------------------------------------------------------
SEED = 42
FINAL_SEEDS = [11, 22, 33, 44, 55]          # independent runs for the final LSTM models
SPACE_VERSION = "v2"                          # change if the Optuna search space / pruning is edited
MIN_EPOCHS = 30                               # early stopping / pruning cannot act before this epoch (slow-start plateau of the LSTM)
OUT = FIG = TAB = PRED = RUNS = MOD = BAK = CON = None
FORCE = False


def init_paths(quick):
    """Create the output tree. --quick uses a separate tree so smoke tests never mix with real runs."""
    global OUT, FIG, TAB, PRED, RUNS, MOD, BAK, CON
    OUT = "outputs_quick" if quick else "outputs"
    FIG, TAB, PRED = (os.path.join(OUT, x) for x in ("figures", "tables", "preds"))
    RUNS, MOD, BAK, CON = (os.path.join(OUT, x) for x in ("runs", "models", "backups", "console"))
    for d in (OUT, FIG, TAB, PRED, RUNS, MOD, BAK, CON):
        os.makedirs(d, exist_ok=True)


TARGET = "Revenue"
LOG_COLS = ["Administrative", "Administrative_Duration", "Informational", "Informational_Duration",
            "ProductRelated", "ProductRelated_Duration", "PageValues"]
NUM_COLS = LOG_COLS + ["BounceRates", "ExitRates", "SpecialDay"]
CAT_COLS = ["Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType"]
BIN_COLS = ["Weekend"]
STRATS = ["none", "class_weight", "ros", "smotenc"]
STRAT_LABEL = {"none": "None", "class_weight": "Class weight", "ros": "Random\noversampling",
               "smotenc": "SMOTE-NC"}
DEFAULT_CFG = dict(hidden=64, layers=1, bidir=False, dropout=0.2, lr=1e-3, wd=1e-5, batch=256)

# Okabe-Ito colour-blind-safe palette
C = dict(blue="#0072B2", orange="#E69F00", green="#009E73", red="#D55E00",
         sky="#56B4E9", pink="#CC79A7", gray="#7F7F7F", black="#000000")

# Figure style: manuscript widths (inches); figures are saved at exactly this size -> insert WITHOUT resizing.
W1, W2 = 85 / 25.4, 174 / 25.4          # single column (85 mm), full width (174 mm)
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.5, "axes.titlesize": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.fontsize": 7.5, "axes.linewidth": 0.7,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 100, "savefig.dpi": 600, "figure.constrained_layout.use": True,
    "pdf.fonttype": 42,
})


def save_fig(fig, name):
    """Saved at exact figure size (no bbox_inches='tight') and 600 dpi -> insert without resizing."""
    path = os.path.join(FIG, name + ".png")
    fig.savefig(path, dpi=600)
    w, h = fig.get_size_inches()
    plt.close(fig)
    print(f"[saved] {path}  ({w * 25.4:.0f} x {h * 25.4:.0f} mm, 600 dpi)")
    log_event(event="figure", file=path, width_mm=round(w * 25.4, 1), height_mm=round(h * 25.4, 1))


def panel_label(ax, s):
    ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontsize=9.5, fontweight="bold", va="bottom", ha="right")


def save_table(df, name, index=True):
    path = os.path.join(TAB, name + ".csv")
    df.to_csv(path, index=index)
    print(f"[saved] {path}")
    log_event(event="table", file=path)


# ----------------------------------------------------------------------------------------------
# TRACKING, CHECKPOINTING, BACKUP
# ----------------------------------------------------------------------------------------------
def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def log_event(**kw):
    kw = dict(time=datetime.now().isoformat(timespec="seconds"), **kw)
    with open(os.path.join(OUT, "run_log.jsonl"), "a") as f:
        f.write(json.dumps(kw, default=_default) + "\n")


def jdump(obj, path):
    """Atomic JSON write (an interruption can never leave a half-written file)."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=_default)
    os.replace(tmp, path)


def npsave(path, **arrays):
    tmp = path[:-4] + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def sig_of(*objs):
    """Short signature of config/seed/data: cached results are reused only if the signature matches."""
    return hashlib.md5(json.dumps(objs, sort_keys=True, default=_default).encode()).hexdigest()[:8]


def file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Tee(io.TextIOBase):
    """Copy everything printed to a per-step console log file."""

    def __init__(self, path):
        self.f, self.o = open(path, "a", buffering=1), sys.stdout      # works in terminal AND Jupyter

    def write(self, s):
        self.o.write(s); self.f.write(s)
        return len(s)

    def flush(self):
        self.o.flush(); self.f.flush()


def save_environment(args):
    env = dict(time=datetime.now().isoformat(timespec="seconds"), command=" ".join(sys.argv),
               python=sys.version.split()[0], platform=platform.platform(), seed=SEED, final_seeds=FINAL_SEEDS)
    for m in ("numpy", "pandas", "sklearn", "scipy", "matplotlib", "torch", "optuna", "imblearn", "xgboost"):
        try:
            env[m] = __import__(m).__version__
        except Exception:
            env[m] = "not installed"
    try:
        env["cuda_available"] = bool(torch.cuda.is_available())
        env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        env["cuda_version"] = torch.version.cuda
    except Exception:
        pass
    if os.path.exists(args.data):
        env["data_file"], env["data_md5"] = args.data, file_md5(args.data)
    jdump(env, os.path.join(OUT, "environment.json"))


def backup(step):
    """Zip snapshot of everything produced so far (backups folder excluded)."""
    path = os.path.join(BAK, f"after_step{step}_{datetime.now():%Y%m%d_%H%M%S}.zip")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(OUT):
            if root.startswith(BAK) or root.startswith(CON):
                continue
            for fn in files:
                if fn.endswith(".tmp") or ".tmp." in fn:
                    continue
                full = os.path.join(root, fn)
                z.write(full, os.path.relpath(full, OUT))
    print(f"[backup] {path}")


# ----------------------------------------------------------------------------------------------
# DATA HELPERS
# ----------------------------------------------------------------------------------------------
def load_raw(path):
    df = pd.read_csv(path)
    df[TARGET] = df[TARGET].astype(int)
    df["Weekend"] = df["Weekend"].astype(int)
    return df


def load_dedup(path):
    return load_raw(path).drop_duplicates().reset_index(drop=True)


def load_processed():
    z = np.load(os.path.join(OUT, "processed.npz"))
    d = {k: z[k] for k in z.files}
    d["sig"] = hashlib.md5(d["Xtr"].tobytes() + d["ytr"].tobytes() + d["Xva"].tobytes()).hexdigest()[:8]
    return d


# ----------------------------------------------------------------------------------------------
# METRICS
# ----------------------------------------------------------------------------------------------
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score, matthews_corrcoef, balanced_accuracy_score,
                             confusion_matrix, precision_recall_curve, roc_curve)


def best_threshold(y, p):
    """Threshold maximising F1 on the VALIDATION set (decision-threshold moving for imbalance)."""
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr[:-1] * rc[:-1] / np.clip(pr[:-1] + rc[:-1], 1e-12, None)
    return float(th[np.argmax(f1)])


def metrics(y, p, thr):
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return dict(Accuracy=accuracy_score(y, yh), Precision=precision_score(y, yh, zero_division=0),
                Recall=recall_score(y, yh), F1=f1_score(y, yh), BalancedAcc=balanced_accuracy_score(y, yh),
                MCC=matthews_corrcoef(y, yh), ROC_AUC=roc_auc_score(y, p), PR_AUC=average_precision_score(y, p),
                TN=tn, FP=fp, FN=fn, TP=tp, Threshold=thr)


def save_preds(name, val_prob, test_prob):
    npsave(os.path.join(PRED, name + ".npz"), val_prob=np.atleast_2d(val_prob), test_prob=np.atleast_2d(test_prob))


def load_preds(name):
    z = np.load(os.path.join(PRED, name + ".npz"))
    return z["val_prob"], z["test_prob"]


def per_run_metrics(name, yva, yte):
    """Metrics on test for every run; threshold chosen on validation of that same run."""
    vp, tp = load_preds(name)
    rows = [metrics(yte, tp[i], best_threshold(yva, vp[i])) for i in range(len(tp))]
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# STEP 1 - DATA AUDIT
# ----------------------------------------------------------------------------------------------
def step1(a):
    df = load_raw(a.data)
    print("Shape:", df.shape)
    print("\nData types:\n", df.dtypes.to_string())
    print("\nMissing values (total):", int(df.isna().sum().sum()))
    n_dup = int(df.duplicated().sum())
    print(f"Duplicate rows: {n_dup} ({n_dup / len(df) * 100:.2f}%) -> removed for all downstream steps")
    dd = df.drop_duplicates()
    print("Shape after de-duplication:", dd.shape)
    vc = dd[TARGET].value_counts().sort_index()
    print("\nClass distribution (after de-duplication):")
    print(pd.DataFrame({"count": vc, "percent": (vc / vc.sum() * 100).round(2)}).to_string())
    print(f"Imbalance ratio (neg:pos) = {vc[0] / vc[1]:.2f} : 1")
    print("\nCategorical cardinality:", {c: int(dd[c].nunique()) for c in CAT_COLS})
    jdump(dict(raw_shape=df.shape, missing=int(df.isna().sum().sum()), duplicates=n_dup, dedup_shape=dd.shape,
               class_counts=vc.to_dict(), imbalance_ratio=float(vc[0] / vc[1])), os.path.join(OUT, "data_audit.json"))

    desc = dd[NUM_COLS].describe().T
    desc["skew"] = dd[NUM_COLS].skew()
    save_table(desc.round(3), "Table1_descriptive_statistics")
    print("\nDescriptive statistics (numerical):\n", desc.round(3).to_string())

    fig, ax = plt.subplots(figsize=(W1, 2.3))
    bars = ax.bar(["No purchase\n(Revenue = 0)", "Purchase\n(Revenue = 1)"], vc.values,
                  color=[C["blue"], C["orange"]], width=0.55)
    for b, v in zip(bars, vc.values):
        ax.text(b.get_x() + b.get_width() / 2, v + vc.max() * 0.015, f"{v:,}\n({v / vc.sum() * 100:.1f}%)",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Number of sessions")
    ax.set_ylim(0, vc.max() * 1.22)
    save_fig(fig, "Fig1_class_distribution")


# ----------------------------------------------------------------------------------------------
# STEP 2 - EXPLORATORY / STATISTICAL ANALYSIS
# ----------------------------------------------------------------------------------------------
def step2(a):
    from scipy.stats import mannwhitneyu, chi2_contingency
    df = load_dedup(a.data)

    # --- Fig 2: Spearman correlation (numerical features + target)
    cols = NUM_COLS + [TARGET]
    short = {"Administrative": "Admin", "Administrative_Duration": "Admin_Dur", "Informational": "Info",
             "Informational_Duration": "Info_Dur", "ProductRelated": "Product", "ProductRelated_Duration": "Product_Dur",
             "BounceRates": "Bounce", "ExitRates": "Exit", "PageValues": "PageVal", "SpecialDay": "SpecialDay",
             "Revenue": "Revenue"}
    cm = df[cols].corr(method="spearman")
    cm.index = cm.columns = [short[c] for c in cols]
    fig, ax = plt.subplots(figsize=(W1 * 1.25, W1 * 1.1))
    im = ax.imshow(cm.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cm))); ax.set_yticks(range(len(cm)))
    ax.set_xticklabels(cm.columns, rotation=45, ha="right"); ax.set_yticklabels(cm.index)
    for i in range(len(cm)):
        for j in range(len(cm)):
            v = cm.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(v) > 0.6 else "black")
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("Spearman ρ", fontsize=8); cb.ax.tick_params(labelsize=7.5)
    save_fig(fig, "Fig2_correlation_heatmap")

    # --- Table 2: association of every feature with the target
    pos, neg = df[df[TARGET] == 1], df[df[TARGET] == 0]
    rows = []
    for c in NUM_COLS:
        u, p = mannwhitneyu(pos[c], neg[c], alternative="two-sided")
        r = 1 - 2 * u / (len(pos) * len(neg))                        # rank-biserial correlation
        rows.append(dict(Feature=c, Type="numerical", Test="Mann-Whitney U", Statistic=u, p_value=p,
                         Effect_size=-r, Effect_name="rank-biserial r",
                         Median_purchase=pos[c].median(), Median_no_purchase=neg[c].median()))
    for c in CAT_COLS + BIN_COLS:
        ct = pd.crosstab(df[c], df[TARGET])
        chi2, p, _, _ = chi2_contingency(ct)
        v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
        rows.append(dict(Feature=c, Type="categorical", Test="Chi-square", Statistic=chi2, p_value=p,
                         Effect_size=v, Effect_name="Cramer's V", Median_purchase=np.nan, Median_no_purchase=np.nan))
    t2 = pd.DataFrame(rows)
    t2["p_bonferroni"] = np.minimum(t2["p_value"] * len(t2), 1.0)
    t2 = t2.sort_values("Effect_size", key=np.abs, ascending=False)
    save_table(t2.round(4), "Table2_feature_target_association", index=False)
    print(t2[["Feature", "Test", "Statistic", "p_bonferroni", "Effect_size", "Effect_name"]].round(4).to_string(index=False))

    # --- Fig 3: purchase rate by Month, VisitorType, Weekend
    order = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    overall = df[TARGET].mean() * 100
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.3), gridspec_kw=dict(width_ratios=[3.2, 1.5, 1.0]))
    for ax, col, lab, ordr, color in zip(
            axs, ["Month", "VisitorType", "Weekend"], ["Month", "Visitor type", "Weekend"],
            [order, ["New_Visitor", "Returning_Visitor", "Other"], [0, 1]], [C["blue"], C["green"], C["orange"]]):
        rate = df.groupby(col)[TARGET].mean().reindex(ordr) * 100
        ax.bar(range(len(rate)), rate.values, color=color, width=0.65)
        ax.axhline(overall, color=C["red"], ls="--", lw=0.9)
        ax.set_xticks(range(len(rate)))
        lbl = [str(x).replace("_Visitor", "").replace("Returning", "Return.") for x in ordr]
        if col == "Weekend":
            lbl = ["No", "Yes"]
        ax.set_xticklabels(lbl, rotation=45 if col == "Month" else 0, ha="right" if col == "Month" else "center")
        ax.set_xlabel(lab)
    axs[0].set_ylabel("Purchase rate (%)")
    axs[0].plot([], [], color=C["red"], ls="--", lw=0.9, label=f"Overall rate ({overall:.1f}%)")
    axs[0].legend(frameon=False, loc="upper left")
    for ax, s in zip(axs, "abc"):
        panel_label(ax, s)
    save_fig(fig, "Fig3_purchase_rate_by_category")


# ----------------------------------------------------------------------------------------------
# STEP 3 - PREPROCESSING AND SPLIT (all statistics learnt on TRAIN only)
# ----------------------------------------------------------------------------------------------
def step3(a):
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    df = load_dedup(a.data)
    y = df[TARGET].values
    Xdf = df.drop(columns=[TARGET])

    idx = np.arange(len(df))
    i_tr, i_tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEED)
    i_va, i_te = train_test_split(i_tmp, test_size=0.50, stratify=y[i_tmp], random_state=SEED)

    def numeric(d):
        d = d[NUM_COLS].copy()
        d[LOG_COLS] = np.log1p(d[LOG_COLS])          # variance-stabilising transform for skewed counts/durations
        return d

    scaler = StandardScaler().fit(numeric(Xdf.iloc[i_tr]))
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(Xdf.iloc[i_tr][CAT_COLS])

    def transform(ii):
        d = Xdf.iloc[ii]
        return np.hstack([scaler.transform(numeric(d)), d[BIN_COLS].values, ohe.transform(d[CAT_COLS])]).astype(np.float32)

    Xtr, Xva, Xte = transform(i_tr), transform(i_va), transform(i_te)
    names = NUM_COLS + BIN_COLS + list(ohe.get_feature_names_out(CAT_COLS))
    groups = NUM_COLS + BIN_COLS + [n for c, cats in zip(CAT_COLS, ohe.categories_) for n in [c] * len(cats)]
    n_cont = len(NUM_COLS)
    cat_idx = np.arange(n_cont, Xtr.shape[1])         # binary columns (Weekend + one-hot) for SMOTE-NC

    np.savez_compressed(os.path.join(OUT, "processed.npz"), Xtr=Xtr, ytr=y[i_tr], Xva=Xva, yva=y[i_va],
                        Xte=Xte, yte=y[i_te], names=np.array(names), groups=np.array(groups), cat_idx=cat_idx,
                        idx_tr=i_tr, idx_va=i_va, idx_te=i_te)         # row indices kept for traceability
    joblib.dump(dict(scaler=scaler, ohe=ohe), os.path.join(OUT, "preprocessors.joblib"))

    rows = [dict(Split=n, Samples=len(yy), Positives=int(yy.sum()), Negatives=int((1 - yy).sum()),
                 Positive_pct=round(yy.mean() * 100, 2)) for n, yy in
            [("Train", y[i_tr]), ("Validation", y[i_va]), ("Test", y[i_te])]]
    t = pd.DataFrame(rows)
    save_table(t, "Table_split_summary", index=False)
    print(t.to_string(index=False))
    print(f"\nFeature matrix: {Xtr.shape[1]} columns ({n_cont} numerical + {len(BIN_COLS)} binary + "
          f"{Xtr.shape[1] - n_cont - len(BIN_COLS)} one-hot)")
    print("LSTM input shape per sample: (timesteps = %d, features = 1)" % Xtr.shape[1])
    log_event(event="preprocess", train=len(i_tr), val=len(i_va), test=len(i_te), n_features=int(Xtr.shape[1]))


# ----------------------------------------------------------------------------------------------
# STEP 4 - BASELINE MACHINE-LEARNING MODELS (CV tuning inside the training set only)
# ----------------------------------------------------------------------------------------------
def step4(a):
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from scipy.stats import randint, uniform, loguniform
    d = load_processed()
    Xtr, ytr, Xva, yva, Xte, yte = d["Xtr"], d["ytr"], d["Xva"], d["yva"], d["Xte"], d["yte"]
    spw = (ytr == 0).sum() / (ytr == 1).sum()
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    n_iter = 3 if a.quick else 25

    models = {
        "LR": (LogisticRegression(class_weight="balanced", max_iter=3000, random_state=SEED),
               dict(C=loguniform(1e-3, 1e2))),
        "RF": (RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
               dict(n_estimators=randint(200, 800), max_depth=randint(4, 25), min_samples_leaf=randint(1, 20),
                    max_features=uniform(0.1, 0.6))),
    }
    try:
        from xgboost import XGBClassifier
        models["XGB"] = (XGBClassifier(scale_pos_weight=spw, eval_metric="aucpr", tree_method="hist",
                                       random_state=SEED, n_jobs=-1),
                         dict(n_estimators=randint(100, 600), max_depth=randint(2, 8), learning_rate=loguniform(0.01, 0.3),
                              subsample=uniform(0.6, 0.4), colsample_bytree=uniform(0.5, 0.5),
                              min_child_weight=randint(1, 10), reg_lambda=loguniform(1e-2, 10)))
    except ImportError:
        print("xgboost not installed -> pip install xgboost"); sys.exit(1)

    rows = []
    for name, (est, space) in models.items():
        key = sig_of(name, n_iter, d["sig"], SEED)
        rp = os.path.join(RUNS, f"s4_{name}_{key}.json")
        if os.path.exists(rp) and os.path.exists(os.path.join(PRED, name + ".npz")) and not FORCE:
            row = json.load(open(rp))
            print(f"{name}: loaded from checkpoint ({rp})")
        else:
            t0 = time.time()
            rs = RandomizedSearchCV(est, space, n_iter=n_iter, scoring="average_precision", cv=cv,
                                    random_state=SEED, n_jobs=1 if name != "LR" else -1, refit=True)
            rs.fit(Xtr, ytr)
            vp, tp = rs.predict_proba(Xva)[:, 1], rs.predict_proba(Xte)[:, 1]
            save_preds(name, vp, tp)
            m = metrics(yte, tp, best_threshold(yva, vp))
            row = dict(Model=name, CV_PR_AUC=rs.best_score_, **{k: m[k] for k in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC"]},
                       best_params=json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rs.best_params_.items()}))
            jdump(row, rp)
            log_event(event="run", step=4, model=name, seed=SEED, n_iter=n_iter, cv_folds=5, seconds=round(time.time() - t0, 1),
                      best_params=rs.best_params_, cv_pr_auc=rs.best_score_, test_pr_auc=m["PR_AUC"], test_f1=m["F1"], file=rp)
        rows.append(row)
        print(f"{name}: CV PR-AUC={row['CV_PR_AUC']:.4f} | test F1={row['F1']:.4f}  PR-AUC={row['PR_AUC']:.4f}  ROC-AUC={row['ROC_AUC']:.4f}")
    t = pd.DataFrame(rows)
    save_table(t.round(4), "Table_baseline_models", index=False)
    print(t.drop(columns="best_params").round(4).to_string(index=False))


# ----------------------------------------------------------------------------------------------
# LSTM CORE
# ----------------------------------------------------------------------------------------------
import torch
import torch.nn as nn

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LSTMNet(nn.Module):
    """Each preprocessed feature is one time step (seq_len = n_features, input_size = 1)."""

    def __init__(self, hidden, layers, bidir, dropout):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, layers, batch_first=True, bidirectional=bidir,
                            dropout=dropout if layers > 1 else 0.0)
        self.bidir = bidir
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * (2 if bidir else 1), 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        h = torch.cat([h[-2], h[-1]], 1) if self.bidir else h[-1]
        return self.fc(self.drop(h)).squeeze(-1)


@torch.no_grad()
def predict(model, X):
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32, device=DEV).unsqueeze(-1)
    out = [torch.sigmoid(model(Xt[i:i + 4096])).cpu().numpy() for i in range(0, len(Xt), 4096)]
    return np.concatenate(out)


_CACHE = {}


def get_train(strategy, d):
    """Training data for a given imbalance strategy (resampling applied to TRAIN only, once, fixed seed)."""
    if strategy in _CACHE:
        return _CACHE[strategy]
    Xtr, ytr = d["Xtr"], d["ytr"]
    if strategy in ("none", "class_weight"):
        res = (Xtr, ytr)
    elif strategy == "ros":
        from imblearn.over_sampling import RandomOverSampler
        res = RandomOverSampler(random_state=SEED).fit_resample(Xtr, ytr)
    elif strategy == "smotenc":
        from imblearn.over_sampling import SMOTENC
        res = SMOTENC(categorical_features=d["cat_idx"].tolist(), random_state=SEED).fit_resample(Xtr, ytr)
    _CACHE[strategy] = res
    return res


def train_lstm(cfg, strategy, d, seed, max_epochs=100, patience=10, trial=None, min_epochs=MIN_EPOCHS):
    """Early stopping on validation PR-AUC. Returns best model, history, best val PR-AUC."""
    import optuna
    set_seed(seed)
    Xtr, ytr = get_train(strategy, d)
    Xva, yva = d["Xva"], d["yva"]
    model = LSTMNet(cfg["hidden"], cfg["layers"], cfg["bidir"], cfg["dropout"]).to(DEV)
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=DEV).unsqueeze(-1)
    yt = torch.tensor(ytr, dtype=torch.float32, device=DEV)
    pw = None
    if strategy == "class_weight":
        pw = torch.tensor((ytr == 0).sum() / (ytr == 1).sum(), dtype=torch.float32, device=DEV)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    plain = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    Xv = torch.tensor(Xva, dtype=torch.float32, device=DEV).unsqueeze(-1)
    yv = torch.tensor(yva, dtype=torch.float32, device=DEV)

    best, best_state, wait, hist = -1.0, None, 0, []
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(len(Xt), device=DEV)
        tot = 0.0
        for i in range(0, len(perm), cfg["batch"]):
            b = perm[i:i + cfg["batch"]]
            opt.zero_grad()
            loss = loss_fn(model(Xt[b]), yt[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            lg = model(Xv)
            vloss = plain(lg, yv).item()
            vp = torch.sigmoid(lg).cpu().numpy()
        ap = average_precision_score(yva, vp)
        hist.append(dict(epoch=ep + 1, train_loss=tot / len(Xt), val_loss=vloss, val_pr_auc=ap))
        if ap > best + 1e-4:
            best, best_state, wait = ap, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
        if trial is not None:
            trial.report(ap, ep)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if wait >= patience and ep + 1 >= min_epochs:
            break
    model.load_state_dict(best_state)
    return model, hist, best


# ----------------------------------------------------------------------------------------------
# STEP 5 - IMBALANCE STRATEGY COMPARISON (default LSTM, validation set only)
# ----------------------------------------------------------------------------------------------
def step5(a):
    d = load_processed()
    seeds = FINAL_SEEDS[:2] if a.quick else FINAL_SEEDS
    ep = 3 if a.quick else 100
    print(f"Device: {DEV} | default config: {DEFAULT_CFG} | seeds: {seeds}")
    rows = []
    for s in STRATS:
        for sd in seeds:
            key = sig_of(DEFAULT_CFG, s, sd, ep, d["sig"], MIN_EPOCHS)
            rp = os.path.join(RUNS, f"s5_{s}_seed{sd}_{key}.json")
            if os.path.exists(rp) and not FORCE:
                rows.append(json.load(open(rp)))
                continue
            t0 = time.time()
            m, h, ap = train_lstm(DEFAULT_CFG, s, d, sd, max_epochs=ep)
            vp = predict(m, d["Xva"])
            f1 = metrics(d["yva"], vp, best_threshold(d["yva"], vp))["F1"]
            row = dict(Strategy=s, Seed=sd, Val_PR_AUC=ap, Val_ROC_AUC=roc_auc_score(d["yva"], vp), Val_F1=f1,
                       Epochs=len(h), Seconds=round(time.time() - t0, 1))
            jdump(row, rp)
            log_event(event="run", step=5, config=DEFAULT_CFG, strategy=s, seed=sd, max_epochs=ep, file=rp, **{
                k: row[k] for k in ("Val_PR_AUC", "Val_ROC_AUC", "Val_F1", "Epochs", "Seconds")})
            rows.append(row)
        print(f"{s:13s} done")
    r = pd.DataFrame(rows)
    save_table(r.round(4), "Table3_imbalance_strategies_raw", index=False)
    g = r.groupby("Strategy")[["Val_PR_AUC", "Val_ROC_AUC", "Val_F1"]].agg(["mean", "std"]).reindex(STRATS)
    save_table(g.round(4), "Table3_imbalance_strategies_summary")
    print(g.round(4).to_string())
    best = r.groupby("Strategy")["Val_PR_AUC"].mean().idxmax()
    jdump(dict(best_strategy=best), os.path.join(OUT, "best_strategy.json"))
    print(f"\nBest strategy for default LSTM (mean validation PR-AUC): {best}")

    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.5))
    cols = [C["gray"], C["blue"], C["orange"], C["green"]]
    for ax, met, lab in zip(axs, ["Val_PR_AUC", "Val_F1"], ["Validation PR-AUC", "Validation F1-score"]):
        mu = g[met]["mean"].values; sd = g[met]["std"].fillna(0).values
        ax.bar(range(4), mu, yerr=sd, color=cols, width=0.6, capsize=2.5, error_kw=dict(lw=0.8))
        for i, v in enumerate(mu):
            ax.text(i, v + sd[i] + 0.005, f"{v:.3f}", ha="center", fontsize=7.5)
        ax.set_xticks(range(4)); ax.set_xticklabels([STRAT_LABEL[s] for s in STRATS])
        ax.set_ylabel(lab); ax.set_ylim(min(mu) * 0.85, max(mu + sd) * 1.07)
    panel_label(axs[0], "a"); panel_label(axs[1], "b")
    save_fig(fig, "Fig4_imbalance_strategies")


# ----------------------------------------------------------------------------------------------
# STEP 6 - HYPERPARAMETER TUNING (Optuna TPE + median pruning; objective = validation PR-AUC)
# ----------------------------------------------------------------------------------------------
def step6(a):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    d = load_processed()
    n_trials = 2 if a.quick else a.n_trials
    ep = 3 if a.quick else 100

    def objective(trial):
        cfg = dict(hidden=trial.suggest_categorical("hidden", [32, 64, 128, 256]),
                   layers=trial.suggest_int("layers", 1, 2),
                   bidir=trial.suggest_categorical("bidir", [False, True]),
                   dropout=trial.suggest_float("dropout", 0.0, 0.5),
                   lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                   wd=trial.suggest_float("wd", 1e-6, 1e-3, log=True),
                   batch=trial.suggest_categorical("batch", [64, 128, 256, 512]))
        strategy = trial.suggest_categorical("strategy", STRATS)
        _, _, ap = train_lstm(cfg, strategy, d, SEED, max_epochs=ep, trial=trial)
        return ap

    # study name carries the data signature + search-space version -> a changed split/space starts a NEW study
    name = f"lstm_shoppers_{SPACE_VERSION}_{d['sig']}_ep{ep}"
    study = optuna.create_study(study_name=name, direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=10),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=MIN_EPOCHS),
                                storage=f"sqlite:///{OUT}/optuna.db", load_if_exists=True)   # resumable
    # a trial that was running when the session died can never finish -> mark it failed, it is re-run as a new trial
    for t in study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.RUNNING]):
        try:
            study._storage.set_trial_state_values(t._trial_id, optuna.trial.TrialState.FAIL)
        except Exception:
            pass
    done = len([t for t in study.trials if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)])
    print(f"Device: {DEV} | study: {name} | finished trials so far: {done} | target: {n_trials}")

    def cb(s, t):
        sec = t.duration.total_seconds() if t.duration is not None else None
        v = t.value if t.value is not None else float("nan")
        log_event(event="trial", step=6, number=t.number, state=t.state.name, val_pr_auc=v, params=t.params, seconds=sec)
        best = s.best_value if any(x.value is not None for x in s.trials) else float("nan")
        print(f"trial {t.number:3d} | {t.state.name:8s} | val PR-AUC={v:.4f} | best={best:.4f}")

    if done < n_trials:
        study.optimize(objective, n_trials=n_trials - done, callbacks=[cb])
    tr = study.trials_dataframe()
    save_table(tr.round(5), "Table_optuna_trials", index=False)
    best = dict(study.best_params)
    jdump(dict(params=best, val_pr_auc=study.best_value, study=name, n_trials=len(tr)), os.path.join(OUT, "best_params.json"))
    n_pr = int((tr["state"] == "PRUNED").sum())
    print(f"\nTrials: {len(tr)} (pruned: {n_pr}) | Best val PR-AUC = {study.best_value:.4f}\nBest params: {best}")

    # Fig 5: optimisation history + hyperparameter importance
    comp = tr[tr["state"] == "COMPLETE"]
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.5), gridspec_kw=dict(width_ratios=[1.15, 1]))
    axs[0].scatter(comp["number"], comp["value"], s=10, color=C["blue"], alpha=0.75, label="Trial")
    axs[0].plot(comp["number"], comp["value"].cummax(), color=C["red"], lw=1.4, label="Best so far")
    axs[0].set_xlabel("Trial"); axs[0].set_ylabel("Validation PR-AUC"); axs[0].legend(frameon=False, loc="lower right")
    try:
        imp = optuna.importance.get_param_importances(study)
        names, vals = list(imp.keys())[::-1], list(imp.values())[::-1]
        axs[1].barh(names, vals, color=C["green"], height=0.65)
        axs[1].set_xlabel("Hyperparameter importance (fANOVA)")
    except Exception as e:
        axs[1].text(0.5, 0.5, "n/a (too few trials)", ha="center"); print("Importance skipped:", e)
    panel_label(axs[0], "a"); panel_label(axs[1], "b")
    save_fig(fig, "Fig5_optuna_history_importance")


# ----------------------------------------------------------------------------------------------
# STEP 7 - FINAL TRAINING: default vs tuned LSTM, several independent seeds
# ----------------------------------------------------------------------------------------------
def step7(a):
    d = load_processed()
    seeds = FINAL_SEEDS[:2] if a.quick else FINAL_SEEDS
    ep = 3 if a.quick else 300
    bp = json.load(open(os.path.join(OUT, "best_params.json")))["params"]
    tuned_strat = bp.pop("strategy")
    tuned_cfg = bp
    def_strat = json.load(open(os.path.join(OUT, "best_strategy.json")))["best_strategy"]
    print(f"Default LSTM : {DEFAULT_CFG} | strategy={def_strat}\nTuned LSTM   : {tuned_cfg} | strategy={tuned_strat}")

    hist_keep, manifest, per_seed = {}, {}, []
    for name, cfg, strat in [("LSTM_default", DEFAULT_CFG, def_strat), ("LSTM_tuned", tuned_cfg, tuned_strat)]:
        vps, tps, manifest[name] = [], [], []
        for k, sd in enumerate(seeds):
            key = sig_of(cfg, strat, sd, ep, d["sig"], MIN_EPOCHS)
            base = f"s7_{name}_seed{sd}_{key}"
            pn, jn = os.path.join(RUNS, base + ".npz"), os.path.join(RUNS, base + ".json")
            mp = os.path.join(MOD, f"{name}_seed{sd}_{key}.pt")
            if os.path.exists(pn) and os.path.exists(jn) and os.path.exists(mp) and not FORCE:
                z, info = np.load(pn), json.load(open(jn))
                vp, tp, h, ap = z["val"], z["test"], info["hist"], info["val_pr_auc"]
                tag = "(checkpoint)"
            else:
                t0 = time.time()
                m, h, ap = train_lstm(cfg, strat, d, sd, max_epochs=ep, patience=15)
                vp, tp = predict(m, d["Xva"]), predict(m, d["Xte"])
                tmp = mp + ".tmp"
                torch.save(dict(state=m.state_dict(), cfg=cfg, strategy=strat, seed=sd), tmp); os.replace(tmp, mp)
                npsave(pn, val=vp, test=tp)
                jdump(dict(hist=h, val_pr_auc=ap, epochs=len(h), config=cfg, strategy=strat, seed=sd,
                           seconds=round(time.time() - t0, 1), model=mp), jn)      # json written LAST = run complete
                log_event(event="run", step=7, model_name=name, config=cfg, strategy=strat, seed=sd, max_epochs=ep,
                          val_pr_auc=ap, epochs=len(h), seconds=round(time.time() - t0, 1), files=[pn, jn, mp])
                tag = ""
            vps.append(vp); tps.append(tp)
            manifest[name].append(dict(seed=sd, model=mp, cfg=cfg, strategy=strat))
            if k == 0:
                hist_keep[name] = h
            tm = metrics(d["yte"], tp, best_threshold(d["yva"], vp))
            per_seed.append(dict(Model=name, Seed=sd, Val_PR_AUC=ap, Epochs=len(h), **{c: tm[c] for c in
                               ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC", "MCC"]}))
            print(f"{name} seed {sd}: val PR-AUC={ap:.4f}, epochs={len(h)} {tag}")
        save_preds(name, np.array(vps), np.array(tps))
        r = per_run_metrics(name, d["yva"], d["yte"])
        print(f"\n{name} test metrics (mean ± SD over {len(seeds)} seeds):")
        print(pd.DataFrame({"mean": r.mean(), "sd": r.std()}).loc[["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC", "MCC"]].round(4).to_string())
    save_table(pd.DataFrame(per_seed).round(4), "Table_final_runs_per_seed", index=False)
    jdump(hist_keep, os.path.join(OUT, "histories.json"))
    jdump(manifest, os.path.join(OUT, "final_manifest.json"))

    # Fig 6: learning curves of the tuned model (first seed)
    h = pd.DataFrame(hist_keep["LSTM_tuned"])
    best_ep = int(h["val_pr_auc"].idxmax()) + 1
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.4))
    axs[0].plot(h["epoch"], h["train_loss"], color=C["blue"], lw=1.4, label="Training")
    axs[0].plot(h["epoch"], h["val_loss"], color=C["orange"], lw=1.4, label="Validation (unweighted)")
    axs[0].set_xlabel("Epoch"); axs[0].set_ylabel("BCE loss"); axs[0].legend(frameon=False)
    axs[1].plot(h["epoch"], h["val_pr_auc"], color=C["green"], lw=1.4)
    axs[1].axvline(best_ep, color=C["red"], ls="--", lw=0.9, label=f"Selected epoch ({best_ep})")
    axs[1].set_xlabel("Epoch"); axs[1].set_ylabel("Validation PR-AUC"); axs[1].legend(frameon=False, loc="lower right")
    panel_label(axs[0], "a"); panel_label(axs[1], "b")
    save_fig(fig, "Fig6_learning_curves")


# ----------------------------------------------------------------------------------------------
# STEP 8 - MODEL COMPARISON
# ----------------------------------------------------------------------------------------------
MODELS = ["LR", "RF", "XGB", "LSTM_default", "LSTM_tuned"]
LABEL = {"LR": "Logistic Regression", "RF": "Random Forest", "XGB": "XGBoost",
         "LSTM_default": "LSTM (default)", "LSTM_tuned": "LSTM (tuned)"}
COLOR = {"LR": C["gray"], "RF": C["green"], "XGB": C["orange"], "LSTM_default": C["sky"], "LSTM_tuned": C["red"]}


def step8(a):
    d = load_processed()
    yva, yte = d["yva"], d["yte"]
    cols = ["Accuracy", "Precision", "Recall", "F1", "BalancedAcc", "MCC", "ROC_AUC", "PR_AUC"]

    # Table 4: test performance (threshold from validation); mean ± SD where several seeds exist
    rows, ens = [], {}
    for m in MODELS:
        r = per_run_metrics(m, yva, yte)
        row = {"Model": LABEL[m]}
        for c in cols:
            row[c] = f"{r[c].mean():.4f}" + (f" ± {r[c].std():.4f}" if len(r) > 1 else "")
        rows.append(row)
        vp, tp = load_preds(m)
        ev, et = vp.mean(0), tp.mean(0)                       # seed-averaged probabilities (LSTM) / identity (baselines)
        ens[m] = (ev, et, best_threshold(yva, ev))
    t4 = pd.DataFrame(rows)
    save_table(t4, "Table4_test_performance", index=False)
    print(t4.to_string(index=False))

    # Fig 7: ROC and PR curves (LSTM curves = seed-averaged probabilities)
    fig, axs = plt.subplots(1, 2, figsize=(W2, 3.0))
    for m in MODELS:
        et = ens[m][1]
        fpr, tpr, _ = roc_curve(yte, et); pr, rc, _ = precision_recall_curve(yte, et)
        lw = 1.8 if m == "LSTM_tuned" else 1.2
        axs[0].plot(fpr, tpr, color=COLOR[m], lw=lw, label=f"{LABEL[m]} ({roc_auc_score(yte, et):.3f})")
        axs[1].plot(rc, pr, color=COLOR[m], lw=lw, label=f"{LABEL[m]} ({average_precision_score(yte, et):.3f})")
    axs[0].plot([0, 1], [0, 1], color="k", lw=0.8, ls=":")
    axs[1].axhline(yte.mean(), color="k", lw=0.8, ls=":")
    axs[0].set_xlabel("False positive rate"); axs[0].set_ylabel("True positive rate")
    axs[1].set_xlabel("Recall"); axs[1].set_ylabel("Precision")
    axs[0].legend(frameon=False, loc="lower right", title="AUC", title_fontsize=7.5)
    axs[1].legend(frameon=False, loc="upper right", title="PR-AUC", title_fontsize=7.5)
    for ax in axs:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    panel_label(axs[0], "a"); panel_label(axs[1], "b")
    save_fig(fig, "Fig7_roc_pr_curves")

    # Fig 8: confusion matrices (best baseline vs tuned LSTM), thresholds from validation
    base_best = max(["LR", "RF", "XGB"], key=lambda m: average_precision_score(yva, ens[m][0]))
    fig, axs = plt.subplots(1, 2, figsize=(W2 * 0.62, 2.7))
    for ax, m, s in zip(axs, [base_best, "LSTM_tuned"], "ab"):
        cmx = confusion_matrix(yte, (ens[m][1] >= ens[m][2]).astype(int))
        ax.imshow(cmx, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cmx[i, j]:,}\n({cmx[i, j] / cmx[i].sum() * 100:.1f}%)", ha="center", va="center", fontsize=8,
                        color="white" if cmx[i, j] > cmx.max() / 2 else "black")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["No", "Yes"]); ax.set_yticklabels(["No", "Yes"])
        ax.set_xlabel("Predicted purchase"); ax.set_ylabel("Actual purchase")
        ax.set_title(LABEL[m], fontsize=8.5)
        for sp in ax.spines.values():
            sp.set_visible(False)
        panel_label(ax, s)
    save_fig(fig, "Fig8_confusion_matrices")

    # Table 5: paired bootstrap of the TEST set (same resample for every model).
    # For LSTM models the metric inside each resample is the MEAN over the independent seeds (each seed with its own
    # validation threshold), i.e. the same quantity reported in Table 4 -> no advantage from seed-ensembling.
    runs = {}
    for m in MODELS:
        vp, tp = load_preds(m)
        runs[m] = [(tp[i], best_threshold(yva, vp[i])) for i in range(len(tp))]
    rng = np.random.default_rng(SEED)
    B = 200 if a.quick else 2000
    boot = {m: dict(PR_AUC=[], F1=[]) for m in MODELS}
    n_ok = 0
    for _ in range(B):
        ii = rng.integers(0, len(yte), len(yte))
        yb = yte[ii]
        if yb.sum() == 0:
            continue
        n_ok += 1
        for m in MODELS:
            boot[m]["PR_AUC"].append(np.mean([average_precision_score(yb, p[ii]) for p, _ in runs[m]]))
            boot[m]["F1"].append(np.mean([f1_score(yb, p[ii] >= t) for p, t in runs[m]]))
    rows = []
    for m in [x for x in MODELS if x != "LSTM_tuned"]:
        for nm in ("PR_AUC", "F1"):
            v = np.array(boot["LSTM_tuned"][nm]) - np.array(boot[m][nm])
            lo, hi = np.percentile(v, [2.5, 97.5])
            rows.append(dict(Comparison=f"LSTM (tuned) - {LABEL[m]}", Metric=nm, Mean_diff=np.mean(v),
                             CI95_low=lo, CI95_high=hi, Significant="yes" if (lo > 0 or hi < 0) else "no"))
    t5 = pd.DataFrame(rows)
    save_table(t5.round(4), "Table5_paired_bootstrap", index=False)
    print(f"\nPaired bootstrap ({n_ok} resamples; positive = tuned LSTM better):\n", t5.round(4).to_string(index=False))
    print(f"\n(Fig 7/8 use seed-averaged LSTM probabilities; Table 4 and Table 5 use per-seed metrics averaged over seeds; "
          f"Fig 8 baseline shown = {LABEL[base_best]})")


# ----------------------------------------------------------------------------------------------
# STEP 9 - PERMUTATION IMPORTANCE (grouped by original feature; PR-AUC drop on the test set; all tuned seeds)
# ----------------------------------------------------------------------------------------------
def step9(a):
    d = load_processed()
    manifest = json.load(open(os.path.join(OUT, "final_manifest.json")))["LSTM_tuned"]
    Xte, yte, groups = d["Xte"], d["yte"], d["groups"]
    R = 3 if a.quick else 20
    feats = list(dict.fromkeys(groups.tolist()))
    drops = {g: [] for g in feats}
    bases = []
    for mi in manifest:
        ck = torch.load(mi["model"], map_location=DEV)
        cfg = ck["cfg"]
        model = LSTMNet(cfg["hidden"], cfg["layers"], cfg["bidir"], cfg["dropout"]).to(DEV)
        model.load_state_dict(ck["state"])
        base = average_precision_score(yte, predict(model, Xte))
        bases.append(base)
        rng = np.random.default_rng(mi["seed"])
        for g in feats:
            cols = np.where(groups == g)[0]
            for _ in range(R):
                Xp = Xte.copy()
                Xp[:, cols] = Xp[rng.permutation(len(Xp))][:, cols]       # shuffle whole feature group jointly
                drops[g].append(base - average_precision_score(yte, predict(model, Xp)))
        print(f"seed {mi['seed']}: baseline test PR-AUC = {base:.4f}")
        log_event(event="run", step=9, seed=mi["seed"], repeats=R, baseline_pr_auc=base, model=mi["model"])
    t = pd.DataFrame([dict(Feature=g, Mean_drop=np.mean(v), SD=np.std(v)) for g, v in drops.items()])
    t = t.sort_values("Mean_drop", ascending=False)
    save_table(t.round(5), "Table6_permutation_importance", index=False)
    print(f"Tuned LSTM, {len(manifest)} seeds x {R} permutations per feature group "
          f"(mean baseline test PR-AUC = {np.mean(bases):.4f})\n", t.round(4).to_string(index=False))

    top = t.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(W1, 2.8))
    ax.barh(top["Feature"], top["Mean_drop"], xerr=top["SD"], color=C["blue"], height=0.65, capsize=2, error_kw=dict(lw=0.8))
    ax.set_xlabel("Decrease in test PR-AUC")
    save_fig(fig, "Fig9_permutation_importance")


# ----------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------
# STEP 10 - SENSITIVITY ANALYSIS: all models re-trained WITHOUT PageValues
#           (PageValues dominates the importance ranking; it is derived from the e-commerce transaction value of
#            pages visited, so it may carry outcome information. Hyperparameters are kept fixed from steps 4/6.)
# ----------------------------------------------------------------------------------------------
def step10(a):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    d = load_processed()
    keep = d["groups"] != "PageValues"
    d2 = dict(d)
    for k in ("Xtr", "Xva", "Xte"):
        d2[k] = d[k][:, keep]
    d2["cat_idx"] = np.arange(len(NUM_COLS) - 1, d2["Xtr"].shape[1])      # binary/one-hot columns after dropping one numeric column
    d2["sig"] = sig_of(d["sig"], "no_PageValues")
    _CACHE.clear()                                                         # resampled train sets of the full feature set must not be reused
    ytr, yva, yte = d["ytr"], d["yva"], d["yte"]
    spw = (ytr == 0).sum() / (ytr == 1).sum()
    ep = 3 if a.quick else 300
    print(f"Features: {d['Xtr'].shape[1]} -> {d2['Xtr'].shape[1]} (PageValues removed)")

    def params(name):
        f = sorted(glob.glob(os.path.join(RUNS, f"s4_{name}_*.json")), key=os.path.getmtime)[-1]
        return json.loads(json.load(open(f))["best_params"])

    builders = {
        "LR": lambda p: LogisticRegression(class_weight="balanced", max_iter=3000, random_state=SEED, **p),
        "RF": lambda p: RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1, random_state=SEED, **p),
        "XGB": lambda p: XGBClassifier(scale_pos_weight=spw, eval_metric="aucpr", tree_method="hist",
                                       random_state=SEED, n_jobs=-1, **p),
    }
    res = {}
    for name, build in builders.items():
        p = params(name)
        pn = os.path.join(RUNS, f"s10_{name}_{sig_of(p, d2['sig'])}.npz")
        if os.path.exists(pn) and not FORCE:
            z = np.load(pn); vp, tp = z["val"], z["test"]
        else:
            t0 = time.time()
            m = build(p).fit(d2["Xtr"], ytr)
            vp, tp = m.predict_proba(d2["Xva"])[:, 1], m.predict_proba(d2["Xte"])[:, 1]
            npsave(pn, val=vp, test=tp)
            log_event(event="run", step=10, model=name, params=p, seconds=round(time.time() - t0, 1), file=pn)
        res[name] = [(vp, tp)]
        print(f"{name} (no PageValues) done")

    manifest = json.load(open(os.path.join(OUT, "final_manifest.json")))["LSTM_tuned"]
    runs = []
    for mi in manifest:
        sd, cfg, strat = mi["seed"], mi["cfg"], mi["strategy"]
        pn = os.path.join(RUNS, f"s10_LSTM_tuned_seed{sd}_{sig_of(cfg, strat, sd, ep, d2['sig'], MIN_EPOCHS)}.npz")
        if os.path.exists(pn) and not FORCE:
            z = np.load(pn); vp, tp = z["val"], z["test"]
        else:
            t0 = time.time()
            m, h, ap = train_lstm(cfg, strat, d2, sd, max_epochs=ep, patience=15)
            vp, tp = predict(m, d2["Xva"]), predict(m, d2["Xte"])
            npsave(pn, val=vp, test=tp)
            log_event(event="run", step=10, model_name="LSTM_tuned", config=cfg, strategy=strat, seed=sd, val_pr_auc=ap,
                      epochs=len(h), seconds=round(time.time() - t0, 1), file=pn)
        runs.append((vp, tp))
        print(f"LSTM_tuned (no PageValues) seed {sd} done")
    res["LSTM_tuned"] = runs
    _CACHE.clear()

    rows, plot = [], {}
    for name in ["LR", "RF", "XGB", "LSTM_tuned"]:
        full = per_run_metrics(name, yva, yte)
        red = pd.DataFrame([metrics(yte, tp, best_threshold(yva, vp)) for vp, tp in res[name]])
        plot[name] = (full["PR_AUC"].mean(), full["PR_AUC"].std() if len(full) > 1 else 0.0,
                      red["PR_AUC"].mean(), red["PR_AUC"].std() if len(red) > 1 else 0.0)
        row = {"Model": LABEL[name]}
        for met in ("PR_AUC", "ROC_AUC", "F1"):
            f = f"{full[met].mean():.4f}" + (f" ± {full[met].std():.4f}" if len(full) > 1 else "")
            r = f"{red[met].mean():.4f}" + (f" ± {red[met].std():.4f}" if len(red) > 1 else "")
            row[f"{met}_all_features"], row[f"{met}_no_PageValues"] = f, r
            row[f"{met}_change"] = round(red[met].mean() - full[met].mean(), 4)
        rows.append(row)
    t7 = pd.DataFrame(rows)
    save_table(t7, "Table7_sensitivity_no_PageValues", index=False)
    print(t7.to_string(index=False))

    fig, ax = plt.subplots(figsize=(W1 * 1.35, 2.5))
    names = list(plot)
    x = np.arange(len(names))
    ax.bar(x - 0.19, [plot[n][0] for n in names], 0.36, yerr=[plot[n][1] for n in names], color=C["blue"],
           capsize=2, error_kw=dict(lw=0.8), label="All features")
    ax.bar(x + 0.19, [plot[n][2] for n in names], 0.36, yerr=[plot[n][3] for n in names], color=C["orange"],
           capsize=2, error_kw=dict(lw=0.8), label="Without PageValues")
    ax.set_xticks(x); ax.set_xticklabels(["LR", "RF", "XGBoost", "LSTM\n(tuned)"])
    ax.set_ylabel("Test PR-AUC"); ax.legend(frameon=False, loc="upper right")
    save_fig(fig, "Fig10_sensitivity_no_PageValues")


STEP_FUNCS = {1: step1, 2: step2, 3: step3, 4: step4, 5: step5, 6: step6, 7: step7, 8: step8, 9: step9, 10: step10}


def parse_steps(txt):
    """'7' -> [7] | '5-9' -> [5..9] | 'all' -> [1..10]"""
    txt = str(txt).strip().lower()
    if txt == "all":
        return list(range(1, 11))
    if "-" in txt:
        lo, hi = (int(x) for x in txt.split("-"))
        return list(range(lo, hi + 1))
    return [int(txt)]


def run_step(n, args):
    _orig_stdout = sys.stdout
    sys.stdout = Tee(os.path.join(CON, f"step{n}_{datetime.now():%Y%m%d_%H%M%S}.txt"))
    t_start = time.time()
    try:
        print(f"=== STEP {n} | device: {DEV} | output folder: {OUT} ===")
        log_event(event="step_start", step=n, args=vars(args))
        STEP_FUNCS[n](args)
        log_event(event="step_end", step=n, seconds=round(time.time() - t_start, 1))
        backup(n)
        print(f"=== STEP {n} finished in {(time.time() - t_start) / 60:.1f} min ===")
    except KeyboardInterrupt:
        log_event(event="step_interrupted", step=n, seconds=round(time.time() - t_start, 1))
        print("\n[interrupted] completed runs are saved - re-run the same command to resume.")
        raise
    except Exception as e:
        log_event(event="step_failed", step=n, error=repr(e), seconds=round(time.time() - t_start, 1))
        raise
    finally:
        sys.stdout = _orig_stdout


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="all", help="one step (e.g. 6), a range (e.g. 5-9) or 'all' (default)")
    ap.add_argument("--data", default="online_shoppers_intention.csv")
    ap.add_argument("--n_trials", type=int, default=50)
    ap.add_argument("--force", action="store_true", help="recompute cached runs of the selected steps")
    ap.add_argument("--quick", action="store_true", help="tiny smoke test, NOT for reported results (separate folder)")
    args = ap.parse_args()
    steps = parse_steps(args.step)
    assert all(n in STEP_FUNCS for n in steps), "steps must be between 1 and 10"
    FORCE = args.force
    init_paths(args.quick)
    save_environment(args)
    try:
        for n in steps:
            run_step(n, args)          # a failure stops the pipeline; re-run the same command to resume
    except KeyboardInterrupt:
        sys.exit(130)
