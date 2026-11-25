"""
Always-edge vs always-cloud vs hybrid (edge screens, cloud confirms) on noisy sensor data.

Datasets : --dataset skab (SKAB, 6 sensors)  |  --dataset nab (Numenta Anomaly Benchmark, 1 sensor each)
EDGE     : median filter + max |z-score| against the machine's own normal calibration data.
CLOUD    : Random Forest trained on labeled data from OTHER recordings (grouped 5-fold, no leakage).
Policies : always-edge | always-cloud | adaptive (first hybrid) | confirm@q (second hybrid)
Optional : --fault-frac F  re-weights evaluation so that faults are only a fraction F of the
           evaluated readings (predictions are still computed on the full, continuous series).

Usage:
    git clone --depth 1 https://github.com/waico/SKAB.git
    git clone --depth 1 --filter=blob:none --sparse https://github.com/numenta/NAB.git
    (cd NAB && git sparse-checkout set data labels)
    python offload_experiment.py --dataset skab --root SKAB/data --seeds 5
    python offload_experiment.py --dataset nab  --root NAB       --seeds 5
    python offload_experiment.py --dataset skab --root SKAB/data --seeds 5 --fault-frac 0.05
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import GroupKFold

# SKAB: Temperature/Thermocouple left out (they warm up slowly in every run and are flagged
# even in clean data). A design choice -- state it in the paper.
SKAB_SENSORS = ["Accelerometer1RMS", "Accelerometer2RMS", "Current", "Pressure",
                "Voltage", "Volume Flow RateRMS"]
NAB_DIRS = ["realKnownCause", "realAWSCloudwatch"]


def load_skab(root):
    out = []
    for folder in ("valve1", "valve2"):
        for f in sorted(glob.glob(os.path.join(root, folder, "*.csv"))):
            df = pd.read_csv(f, sep=";", index_col="datetime", parse_dates=True)
            out.append(dict(name=f"{folder}/{os.path.basename(f)}",
                            x=df[SKAB_SENSORS].to_numpy(float),
                            y=df["anomaly"].to_numpy(int), n_train=400))
    return out


def load_nab(root):
    labels = json.load(open(os.path.join(root, "labels", "combined_windows.json")))
    out = []
    for d in NAB_DIRS:
        for f in sorted(glob.glob(os.path.join(root, "data", d, "*.csv"))):
            key = f"{d}/{os.path.basename(f)}"
            df = pd.read_csv(f, parse_dates=["timestamp"])
            ts, y = df["timestamp"], np.zeros(len(df), int)
            for s, e in labels.get(key, []):
                y[((ts >= pd.Timestamp(s)) & (ts <= pd.Timestamp(e))).to_numpy()] = 1
            n_train = min(int(0.15 * len(df)), 5000)  # NAB's 'probationary' period
            out.append(dict(name=key, x=df[["value"]].to_numpy(float), y=y, n_train=n_train))
    return out


def add_noise(x, level, rng):
    return x + rng.normal(0.0, 1.0, x.shape) * (level * x.std(axis=0))


def edge_score(x, mu, sd, window=5):
    sm = median_filter(x, size=(window, 1), mode="nearest")
    return np.abs((sm - mu) / sd).max(axis=1)


def cloud_features(x, filt=9, roll=15):
    f = median_filter(x, size=(filt, 1), mode="nearest")
    d = pd.DataFrame(f)
    m = d.rolling(roll, min_periods=1).mean().to_numpy()
    s = d.rolling(roll, min_periods=2).std().fillna(0).to_numpy()
    return np.hstack([f, m, s])


def prep(rec, noise, rng):
    x_clean, n = rec["x"], rec["n_train"]
    mu, sd = x_clean[:n].mean(0), x_clean[:n].std(0) + 1e-9
    x_noisy = add_noise(x_clean, noise, rng)
    return dict(x_clean=x_clean, x_noisy=x_noisy, y=rec["y"], n=n, mu=mu, sd=sd,
                feats=cloud_features((x_noisy - mu) / sd))


def eval_mask(y, fault_frac, rng):
    """Boolean mask over evaluation rows. None -> keep all. Otherwise keep all normal rows
    and a random subset of fault rows so that faults are ~fault_frac of the kept rows."""
    keep = np.ones(len(y), bool)
    if fault_frac is None:
        return keep
    n_norm, n_anom = (y == 0).sum(), (y == 1).sum()
    if n_anom == 0 or fault_frac * n_norm >= (1 - fault_frac) * n_anom:
        return keep
    q = fault_frac / (1 - fault_frac) * n_norm / n_anom
    keep[(y == 1) & (rng.random(len(y)) > q)] = False
    return keep


def evaluate(p, cloud_alarm, qlows, edge_ms, cloud_ms, fault_frac, rng, q_low=0.95, q_high=0.9995):
    n = p["n"]
    s_train = edge_score(p["x_clean"][:n], p["mu"], p["sd"])
    t_low, t_high = np.quantile(s_train, q_low), np.quantile(s_train, q_high)
    t_edge = np.quantile(s_train, 0.99)
    s = edge_score(p["x_noisy"], p["mu"], p["sd"])[n:]
    y = p["y"][n:]
    keep = eval_mask(y, fault_frac, rng)
    amb = (s >= t_low) & (s <= t_high)
    pred = {"always-cloud": cloud_alarm, "always-edge": s > t_edge,
            "adaptive": np.where(amb, cloud_alarm, s > t_high)}
    sent = {"always-cloud": np.ones(len(s), bool), "always-edge": np.zeros(len(s), bool),
            "adaptive": amb}
    for q in qlows:
        send = s >= np.quantile(s_train, q)
        pred[f"confirm@{q}"] = np.where(send, cloud_alarm, False)
        sent[f"confirm@{q}"] = send
    yk, normal = y[keep], y[keep] == 0
    rows = []
    for m, pr in pred.items():
        prk = pr[keep]
        frac = sent[m][keep].mean()
        ms = cloud_ms if m == "always-cloud" else edge_ms + frac * cloud_ms
        rows.append(dict(method=m,
                         false_alarm_rate=prk[normal].mean() if normal.any() else np.nan,
                         recall=recall_score(yk, prk, zero_division=0) if yk.any() else np.nan,
                         f1=f1_score(yk, prk, zero_division=0),
                         frac_sent_to_cloud=frac, est_ms_per_reading=ms,
                         fault_share=yk.mean()))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["skab", "nab"], required=True)
    ap.add_argument("--root", required=True, help="SKAB: path to SKAB/data | NAB: path to NAB folder")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--noise", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0])
    ap.add_argument("--cloud", choices=["pooled", "local"], default="pooled")
    ap.add_argument("--qlows", type=float, nargs="+", default=[0.9, 0.95, 0.99])
    ap.add_argument("--fault-frac", type=float, default=None)
    ap.add_argument("--balanced", action="store_true",
                    help="class_weight='balanced_subsample' for the cloud Random Forest (helps when faults are rare)")
    ap.add_argument("--edge-ms", type=float, default=1.0, help="ASSUMED edge time per reading")
    ap.add_argument("--cloud-ms", type=float, default=100.0, help="ASSUMED cloud round trip")
    ap.add_argument("--out", default="results.csv")
    a = ap.parse_args()

    data = load_skab(a.root) if a.dataset == "skab" else load_nab(a.root)
    print(f"Loaded {len(data)} {a.dataset.upper()} recordings | cloud = {a.cloud} | fault-frac = {a.fault_frac}")
    rows = []
    for noise in a.noise:
        for seed in range(a.seeds):
            rng = np.random.default_rng(seed)
            P = [prep(r, noise, rng) for r in data]
            alarms = {}
            if a.cloud == "pooled":
                idx = np.arange(len(P))
                for tr, te in GroupKFold(n_splits=5).split(idx, groups=idx):
                    X = np.vstack([P[i]["feats"] for i in tr])
                    Y = np.concatenate([P[i]["y"] for i in tr])
                    clf = RandomForestClassifier(n_estimators=60, max_depth=12, n_jobs=-1, random_state=seed,
                                                 class_weight="balanced_subsample" if a.balanced else None).fit(X, Y)
                    for i in te:
                        alarms[i] = clf.predict_proba(P[i]["feats"][P[i]["n"]:])[:, 1] > 0.5
            else:
                for i, p in enumerate(P):
                    ftr = cloud_features(p["x_clean"][:p["n"]])
                    m = IsolationForest(n_estimators=100, random_state=0).fit(ftr)
                    t = np.quantile(m.score_samples(ftr), 0.01)
                    alarms[i] = m.score_samples(cloud_features(p["x_noisy"])[p["n"]:]) < t
            for i, p in enumerate(P):
                for r in evaluate(p, alarms[i], a.qlows, a.edge_ms, a.cloud_ms, a.fault_frac, rng):
                    r.update(noise=noise, seed=seed, file=data[i]["name"])
                    rows.append(r)
        print(f"  noise {noise} done", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(a.out, index=False)
    per_seed = res.groupby(["noise", "method", "seed"]).mean(numeric_only=True).reset_index()
    summary = per_seed.groupby(["noise", "method"]).agg(
        false_alarm_rate=("false_alarm_rate", "mean"), false_alarm_std=("false_alarm_rate", "std"),
        recall=("recall", "mean"), f1=("f1", "mean"),
        sent_to_cloud=("frac_sent_to_cloud", "mean"), est_ms=("est_ms_per_reading", "mean"),
        fault_share=("fault_share", "mean")).round(3)
    print(summary.to_string())
    print(f"\nPer-run results saved to {a.out}")


if __name__ == "__main__":
    main()
