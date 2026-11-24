"""
Always-edge vs always-cloud vs adaptive offloading on noisy sensor data (SKAB).

EDGE detector : median filter + max |z-score| against the machine's own normal data.
CLOUD detector:
  --cloud pooled : Random Forest trained on labeled data from OTHER machines (files),
                   tested on the held-out machine (grouped 5-fold, no leakage).
  --cloud local  : Isolation Forest trained only on the machine's own first 400 rows.
ADAPTIVE       : score < low -> normal (edge) | score > high -> alarm (edge)
                 otherwise   -> reading is sent to the cloud, which decides.

Usage:
    python offload_experiment.py --skab SKAB/data --seeds 5 --cloud pooled
    python offload_experiment.py --skab SKAB/data --seeds 5 --cloud local
"""
import argparse, glob, os
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import GroupKFold

# Temperature/Thermocouple left out: they warm up slowly in every run and get
# flagged even in clean data. A design choice -- state it in the paper.
SENSORS = ["Accelerometer1RMS", "Accelerometer2RMS", "Current", "Pressure",
           "Voltage", "Volume Flow RateRMS"]
N_TRAIN = 400  # first rows of every SKAB file are normal operation


def load_skab(root, folders=("valve1", "valve2")):
    files = []
    for folder in folders:
        files += sorted(glob.glob(os.path.join(root, folder, "*.csv")))
    out = []
    for f in files:
        df = pd.read_csv(f, sep=";", index_col="datetime", parse_dates=True)
        out.append((f"{os.path.basename(os.path.dirname(f))}/{os.path.basename(f)}", df))
    return out


def add_noise(x, level, rng):
    return x + rng.normal(0.0, 1.0, x.shape) * (level * x.std(axis=0))


def edge_score(x, mu, sd, window=5):
    smoothed = median_filter(x, size=(window, 1), mode="nearest")
    return np.abs((smoothed - mu) / sd).max(axis=1)


def cloud_features(x, filt=9, roll=15):
    """Longer windows: median-filtered values + rolling mean + rolling std."""
    f = median_filter(x, size=(filt, 1), mode="nearest")
    d = pd.DataFrame(f)
    m = d.rolling(roll, min_periods=1).mean().to_numpy()
    s = d.rolling(roll, min_periods=2).std().fillna(0).to_numpy()
    return np.hstack([f, m, s])


def prep(df, noise, rng):
    x_clean = df[SENSORS].to_numpy(float)
    y = df["anomaly"].to_numpy(int)
    mu, sd = x_clean[:N_TRAIN].mean(0), x_clean[:N_TRAIN].std(0) + 1e-9
    x_noisy = add_noise(x_clean, noise, rng)  # noise on the whole recording
    return dict(x_clean=x_clean, x_noisy=x_noisy, y=y, mu=mu, sd=sd,
                feats=cloud_features((x_noisy - mu) / sd))


def evaluate(p, cloud_alarm, qlows, edge_ms, cloud_ms, q_low=0.95, q_high=0.9995):
    """cloud_alarm: boolean array for rows N_TRAIN: of this file.
    Methods:
      adaptive          machine decides alone if clearly normal OR clearly faulty;
                        only the in-between readings go to the cloud (first version)
      confirm@q         machine decides alone ONLY if clearly normal (score below the
                        q-th percentile of its normal training scores); everything
                        else is confirmed by the cloud
    Time is ASSUMED, not measured: edge_ms per reading on the machine, plus
    cloud_ms for every reading sent to the cloud (always-cloud = cloud_ms only)."""
    s_train = edge_score(p["x_clean"][:N_TRAIN], p["mu"], p["sd"])
    t_low, t_high = np.quantile(s_train, q_low), np.quantile(s_train, q_high)
    t_edge = np.quantile(s_train, 0.99)
    s = edge_score(p["x_noisy"], p["mu"], p["sd"])[N_TRAIN:]
    y = p["y"][N_TRAIN:]
    amb = (s >= t_low) & (s <= t_high)
    pred = {"always-cloud": cloud_alarm, "always-edge": s > t_edge,
            "adaptive": np.where(amb, cloud_alarm, s > t_high)}
    sent = {"always-cloud": 1.0, "always-edge": 0.0, "adaptive": amb.mean()}
    for q in qlows:
        t = np.quantile(s_train, q)
        send = s >= t
        pred[f"confirm@{q}"] = np.where(send, cloud_alarm, False)
        sent[f"confirm@{q}"] = send.mean()
    normal = y == 0
    rows = []
    for m, q in pred.items():
        ms = cloud_ms if m == "always-cloud" else edge_ms + sent[m] * cloud_ms
        rows.append(dict(method=m, false_alarm_rate=q[normal].mean(),
                         recall=recall_score(y, q, zero_division=0),
                         f1=f1_score(y, q, zero_division=0),
                         frac_sent_to_cloud=sent[m], est_ms_per_reading=ms))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skab", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--noise", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0])
    ap.add_argument("--cloud", choices=["pooled", "local"], default="pooled")
    ap.add_argument("--qlows", type=float, nargs="+", default=[0.9, 0.95, 0.99],
                    help="'clearly normal' cutoffs (percentile of normal training scores) for confirm@q")
    ap.add_argument("--edge-ms", type=float, default=1.0, help="ASSUMED edge time per reading")
    ap.add_argument("--cloud-ms", type=float, default=100.0, help="ASSUMED cloud round trip per reading")
    ap.add_argument("--out", default="results.csv")
    a = ap.parse_args()

    data = load_skab(a.skab)
    names = [n for n, _ in data]
    print(f"Loaded {len(data)} SKAB files | cloud = {a.cloud}")
    rows = []
    for noise in a.noise:
        for seed in range(a.seeds):
            rng = np.random.default_rng(seed)
            P = [prep(df, noise, rng) for _, df in data]
            alarms = {}
            if a.cloud == "pooled":
                idx = np.arange(len(P))
                for tr, te in GroupKFold(n_splits=5).split(idx, groups=idx):
                    X = np.vstack([P[i]["feats"] for i in tr])
                    Y = np.concatenate([P[i]["y"] for i in tr])
                    clf = RandomForestClassifier(n_estimators=60, max_depth=12,
                                                 n_jobs=-1, random_state=seed).fit(X, Y)
                    for i in te:
                        alarms[i] = clf.predict_proba(P[i]["feats"][N_TRAIN:])[:, 1] > 0.5
            else:
                for i, p in enumerate(P):
                    ftr = cloud_features(p["x_clean"][:N_TRAIN])
                    m = IsolationForest(n_estimators=100, random_state=0).fit(ftr)
                    t = np.quantile(m.score_samples(ftr), 0.01)
                    alarms[i] = m.score_samples(cloud_features(p["x_noisy"])[N_TRAIN:]) < t
            for i, p in enumerate(P):
                for r in evaluate(p, alarms[i], a.qlows, a.edge_ms, a.cloud_ms):
                    r.update(noise=noise, seed=seed, file=names[i])
                    rows.append(r)
        print(f"  noise {noise} done")

    res = pd.DataFrame(rows)
    res.to_csv(a.out, index=False)
    per_seed = res.groupby(["noise", "method", "seed"]).mean(numeric_only=True).reset_index()
    summary = per_seed.groupby(["noise", "method"]).agg(
        false_alarm_rate=("false_alarm_rate", "mean"),
        false_alarm_std=("false_alarm_rate", "std"),
        recall=("recall", "mean"), f1=("f1", "mean"),
        sent_to_cloud=("frac_sent_to_cloud", "mean"),
        est_ms=("est_ms_per_reading", "mean")).round(3)
    print(summary.to_string())
    print(f"\nPer-run results saved to {a.out}")


if __name__ == "__main__":
    main()
