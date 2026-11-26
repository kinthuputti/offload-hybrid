r"""
Measure the ACTUAL compute time of the edge and cloud detectors on this machine.
(Network delay is not measured: it is still the assumed 100 ms in the paper.)

Usage:   python measure_latency.py --root ..\SKAB\data
Put this file in the same folder as offload_experiment.py.
"""
import argparse, statistics, time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import offload_experiment as ox


def median_time(fn, repeats):
    ts = []
    for _ in range(repeats):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="path to SKAB/data")
    a = ap.parse_args()

    rng = np.random.default_rng(0)
    P = [ox.prep(r, 0.5, rng) for r in ox.load_skab(a.root)]
    X = np.vstack([p["feats"] for p in P[4:]]); Y = np.concatenate([p["y"] for p in P[4:]])
    clf = RandomForestClassifier(n_estimators=60, max_depth=12, n_jobs=-1, random_state=0).fit(X, Y)

    p = P[0]; n = p["n"]
    window = p["x_noisy"][n:n + 5]           # the last 5 readings an edge device would hold
    rows = p["x_noisy"][n:n + 1000]
    z = (p["x_noisy"] - p["mu"]) / p["sd"]
    feats1 = p["feats"][n:n + 1]
    feats1000 = p["feats"][n:n + 1000]

    edge = median_time(lambda: ox.edge_score(window, p["mu"], p["sd"]), 2000)
    cloud_one = median_time(lambda: clf.predict_proba(feats1), 200)
    cloud_batch = median_time(lambda: clf.predict_proba(feats1000), 50) / 1000
    feat_batch = median_time(lambda: ox.cloud_features(z[n:n + 1000]), 50) / 1000

    print(f"Edge score for one reading (5-reading window):      {edge * 1000:.4f} ms")
    print(f"Cloud classifier, one reading at a time:            {cloud_one * 1000:.4f} ms")
    print(f"Cloud classifier, per reading in a batch of 1000:   {cloud_batch * 1000:.5f} ms")
    print(f"Cloud feature extraction, per reading in a batch:   {feat_batch * 1000:.5f} ms")
    print("Network delay is NOT measured (assumed 100 ms in the paper).")


if __name__ == "__main__":
    main()
