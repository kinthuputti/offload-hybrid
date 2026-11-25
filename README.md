# Edge-cloud hybrid anomaly detection on noisy sensor streams

Experiment code for the paper. Compares always-edge, always-cloud, and two edge-cloud hybrid
policies on noisy sensor data from SKAB and NAB.

## Setup

```
pip install numpy pandas scipy scikit-learn

git clone --depth 1 https://github.com/waico/SKAB.git
git clone --depth 1 --filter=blob:none --sparse https://github.com/numenta/NAB.git
cd NAB
git sparse-checkout set data labels
cd ..
```

## Run

```
# SKAB, all faults (Table 1)
python offload_experiment.py --dataset skab --root SKAB/data --seeds 5 --out results_skab.csv

# SKAB with faults down-weighted to about 5% (Table 2)
python offload_experiment.py --dataset skab --root SKAB/data --seeds 5 --fault-frac 0.05 --out results_skab_rare.csv

# NAB (Table 3)
python offload_experiment.py --dataset nab --root NAB --seeds 3 --out results_nab.csv
```

Each run prints a summary table and saves the per-run results to the CSV file.
Latency numbers are assumed (1 ms edge, 100 ms per cloud call), not measured; change them with
`--edge-ms` and `--cloud-ms`.

## Notes

- Temperature and Thermocouple are excluded from the SKAB sensors (they drift as the rig warms up).
- The cloud detector is a Random Forest trained on labeled data from other recordings (grouped
  5-fold cross-validation, so it never sees the recording it is tested on).
- `--balanced` (class-weighted Random Forest) was tried only in one exploratory run and is not
  used for any reported result.
