# Edge-cloud hybrid anomaly detection on noisy sensor streams

Experiment code for the paper. Data: SKAB (https://github.com/waico/SKAB).

    git clone --depth 1 https://github.com/waico/SKAB.git
    pip install numpy pandas scipy scikit-learn
    python offload_experiment.py --skab SKAB/data --seeds 5

results.csv holds the per-run results reported in the paper.
