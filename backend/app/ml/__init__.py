"""Razorpay AI Recovery Orchestrator -- ML layer.

Production-oriented pipeline that predicts, for a specific failed payment and a
specific recovery action:

    P(recovery | point-in-time customer/payment context, action)

Sub-packages (each with a single responsibility):

    features/       point-in-time-correct feature construction (train == inference)
    datasets/       training-dataset assembly + action-coverage reporting
    preprocessing/  reproducible sklearn ColumnTransformer
    training/       data-quality / leakage validation, splitting, training driver
    evaluation/     classification + calibration metrics
    models/         candidate model registry + artifact (de)serialization
    inference/      RecoveryModel -- load artifact, predict, explain
    monitoring/     feature baseline, PSI drift, prediction logging

The ML model is a *decision tool*, not the agent. It is consumed through
``app.services.ml_recovery_policy.MLRecoveryPolicy`` (a drop-in
``RecoveryPolicy``), which falls back to ``RulesBasedRecoveryPolicy`` whenever
the artifact is missing or invalid.

WARNING: the current training data is produced by a synthetic generator with a
known outcome process. Metrics on it are a *synthetic benchmark* and must not be
read as real-world predictive performance.
"""
