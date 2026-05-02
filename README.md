Network Intrusion Detection using Probabilistic Machine LearningThis project builds a multi-milestone machine learning pipeline for detecting network intrusions in Software-Defined Networking (SDN) environments, using statistical modeling and Naive Bayes classification

📁 Project Structure
The project is divided into three milestones, each building on the previous:
Milestone 1 – Data Cleaning & Preprocessing
Covers loading the raw network traffic dataset, correcting data types, standardizing label formats (e.g., TCP-SYN, BLACKHOLE, PORTSCAN), removing duplicates, handling negative/sentinel values, and encoding binary labels (NORMAL vs ATTACK).
Milestone 2 (updatedM2.py) – Statistical Modeling
Fits probability distributions to the training data, split by attack type:

PDF Fitting – Tries a broad set of continuous distributions (via SciPy) per feature and per attack type, selecting the best fit by MSE. Saves fitted models to JSON and generates histogram plots.
PMF Fitting – Computes probability mass functions for discrete features across overall, normal, all-attack, and per-attack-type subsets. Exports models to both JSON and CSV.
Z-Score Anomaly Detection – Sweeps thresholds to evaluate accuracy, precision, and recall, saving a performance plot.

Milestone 3 (M3.py) – Naive Bayes Classification

Task 1 – From-Scratch Naive Bayes (Binary): Implements a full Naive Bayes classifier from scratch using the PDF/PMF models generated in Milestone 2, computing log-likelihoods per feature and classifying traffic as NORMAL or ATTACK.
Task 2 – Scikit-learn Naive Bayes Comparison: Trains and evaluates GaussianNB, MultinomialNB, and BernoulliNB on the same dataset, compares them against the from-scratch implementation using accuracy, precision, recall, and F1-score.

📊 Outputs

Trained probability models in JSON/CSV format
PNG plots for Z-score performance, PDF fits, and PMF comparisons (overall and per attack type)
milestone3_task2_predictions.csv – Final predictions on the test set
