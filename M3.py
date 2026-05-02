import json
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import *
import warnings
import sys
# For multi-class report (bonus)
from sklearn.metrics import classification_report, accuracy_score

warnings.filterwarnings('ignore')





def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the exact Milestone 1 cleaning procedure to a DataFrame.
    Returns the cleaned DataFrame with Binary_Label_Num added.
    """
    # 1b. List the data fields
    print("Data fields:", df.columns.tolist())
    print("\nColumns dtypes:\n", df.dtypes)
    
    # 1c. Missing data and inf check
    print("\nMissing data check:\n", df.isnull().sum())
    print("Inf data check:", np.isinf(df.select_dtypes(include=['int64', 'float64'])).sum().sum())
    
    # 1d. Correct data types for numeric/string columns
    known_categorical = ['Switch ID', 'Port Number', 'Label', 'Binary Label']
    known_categorical = [c for c in known_categorical if c in df.columns]
    
    numeric_cols = [col for col in df.columns if col not in known_categorical and df[col].dtype == 'object']
    
    changes_made = False
    for column in numeric_cols:
        original_dtype = df[column].dtype
        df[column] = pd.to_numeric(df[column], errors='coerce')
        if df[column].dtype in ['int64', 'float64'] and original_dtype != df[column].dtype:
            print(f"Converted {column} to numeric type: {df[column].dtype}")
            changes_made = True
    
    if not changes_made:
        print("No data type conversions were needed. All data types are correct and unchanged.")
    
    print("\nUpdated data types:\n", df.dtypes)
    
    # 1e. Standardization of Label + Binary Label
    letter_mappings = {
        frozenset(['t','c','p','s','y','n']): 'TCP-SYN',
        frozenset(['d','i','v','e','r','s','i','o','n']): 'DIVERSION',
        frozenset(['p','o','r','t','s','c','a','n']): 'PORTSCAN',
        frozenset(['a','t','t','a','c','k']): 'ATTACK',
        frozenset(['b','l','a','c','k','h','o','l','e']): 'BLACKHOLE',
        frozenset(['o','v','e','r','f','l','o','w']): 'OVERFLOW',
        frozenset(['n','o','r','m','a','l']): 'NORMAL'
    }

    def standardize_text(text):
        if pd.isna(text) or not isinstance(text, str):
            return text
        text = text.lower().strip()
        parts = text.split('-')
        all_letters = frozenset(c for part in parts for c in part if c.isalpha())
        for required_letters, standard_value in letter_mappings.items():
            if all_letters.issubset(required_letters):
                if standard_value == 'TCP-SYN' and any('tcp' in p or 'syn' in p for p in parts):
                    return standard_value
                return standard_value
        return text

    for col in ['Label', 'Binary Label']:
        if col in df.columns:
            df[col] = df[col].apply(standardize_text)

    print("\nUpdated sample of categorical columns:")
    if 'Label' in df.columns and 'Binary Label' in df.columns:
        print(df[['Label', 'Binary Label']].head())
    
    # 1f. Universal inconsistency correction
    print("\n==================================================")
    print("1f. UNIVERSAL NEGATIVE VALUE CORRECTION")
    print("==================================================")

    dup = df.duplicated().sum()
    print(f"Duplicate rows: {dup}")
    if dup:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"Removed {dup} duplicates (keeping first). New shape: {df.shape}")
    else:
        print("No duplicates found.")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    true_neg_total = 0
    sentinel_total = 0

    for col in numeric_cols:
        true_neg = ((df[col] < 0) & (df[col] != -1)).sum()
        sentinel = (df[col] == -1).sum()
        if true_neg or sentinel:
            print(f"\nColumn: {col}")
            if true_neg:
                print(f"  {true_neg} true negatives (<0, != -1) -> replaced with 0")
                true_neg_total += true_neg
            if sentinel:
                print(f"  {sentinel} sentinel values (-1) -> replaced with NaN")
                sentinel_total += sentinel
        df[col] = df[col].apply(lambda x: 0 if (x < 0 and x != -1) else (np.nan if x == -1 else x))

    print("\nSummary:")
    print(f"  Total true negatives fixed: {true_neg_total}")
    print(f"  Total -1 sentinels converted to NaN: {sentinel_total}")
    print("==================================================")
    
    # 1g. Unique Categories
    categorical_cols = df.select_dtypes(include=['object', 'bool']).columns

    print("\n=== Task 1g: Unique Category Counts ===")
    for col in categorical_cols:
        print(f"- {col}: {df[col].nunique()} unique values")
    print("=== End of Task 1g ===")
    
    # Add numeric binary label (0 = NORMAL, 1 = ATTACK)
    if 'Binary Label' in df.columns:
        df['Binary_Label_Num'] = (df['Binary Label'] == 'ATTACK').astype(int)
        print("Added 'Binary_Label_Num' column (0 = NORMAL, 1 = ATTACK)")
    
    return df

# ======================================================================
# LOAD DATA AND MODELS
# ======================================================================

print("==============================================================================")
print("LOADING TEST DATA")
print("==============================================================================")
raw_test_df = pd.read_csv('test.csv')
print(f"Raw test data loaded: {raw_test_df.shape}")

test_df = clean_dataset(raw_test_df.copy())
print(f"Cleaned test data: {test_df.shape}")

print("\n==============================================================================")
print("LOADING MODELS FROM MILESTONE 2")
print("==============================================================================")

print("Loading PMF models...")
with open('milestone2_pmf_models_per_attack.json', 'r') as f:
    pmf_models = json.load(f)
print("PMF models loaded successfully!")

print("Loading PDF models...")
with open('milestone2_pdf_models_per_attack.json', 'r') as f:
    pdf_models = json.load(f)
print("PDF models loaded successfully!")

# Feature names
discrete_features = list(pmf_models.keys())
continuous_features = list(pdf_models.keys())

print(f"\nDiscrete features: {len(discrete_features)}")
print(f"Continuous features: {len(continuous_features)}")
print(f"Test samples: {len(test_df)}")

common_discrete = [f for f in discrete_features if f in test_df.columns]
common_continuous = [f for f in continuous_features if f in test_df.columns]

print(f"Using {len(common_continuous)} continuous and {len(common_discrete)} discrete features")

# ======================================================================
# MILESTONE 3 TASK 1: FROM-SCRATCH NAIVE BAYES (BINARY)
# ======================================================================

print("\n==============================================================================")
print("MILESTONE 3 TASK 1: FROM-SCRATCH NAIVE BAYES (BINARY CLASSIFICATION)")
print("==============================================================================")

def get_dist(dist_name, params):
    try:
        return getattr(stats, dist_name)(*params)
    except:
        return None

# STEP 1: PRIORS
print("\n--- Calculating Priors from Training Data ---")
train_raw = pd.read_csv('Train_data_updated.csv')
train_df_temp = clean_dataset(train_raw.copy())

prior_attack = train_df_temp['Binary_Label_Num'].mean()
prior_normal = 1 - prior_attack

print(f"\nPrior P(Attack) = {prior_attack:.4f}")
print(f"Prior P(Normal) = {prior_normal:.4f}")

del train_df_temp, train_raw

# STEP 2: BINARY CLASSIFICATION
print("\n--- Running Binary Classification on Test Data ---")

binary_predictions = []
prob_attack_list = []
prob_normal_list = []

for idx, row in test_df.iterrows():
    if idx % 1000 == 0:
        print(f"Processing row {idx}/{len(test_df)}")
    
    log_prob_attack = np.log(prior_attack + 1e-10)
    
    for f in common_discrete:
        val = str(row[f])
        prob = pmf_models[f]["PMF_Attack"].get(val, 1e-10)
        log_prob_attack += np.log(prob)
    
    for f in common_continuous:
        val = row[f]
        if pd.isna(val):
            continue
        model = pdf_models[f].get("Attack", pdf_models[f].get("Overall"))
        if model:
            dist = get_dist(model['distribution'], model['params'])
            if dist:
                pdf_val = dist.pdf(val)
                log_prob_attack += np.log(max(pdf_val, 1e-10))

    log_prob_normal = np.log(prior_normal + 1e-10)
    
    for f in common_discrete:
        val = str(row[f])
        prob = pmf_models[f]["PMF_Normal"].get(val, 1e-10)
        log_prob_normal += np.log(prob)
    
    for f in common_continuous:
        val = row[f]
        if pd.isna(val):
            continue
        model = pdf_models[f].get("Normal", pdf_models[f].get("Overall"))
        if model:
            dist = get_dist(model['distribution'], model['params'])
            if dist:
                pdf_val = dist.pdf(val)
                log_prob_normal += np.log(max(pdf_val, 1e-10))

    log_sum = np.logaddexp(log_prob_attack, log_prob_normal)
    prob_attack = np.exp(log_prob_attack - log_sum)
    prob_normal = np.exp(log_prob_normal - log_sum)
    
    prob_attack_list.append(prob_attack)
    prob_normal_list.append(prob_normal)

    binary_predictions.append(1 if log_prob_attack > log_prob_normal else 0)

test_df['Predicted_Binary'] = binary_predictions
test_df['Prob_Attack'] = prob_attack_list
test_df['Prob_Normal'] = prob_normal_list

# ======================================================================
# STEP 3: METRICS
# ======================================================================

print("\n--- Calculating Performance Metrics ---")

y_true = test_df['Binary_Label_Num'].values
y_pred = test_df['Predicted_Binary'].values

TP = np.sum((y_pred == 1) & (y_true == 1))
TN = np.sum((y_pred == 0) & (y_true == 0))
FP = np.sum((y_pred == 1) & (y_true == 0))
FN = np.sum((y_pred == 0) & (y_true == 1))

total = len(y_true)

accuracy = (TP + TN) / total
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print("==============================================================================")
print("BINARY CLASSIFICATION RESULTS (FROM-SCRATCH NAIVE BAYES)")
print("==============================================================================")

print("\nCONFUSION MATRIX:")
print("                  Predicted Normal | Predicted Attack")
print(f"Actual Normal     {TN:15} | {FP:15}")
print(f"Actual Attack     {FN:15} | {TP:15}")

print("\nDETAILED BREAKDOWN:")
print(f"True Positives (TP):  {TP}")
print(f"True Negatives (TN):  {TN}")
print(f"False Positives (FP): {FP}")
print(f"False Negatives (FN): {FN}")

print("\n==============================================================================")
print("PERFORMANCE METRICS (MILESTONE 3)")
print("==============================================================================")

print(f"\nAccuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"F1-Score:  {f1_score:.4f}")

print("\n==============================================================================")
print("SAMPLE PREDICTIONS (FIRST 10 ROWS)")
print("==============================================================================")

sample_df = test_df[['Binary Label', 'Predicted_Binary', 'Prob_Attack', 'Prob_Normal']].head(10).copy()
sample_df['Predicted_Label'] = sample_df['Predicted_Binary'].map({0: 'NORMAL', 1: 'ATTACK'})
print(sample_df[['Binary Label', 'Predicted_Label', 'Prob_Attack', 'Prob_Normal']].to_string())

print("\nMILESTONE 3 TASK 1 COMPLETE!")

# ======================================================================
# BONUS: MULTI CLASS CLASSIFICATION
# ======================================================================

print("\n==============================================================================")
print("BONUS: MULTI-CLASS CLASSIFICATION (6 Attack Types)")
print("==============================================================================")

attack_types = ['NORMAL', 'TCP-SYN', 'BLACKHOLE', 'DIVERSION', 'OVERFLOW', 'PORTSCAN']

attack_to_pmf_key = {
    'NORMAL': 'PMF_NORMAL',
    'TCP-SYN': 'PMF_TCP_SYN',
    'BLACKHOLE': 'PMF_BLACKHOLE',
    'DIVERSION': 'PMF_DIVERSION',
    'OVERFLOW': 'PMF_OVERFLOW',
    'PORTSCAN': 'PMF_PORTSCAN'
}

attack_to_pdf_key = {
    'NORMAL': 'NORMAL',
    'TCP-SYN': 'TCP_SYN',
    'BLACKHOLE': 'BLACKHOLE',
    'DIVERSION': 'DIVERSION',
    'OVERFLOW': 'OVERFLOW',
    'PORTSCAN': 'PORTSCAN'
}

multi_predictions = []

for idx, row in test_df.iterrows():
    if idx % 1000 == 0:
        print(f"Processing row {idx}/{len(test_df)} (multi-class)")
    
    log_probs = {}
    
    for attack in attack_types:
        log_prob = 0.0
        
        pmf_key = attack_to_pmf_key[attack]
        for f in common_discrete:
            val = str(row[f])
            prob = pmf_models[f][pmf_key].get(val, 1e-10)
            log_prob += np.log(prob)
        
        pdf_key = attack_to_pdf_key[attack]
        for f in common_continuous:
            val = row[f]
            if pd.isna(val):
                continue
            model = pdf_models[f].get(pdf_key, pdf_models[f].get("Overall"))
            if model:
                dist = get_dist(model['distribution'], model['params'])
                if dist:
                    pdf_val = dist.pdf(val)
                    log_prob += np.log(max(pdf_val, 1e-10))
        
        log_probs[attack] = log_prob
    
    predicted = max(log_probs, key=log_probs.get)
    multi_predictions.append(predicted)

test_df['Predicted_Multi_Class'] = multi_predictions

print("\n--- Multi-Class Classification Report ---")
print(classification_report(test_df['Label'], test_df['Predicted_Multi_Class'], digits=4))

multi_acc = accuracy_score(test_df['Label'], test_df['Predicted_Multi_Class'])
print(f"\nMulti-Class Accuracy: {multi_acc:.4f} ({multi_acc*100:.2f}%)")

print("\n--- Per-Attack-Type Accuracy ---")
for attack in attack_types:
    mask = test_df['Label'] == attack
    if mask.sum() > 0:
        correct = (test_df.loc[mask, 'Label'] == test_df.loc[mask, 'Predicted_Multi_Class']).sum()
        acc = correct / mask.sum()
        print(f"{attack:12}: {acc:.4f} ({correct}/{mask.sum()} correct)")

test_df.to_csv("milestone3_task1_predictions.csv", index=False)
print("\nAll predictions saved to milestone3_task1_predictions.csv")
print("MILESTONE 3 TASK 1 COMPLETE (BINARY + MULTI-CLASS)")





# ============================================================
# MILESTONE 3 TASK 2: SCIKIT-LEARN NAIVE BAYES
# Train on Train_data_updated.csv, Test on test.csv
# ============================================================

print("\n" + "="*80)
print("MILESTONE 3 TASK 2: SCIKIT-LEARN NAIVE BAYES MODELS")
print("="*80)

from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# -----------------------------
# STEP 1: LOAD AND PREPARE TRAINING DATA
# -----------------------------
print("\n--- Loading and Preparing Training Data ---")
train_raw = pd.read_csv('Train_data_updated.csv')
train_df = clean_dataset(train_raw.copy())
print(f"Training data shape: {train_df.shape}")

# Use same features as test (common ones already defined)
X_train_continuous = train_df[common_continuous].fillna(0)
X_train_discrete = train_df[common_discrete]

print(f"Training continuous features: {X_train_continuous.shape}")
print(f"Training discrete features: {X_train_discrete.shape}")

# -----------------------------
# STEP 2: ONE-HOT ENCODE DISCRETE FEATURES (FIT ON TRAIN ONLY)
# -----------------------------
print("\n--- One-Hot Encoding Discrete Features ---")
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoder.fit(X_train_discrete)

X_train_discrete_encoded = pd.DataFrame(
    encoder.transform(X_train_discrete),
    columns=encoder.get_feature_names_out(common_discrete),
    index=train_df.index
)

print(f"One-hot encoded discrete features: {X_train_discrete_encoded.shape}")

# Combine continuous and encoded discrete features
X_train = pd.concat([X_train_continuous, X_train_discrete_encoded], axis=1)
y_train = train_df['Binary_Label_Num']

print(f"Final training feature matrix: {X_train.shape}")
print(f"Training labels: {y_train.shape}")
print(f"  Normal (0): {(y_train == 0).sum()}")
print(f"  Attack (1): {(y_train == 1).sum()}")

# -----------------------------
# STEP 3: PREPARE TEST DATA (TRANSFORM ONLY, NO FITTING)
# -----------------------------
print("\n--- Preparing Test Data ---")
X_test_continuous = test_df[common_continuous].fillna(0)
X_test_discrete = test_df[common_discrete]

# Transform test data using fitted encoder (DO NOT FIT AGAIN)
X_test_discrete_encoded = pd.DataFrame(
    encoder.transform(X_test_discrete),
    columns=encoder.get_feature_names_out(common_discrete),
    index=test_df.index
)

X_test = pd.concat([X_test_continuous, X_test_discrete_encoded], axis=1)
y_test = test_df['Binary_Label_Num']

print(f"Final test feature matrix: {X_test.shape}")
print(f"Test labels: {y_test.shape}")
print(f"  Normal (0): {(y_test == 0).sum()}")
print(f"  Attack (1): {(y_test == 1).sum()}")

# -----------------------------
# STEP 4: TRAIN AND EVALUATE NAIVE BAYES MODELS
# -----------------------------
print("\n" + "="*80)
print("TRAINING AND EVALUATING SCIKIT-LEARN NAIVE BAYES MODELS")
print("="*80)

# Define models
models = {
    "GaussianNB": GaussianNB(),
    "MultinomialNB": MultinomialNB(),
    "BernoulliNB": BernoulliNB()
}

results = {}

for name, model in models.items():
    print("\n" + "-"*80)
    print(f"TRAINING {name}")
    print("-"*80)
    
    # Train the model
    model.fit(X_train, y_train)
    print(f"Model trained on {len(X_train)} samples")
    
    # Make predictions on test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    accuracy = accuracy_score(y_test, y_pred)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Store results
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn
    }
    
    # Print results
    print(f"\nCONFUSION MATRIX:")
    print("                  Predicted Normal | Predicted Attack")
    print(f"Actual Normal     {tn:15} | {fp:15}")
    print(f"Actual Attack     {fn:15} | {tp:15}")
    
    print(f"\nPERFORMANCE METRICS:")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score:  {f1:.4f}")
    
    print(f"\nCLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred, digits=4, target_names=['Normal', 'Attack']))

# -----------------------------
# STEP 5: COMPARE ALL MODELS
# -----------------------------
print("\n" + "="*80)
print("COMPARISON OF ALL SCIKIT-LEARN MODELS")
print("="*80)

print("\nSUMMARY TABLE:")
print("-"*80)
print(f"{'Model':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
print("-"*80)
for name, metrics in results.items():
    print(f"{name:<15} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} {metrics['f1']:<12.4f}")
print("-"*80)

# Find best model
best_accuracy_model = max(results.items(), key=lambda x: x[1]['accuracy'])
best_precision_model = max(results.items(), key=lambda x: x[1]['precision'])
best_recall_model = max(results.items(), key=lambda x: x[1]['recall'])
best_f1_model = max(results.items(), key=lambda x: x[1]['f1'])

print(f"\nBEST MODELS BY METRIC:")
print(f"  Best Accuracy:  {best_accuracy_model[0]} ({best_accuracy_model[1]['accuracy']:.4f})")
print(f"  Best Precision: {best_precision_model[0]} ({best_precision_model[1]['precision']:.4f})")
print(f"  Best Recall:    {best_recall_model[0]} ({best_recall_model[1]['recall']:.4f})")
print(f"  Best F1-Score:  {best_f1_model[0]} ({best_f1_model[1]['f1']:.4f})")

# -----------------------------
# STEP 6: COMPARE WITH TASK 1 FROM-SCRATCH MODEL
# -----------------------------
print("\n" + "="*80)
print("COMPARISON: SCIKIT-LEARN vs FROM-SCRATCH NAIVE BAYES")
print("="*80)

# Get Task 1 results (already calculated)
task1_accuracy = accuracy_score(y_test, test_df['Predicted_Binary'])
task1_y_pred = test_df['Predicted_Binary'].values
task1_tn, task1_fp, task1_fn, task1_tp = confusion_matrix(y_test, task1_y_pred).ravel()
task1_precision = task1_tp / (task1_tp + task1_fp) if (task1_tp + task1_fp) > 0 else 0
task1_recall = task1_tp / (task1_tp + task1_fn) if (task1_tp + task1_fn) > 0 else 0
task1_f1 = 2 * (task1_precision * task1_recall) / (task1_precision + task1_recall) if (task1_precision + task1_recall) > 0 else 0

print("\nCOMPARATIVE RESULTS:")
print("-"*80)
print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
print("-"*80)
print(f"{'From-Scratch NB':<20} {task1_accuracy:<12.4f} {task1_precision:<12.4f} {task1_recall:<12.4f} {task1_f1:<12.4f}")
for name, metrics in results.items():
    print(f"{name:<20} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} {metrics['f1']:<12.4f}")
print("-"*80)

# -----------------------------
# STEP 7: ANALYSIS AND RECOMMENDATIONS
# -----------------------------
print("\n" + "="*80)
print("ANALYSIS AND RECOMMENDATIONS")
print("="*80)

print("\nKEY FINDINGS:")

# Which sklearn model is best?
overall_best = max(results.items(), key=lambda x: x[1]['f1'])
print(f"\n1. Best Scikit-Learn Model: {overall_best[0]}")
print(f"   - Achieves F1-Score of {overall_best[1]['f1']:.4f}")
print(f"   - Accuracy: {overall_best[1]['accuracy']:.4f}")

# Compare with from-scratch
if overall_best[1]['f1'] > task1_f1:
    diff = (overall_best[1]['f1'] - task1_f1) * 100
    print(f"\n2. Scikit-Learn models outperform from-scratch implementation")
    print(f"   - {overall_best[0]} has {diff:.2f}% higher F1-Score")
elif task1_f1 > overall_best[1]['f1']:
    diff = (task1_f1 - overall_best[1]['f1']) * 100
    print(f"\n2. From-Scratch implementation outperforms Scikit-Learn models")
    print(f"   - From-scratch has {diff:.2f}% higher F1-Score")
else:
    print(f"\n2. From-Scratch and Scikit-Learn models perform similarly")

# Model characteristics
print(f"\n3. Model Characteristics:")
print(f"   - GaussianNB: Assumes features follow normal distribution")
print(f"   - MultinomialNB: Good for discrete/count features")
print(f"   - BernoulliNB: Good for binary features")

# Metric importance
print(f"\n4. Metric Importance for Anomaly Detection:")
print(f"   - Recall is crucial: Missing attacks (FN) is costly")
print(f"   - Precision matters: Too many false alarms (FP) reduce trust")
print(f"   - F1-Score balances both concerns")

print("\nRECOMMENDED MODEL:")
if recall > 0.9:
    best_for_task = max(results.items(), key=lambda x: x[1]['recall'])
    print(f"  {best_for_task[0]} - Prioritizes detecting attacks (high recall)")
else:
    print(f"  {overall_best[0]} - Best overall balance of metrics")

# Save sklearn predictions
best_model_name = overall_best[0]
best_model = models[best_model_name]
test_df[f'Predicted_{best_model_name}'] = best_model.predict(X_test)

test_df.to_csv("milestone3_task2_predictions.csv", index=False)
print(f"\nPredictions saved to milestone3_task2_predictions.csv")
print("MILESTONE 3 TASK 2 COMPLETE!")