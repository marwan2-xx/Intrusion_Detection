import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import scipy.stats as st
from scipy.stats._continuous_distns import _distn_names
import json

warnings.filterwarnings('ignore')

# ============================================================
# 1. LOAD & CLEAN DATA (Same as Milestone 1)
# ============================================================

df = pd.read_csv('Train_data_updated.csv')
print(f"Loaded data: {df.shape}")

# --- Cleaning steps (exactly like before) ---
categorical_cols = ['Switch ID', 'Port Number', 'Label', 'Binary Label']
numeric_cols = [col for col in df.columns if col not in categorical_cols and df[col].dtype == 'object']

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Standardize labels
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
    for required, standard in letter_mappings.items():
        if all_letters.issubset(required):
            if standard == 'TCP-SYN' and any('tcp' in p or 'syn' in p for p in parts):
                return standard
            return standard
    return text.upper()

for col in ['Label', 'Binary Label']:
    df[col] = df[col].apply(standardize_text)

# Fix negative values
dup = df.duplicated().sum()
if dup:
    df = df.drop_duplicates().reset_index(drop=True)

for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].apply(lambda x: 0 if (x < 0 and x != -1) else (np.nan if x == -1 else x))

df_copy = df.copy()
print("Data cleaned and ready!")

# ============================================================
# 2. Z-SCORE TASK (Task 1)
# ============================================================

def task1_zscore_anomaly_with_plot(df_input):
    df_work = df_input.copy()
    y = (df_work['Binary Label'] == 'ATTACK').astype(int)
    X = df_work.drop(columns=['Label', 'Binary Label'], errors='ignore').select_dtypes(include=[np.number])
    constant_cols = [c for c in X.columns if X[c].std() == 0]
    X = X.drop(columns=constant_cols)

    np.random.seed(42)
    idx = np.random.permutation(len(X))
    split = int(0.7 * len(X))
    X_train, X_test = X.iloc[idx[:split]], X.iloc[idx[split:]]
    y_test = y.iloc[idx[split:]]

    mu = X_train.mean()
    sigma = X_train.std().replace(0, 1)
    z_scores = ((X_test - mu) / sigma).abs().max(axis=1)

    thresholds = np.arange(0.5, 4.1, 0.25)
    results = []
    for th in thresholds:
        pred = (z_scores > th).astype(int)
        tp = ((pred == 1) & (y_test == 1)).sum()
        fp = ((pred == 1) & (y_test == 0)).sum()
        tn = ((pred == 0) & (y_test == 0)).sum()
        fn = ((pred == 0) & (y_test == 1)).sum()
        acc = (tp + tn) / len(y_test)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        results.append({"Threshold": th, "Accuracy": acc, "Precision": prec, "Recall": rec})

    results_df = pd.DataFrame(results)
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['Threshold'], results_df['Accuracy'], 'o-', label='Accuracy')
    plt.plot(results_df['Threshold'], results_df['Precision'], 's-', label='Precision')
    plt.plot(results_df['Threshold'], results_df['Recall'], '^-', label='Recall')
    plt.axvline(results_df.loc[results_df['Recall'].idxmax(), 'Threshold'], color='red', linestyle='--')
    plt.title('Z-Score Anomaly Detection Performance')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('task1_zscore_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Z-score plot saved!")
    return results_df

# ============================================================
# 3. PDF FITTING
# ============================================================

# ============================================================
# MODIFIED: PDF FITTING WITH PER-ATTACK-TYPE STORAGE
# ============================================================

def generate_pdf_plots_with_per_attack():
    """
    Modified PDF fitting that saves models for EACH attack type separately
    """
    print("\n" + "="*80)
    print("STARTING PDF FITTING WITH PER-ATTACK-TYPE STORAGE")
    print("="*80)
    
    global DISTRIBUTIONS
    exclude_dists = {
        'levy_stable', 'studentized_range',
        'norm', 'anglit', 'cauchy', 'cosine', 'dgamma', 'dweibull', 'exponnorm',
        'truncweibull_min', 'weibull_max', 'genlogistic', 'gennorm', 'genpareto',
        'genhalflogistic', 'gumbel_l', 'gumbel_r', 'halfnorm', 'laplace', 'laplace_asymmetric',
        'logistic', 'maxwell', 'mielke', 'nakagami', 'ncf', 'nct', 'ncx2',
        'pearson3', 'powerlaw', 'powerlognorm', 'powernorm', 'rdist', 'recipinvgauss',
        'reciprocal', 'rice', 'skewcauchy', 'trapz', 'triang', 'truncnorm',
        'tukeylambda', 'uniform', 'vonmises', 'vonmises_line', 'wrapcauchy',
    }
    
    DISTRIBUTIONS = [getattr(st, d) for d in _distn_names if d not in exclude_dists]
    print(f"Trying {len(DISTRIBUTIONS)} distributions\n")
    
    def safe_filename(name):
        return name.replace('/', 'per').replace(' ', '').replace('\\', '').replace(':', '').replace('-', '_')
    
    def best_fit_distribution(data, bins=80, ax=None):
        y, x = np.histogram(data, bins=bins, density=True)
        x = (x[:-1] + x[1:]) / 2.0
        fits = []
        n_bins = len(y)
        for dist in DISTRIBUTIONS:
            try:
                params = dist.fit(data)
                pdf = dist.pdf(x, *params[:-2], loc=params[-2], scale=params[-1])
                sse = np.sum((y - pdf)**2)
                mse = sse / n_bins if n_bins > 0 else sse
                if ax is not None:
                    ax.plot(x, pdf, alpha=0.5, lw=0.8)
                fits.append((dist, params, mse))
            except:
                pass
        return sorted(fits, key=lambda x: x[2])
    
    def plot_column(series, col_name, suffix="", condition_name=""):
        data = series.dropna()
        if len(data) < 50:
            print(f"   SKIPPED: {col_name}{suffix} (only {len(data)} points)")
            return None
        
        name = safe_filename(col_name + suffix)
        print(f"   -> {col_name}{suffix} ({len(data)} points)")
        
        plt.figure(figsize=(16, 10))
        ax = plt.gca()
        ax.hist(data, bins=80, density=True, alpha=0.6, color='lightblue', 
                edgecolor='white', label='Data', zorder=5)
        ylim = ax.get_ylim()
        
        fits = best_fit_distribution(data, bins=80, ax=ax)
        ax.set_ylim(ylim)
        
        if not fits:
            print(f"   NO FITS FOUND for {col_name}{suffix}")
            plt.close()
            return None
        
        best_dist, best_params, best_mse = fits[0]
        
        y_h, edges = np.histogram(data, bins=80, density=True)
        x_mid = (edges[:-1] + edges[1:]) / 2
        pdf_best = best_dist.pdf(x_mid, *best_params[:-2], loc=best_params[-2], scale=best_params[-1])
        ax.plot(x_mid, pdf_best, color='red', lw=4, label=f'BEST: {best_dist.name.upper()}', zorder=10)
        
        shapes = best_dist.shapes.split(', ') if best_dist.shapes else []
        names = shapes + ['loc', 'scale']
        param_str = ', '.join([f'{n}={v:.4g}' for n, v in zip(names, best_params)])
        
        ax.set_title(f'{col_name} - {condition_name}\n'
                    f'Best: {best_dist.name.upper()}({param_str}) | MSE={best_mse:.2e}',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel(col_name, fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{name}_rainbow_spaghetti.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"      BEST: {best_dist.name.upper()} | MSE={best_mse:.2e}")
        
        return {'dist': best_dist.name, 'params': best_params, 'mse': best_mse}
    
    df_work = df_copy.copy()
    if df_work['Binary Label'].dtype == 'object':
        df_work['Binary Label'] = (df_work['Binary Label'] == 'ATTACK').astype(int)
    
    cols = [c for c in df_work.select_dtypes(include=np.number).columns 
            if df_work[c].nunique() > 50 and c not in ['Binary Label']]
    
    print(f"\nFound {len(cols)} continuous columns to analyze\n")
    
    pdf_summary = []
    pdf_models = {}
    
    # Get all unique attack types
    attack_types = df_work['Label'].dropna().unique().tolist()
    print(f"Found {len(attack_types)} attack types: {attack_types}\n")
    
    for idx, col in enumerate(cols, 1):
        print(f"\n[{idx}/{len(cols)}] Processing: {col}")
        print("-" * 80)
        
        col_results = {'Column': col}
        col_models = {}
        
        # OVERALL
        result = plot_column(df_work[col], col, "_OVERALL", "Overall Data")
        if result:
            col_results['Best_Overall'] = result['dist']
            col_results['MSE_Overall'] = f"{result['mse']:.6e}"
            col_models['Overall'] = {
                'distribution': result['dist'],
                'params': [float(p) for p in (result['params'].tolist() if isinstance(result['params'], np.ndarray) else list(result['params']))],
                'mse': float(result['mse'])
            }
        
        # NORMAL
        normal_data = df_work[df_work['Binary Label'] == 0]
        if len(normal_data) > 100:
            result = plot_column(normal_data[col], col, "_NORMAL", "Normal Traffic")
            if result:
                col_results['Best_Normal'] = result['dist']
                col_results['MSE_Normal'] = f"{result['mse']:.6e}"
                col_models['Normal'] = {
                    'distribution': result['dist'],
                    'params': [float(p) for p in (result['params'].tolist() if isinstance(result['params'], np.ndarray) else list(result['params']))],
                    'mse': float(result['mse'])
                }
        
        # ALL ATTACKS COMBINED
        attack_data = df_work[df_work['Binary Label'] == 1]
        if len(attack_data) > 100:
            result = plot_column(attack_data[col], col, "_ATTACK", "All Attacks Combined")
            if result:
                col_results['Best_Attack'] = result['dist']
                col_results['MSE_Attack'] = f"{result['mse']:.6e}"
                col_models['Attack'] = {
                    'distribution': result['dist'],
                    'params': [float(p) for p in (result['params'].tolist() if isinstance(result['params'], np.ndarray) else list(result['params']))],
                    'mse': float(result['mse'])
                }
        
        #  NEW: PER ATTACK TYPE
        for attack in attack_types:
            attack_subset = df_work[df_work['Label'] == attack]
            if len(attack_subset) < 100:
                continue
            
            attack_name_safe = safe_filename(attack.upper().replace(' ', '_'))
            result = plot_column(attack_subset[col], col, 
                               f"_{attack_name_safe}", 
                               f"{attack.upper()} Attack")
            
            if result:
                # Add to summary
                col_results[f'Best_{attack_name_safe}'] = result['dist']
                col_results[f'MSE_{attack_name_safe}'] = f"{result['mse']:.6e}"
                
                # Add to models
                col_models[attack_name_safe] = {
                    'distribution': result['dist'],
                    'params': [float(p) for p in (result['params'].tolist() if isinstance(result['params'], np.ndarray) else list(result['params']))],
                    'mse': float(result['mse'])
                }
        
        pdf_summary.append(col_results)
        if col_models:
            pdf_models[col] = col_models
    
    # Print summary
    print("\n" + "="*80)
    print("PDF SUMMARY TABLE")
    print("="*80)
    summary_df = pd.DataFrame(pdf_summary)
    print(summary_df.to_string(index=False))
    
    # Save to JSON
    import json
    
    def convert_to_native(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return [convert_to_native(item) for item in obj.tolist()]
        elif isinstance(obj, dict):
            return {key: convert_to_native(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        else:
            return obj
    
    pdf_models_serializable = convert_to_native(pdf_models)
    
    with open('milestone2_pdf_models_per_attack.json', 'w') as f:
        json.dump(pdf_models_serializable, f, indent=2)
    print(f"\n Saved PDF models (with per-attack types) to: milestone2_pdf_models_per_attack.json")
    
    # Save to CSV
    pdf_csv_rows = []
    for col, col_models in pdf_models.items():
        for condition in col_models.keys():
            model = col_models[condition]
            params_str = ', '.join([f'{p:.6f}' for p in model['params']])
            pdf_csv_rows.append({
                'Column': col,
                'Condition': condition,
                'Distribution': model['distribution'],
                'Parameters': params_str,
                'MSE': model['mse']
            })
    
    if pdf_csv_rows:
        pdf_csv_df = pd.DataFrame(pdf_csv_rows)
        pdf_csv_df.to_csv('milestone2_pdf_models_per_attack.csv', index=False)
        print(f" Saved PDF models (with per-attack types) to: milestone2_pdf_models_per_attack.csv")
    
    print("\n" + "="*80)
    print("PDF ANALYSIS COMPLETE - Saved with per-attack-type breakdowns!")
    print("="*80)
    
    return summary_df, pdf_models


# ============================================================
# MODIFIED: PMF PLOTS WITH PER-ATTACK-TYPE STORAGE
# ============================================================

def generate_pmf_plots_with_per_attack():
    """
    Modified PMF that saves probabilities for EACH attack type separately
    """
    print("\n" + "="*80)
    print("PMF ANALYSIS WITH PER-ATTACK-TYPE STORAGE")
    print("="*80)
    
    df_work = df_copy.copy()
    
    if df_work['Binary Label'].dtype == 'object':
        df_work['Binary Label'] = (df_work['Binary Label'] == 'ATTACK').astype(int)
    
    discrete_cols = [
        c for c in df_work.columns
        if df_work[c].nunique() <= 20
        and c not in ['Binary Label', 'Label']
        and not c.startswith('Attack_')
        and not df_work[c].isna().all()
    ]
    print(f"Found {len(discrete_cols)} discrete columns: {discrete_cols}\n")
    
    def safe_filename(name):
        return "".join(c if c.isalnum() or c == "_" else "_" for c in str(name))
    
    # Get all unique attack types
    attack_types = df_work['Label'].dropna().unique().tolist()
    print(f"Found {len(attack_types)} attack types: {attack_types}\n")
    
    PMF_STORAGE = {}
    
    for col in discrete_cols:
        all_values = sorted(df_work[col].dropna().unique())
        
        # Overall, Normal, All Attacks
        pmf_all = df_work[col].value_counts(normalize=True).reindex(all_values, fill_value=0)
        pmf_normal = df_work[df_work['Binary Label'] == 0][col].value_counts(normalize=True).reindex(all_values, fill_value=0)
        pmf_attack = df_work[df_work['Binary Label'] == 1][col].value_counts(normalize=True).reindex(all_values, fill_value=0)
        
        #  NEW: Per attack type
        pmf_per_attack = {}
        for attack in attack_types:
            attack_subset = df_work[df_work['Label'] == attack]
            if len(attack_subset) >= 50:
                attack_name_safe = safe_filename(attack.upper().replace(' ', '_').replace('-', '_'))
                pmf_per_attack[attack_name_safe] = attack_subset[col].value_counts(normalize=True).reindex(all_values, fill_value=0).to_dict()
        
        PMF_STORAGE[col] = {
            'categories': all_values,
            'PMF_Overall': pmf_all.to_dict(),
            'PMF_Normal': pmf_normal.to_dict(),
            'PMF_Attack': pmf_attack.to_dict(),
            **{f'PMF_{attack_name}': pmf for attack_name, pmf in pmf_per_attack.items()}
        }
    
    # Generate plots (main comparison)
    print("\nGenerating comparison figures...")
    for col in discrete_cols:
        all_values = sorted(df_work[col].dropna().unique())
        pmf_all = df_work[col].value_counts(normalize=True).reindex(all_values, fill_value=0)
        pmf_normal = df_work[df_work['Binary Label'] == 0][col].value_counts(normalize=True).reindex(all_values, fill_value=0)
        pmf_attack = df_work[df_work['Binary Label'] == 1][col].value_counts(normalize=True).reindex(all_values, fill_value=0)
        
        fig, axes = plt.subplots(1, 3, figsize=(28, 7), sharey=True)
        fig.suptitle(f"PMF Comparison for {col}", fontsize=20, fontweight='bold')
        
        axes[0].bar(all_values, pmf_all, color='#1f77b4', edgecolor='black')
        axes[0].set_title("Overall PMF")
        axes[0].set_ylabel("Probability")
        axes[0].grid(axis='y', alpha=0.3)
        
        axes[1].bar(all_values, pmf_normal, color='#2ca02c', edgecolor='black')
        axes[1].set_title("Normal PMF")
        axes[1].grid(axis='y', alpha=0.3)
        
        axes[2].bar(all_values, pmf_attack, color='#d62728', edgecolor='black')
        axes[2].set_title("All Attacks PMF")
        axes[2].grid(axis='y', alpha=0.3)
        
        for ax in axes:
            ax.set_xticks(all_values)
            ax.set_xticklabels(all_values, rotation=60, ha='right')
        
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        filename = f"PMF_COMPARISON_{safe_filename(col)}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f" Saved: {filename}")
    
    # Generate per-attack-type plots
    print("\nGenerating per-attack-type figures...")
    for attack in attack_types:
        sub = df_work[df_work['Label'] == attack]
        if len(sub) < 50:
            continue
        
        attack_name_safe = safe_filename(attack.upper().replace(' ', '_').replace('-', '_'))
        print(f"\n-> {attack} ({len(sub)} samples)")
        
        for col in discrete_cols:
            all_values = sorted(df_work[col].dropna().unique())
            pmf_all = df_work[col].value_counts(normalize=True).reindex(all_values, fill_value=0)
            pmf_normal = df_work[df_work['Binary Label'] == 0][col].value_counts(normalize=True).reindex(all_values, fill_value=0)
            pmf_specific = sub[col].value_counts(normalize=True).reindex(all_values, fill_value=0)
            
            fig, axes = plt.subplots(1, 3, figsize=(28, 7), sharey=True)
            fig.suptitle(f"PMF for {col} - Attack: {attack}", fontsize=20, fontweight='bold')
            
            axes[0].bar(all_values, pmf_all, color='#1f77b4', edgecolor='black')
            axes[0].set_title("Overall")
            axes[0].set_ylabel("Probability")
            
            axes[1].bar(all_values, pmf_normal, color='#2ca02c', edgecolor='black')
            axes[1].set_title("Normal")
            
            axes[2].bar(all_values, pmf_specific, color='#d62728', edgecolor='black')
            axes[2].set_title(f"{attack}")
            
            for ax in axes:
                ax.set_xticks(all_values)
                ax.set_xticklabels(all_values, rotation=60, ha='right')
                ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout(rect=[0, 0, 1, 0.93])
            filename = f"PMF_{safe_filename(col)}_{attack_name_safe}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   Saved: {filename}")
    
    # Save to JSON
    import json
    pmf_models_saveable = {}
    for col, data in PMF_STORAGE.items():
        pmf_models_saveable[col] = {
            'categories': [str(v) for v in data['categories']],
            **{key: {str(k): float(v) for k, v in value.items()} 
               for key, value in data.items() if key != 'categories'}
        }
    
    with open('milestone2_pmf_models_per_attack.json', 'w') as f:
        json.dump(pmf_models_saveable, f, indent=2)
    print(f"\n Saved PMF models (with per-attack types) to: milestone2_pmf_models_per_attack.json")
    
    # Save to CSV
    pmf_csv_rows = []
    for col, data in PMF_STORAGE.items():
        categories = data['categories']
        
        for category in categories:
            row = {
                'Column': col,
                'Category': str(category),
                'PMF_Overall': data['PMF_Overall'].get(category, 0.0),
                'PMF_Normal': data['PMF_Normal'].get(category, 0.0),
                'PMF_Attack': data['PMF_Attack'].get(category, 0.0),
            }
            
            # Add per-attack-type PMFs
            for key in data.keys():
                if key.startswith('PMF_') and key not in ['PMF_Overall', 'PMF_Normal', 'PMF_Attack']:
                    row[key] = data[key].get(category, 0.0)
            
            pmf_csv_rows.append(row)
    
    if pmf_csv_rows:
        pmf_csv_df = pd.DataFrame(pmf_csv_rows)
        pmf_csv_df.to_csv('milestone2_pmf_models_per_attack.csv', index=False)
        print(f" Saved PMF models (with per-attack types) to: milestone2_pmf_models_per_attack.csv")
    
    print("\n" + "="*80)
    print("PMF ANALYSIS COMPLETE - Saved with per-attack-type breakdowns!")
    print("="*80)
    
    return PMF_STORAGE


# ============================================================
# USAGE - Replace your existing function calls with these:
# ============================================================

# Task 2a - PDF with per-attack-type
print("\n>>> TASK 2a: PDF Fitting (with per-attack-type)")
pdf_results, pdf_models = generate_pdf_plots_with_per_attack()

# Task 2b - PMF with per-attack-type  
print("\n>>> TASK 2b: PMF Fitting (with per-attack-type)")
PMF_STORAGE = generate_pmf_plots_with_per_attack()