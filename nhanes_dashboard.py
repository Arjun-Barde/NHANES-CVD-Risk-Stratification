"""
NHANES CVD Risk Stratification — Interactive Dashboard
Self-contained: downloads NHANES XPT files directly from CDC servers.

Deploy to Streamlit Cloud:
1. Push this file + requirements.txt to GitHub
2. Connect repo at share.streamlit.io
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="NHANES CVD Risk Dashboard",
    page_icon="🫀",
    layout="wide"
)


@st.cache_data(show_spinner="Downloading NHANES data from CDC...")
def load_and_prepare_data():
    import io, requests

    CDC_BASE_P = "https://wwwn.cdc.gov/Nchs/Nhanes/2017-2020/PrePandemic"
    files = {
        'demo':  f"{CDC_BASE_P}/P_DEMO.XPT",
        'bmx':   f"{CDC_BASE_P}/P_BMX.XPT",
        'bpxo':  f"{CDC_BASE_P}/P_BPXO.XPT",
        'tchol': f"{CDC_BASE_P}/P_TCHOL.XPT",
        'diq':   f"{CDC_BASE_P}/P_DIQ.XPT",
        'cdq':   f"{CDC_BASE_P}/P_CDQ.XPT",
    }
    dfs = {}
    for key, url in files.items():
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        dfs[key] = pd.read_sas(io.BytesIO(r.content), format='xport', encoding='utf-8')

    demo, bmx, bpxo, tchol, diq, cdq = (dfs[k] for k in
                                          ['demo','bmx','bpxo','tchol','diq','cdq'])

    df = (cdq[['SEQN','CDQ001','CDQ010']]
          .merge(demo[['SEQN','RIDAGEYR','RIAGENDR','RIDRETH3','INDFMPIR',
                        'WTMECPRP','SDMVPSU','SDMVSTRA']], on='SEQN', how='left')
          .merge(bmx[['SEQN','BMXBMI','BMXWAIST']], on='SEQN', how='left')
          .merge(bpxo[['SEQN','BPXOSY1','BPXODI1']], on='SEQN', how='left')
          .merge(tchol[['SEQN','LBXTC']], on='SEQN', how='left')
          .merge(diq[['SEQN','DIQ010']], on='SEQN', how='left'))

    df = df[df['CDQ001'] != 9.0].copy()
    df['cvd_symptom']    = (df['CDQ001'] == 1.0).astype(int)
    df['sob']            = (df['CDQ010'] == 1.0).astype(int)
    df['male']           = (df['RIAGENDR'] == 1.0).astype(int)
    df['diabetes']       = (df['DIQ010'] == 1.0).astype(int)

    race_map = {1.0:'Hispanic', 2.0:'Hispanic', 3.0:'NH_White',
                4.0:'NH_Black', 6.0:'NH_Asian', 7.0:'Other'}
    df['race_eth'] = df['RIDRETH3'].map(race_map)

    df.rename(columns={
        'RIDAGEYR':'age', 'BMXBMI':'bmi', 'BMXWAIST':'waist_cm',
        'BPXOSY1':'sbp', 'BPXODI1':'dbp', 'LBXTC':'total_chol',
        'INDFMPIR':'income_pir'
    }, inplace=True)

    for col in ['bmi','waist_cm','sbp','dbp','total_chol','income_pir']:
        df[col] = df[col].fillna(df[col].median())

    df['hypertension']   = (df['sbp'] >= 130).astype(int)
    df['obese']          = (df['bmi'] >= 30).astype(int)
    df['high_chol']      = (df['total_chol'] >= 200).astype(int)
    df['pulse_pressure'] = df['sbp'] - df['dbp']
    df['age_x_male']     = df['age'] * df['male']
    df['age_x_diabetes'] = df['age'] * df['diabetes']
    df['sbp_x_chol']     = df['sbp'] * df['total_chol']
    df['bmi_x_diabetes'] = df['bmi'] * df['diabetes']

    df['income_bracket'] = pd.cut(
        df['income_pir'],
        bins=[0, 1.0, 2.0, 3.5, 5.1],
        labels=['<100% FPL','100-200% FPL','200-350% FPL','>350% FPL'],
        include_lowest=True
    )
    return df


@st.cache_resource(show_spinner="Training models (first load only)...")
def train_models(_df):
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import roc_auc_score, roc_curve
    from xgboost import XGBClassifier
    import shap

    df = _df
    df_model = pd.get_dummies(df, columns=['race_eth'], drop_first=True)
    drop_cols = ['SEQN','WTMECPRP','SDMVPSU','SDMVSTRA','cvd_symptom',
                 'CDQ001','CDQ010','RIAGENDR','RIDRETH3','DIQ010','income_bracket']
    feature_cols = [c for c in df_model.columns if c not in drop_cols]

    X = df_model[feature_cols]
    y = df_model['cvd_symptom']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    test_idx = X_test.index
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models_def = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        'Random Forest': RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=10,
            random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            scale_pos_weight=(y_train==0).sum()/(y_train==1).sum(),
            eval_metric='logloss', random_state=42, n_jobs=-1)
    }

    results = {}
    for name, model in models_def.items():
        cv_aucs = cross_val_score(model, X_train, y_train, cv=cv,
                                   scoring='roc_auc', n_jobs=-1)
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:,1]
        test_auc = roc_auc_score(y_test, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        results[name] = {
            'model': model, 'y_prob': y_prob,
            'cv_auc': cv_aucs.mean(), 'cv_std': cv_aucs.std(),
            'test_auc': test_auc, 'fpr': fpr, 'tpr': tpr
        }

    best_name = max(results, key=lambda x: results[x]['test_auc'])

    # SHAP
    rf = results['Random Forest']['model']
    explainer = shap.TreeExplainer(rf)
    sv_raw = explainer.shap_values(X_test)
    if isinstance(sv_raw, list):
        sv = sv_raw[1]
    elif np.array(sv_raw).ndim == 3:
        sv = sv_raw[:, :, 1]
    else:
        sv = sv_raw

    # Subgroup AUC
    df_test = df.loc[test_idx].copy()
    df_test['y_true'] = y_test.values
    df_test['y_prob'] = results[best_name]['y_prob']

    sub_race, sub_income = [], []
    for group, grp in df_test.groupby('race_eth', observed=True):
        if grp['y_true'].nunique() < 2 or len(grp) < 20:
            continue
        auc = roc_auc_score(grp['y_true'], grp['y_prob'])
        sub_race.append({'Group': group, 'N': len(grp), 'AUC': round(auc, 4)})

    for group, grp in df_test.groupby('income_bracket', observed=True):
        if grp['y_true'].nunique() < 2 or len(grp) < 20:
            continue
        auc = roc_auc_score(grp['y_true'], grp['y_prob'])
        sub_income.append({'Group': str(group), 'N': len(grp), 'AUC': round(auc, 4)})

    return {
        'results': results, 'best_name': best_name,
        'best_model': results[best_name]['model'],
        'feature_cols': feature_cols,
        'X_test': X_test, 'y_test': y_test,
        'shap_values': sv,
        'sub_race': pd.DataFrame(sub_race),
        'sub_income': pd.DataFrame(sub_income),
    }


def weighted_prevalence(data, group_col):
    rows = []
    for group, grp in data.groupby(group_col, observed=True):
        w = grp['WTMECPRP']
        y = grp['cvd_symptom']
        prev = (y * w).sum() / w.sum()
        rows.append({'Group': str(group), 'N': len(grp),
                     'Weighted Prevalence (%)': round(prev * 100, 2)})
    return pd.DataFrame(rows).sort_values('Weighted Prevalence (%)', ascending=False)


# ── Load & train ──────────────────────────────────────────────────────────────
st.title("🫀 NHANES CVD Risk Stratification Dashboard")
st.markdown(
    "**Dataset:** CDC NHANES 2017–March 2020 Pre-Pandemic | "
    "**Population:** U.S. Adults 40+ | **N ≈ 6,400+**"
)
st.markdown("---")

try:
    df = load_and_prepare_data()
    art = train_models(df)
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

results      = art['results']
best_name    = art['best_name']
best_model   = art['best_model']
feature_cols = art['feature_cols']
X_test       = art['X_test']
y_test       = art['y_test']
sv           = art['shap_values']
sub_race     = art['sub_race']
sub_income   = art['sub_income']
overall_auc  = results[best_name]['test_auc']
race_prev    = weighted_prevalence(df, 'race_eth')
income_prev  = weighted_prevalence(df, 'income_bracket')

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Health Equity", "🤖 Model Performance",
    "🔍 SHAP Analysis", "🧮 Risk Calculator"
])

# ── TAB 1 ─────────────────────────────────────────────────────────────────────
with tab1:
    st.header("Survey-Weighted CVD Prevalence")
    st.markdown(
        "Survey weights (`WTMECPRP`) correct for NHANES complex multistage sampling, "
        "producing **nationally representative** prevalence estimates."
    )
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("By Race/Ethnicity")
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = sns.color_palette("Set2", len(race_prev))
        bars = ax.barh(race_prev['Group'], race_prev['Weighted Prevalence (%)'],
                       color=colors, height=0.5)
        ax.set_xlabel("Weighted CVD Prevalence (%)")
        ax.set_title("CVD Prevalence by Race/Ethnicity", fontweight='bold')
        for bar, val in zip(bars, race_prev['Weighted Prevalence (%)']):
            ax.text(bar.get_width()+0.05, bar.get_y()+bar.get_height()/2,
                    f'{val:.1f}%', va='center', fontsize=10)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.subheader("By Income Bracket")
        fig, ax = plt.subplots(figsize=(7, 4))
        colors2 = sns.color_palette("OrRd", len(income_prev))
        bars2 = ax.barh(income_prev['Group'], income_prev['Weighted Prevalence (%)'],
                        color=colors2, height=0.5)
        ax.set_xlabel("Weighted CVD Prevalence (%)")
        ax.set_title("CVD Prevalence by Income-to-Poverty Ratio", fontweight='bold')
        for bar, val in zip(bars2, income_prev['Weighted Prevalence (%)']):
            ax.text(bar.get_width()+0.05, bar.get_y()+bar.get_height()/2,
                    f'{val:.1f}%', va='center', fontsize=10)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.subheader("Race × Income Heatmap")
    col_order = ['<100% FPL','100-200% FPL','200-350% FPL','>350% FPL']
    eq_rows = []
    for race, rgrp in df.groupby('race_eth', observed=True):
        for inc, igrp in rgrp.groupby('income_bracket', observed=True):
            if len(igrp) < 10: continue
            w, y = igrp['WTMECPRP'], igrp['cvd_symptom']
            eq_rows.append({'Race/Ethnicity': race, 'Income Bracket': str(inc),
                             'Weighted CVD Prevalence (%)': round((y*w).sum()/w.sum()*100, 2)})
    eq_df = pd.DataFrame(eq_rows)
    pivot = eq_df.pivot(index='Race/Ethnicity', columns='Income Bracket',
                         values='Weighted CVD Prevalence (%)')
    pivot = pivot[[c for c in col_order if c in pivot.columns]]
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax,
                linewidths=0.5, cbar_kws={'label': 'Weighted CVD Prevalence (%)'})
    ax.set_title("Survey-Weighted CVD Prevalence (%)\nRace/Ethnicity × Income Bracket",
                 fontweight='bold', fontsize=12)
    ax.set_xlabel("Income Bracket (% Federal Poverty Level)")
    ax.set_ylabel("")
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ── TAB 2 ─────────────────────────────────────────────────────────────────────
with tab2:
    st.header("Model Performance")
    c1, c2, c3 = st.columns(3)
    for col, (name, res) in zip([c1,c2,c3], results.items()):
        col.metric(name, f"{res['test_auc']:.4f}",
                   f"CV: {res['cv_auc']:.4f} ± {res['cv_std']:.4f}")

    st.markdown("---")
    st.subheader("ROC Curves")
    fig, ax = plt.subplots(figsize=(7, 6))
    for (name, res), color in zip(results.items(), ['#4C72B0','#55A868','#C44E52']):
        ax.plot(res['fpr'], res['tpr'],
                label=f"{name} (AUC={res['test_auc']:.4f})", color=color, linewidth=2)
    ax.plot([0,1],[0,1],'k--', linewidth=1, alpha=0.5, label='Random classifier')
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves — CVD Risk Prediction', fontweight='bold')
    ax.legend(loc='lower right', fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("Subgroup AUC — Equity Diagnostics")
    st.markdown(f"Best model: **{best_name}** | Red = below overall AUC")
    c1, c2 = st.columns(2)
    for col, df_sub, title in zip([c1,c2], [sub_race, sub_income],
                                   ['By Race/Ethnicity','By Income Bracket']):
        with col:
            fig, ax = plt.subplots(figsize=(6, 4))
            bar_cols = ['#C44E52' if v < overall_auc else '#55A868' for v in df_sub['AUC']]
            bars = ax.barh(df_sub['Group'], df_sub['AUC'], color=bar_cols, height=0.5)
            ax.axvline(overall_auc, color='#333', linestyle='--', linewidth=1.5,
                       label=f'Overall: {overall_auc:.4f}')
            ax.set_xlim(0.45, 1.0); ax.set_xlabel('ROC-AUC')
            ax.set_title(f'Subgroup AUC {title}', fontweight='bold')
            for bar, auc in zip(bars, df_sub['AUC']):
                ax.text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2,
                        f'{auc:.4f}', va='center', fontsize=9)
            ax.legend(fontsize=8)
            plt.tight_layout(); st.pyplot(fig); plt.close()

# ── TAB 3 ─────────────────────────────────────────────────────────────────────
with tab3:
    st.header("SHAP Feature Importance")
    st.markdown("Positive SHAP = increases predicted CVD risk. Based on Random Forest.")

    mean_shap = pd.DataFrame({
        'Feature': X_test.columns,
        'Mean |SHAP|': np.abs(sv).mean(axis=0)
    }).sort_values('Mean |SHAP|', ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(mean_shap['Feature'], mean_shap['Mean |SHAP|'], color='#4C72B0')
    ax.set_xlabel('Mean |SHAP Value|')
    ax.set_title('Top 15 Features by Mean |SHAP| (Random Forest)', fontweight='bold')
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("SHAP Interaction: Income-to-Poverty Ratio × Age")
    income_idx = list(X_test.columns).index('income_pir')
    age_idx    = list(X_test.columns).index('age')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sc1 = axes[0].scatter(X_test.iloc[:,income_idx], sv[:,income_idx],
                          c=X_test.iloc[:,age_idx], cmap='coolwarm', alpha=0.35, s=12)
    axes[0].axhline(0, color='gray', linestyle='--', linewidth=0.8)
    axes[0].set_xlabel('Income-to-Poverty Ratio'); axes[0].set_ylabel('SHAP Value')
    axes[0].set_title('Income PIR Effect\n(colored by Age)', fontweight='bold')
    plt.colorbar(sc1, ax=axes[0], label='Age')

    sc2 = axes[1].scatter(X_test.iloc[:,age_idx], sv[:,age_idx],
                          c=X_test.iloc[:,income_idx], cmap='RdYlGn', alpha=0.35, s=12)
    axes[1].axhline(0, color='gray', linestyle='--', linewidth=0.8)
    axes[1].set_xlabel('Age (years)'); axes[1].set_ylabel('SHAP Value')
    axes[1].set_title('Age Effect\n(colored by Income PIR)', fontweight='bold')
    plt.colorbar(sc2, ax=axes[1], label='Income PIR')

    plt.suptitle('SDOH × Clinical Risk Interaction Analysis', fontsize=12,
                 fontweight='bold', y=1.02)
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ── TAB 4 ─────────────────────────────────────────────────────────────────────
with tab4:
    st.header("Individual CVD Risk Calculator")
    st.markdown(f"Uses **{best_name}**. _Educational only — not a clinical tool._")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Demographics")
        age    = st.slider("Age", 40, 85, 55)
        male   = st.selectbox("Sex", ["Female","Male"]) == "Male"
        race   = st.selectbox("Race/Ethnicity",
                               ["NH_White","NH_Black","Hispanic","NH_Asian","Other"])
        income = st.slider("Income-to-Poverty Ratio", 0.0, 5.0, 2.5, 0.1)
    with c2:
        st.subheader("Clinical Measurements")
        bmi   = st.slider("BMI (kg/m²)", 15.0, 50.0, 27.0, 0.5)
        waist = st.slider("Waist circumference (cm)", 60.0, 150.0, 90.0, 1.0)
        sbp   = st.slider("Systolic BP (mmHg)", 80, 200, 125)
        dbp   = st.slider("Diastolic BP (mmHg)", 50, 120, 80)
        chol  = st.slider("Total Cholesterol (mg/dL)", 100, 350, 200)
    with c3:
        st.subheader("Medical History")
        diabetes = st.checkbox("Diagnosed Diabetes")
        sob      = st.checkbox("Shortness of Breath on Exertion")

    if st.button("Calculate CVD Risk", type="primary"):
        input_dict = {
            'age': age, 'bmi': bmi, 'waist_cm': waist, 'sbp': sbp, 'dbp': dbp,
            'total_chol': chol, 'income_pir': income, 'male': int(male),
            'diabetes': int(diabetes), 'sob': int(sob),
            'hypertension': int(sbp >= 130), 'obese': int(bmi >= 30),
            'high_chol': int(chol >= 200), 'pulse_pressure': sbp - dbp,
            'age_x_male': age * int(male), 'age_x_diabetes': age * int(diabetes),
            'sbp_x_chol': sbp * chol, 'bmi_x_diabetes': bmi * int(diabetes),
            'race_eth_NH_Asian': int(race=='NH_Asian'),
            'race_eth_NH_Black': int(race=='NH_Black'),
            'race_eth_NH_White': int(race=='NH_White'),
            'race_eth_Other': int(race=='Other'),
        }
        input_df = pd.DataFrame([input_dict])
        for c in feature_cols:
            if c not in input_df.columns:
                input_df[c] = 0
        input_df = input_df[feature_cols]

        prob = best_model.predict_proba(input_df)[0, 1]
        risk_level = "Low" if prob < 0.15 else "Moderate" if prob < 0.30 else "High"
        emoji = {"Low":"🟢","Moderate":"🟡","High":"🔴"}[risk_level]

        st.markdown("---")
        r1, r2, r3 = st.columns(3)
        r1.metric("Predicted CVD Risk", f"{prob:.1%}")
        r2.metric("Risk Level", f"{emoji} {risk_level}")
        r3.metric("Model", best_name)

        flags = []
        if age >= 65:    flags.append("Age 65+")
        if sbp >= 130:   flags.append("Hypertension (SBP >= 130)")
        if bmi >= 30:    flags.append("Obesity (BMI >= 30)")
        if chol >= 200:  flags.append("High Cholesterol (>= 200 mg/dL)")
        if diabetes:     flags.append("Diabetes")
        if sob:          flags.append("Shortness of breath on exertion")
        if income < 1.0: flags.append("Below poverty level (PIR < 1.0)")

        st.markdown("**Risk factors present:**")
        for flag in flags: st.markdown(f"- {flag}")
        if not flags: st.markdown("- No major risk flags detected")
        st.caption("AUC ~0.69. Not for clinical use.")

st.markdown("---")
st.markdown(
    "**Data:** CDC NHANES 2017–March 2020 | "
    "**GitHub:** [github.com/Arjun-Barde/NHANES-CVD-Risk-Stratification]"
    "(https://github.com/Arjun-Barde/NHANES-CVD-Risk-Stratification) | "
    "Arjun Barde | M.S. Health Informatics, University of Pittsburgh"
)
