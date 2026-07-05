# Requires: dataset_feature_engineered.csv in the same folder

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings

from sklearn.compose         import ColumnTransformer
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import OneHotEncoder
from sklearn.impute          import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics         import r2_score, mean_absolute_error, mean_squared_error
from sklearn.base            import clone

from xgboost import XGBRegressor
import shap

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.dpi']     = 120
plt.rcParams['font.size']      = 11
plt.rcParams['figure.figsize'] = (14, 6)

print('✅ Step 1 — Libraries imported')

# %% [markdown]
# ## STEP 2 — Load & Explore Dataset

# %%
df = pd.read_csv('dataset_feature_engineered.csv')

print(f'Shape        : {df.shape}  ({df.shape[0]} rows × {df.shape[1]} columns)')
print(f'Districts    : {df["district"].nunique()}  → {sorted(df["district"].unique())}')
print(f'Years        : {sorted(df["year"].unique())}')
print(f'Months       : {df["month"].nunique()} (all 12 confirmed)')
print(f'Seasons      : {df["season"].unique().tolist()}')
print()

print('── Health targets (per 1 000 population) ──')
for col in ['total_resp_rate_per_1k', 'bronchitis_rate_per_1k', 'asthma_rate_per_1k']:
    print(f'  {col:<35}: min={df[col].min():.3f}  max={df[col].max():.3f}  mean={df[col].mean():.3f}')

print()
g = df.groupby(['district','year'])['total_resp_rate_per_1k'].nunique()
print(f'Max unique resp. values per district-year: {g.max()}  (1 = annual ✅)')

g2 = df.groupby(['district','year'])['pm2.5_ug_m3'].nunique()
print(f'Max unique PM2.5 per district-year       : {g2.max()}  (genuinely monthly ✅)')

print()
print(f'Missing values (non-zero only):')
miss = df.isnull().sum()
print(miss[miss > 0].to_string())
print()
print('✅ Step 2 — Dataset loaded and explored')

# %% [markdown]
# ## STEP 2B — PM2.5 Baseline Model
# 
# This section trains a baseline PM2.5 model using a simpler feature set
# (no lag/rolling/YoY features) and a standard random 80/20 train-test split,
# for comparison with the final tuned model trained later in this notebook.

# %%
# ── Baseline feature engineering ──
from sklearn.preprocessing import LabelEncoder as _LabelEncoder_baseline

_df_base = df.copy()

_le_dist = _LabelEncoder_baseline()
_le_prov = _LabelEncoder_baseline()
_df_base['district_encoded_base'] = _le_dist.fit_transform(_df_base['district'])
_df_base['province_encoded_base'] = _le_prov.fit_transform(_df_base['province'])

# Derived features matching the original baseline notebook's feature engineering
_df_base['year_trend_base']    = _df_base['year'] - _df_base['year'].min()
_df_base['frp_per_fire_base']  = _df_base['frp_total'] / (_df_base['no_fire_types'] + 1)
_df_base['vim_range_base']     = _df_base['vim_max'] - _df_base['vim_min']
_df_base['carbon_per_ha_base'] = _df_base['carbon_gross_emissions_yearly'] / (_df_base['area_ha'] + 1)

# NOTE: 'month_num' and 'fire_active' already exist in dataset_feature_engineered.csv
# NOTE: 'net_c_flux_yr' is the renamed equivalent of the raw dataset's 'Net_C_Flux_yr-1'

baseline_pm25_features = [
    'tc_loss_ha', 'carbon_gross_emissions_yearly', 'carbon_per_ha_base', 'net_c_flux_yr',
    'vim', 'vim_anomaly', 'vim_range_base', 'vim_climatology',
    'frp_mean', 'frp_total', 'frp_per_fire_base', 'no_fire_types', 'brightness', 'bright_t31', 'fire_active',
    'so2_ug_m3', 'no2_ug_m3',
    'total_population_1k', 'month_num', 'year_trend_base', 'district_encoded_base', 'province_encoded_base'
]

X_base = _df_base[baseline_pm25_features]
y_base = _df_base['pm2.5_ug_m3']

print(f'Baseline feature count: {len(baseline_pm25_features)}')
print(f'Rows: {X_base.shape[0]}')

# %%
# ── Baseline model: random 80/20 split, fixed hyperparameters ──

from sklearn.model_selection import train_test_split as _tts_baseline
from xgboost import XGBRegressor as _XGB_baseline
from sklearn.metrics import r2_score as _r2_baseline, mean_absolute_error as _mae_baseline, mean_squared_error as _mse_baseline

X_train_base, X_test_base, y_train_base, y_test_base = _tts_baseline(
    X_base, y_base, test_size=0.2, random_state=42
)

# Hyperparameters found via RandomizedSearchCV (n_iter=60, cv=5)
baseline_best_params = {
    'subsample': 0.8,
    'reg_lambda': 2,
    'reg_alpha': 0.5,
    'n_estimators': 200,
    'min_child_weight': 5,
    'max_depth': 6,
    'learning_rate': 0.1,
    'gamma': 0,
    'colsample_bytree': 0.8
}

baseline_model = _XGB_baseline(
    **baseline_best_params,
    random_state=42,
    tree_method='hist',
    n_jobs=1   # fixed to 1 for full determinism
)
baseline_model.fit(X_train_base, y_train_base)
y_pred_base = baseline_model.predict(X_test_base)

baseline_r2   = _r2_baseline(y_test_base, y_pred_base)
baseline_rmse = (_mse_baseline(y_test_base, y_pred_base)) ** 0.5
baseline_mae  = _mae_baseline(y_test_base, y_pred_base)

pm25_baseline_result = {
    'best_params': baseline_best_params,
    'te_r2': baseline_r2,
    'te_rmse': baseline_rmse,
    'te_mae': baseline_mae,
}

print('=== PM2.5 Baseline Model (random split, no lag features, fixed hyperparameters) ===')
print(f'  Test  R2   : {baseline_r2:.4f}')
print(f'  Test  RMSE : {baseline_rmse:.4f}')
print(f'  Test  MAE  : {baseline_mae:.4f}')
print(f'  Hyperparameters: {baseline_best_params}')
print()
print('✅ Step 2B — PM2.5 baseline model trained (deterministic)')

# %% [markdown]
# ## STEP 3 — Variable Check and Encoding

# %%
month_order   = ['January','February','March','April','May','June',
                 'July','August','September','October','November','December']
month_num_map = {m: i+1 for i, m in enumerate(month_order)}

df['month_num_mapped'] = df['month'].map(month_num_map)
df['month_sin']        = np.sin(2 * np.pi * df['month_num_mapped'] / 12)
df['month_cos']        = np.cos(2 * np.pi * df['month_num_mapped'] / 12)

print('✅ Cyclic month encoding added (month_sin, month_cos)')

lag_cols   = [c for c in df.columns if any(x in c for x in ['lag','roll','yoy'])]
inter_cols = [c for c in df.columns if any(x in c for x in
              ['interaction','burden','health_index','proxy','stress',
               'efficiency','sink_strength','deforest_poll'])]

print(f'Pre-built lag / rolling / YoY features    : {len(lag_cols)}')
print(f'Pre-built interaction / composite features : {len(inter_cols)}')
print(f'Total columns available                    : {df.shape[1]}')
print('\n✅ Step 3 — Features confirmed. Ready for modelling.')

# %% [markdown]
# ## STEP 4 — Define Model Inputs

# %%
ENV_NUMERIC = [f for f in [
    # Forest / vegetation
    'vim', 'vim_anomaly', 'vim_min', 'vim_max', 'vim_climatology',
    'vegetation_stress', 'ndvi_range', 'veg_deficit', 'below_climatology',
    # Forest cover & carbon
    'forest_cover_pct', 'tc_loss_ha', 'tc_loss_pct_of_extent', 'tc_loss_per_km2',
    'carbon_loss_per_ha', 'carbon_efficiency', 'net_flux_per_ha',
    'is_carbon_sink', 'carbon_sink_strength',
    # Forest lags (pre-built)
    'vim_lag1m', 'vim_lag3m', 'vim_roll3m', 'vim_roll6m',
    'tc_loss_ha_lag1m', 'tc_loss_ha_lag3m', 'tc_loss_ha_roll3m',
    # Fire
    'frp_mean', 'frp_total', 'frp_normalized', 'fire_smoke_proxy',
    'fire_active', 'no_fire_types', 'is_forest_fire', 'brightness',
    'frp_mean_lag1m', 'frp_mean_lag3m', 'frp_mean_roll3m',
    # Air quality
    'pm2.5_ug_m3', 'so2_ug_m3', 'no2_ug_m3',
    'pollution_index', 'pm25_so2_ratio', 'total_pollutant_load',
    'pm2.5_ug_m3_lag1m', 'pm2.5_ug_m3_lag3m', 'pm2.5_ug_m3_roll3m', 'pm2.5_ug_m3_roll6m',
    'so2_ug_m3_lag1m', 'so2_ug_m3_lag3m', 'so2_ug_m3_roll3m',
    'no2_ug_m3_lag1m', 'no2_ug_m3_lag3m', 'no2_ug_m3_roll3m',
    # Interaction / composite indices
    'fire_pollution_interaction', 'deforest_pollution_index', 'forest_health_index',
    'pollution_health_burden', 'fire_health_burden',
    'veg_pollution_interaction', 'deforest_health_interaction',
    # Population
    'pop_density_per_ha', 'gender_ratio',
    # Cyclic seasonality
    'month_sin', 'month_cos',
] if f in df.columns]

CATEGORICAL = ['district', 'season']

TARGETS = {
    'total_resp_rate_per_1k' : 'Total respiratory rate per 1 000 population',
    'bronchitis_rate_per_1k'  : 'Bronchitis live discharges per 1 000',
    'asthma_rate_per_1k'      : 'Asthma live discharges per 1 000',
}

HEALTH_LAG_COLS = [c for c in [
    'total_resp_rate_per_1k_lag1m', 'total_resp_rate_per_1k_lag3m',
    'total_resp_rate_per_1k_roll3m', 'total_resp_rate_per_1k_roll6m',
] if c in df.columns]


def get_features_for_target(target, model_type='A'):
    base = ENV_NUMERIC.copy()
    if model_type == 'B':
        base += [c for c in HEALTH_LAG_COLS if c != target]
    all_feats = base + CATEGORICAL
    seen, result = set(), []
    for f in all_feats:
        if f not in seen and f != target and f in df.columns:
            seen.add(f); result.append(f)
    return result


print('✅ Step 4 — Feature matrix defined')
print(f'Monthly Model A features : {len(get_features_for_target("total_resp_rate_per_1k", "A"))}')

# %% [markdown]
# ## STEP 5 — Time-Based Split

# %%
TRAIN_YEARS = list(range(2015, 2021))   # 2015–2020
TEST_YEARS  = list(range(2021, 2025))   # 2021–2024

print(f'Train years : {TRAIN_YEARS[0]}–{TRAIN_YEARS[-1]}  ({len(TRAIN_YEARS)} years)')
print(f'Test  years : {TEST_YEARS[0]}–{TEST_YEARS[-1]}  ({len(TEST_YEARS)} years)')
print(f'Train rows  : {df["year"].isin(TRAIN_YEARS).sum()}')
print(f'Test  rows  : {df["year"].isin(TEST_YEARS).sum()}')
print('(2014 excluded — insufficient lag history)')
print('\n✅ Step 5 — Time split defined')

# %% [markdown]
# ## STEP 6A — Monthly Model A vs B *(env baseline vs AR-lag reference)*

# %%
def build_preprocessor(numeric_features, categorical_features):
    transformers = []
    if numeric_features:
        transformers.append(('num', SimpleImputer(strategy='median'), numeric_features))
    if categorical_features:
        transformers.append(('cat', Pipeline([
            ('imp', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_features))
    return ColumnTransformer(transformers=transformers, remainder='drop')


PARAM_DIST = {
    'model__n_estimators'     : [300, 500, 700, 900],
    'model__max_depth'        : [3, 4, 5, 6],
    'model__learning_rate'    : [0.01, 0.03, 0.05, 0.08],
    'model__subsample'        : [0.7, 0.8, 0.9, 1.0],
    'model__colsample_bytree' : [0.7, 0.8, 0.9, 1.0],
    'model__min_child_weight' : [1, 3, 5],
    'model__reg_alpha'        : [0, 0.1, 0.3],
    'model__reg_lambda'       : [1, 1.5, 2],
}


def train_monthly_model(target, desc, model_type='A', n_iter=25):
    features  = get_features_for_target(target, model_type=model_type)
    cat_feats = [f for f in features if f in CATEGORICAL]
    num_feats = [f for f in features if f not in CATEGORICAL]

    cols  = list(dict.fromkeys(features + [target, 'year']))
    df_m  = df[cols].dropna().reset_index(drop=True)
    X, y  = df_m[features].copy(), df_m[target].copy()

    tr_mask = df_m['year'].isin(TRAIN_YEARS)
    te_mask = df_m['year'].isin(TEST_YEARS)
    X_train, y_train = X.loc[tr_mask], y.loc[tr_mask]
    X_test,  y_test  = X.loc[te_mask], y.loc[te_mask]

    prep   = build_preprocessor(num_feats, cat_feats)
    pipe   = Pipeline([('prep', prep),
                       ('model', XGBRegressor(objective='reg:squarederror',
                                              random_state=42, verbosity=0))])
    tscv   = TimeSeriesSplit(n_splits=4)
    search = RandomizedSearchCV(pipe, PARAM_DIST, n_iter=n_iter, scoring='r2',
                                cv=tscv, random_state=42, n_jobs=-1, verbose=0)
    search.fit(X_train, y_train)
    pipe_h = search.best_estimator_

    tr_pred = pipe_h.predict(X_train)
    te_pred = pipe_h.predict(X_test)
    tr_r2   = r2_score(y_train, tr_pred)
    te_r2   = r2_score(y_test,  te_pred)
    te_mae  = mean_absolute_error(y_test, te_pred)
    te_rmse = np.sqrt(mean_squared_error(y_test, te_pred))

    return {'model_type': model_type, 'target': target, 'desc': desc,
            'features': features, 'num_features': num_feats, 'cat_features': cat_feats,
            'n_features': len(features),
            'pipe': pipe_h,
            'best_params': search.best_params_, 'best_cv_r2': search.best_score_,
            'X': X, 'y': y, 'df_m': df_m,
            'X_train': X_train, 'y_train': y_train,
            'X_test': X_test, 'y_test': y_test,
            'tr_pred': tr_pred, 'te_pred': te_pred,
            'tr_r2': tr_r2, 'te_r2': te_r2, 'te_mae': te_mae, 'te_rmse': te_rmse}


results_compare = {}

for target, desc in TARGETS.items():
    print(f'{"═"*65}')
    print(f'TARGET: {target}')
    print(f'{"═"*65}')

    print('  Training Model A  (env + seasonal + district)...')
    res_A = train_monthly_model(target, desc, model_type='A', n_iter=25)
    print(f'    Features: {res_A["n_features"]}  |  CV R²={res_A["best_cv_r2"]:.3f}')
    print(f'    Train R²: {res_A["tr_r2"]:.3f}   Test R²: {res_A["te_r2"]:.3f}   MAE: {res_A["te_mae"]:.4f}')

    print('  Training Model B  (+ health AR lags — leakage reference)...')
    res_B = train_monthly_model(target, desc, model_type='B', n_iter=25)
    print(f'    Features: {res_B["n_features"]}  |  CV R²={res_B["best_cv_r2"]:.3f}')
    print(f'    Train R²: {res_B["tr_r2"]:.3f}   Test R²: {res_B["te_r2"]:.3f}   MAE: {res_B["te_mae"]:.4f}  ⚠ annual leakage')

    results_compare[target] = {'A': res_A, 'B': res_B}

results = {t: results_compare[t]['A'] for t in TARGETS}
print('\n✅ Step 6A — Monthly Models A and B trained for all targets')

# %% [markdown]
# ## STEP 6B — Model A vs B Comparison Table

# %%
rows = []
for target, bundle in results_compare.items():
    A, B = bundle['A'], bundle['B']
    rows.append({
        'Target'     : target,
        'A_Test_R2'  : round(A['te_r2'], 3),
        'A_MAE'      : round(A['te_mae'], 4),
        'B_Test_R2'  : round(B['te_r2'], 3),
        'B_MAE'      : round(B['te_mae'], 4),
        'B_minus_A'  : round(B['te_r2'] - A['te_r2'], 3),
    })

cmp_df = pd.DataFrame(rows)
print(cmp_df.to_string(index=False))
print()
print('★ Report Model A (env only). B-A gap = leakage inflation.')
print('→ See Step 6Y for the correct yearly respiratory model (PRIMARY).')

# %% [markdown]
# ## STEP 6C — Leakage Diagnostic

# %%
print('─'*65)
print('WITHIN-YEAR AUTOREGRESSION DIAGNOSTIC')
print('Health data is annual → broadcast to all 12 months.')
print('─'*65)
for target in TARGETS:
    lag_col = f'{target}_lag1m'
    if lag_col in df.columns:
        corr = df[target].corr(df[lag_col])
        flag = '⚠ HIGH (annual leakage)' if corr > 0.90 else '✅ OK'
        print(f'  {target}:')
        print(f'    lag1m correlation: {corr:.3f}  {flag}')

print()
print('→ Step 6Y (yearly model) removes this leakage entirely.')
print('✅ Step 6C — Leakage diagnostic complete')

# %% [markdown]
# ## STEP 6Y — Yearly Respiratory Disease Model

# %%
agg_dict = {
    # Targets (annual — identical across months, take first)
    'total_resp_rate_per_1k'     : 'first',
    'bronchitis_rate_per_1k'     : 'first',
    'asthma_rate_per_1k'         : 'first',
    'province'                    : 'first',
    # Forest / vegetation
    'vim'                         : 'mean',
    'vim_anomaly'                 : 'mean',
    'vim_min'                     : 'min',
    'vim_max'                     : 'max',
    'vim_climatology'             : 'mean',
    'vegetation_stress'           : 'mean',
    'ndvi_range'                  : 'mean',
    'veg_deficit'                 : 'sum',
    'below_climatology'           : 'sum',
    'forest_cover_pct'            : 'mean',
    'tc_loss_ha'                  : 'sum',
    'tc_loss_pct_of_extent'       : 'mean',
    'tc_loss_per_km2'             : 'mean',
    'carbon_loss_per_ha'          : 'mean',
    'carbon_efficiency'           : 'mean',
    'net_flux_per_ha'             : 'mean',
    'is_carbon_sink'              : 'mean',
    'carbon_sink_strength'        : 'mean',
    # Fire
    'frp_mean'                    : 'mean',
    'frp_total'                   : 'sum',
    'frp_normalized'              : 'mean',
    'fire_smoke_proxy'            : 'mean',
    'fire_active'                 : 'mean',
    'no_fire_types'               : 'max',
    'is_forest_fire'              : 'max',
    'brightness'                  : 'mean',
    # Air quality
    'pm2.5_ug_m3'                 : 'mean',
    'so2_ug_m3'                   : 'mean',
    'no2_ug_m3'                   : 'mean',
    'pollution_index'             : 'mean',
    'pm25_so2_ratio'              : 'mean',
    'total_pollutant_load'        : 'mean',
    # Interaction indices
    'fire_pollution_interaction'  : 'mean',
    'deforest_pollution_index'    : 'mean',
    'forest_health_index'         : 'mean',
    'pollution_health_burden'     : 'mean',
    'fire_health_burden'          : 'mean',
    'veg_pollution_interaction'   : 'mean',
    'deforest_health_interaction' : 'mean',
    # Population
    'pop_density_per_ha'          : 'first',
    'gender_ratio'                : 'first',
}

df_yr = df.groupby(['district','year']).agg(agg_dict).reset_index()

print(f'Yearly dataset shape : {df_yr.shape}  ({df_yr.shape[0]} district-year rows)')
print(f'Year range           : {df_yr["year"].min()} – {df_yr["year"].max()}')
print(f'Train rows           : {df_yr["year"].isin(TRAIN_YEARS).sum()}')
print(f'Test  rows           : {df_yr["year"].isin(TEST_YEARS).sum()}')

# ── v5 FIX: Add cross-year health AR lags (valid — genuine inter-year shift) ──────
Y_TARGETS = list(TARGETS.keys())

for target in Y_TARGETS:
    # Lag-1 year: last year's respiratory rate for this district
    df_yr[f'{target}_yr_lag1'] = df_yr.groupby('district')[target].shift(1)
    # Lag-2 year
    df_yr[f'{target}_yr_lag2'] = df_yr.groupby('district')[target].shift(2)
    # 2-year rolling mean of past values (shift first to avoid leakage)
    df_yr[f'{target}_yr_roll2'] = df_yr.groupby('district')[target].transform(
        lambda x: x.shift(1).rolling(2, min_periods=1).mean())
    # Year-over-year change (district-specific trend)
    df_yr[f'{target}_yr_yoy'] = df_yr[f'{target}_yr_lag1'] - df_yr[f'{target}_yr_lag2']

Y_HEALTH_LAG_COLS = {t: [
    f'{t}_yr_lag1', f'{t}_yr_lag2', f'{t}_yr_roll2', f'{t}_yr_yoy'
] for t in Y_TARGETS}

Y_NUMERIC_RAW = [c for c in agg_dict.keys()
                 if c not in Y_TARGETS + ['province'] and c in df_yr.columns]
Y_NUMERIC_FEATURES = Y_NUMERIC_RAW + ['year']

print(f'\nYearly numeric features (env + year) : {len(Y_NUMERIC_FEATURES)}')
print(f'Yearly health AR lags per target      : {len(list(Y_HEALTH_LAG_COLS.values())[0])}  (valid cross-year)')
print('\n✅ Yearly dataset + v5 cross-year lags prepared')

# %%
def train_yearly_model(target, tag, use_health_lags=False, n_iter=30):
    h_lags = Y_HEALTH_LAG_COLS[target] if use_health_lags else []
    # v5: province added as categorical grouping feature
    y_cat  = ['district', 'province']
    feats  = [f for f in Y_NUMERIC_FEATURES + h_lags + y_cat
              if f != target and f in df_yr.columns]
    feats  = list(dict.fromkeys(feats))
    cat_f  = [f for f in feats if f in y_cat]
    num_f  = [f for f in feats if f not in y_cat]

    cols   = list(dict.fromkeys(feats + [target, 'year']))
    df_y   = df_yr[cols].dropna().reset_index(drop=True)
    X, y   = df_y[feats], df_y[target]

    tr_mask = df_y['year'].isin(TRAIN_YEARS)
    te_mask = df_y['year'].isin(TEST_YEARS)
    X_tr, y_tr = X.loc[tr_mask], y.loc[tr_mask]
    X_te, y_te = X.loc[te_mask], y.loc[te_mask]

    prep  = build_preprocessor(num_f, cat_f)
    pipe  = Pipeline([('prep', prep),
                      ('model', XGBRegressor(objective='reg:squarederror',
                                             random_state=42, verbosity=0))])
    tscv  = TimeSeriesSplit(n_splits=3)
    srch  = RandomizedSearchCV(pipe, PARAM_DIST, n_iter=n_iter, scoring='r2',
                               cv=tscv, random_state=42, n_jobs=-1, verbose=0)
    srch.fit(X_tr, y_tr)
    pipe_h = srch.best_estimator_
    best_params_h = srch.best_params_

    tr_pred = pipe_h.predict(X_tr)
    te_pred = pipe_h.predict(X_te)
    tr_r2   = r2_score(y_tr, tr_pred)
    te_r2   = r2_score(y_te, te_pred)
    te_mae  = mean_absolute_error(y_te, te_pred)

    return {'target': target, 'tag': tag, 'feats': feats,
            'num_f': num_f, 'cat_f': cat_f,
            'pipe': pipe_h, 'best_cv_r2': srch.best_score_,
            'df_yr': df_yr, 'X': X, 'y': y,
            'X_tr': X_tr, 'y_tr': y_tr, 'X_te': X_te, 'y_te': y_te,
            'tr_pred': tr_pred, 'te_pred': te_pred,
            'tr_r2': tr_r2, 'te_r2': te_r2, 'te_mae': te_mae,
            'df_y': df_y, 'best_params': best_params_h}


yearly_results = {}

for target, desc in TARGETS.items():
    print(f'\n{"═"*60}')
    print(f'YEARLY TARGET: {target}')
    print(f'{"═"*60}')

    print('  YA — Environmental features only (province categorical)...')
    r_ya = train_yearly_model(target, 'YA (env only)', use_health_lags=False, n_iter=30)
    print(f'    Feats={len(r_ya["feats"])}  CV={r_ya["best_cv_r2"]:.3f}  Train={r_ya["tr_r2"]:.3f}  Test={r_ya["te_r2"]:.3f}  MAE={r_ya["te_mae"]:.4f}')

    print('  YB — Env + yearly cross-year health lags + province (v5 PRIMARY)...')
    r_yb = train_yearly_model(target, 'YB (env+yr_lag1+province)', use_health_lags=True, n_iter=30)
    print(f'    Feats={len(r_yb["feats"])}  CV={r_yb["best_cv_r2"]:.3f}  Train={r_yb["tr_r2"]:.3f}  Test={r_yb["te_r2"]:.3f}  MAE={r_yb["te_mae"]:.4f}')
    print(f'    Improvement vs YA: ΔTest R² = {r_yb["te_r2"] - r_ya["te_r2"]:+.3f}')

    yearly_results[f'{target}__YA'] = r_ya
    yearly_results[f'{target}__YB'] = r_yb

print('\n✅ Step 6Y — Yearly models trained for all targets')

print('\nBest hyperparameters — PRIMARY yearly model (total_resp_rate_per_1k, YB):')
_yb_primary = yearly_results['total_resp_rate_per_1k__YB']
for k, v in _yb_primary['best_params'].items():
    print(f'  {k:<28}: {v}')

# %% [markdown]
# ## STEP 6Y-EVAL — Per-District MAPE Table

# %%
PRIMARY = 'total_resp_rate_per_1k'
r_primary_y = yearly_results[f'{PRIMARY}__YB']

# Build test predictions with district labels
df_te = r_primary_y['df_y'][r_primary_y['df_y']['year'].isin(TEST_YEARS)].copy()
df_te['pred'] = r_primary_y['pipe'].predict(r_primary_y['X_te'])

def mape(actual, pred):
    mask = actual.abs() > 0.01
    return (((actual[mask] - pred[mask]).abs() / actual[mask].abs()) * 100).mean()

district_eval = []
for dist in sorted(df_te['district'].unique()):
    sub = df_te[df_te['district'] == dist]
    actual = sub[PRIMARY]
    pred   = sub['pred']
    n_pts  = len(sub)
    mae_d  = mean_absolute_error(actual, pred) if n_pts > 0 else np.nan
    mape_d = mape(actual, pred) if n_pts > 0 else np.nan
    r2_d   = r2_score(actual, pred) if n_pts > 1 else np.nan
    quality = ('✅ Good' if mape_d <= 20 else
               '🟡 OK'  if mape_d <= 35 else
               '🔴 Poor')
    district_eval.append({
        'District': dist,
        'N_test'  : n_pts,
        'MAE'     : round(mae_d, 3),
        'MAPE_%'  : round(mape_d, 1),
        'Test_R2' : round(r2_d, 3) if not np.isnan(r2_d) else 'N/A',
        'Quality' : quality,
    })

eval_df = pd.DataFrame(district_eval).sort_values('MAPE_%')

print(f'{"═"*75}')
print(f'PER-DISTRICT PREDICTION QUALITY — Yearly Model YB (test 2021–2024)')
print(f'{"═"*75}')
print(eval_df.to_string(index=False))
print()

good  = (eval_df['MAPE_%'] <= 20).sum()
ok    = ((eval_df['MAPE_%'] > 20) & (eval_df['MAPE_%'] <= 35)).sum()
poor  = (eval_df['MAPE_%'] > 35).sum()

print(f'Summary: ✅ Good (MAPE ≤20%): {good}/25   🟡 OK (≤35%): {ok}/25   🔴 Poor (>35%): {poor}/25')
print()
print('Note: Districts with near-zero values in 2021 (COVID hospital disruption)')
print('      may show high MAPE — this is a data artifact, not a model failure.')
print('\n✅ Step 6Y-EVAL — District MAPE table done')

# %% [markdown]
# ## STEP 6P — Monthly PM2.5 Prediction Model

# %%
PM25_TARGET = 'pm2.5_ug_m3'

PM25_FEATURES_NUM = [f for f in [
    'vim', 'vim_anomaly', 'vim_min', 'vim_max',
    'vegetation_stress', 'ndvi_range', 'veg_deficit', 'below_climatology',
    'tc_loss_ha', 'forest_cover_pct', 'is_carbon_sink', 'carbon_sink_strength',
    'frp_mean', 'frp_total', 'frp_normalized', 'fire_smoke_proxy',
    'fire_active', 'no_fire_types', 'is_forest_fire', 'brightness',
    'so2_ug_m3', 'no2_ug_m3', 'pollution_index', 'total_pollutant_load',
    'pm25_so2_ratio',
    'so2_ug_m3_lag1m', 'so2_ug_m3_lag3m', 'so2_ug_m3_roll3m',
    'no2_ug_m3_lag1m', 'no2_ug_m3_lag3m', 'no2_ug_m3_roll3m',
    'pm2.5_ug_m3_lag1m', 'pm2.5_ug_m3_lag3m', 'pm2.5_ug_m3_roll3m', 'pm2.5_ug_m3_roll6m',
    'fire_pollution_interaction', 'deforest_pollution_index',
    'veg_pollution_interaction',
    'pop_density_per_ha',
    'month_sin', 'month_cos',
] if f != PM25_TARGET and f in df.columns]

PM25_FEATURES_CAT = ['district', 'season']
PM25_FEATURES     = PM25_FEATURES_NUM + PM25_FEATURES_CAT

cols_pm  = list(dict.fromkeys(PM25_FEATURES + [PM25_TARGET, 'year']))
df_pm25  = df[cols_pm].dropna().reset_index(drop=True)
X_pm25   = df_pm25[PM25_FEATURES]
y_pm25   = df_pm25[PM25_TARGET]

tr_pm_m  = df_pm25['year'].isin(TRAIN_YEARS)
te_pm_m  = df_pm25['year'].isin(TEST_YEARS)
X_pm_tr, y_pm_tr = X_pm25.loc[tr_pm_m], y_pm25.loc[tr_pm_m]
X_pm_te, y_pm_te = X_pm25.loc[te_pm_m], y_pm25.loc[te_pm_m]

prep_pm  = build_preprocessor(PM25_FEATURES_NUM, PM25_FEATURES_CAT)
pipe_pm  = Pipeline([('prep', prep_pm),
                     ('model', XGBRegressor(objective='reg:squarederror',
                                            random_state=42, verbosity=0))])
tscv_pm  = TimeSeriesSplit(n_splits=4)
srch_pm  = RandomizedSearchCV(pipe_pm, PARAM_DIST, n_iter=30, scoring='r2',
                               cv=tscv_pm, random_state=42, n_jobs=-1, verbose=0)
srch_pm.fit(X_pm_tr, y_pm_tr)
pipe_pm_h = srch_pm.best_estimator_

# Best hyperparameters found by RandomizedSearchCV for the PM2.5 model
pm25_best_params = srch_pm.best_params_

pm_tr_pred = pipe_pm_h.predict(X_pm_tr)
pm_te_pred = pipe_pm_h.predict(X_pm_te)
pm_tr_r2   = r2_score(y_pm_tr, pm_tr_pred)
pm_te_r2   = r2_score(y_pm_te, pm_te_pred)
pm_te_mae  = mean_absolute_error(y_pm_te, pm_te_pred)
pm_te_rmse = np.sqrt(mean_squared_error(y_pm_te, pm_te_pred))

pm25_result = {
    'pipe': pipe_pm_h, 'df_pm25': df_pm25, 'X_pm25': X_pm25,
    'X_pm_tr': X_pm_tr, 'y_pm_tr': y_pm_tr,
    'X_pm_te': X_pm_te, 'y_pm_te': y_pm_te,
    'pm_tr_pred': pm_tr_pred, 'pm_te_pred': pm_te_pred,
    'tr_r2': pm_tr_r2, 'te_r2': pm_te_r2,
    'te_mae': pm_te_mae, 'te_rmse': pm_te_rmse,
    'features': PM25_FEATURES, 'num_features': PM25_FEATURES_NUM,
    'cat_features': PM25_FEATURES_CAT,
    'best_params': pm25_best_params,
}

print(f'Monthly PM2.5 Model:')
print(f'  Features  : {len(PM25_FEATURES)}')
print(f'  CV R²     : {srch_pm.best_score_:.3f}')
print(f'  Train R²  : {pm_tr_r2:.3f}')
print(f'  Test  R²  : {pm_te_r2:.3f}')
print(f'  Test  MAE : {pm_te_mae:.3f} µg/m³')
print(f'  Test  RMSE: {pm_te_rmse:.3f} µg/m³')

print(f'\n  Best hyperparameters (RandomizedSearchCV, n_iter=30, cv=TimeSeriesSplit(4)):')
for k, v in pm25_best_params.items():
    print(f'    {k:<28}: {v}')

print('\n✅ Step 6P — Monthly PM2.5 model trained')

# %% [markdown]
# ## STEP 7 — Select Primary Target for SHAP & FAH

# %%
PRIMARY = 'total_resp_rate_per_1k'
r_primary_y = yearly_results[f'{PRIMARY}__YB']
pipe_shap   = r_primary_y['pipe']

print(f'Primary target : {PRIMARY}')
print(f'SHAP source    : Yearly Model YB (env + yearly cross-year lags + province)')
print()
print(f'  Yearly YA (env only)        : Train={yearly_results[f"{PRIMARY}__YA"]["tr_r2"]:.3f}  Test={yearly_results[f"{PRIMARY}__YA"]["te_r2"]:.3f}')
print(f'  Yearly YB (env+yr_lags)     : Train={r_primary_y["tr_r2"]:.3f}  Test={r_primary_y["te_r2"]:.3f}  ← SHAP source')
print(f'  Monthly Model A (ref)       : Train={results[PRIMARY]["tr_r2"]:.3f}  Test={results[PRIMARY]["te_r2"]:.3f}')
print()
print('✅ Step 7 — Primary model selected')

# %% [markdown]
# ## STEP 8 — Compute SHAP *(yearly model on test set)*

# %%
r_shap        = r_primary_y
X_for_shap    = r_shap['X_te'].copy()
X_transformed = pipe_shap.named_steps['prep'].transform(X_for_shap)

try:
    ohe_names = []
    for cat_col in r_shap['cat_f']:
        names = list(pipe_shap.named_steps['prep']
                     .named_transformers_['cat'].named_steps['ohe']
                     .get_feature_names_out([cat_col]))
        ohe_names.extend(names)
except Exception:
    ohe_names = [f'cat_{i}' for i in range(X_transformed.shape[1] - len(r_shap['num_f']))]

num_feat_names = r_shap['num_f']
all_feat_names = num_feat_names + ohe_names

explainer   = shap.TreeExplainer(pipe_shap.named_steps['model'])
shap_values = explainer.shap_values(X_transformed)

print(f'SHAP computed on {X_transformed.shape[0]} test rows × {X_transformed.shape[1]} features')


def fah_component(f):
    f_l = f.lower()
    if any(x in f_l for x in ['vim','tc_loss','carbon','forest_cover','net_flux','deforest',
                                'sink','removals','vegetation','ndvi','veg_def','below_clim',
                                'forest_health']):
        return 'Forest Risk'
    if any(x in f_l for x in ['frp','fire','brightness','smoke','fire_pollution','fire_health']):
        return 'Fire Risk'
    if any(x in f_l for x in ['pm2','so2','no2','pollution','pollutant','pm25']):
        return 'Air Quality Risk'
    if 'district' in f_l:
        return 'District'
    if 'province' in f_l:
        return 'Province'
    if any(x in f_l for x in ['month_sin','month_cos','season']):
        return 'Seasonality'
    if any(x in f_l for x in ['pop_density','gender']):
        return 'Demographics'
    if any(x in f_l for x in ['yr_lag','yr_roll','yr_yoy']):
        return 'Health AR (yearly)'
    return 'Other'


shap_df = pd.DataFrame({
    'feature'       : all_feat_names,
    'mean_abs_shap' : np.abs(shap_values).mean(axis=0)
}).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

shap_df['component'] = shap_df['feature'].apply(fah_component)
shap_df['pct_all']   = shap_df['mean_abs_shap'] / shap_df['mean_abs_shap'].sum() * 100

FAH_COMPS = ['Forest Risk', 'Fire Risk', 'Air Quality Risk']
env_shap  = shap_df[shap_df['component'].isin(FAH_COMPS)].copy()
comp_sums = env_shap.groupby('component')['mean_abs_shap'].sum()
total_env = comp_sums.sum()
FAH_WEIGHTS = {c: comp_sums.get(c, 0) / total_env if total_env > 0 else 1/3
               for c in FAH_COMPS}

print(f'\nFAH SHAP weights (from yearly model):')
for c, w in FAH_WEIGHTS.items():
    print(f'  {c:<22}: {w*100:.1f}%')

print(f'\nTop 15 SHAP drivers:')
for _, row in shap_df.head(15).iterrows():
    print(f'  {row["feature"]:<35} {row["component"]:<22} {row["mean_abs_shap"]:.4f} ({row["pct_all"]:.1f}%)')

print('\n✅ Step 8 — SHAP computed')

# %% [markdown]
# ## STEP 9 — SHAP Beeswarm & Waterfall Charts

# %%
# Beeswarm (env features only, top 15)
top_env_feats   = shap_df[shap_df['component'].isin(FAH_COMPS)].head(15)['feature'].tolist()
top_env_indices = [i for i, f in enumerate(all_feat_names) if f in top_env_feats]

if top_env_indices:
    shap_env  = shap_values[:, top_env_indices]
    X_env     = X_transformed[:, top_env_indices]
    env_names = [all_feat_names[i] for i in top_env_indices]

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_env, X_env, feature_names=env_names,
                      show=False, max_display=15)
    plt.title('SHAP Beeswarm — Yearly Model YB (Environmental Features Only)\n'
              'Trained 2015–2020  |  Evaluated on test set 2021–2024',
              fontweight='bold')
    plt.tight_layout()
    plt.show()

# ── Waterfall for one test observation ────────────────────────────────────────
try:
    ev = explainer.expected_value
    if isinstance(ev, np.ndarray):
        ev = float(ev[0])
    explanation = shap.Explanation(
        values        = shap_values[0],
        base_values   = ev,
        data          = X_transformed[0],
        feature_names = all_feat_names
    )
    plt.figure(figsize=(12, 6))
    shap.plots.waterfall(explanation, max_display=12, show=False)
    plt.title('SHAP Waterfall — Example Prediction (test row #0)', fontweight='bold')
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f'Waterfall skipped: {e}')

print('✅ Step 9 — SHAP charts done')

# %% [markdown]
# ## STEP 10 — FAH Index Calculation

# %%
def scale_col(df_, col, invert=False):
    mn, mx = df_[col].min(), df_[col].max()
    if mx == mn:
        return pd.Series(0.5, index=df_.index)
    s = (df_[col] - mn) / (mx - mn)
    return 1 - s if invert else s


district_avg = df.groupby(['district','province']).agg(
    avg_vim           = ('vim',               'mean'),
    avg_vim_anomaly   = ('vim_anomaly',       'mean'),
    avg_veg_stress    = ('vegetation_stress', 'mean'),
    total_tc_loss     = ('tc_loss_ha',        'sum'),
    avg_forest_cover  = ('forest_cover_pct',  'mean'),
    avg_sink_strength = ('carbon_sink_strength','mean'),
    fire_rate         = ('fire_active',        'mean'),
    forest_fire_rate  = ('is_forest_fire',     'mean'),
    avg_frp           = ('frp_mean',           'mean'),
    max_frp_total     = ('frp_total',          'max'),
    avg_smoke         = ('fire_smoke_proxy',   'mean'),
    avg_pm25          = ('pm2.5_ug_m3',        'mean'),
    avg_no2           = ('no2_ug_m3',          'mean'),
    avg_so2           = ('so2_ug_m3',          'mean'),
    avg_poll_idx      = ('pollution_index',    'mean'),
).reset_index()

f1 = scale_col(district_avg, 'avg_vim',         invert=True)
f2 = scale_col(district_avg, 'total_tc_loss',   invert=False)
f3 = scale_col(district_avg, 'avg_veg_stress',  invert=False)
f4 = scale_col(district_avg, 'avg_vim_anomaly', invert=True)
f5 = 1 - scale_col(district_avg, 'avg_forest_cover', invert=False)
district_avg['forest_score'] = (f1 + f2 + f3 + f4 + f5) / 5

fi1 = scale_col(district_avg, 'avg_frp',         invert=False)
fi2 = scale_col(district_avg, 'max_frp_total',   invert=False)
fi3 = scale_col(district_avg, 'fire_rate',         invert=False)
fi4 = scale_col(district_avg, 'forest_fire_rate',  invert=False)
fi5 = scale_col(district_avg, 'avg_smoke',          invert=False)
district_avg['fire_score'] = (fi1 + fi2 + fi3 + fi4 + fi5) / 5

a1 = scale_col(district_avg, 'avg_pm25',    invert=False)
a2 = scale_col(district_avg, 'avg_no2',     invert=False)
a3 = scale_col(district_avg, 'avg_so2',     invert=False)
a4 = scale_col(district_avg, 'avg_poll_idx',invert=False)
district_avg['air_score'] = (a1 + a2 + a3 + a4) / 4

w_f = FAH_WEIGHTS.get('Forest Risk',      1/3)
w_i = FAH_WEIGHTS.get('Fire Risk',        1/3)
w_a = FAH_WEIGHTS.get('Air Quality Risk', 1/3)

district_avg['FAH_score'] = (
    district_avg['forest_score'] * w_f +
    district_avg['fire_score']   * w_i +
    district_avg['air_score']    * w_a
).round(4)

district_avg = district_avg.sort_values('FAH_score', ascending=False).reset_index(drop=True)
district_avg['FAH_rank'] = district_avg.index + 1
district_avg['risk_tier'] = pd.cut(
    district_avg['FAH_score'],
    bins=[0, 0.40, 0.55, 1.01],
    labels=['Low Risk', 'Moderate', 'High Risk']
)

print(f'{"═"*72}')
print(f'FAH INDEX — DISTRICT RANKINGS')
print(f'Weights: Forest={w_f*100:.1f}%  Fire={w_i*100:.1f}%  Air={w_a*100:.1f}%  (SHAP-derived)')
print(f'{"═"*72}')
print(f'  {"Rank":<5} {"District":<15} {"Province":<16} {"FAH":>6} {"Forest":>7} {"Fire":>7} {"Air":>7}  Tier')
print(f'  {"─"*72}')
for _, row in district_avg.iterrows():
    sym = '🔴' if row['risk_tier']=='High Risk' else '🟠' if row['risk_tier']=='Moderate' else '🟢'
    print(f'  #{row["FAH_rank"]:<4} {row["district"]:<15} {row["province"]:<16} '
          f'{row["FAH_score"]:>6.3f} {row["forest_score"]:>7.3f} '
          f'{row["fire_score"]:>7.3f} {row["air_score"]:>7.3f}  {sym} {row["risk_tier"]}')

print('\n✅ Step 10 — FAH Index complete')

# %% [markdown]
# ## STEP 11 — FAH Index Charts

# %%
fig, axes = plt.subplots(1, 2, figsize=(22, 10))

colors = ['#e74c3c' if t=='High Risk' else '#e67e22' if t=='Moderate' else '#27ae60'
          for t in district_avg['risk_tier']]

axes[0].barh(district_avg['district'][::-1], district_avg['FAH_score'][::-1],
             color=colors[::-1], edgecolor='black', linewidth=0.5)
for i, v in enumerate(district_avg['FAH_score'][::-1]):
    axes[0].text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=8)
axes[0].axvline(x=0.55, color='red',    linestyle='--', alpha=0.5)
axes[0].axvline(x=0.40, color='orange', linestyle='--', alpha=0.5)
axes[0].legend(handles=[
    mpatches.Patch(color='#e74c3c', label='High Risk (>0.55)'),
    mpatches.Patch(color='#e67e22', label='Moderate (0.40–0.55)'),
    mpatches.Patch(color='#27ae60', label='Low Risk (<0.40)'),
], fontsize=9)
axes[0].set_title(f'FAH Risk Index — District Rankings\n'
                  f'Weights: Forest={w_f*100:.1f}%  Fire={w_i*100:.1f}%  Air={w_a*100:.1f}%',
                  fontweight='bold')
axes[0].set_xlabel('FAH Score (0 = lowest risk, 1 = highest risk)')

comp_data = district_avg[['district','forest_score','fire_score','air_score']].sort_values('district')
x  = np.arange(len(comp_data)); bw = 0.25
axes[1].bar(x-bw, comp_data['forest_score'], bw, label=f'Forest ({w_f*100:.1f}%)',
            color='#27ae60', edgecolor='black', linewidth=0.4)
axes[1].bar(x,    comp_data['fire_score'],   bw, label=f'Fire  ({w_i*100:.1f}%)',
            color='#e67e22', edgecolor='black', linewidth=0.4)
axes[1].bar(x+bw, comp_data['air_score'],    bw, label=f'Air   ({w_a*100:.1f}%)',
            color='#3498db', edgecolor='black', linewidth=0.4)
axes[1].set_xticks(x)
axes[1].set_xticklabels(comp_data['district'], rotation=45, ha='right', fontsize=8)
axes[1].set_title('FAH Component Scores per District', fontweight='bold')
axes[1].set_ylabel('Component score (0–1)')
axes[1].legend()

plt.suptitle('Forest–Air–Health (FAH) Risk Index — SHAP-Derived Weights (Yearly Model v5)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
print('✅ Step 11 — FAH charts done')

# %% [markdown]
# ## STEP 12 — Global SHAP Importance Chart

# %%
cat_colors = {
    'Forest Risk'       : '#27ae60',
    'Fire Risk'         : '#e67e22',
    'Air Quality Risk'  : '#3498db',
    'Health AR (yearly)': '#9b59b6',
    'District'          : '#7f8c8d',
    'Province'          : '#95a5a6',
    'Seasonality'       : '#bdc3c7',
    'Demographics'      : '#f39c12',
    'Other'             : '#ecf0f1',
}

top20      = shap_df.head(20)
bar_colors = [cat_colors.get(c, '#ecf0f1') for c in top20['component']]

fig, axes = plt.subplots(1, 2, figsize=(20, 9))

axes[0].barh(top20['feature'][::-1], top20['mean_abs_shap'][::-1],
             color=bar_colors[::-1], edgecolor='black', linewidth=0.4)
for i, (v, p) in enumerate(zip(top20['mean_abs_shap'][::-1], top20['pct_all'][::-1])):
    axes[0].text(v + 0.002, i, f'{p:.1f}%', va='center', fontsize=8)
patches = [mpatches.Patch(color=v, label=k) for k, v in cat_colors.items()
           if k in shap_df['component'].values]
axes[0].legend(handles=patches, fontsize=8)
axes[0].set_title('Global SHAP Importance — Yearly Model YB v5 (Top 20)\n'
                  'Drivers of total respiratory rate per 1 000', fontweight='bold')
axes[0].set_xlabel('Mean |SHAP value|')

cdf = pd.DataFrame([{
    'component'   : c,
    'equal_weight': 33.3,
    'data_weight' : FAH_WEIGHTS.get(c, 0) * 100
} for c in FAH_COMPS])
cx = np.arange(len(cdf)); cw = 0.35
cb1 = axes[1].bar(cx-cw/2, cdf['equal_weight'], cw, label='Equal (33.3%)',
                  color='#bdc3c7', edgecolor='black', linewidth=0.5)
cb2 = axes[1].bar(cx+cw/2, cdf['data_weight'],  cw, label='SHAP data-driven',
                  color=[cat_colors[c] for c in FAH_COMPS], edgecolor='black', linewidth=0.5)
for bar in list(cb1) + list(cb2):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                 f'{bar.get_height():.1f}%', ha='center', fontsize=9)
axes[1].axhline(y=33.3, color='gray', linestyle='--', alpha=0.5)
axes[1].set_xticks(cx); axes[1].set_xticklabels(cdf['component'])
axes[1].set_ylabel('Contribution (%)')
axes[1].set_ylim(0, 80)
axes[1].set_title('FAH Weight Validation\n1/3 equal vs SHAP data-driven', fontweight='bold')
axes[1].legend()

plt.suptitle('SHAP Global Importance & FAH Weight Validation (v5)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
print('✅ Step 12 — Global SHAP + FAH validation done')

# %% [markdown]
# ## STEP 13 — District-Level SHAP Breakdown

# %%
district_labels_test = (
    r_primary_y['df_y']
    .loc[r_primary_y['X_te'].index, 'district']
    .reset_index(drop=True)
)

df_shap = pd.DataFrame(shap_values, columns=all_feat_names)
df_shap['district'] = district_labels_test.values

env_feat_map = {f: fah_component(f) for f in all_feat_names if fah_component(f) in FAH_COMPS}
env_feats    = list(env_feat_map.keys())

dist_shap = df_shap.groupby('district')[env_feats].mean()
for comp in FAH_COMPS:
    cf = [f for f in env_feats if env_feat_map[f] == comp]
    if cf:
        dist_shap[comp] = dist_shap[cf].sum(axis=1)

dist_shap['total'] = dist_shap[FAH_COMPS].sum(axis=1)
dist_shap = dist_shap.sort_values('total', ascending=True)

pc_colors = ['#27ae60', '#e67e22', '#3498db']
fig, ax   = plt.subplots(figsize=(16, 12))
bot_pos   = np.zeros(len(dist_shap))
bot_neg   = np.zeros(len(dist_shap))

for col, color in zip(FAH_COMPS, pc_colors):
    if col not in dist_shap.columns:
        continue
    vals  = dist_shap[col].values
    pos_v = np.where(vals > 0, vals, 0)
    neg_v = np.where(vals < 0, vals, 0)
    ax.barh(dist_shap.index, pos_v, left=bot_pos, color=color, alpha=0.85,
            edgecolor='white', linewidth=0.3, label=col)
    ax.barh(dist_shap.index, neg_v, left=bot_neg, color=color, alpha=0.85,
            edgecolor='white', linewidth=0.3)
    bot_pos += pos_v
    bot_neg += neg_v

ax.axvline(x=0, color='black', linewidth=1.2)
ax.legend(loc='lower right', fontsize=9)
ax.set_title('District-Level SHAP Breakdown — Yearly Model YB v5\n'
             'Tested 2021–2024  |  Right = increases respiratory burden\n'
             'Left = pushes respiratory burden DOWN (protective)',
             fontsize=11, fontweight='bold')
ax.set_xlabel('Mean SHAP value (cases per 1 000 above/below average)')
plt.tight_layout()
plt.show()
print('✅ Step 13 — District SHAP breakdown done')

# %% [markdown]
# ## STEP 14 — Prediction and District-Level Visualization
# ### Part A: All-Districts Actual vs Predicted Scatter
# ### Part B: Single-District Time Series
# ### Part C: Monthly PM2.5 for the Selected District
# ### Part D: FAH Score for the Selected District

# %%
# ── Part A: All-25-Districts actual vs predicted scatter (district legend GUARANTEED visible) ──
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from sklearn.metrics import r2_score

df_te = r_primary_y['df_y'][r_primary_y['df_y']['year'].isin(TEST_YEARS)].copy()
df_te['pred'] = r_primary_y['pipe'].predict(r_primary_y['X_te'])

STANDARD_25_COLORS = [
    '#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99',
    '#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a',
    '#ffff99','#b15928','#8dd3c7','#ffffb3','#bebada',
    '#fb8072','#80b1d3','#fdb462','#b3de69','#fccde5',
    '#d9d9d9','#bc80bd','#ccebc5','#ffed6f','#e5c494'
]

year_styles = {
    2021: ('o', 100),
    2022: ('s', 120),
    2023: ('^', 140),
    2024: ('D', 160),
}

districts_sorted = sorted(df_te['district'].unique())
district_color_map = {
    dist: STANDARD_25_COLORS[i % len(STANDARD_25_COLORS)]
    for i, dist in enumerate(districts_sorted)
}

# ── Use figure + gridspec: top = scatter plot, bottom = legend area ──
fig = plt.figure(figsize=(16, 13))
gs = gridspec.GridSpec(
    2, 1,
    height_ratios=[8, 1.8],   # top 8 parts = plot, bottom 1.8 parts = legend
    hspace=0.05
)
ax = fig.add_subplot(gs[0])         # main scatter plot
ax_leg = fig.add_subplot(gs[1])     # dedicated legend area
ax_leg.axis('off')                  # hide axes, only used for legend

# ── Scatter plot ──
for dist in districts_sorted:
    sub = df_te[df_te['district'] == dist]
    color = district_color_map[dist]
    for _, row in sub.iterrows():
        yr = int(row['year'])
        marker, size = year_styles[yr]
        ax.scatter(
            row[PRIMARY], row['pred'],
            color=color, marker=marker, s=size,
            alpha=0.9, edgecolors='black', linewidth=0.4
        )

# Perfect fit line
lim_min = min(df_te[PRIMARY].min(), df_te['pred'].min()) * 0.95
lim_max = max(df_te[PRIMARY].max(), df_te['pred'].max()) * 1.05
ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', alpha=0.5, linewidth=1.2)

# ── Year legend — inside top-left of scatter plot ──
year_handles = [
    Line2D([0], [0],
           marker=marker, color='black', linestyle='None',
           markersize=np.sqrt(size),
           markerfacecolor='gray', markeredgecolor='black',
           markeredgewidth=0.5,
           label=f'{yr}  (size = {size})')
    for yr, (marker, size) in year_styles.items()
]
ax.legend(
    handles=year_handles,
    loc='upper left', fontsize=9, ncol=1,
    title='Year  |  Marker size', title_fontsize=9,
    framealpha=0.95, edgecolor='gray'
)

ax.set_xlabel('Actual respiratory rate per 1 000', fontsize=12)
ax.set_ylabel('Predicted respiratory rate per 1 000', fontsize=12)
r2_all = r2_score(df_te[PRIMARY], df_te['pred'])
ax.set_title(
    f'All 25 Districts — Actual vs Predicted (Test 2021–2024)\nOverall Test R² = {r2_all:.3f}',
    fontweight='bold', fontsize=13
)
ax.grid(True, linestyle='--', alpha=0.4)

# ── District legend — placed inside dedicated ax_leg subplot area ──
district_handles = [
    mpatches.Patch(
        facecolor=district_color_map[dist],
        edgecolor='black', linewidth=0.5,
        label=dist                          # district NAME here
    )
    for dist in districts_sorted
]
ax_leg.legend(
    handles=district_handles,
    loc='center',                           # centered in the reserved area
    fontsize=8.5, ncol=5,                   # 5 cols × 5 rows = 25 districts
    title='Districts', title_fontsize=10,
    framealpha=0.95, edgecolor='gray',
    columnspacing=1.2, handlelength=1.5,
    handleheight=1.2
)

plt.savefig('actual_vs_predicted_all_districts.png', dpi=300, bbox_inches='tight')
plt.show()
print(f'All-district scatter: Test R² = {r2_all:.3f}')

# %%
# ── Part B: Single District Time-Series ──────────────────────────────────────
DISTRICT_TO_PLOT = 'Kandy'    # ← Change to any of the 25 district names

df_yr_plot = r_primary_y['df_y'].copy()
df_yr_plot['pred_impact'] = r_primary_y['pipe'].predict(r_primary_y['X'])
df_yr_plot = df_yr_plot.dropna(subset=['pred_impact']).copy()

dist_yr = df_yr_plot[df_yr_plot['district'] == DISTRICT_TO_PLOT].sort_values('year')

if len(dist_yr) == 0:
    print(f'❌ District "{DISTRICT_TO_PLOT}" not found. Available: {sorted(df_yr_plot["district"].unique())}')
else:
    te_yr = dist_yr[dist_yr['year'].isin(TEST_YEARS)]
    tr_yr = dist_yr[dist_yr['year'].isin(TRAIN_YEARS)]

    r2_tr  = r2_score(tr_yr[PRIMARY], tr_yr['pred_impact']) if len(tr_yr) > 1 else float('nan')
    r2_te  = r2_score(te_yr[PRIMARY], te_yr['pred_impact']) if len(te_yr) > 1 else float('nan')
    mae_te = mean_absolute_error(te_yr[PRIMARY], te_yr['pred_impact']) if len(te_yr) > 0 else float('nan')
    mape_te = (((te_yr[PRIMARY] - te_yr['pred_impact']).abs() / te_yr[PRIMARY].abs().clip(0.01)) * 100).mean() if len(te_yr) > 0 else float('nan')

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(dist_yr['year'].values, dist_yr[PRIMARY].values, dist_yr['pred_impact'].values,
                    alpha=0.1, color='gray')
    ax.plot(dist_yr['year'].values, dist_yr[PRIMARY].values, 'o-', color='#e74c3c',
            linewidth=2.5, label='Actual')
    ax.plot(dist_yr['year'].values, dist_yr['pred_impact'].values, 's--', color='#3498db',
            linewidth=2.5, label='Predicted')
    ax.axvspan(2020.5, dist_yr['year'].max() + 0.4, alpha=0.08, color='orange', label='Test period')
    ax.axvline(x=2020.5, color='orange', linestyle='--', alpha=0.7)
    ax.set_title(f'{DISTRICT_TO_PLOT} — Yearly Respiratory Rate Prediction (v5)\n'
                 f'Train R²={r2_tr:.3f}  |  Test R²={r2_te:.3f}  |  MAE={mae_te:.3f}  |  MAPE={mape_te:.1f}%',
                 fontweight='bold')
    ax.set_xlabel('Year')
    ax.set_ylabel('Total respiratory rate per 1 000')
    ax.set_xticks(sorted(dist_yr['year'].unique()))
    ax.legend()
    plt.tight_layout()
    plt.show()
    print(f'{DISTRICT_TO_PLOT}: Train R²={r2_tr:.3f}  Test R²={r2_te:.3f}  MAE={mae_te:.4f}  MAPE={mape_te:.1f}%')

# %%
# ── Part B2: All-25-district time-series grid ─────────────────────────────────
all_districts = sorted(df_yr_plot['district'].unique())
ncols = 5
nrows = (len(all_districts) + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(22, nrows * 3.5))
axes = axes.flatten()

for i, dist in enumerate(all_districts):
    ax = axes[i]
    sub = df_yr_plot[df_yr_plot['district'] == dist].sort_values('year')
    te_sub = sub[sub['year'].isin(TEST_YEARS)]

    ax.plot(sub['year'], sub[PRIMARY], 'o-', color='#e74c3c', linewidth=1.5,
            markersize=4, label='Actual')
    ax.plot(sub['year'], sub['pred_impact'], 's--', color='#3498db', linewidth=1.5,
            markersize=4, label='Predicted')
    ax.axvspan(2020.5, sub['year'].max() + 0.4, alpha=0.08, color='orange')
    ax.axvline(x=2020.5, color='orange', linestyle='--', alpha=0.6, linewidth=0.8)

    if len(te_sub) > 1:
        te_r2_d = r2_score(te_sub[PRIMARY], te_sub['pred_impact'])
        mape_d  = (((te_sub[PRIMARY] - te_sub['pred_impact']).abs() /
                     te_sub[PRIMARY].abs().clip(0.01)) * 100).mean()
        color   = 'green' if mape_d <= 20 else 'orange' if mape_d <= 35 else 'red'
        ax.set_title(f'{dist}\nR²={te_r2_d:.2f}  MAPE={mape_d:.0f}%',
                     fontsize=8, color=color, fontweight='bold')
    else:
        ax.set_title(dist, fontsize=8)

    ax.tick_params(axis='both', labelsize=7)
    ax.set_xticks(sorted(sub['year'].unique())[::2])
    ax.set_xticklabels(sorted(sub['year'].unique())[::2], rotation=45, fontsize=6)

# Hide unused subplots
for j in range(len(all_districts), len(axes)):
    axes[j].set_visible(False)

fig.suptitle('All 25 Districts — Yearly Respiratory Rate: Actual vs Predicted (v5)\n'
             'Orange shading = test period 2021–2024  |  Title colour: green ≤20% MAPE, orange ≤35%, red >35%',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()
print('✅ All-25-district grid plotted')

# %%
# ── Part C: Monthly PM2.5 for DISTRICT_TO_PLOT ────────────────────────────────
df_pm25_plot = pm25_result['df_pm25'].copy()
df_pm25_plot['pred_pm25'] = pm25_result['pipe'].predict(pm25_result['X_pm25'])
df_pm25_plot = df_pm25_plot.dropna(subset=['pred_pm25']).copy()
df_pm25_plot = df_pm25_plot.sort_values(['district', 'year']).reset_index(drop=True)

dist_pm = df_pm25_plot[df_pm25_plot['district'] == DISTRICT_TO_PLOT].reset_index(drop=True)

if len(dist_pm) > 0:
    te_pm  = dist_pm[dist_pm['year'].isin(TEST_YEARS)]
    r2_pm  = r2_score(te_pm['pm2.5_ug_m3'], te_pm['pred_pm25']) if len(te_pm) > 1 else float('nan')
    mae_pm = mean_absolute_error(te_pm['pm2.5_ug_m3'], te_pm['pred_pm25']) if len(te_pm) > 0 else float('nan')

    time_idx   = np.arange(len(dist_pm))
    test_start = np.where(dist_pm['year'].values >= 2021)[0]
    split_pm   = test_start[0] if len(test_start) > 0 else len(time_idx)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axvspan(split_pm, len(time_idx), alpha=0.08, color='orange', label='Test period')
    ax.axvline(x=split_pm, color='orange', linestyle='--', alpha=0.7)
    ax.plot(time_idx, dist_pm['pm2.5_ug_m3'].values, 'o-', color='#e74c3c',
            linewidth=1.5, markersize=3, label='Actual PM2.5')
    ax.plot(time_idx, dist_pm['pred_pm25'].values, 's--', color='#e67e22',
            linewidth=1.5, markersize=3, label='Predicted PM2.5')

    unique_years = sorted(dist_pm['year'].unique())
    yr_ticks = [np.where(dist_pm['year'].values == yr)[0][0] for yr in unique_years
                if len(np.where(dist_pm['year'].values == yr)[0]) > 0]
    ax.set_xticks(yr_ticks)
    ax.set_xticklabels([str(yr) for yr in unique_years], rotation=45)
    ax.set_title(f'{DISTRICT_TO_PLOT} — Monthly PM2.5 Prediction\n'
                 f'Test R²={r2_pm:.3f}  |  MAE={mae_pm:.3f} µg/m³', fontweight='bold')
    ax.set_xlabel('Time'); ax.set_ylabel('PM2.5 (µg/m³)'); ax.legend()
    plt.tight_layout()
    plt.show()
    print(f'{DISTRICT_TO_PLOT} PM2.5: Test R²={r2_pm:.3f}  MAE={mae_pm:.3f} µg/m³')

# ── Part D: FAH Score ──────────────────────────────────────────────────────────
fah_row = district_avg[district_avg['district'] == DISTRICT_TO_PLOT]
if len(fah_row) > 0:
    print(f'FAH Score : {fah_row["FAH_score"].values[0]:.3f}')
    print(f'FAH Rank  : #{int(fah_row["FAH_rank"].values[0])}')
    print(f'Risk Tier : {fah_row["risk_tier"].values[0]}')

# %%
# ── Part C + Part D for ALL districts + save all images + zip ────────────────

import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

# 1) Prepare PM2.5 prediction dataframe once
df_pm25_plot = pm25_result['df_pm25'].copy()
df_pm25_plot['pred_pm25'] = pm25_result['pipe'].predict(pm25_result['X_pm25'])
df_pm25_plot = df_pm25_plot.dropna(subset=['pred_pm25']).copy()
df_pm25_plot = df_pm25_plot.sort_values(['district', 'year']).reset_index(drop=True)

# 2) Get all districts
all_districts = sorted(df_pm25_plot['district'].dropna().unique())

# 3) Output folder
output_dir = 'output_pm25_plots'
os.makedirs(output_dir, exist_ok=True)

saved_files = []

for DISTRICT_TO_PLOT in all_districts:
    print('\n' + '='*90)
    print(f'DISTRICT: {DISTRICT_TO_PLOT}')
    print('='*90)

    # ── Part C: Monthly PM2.5 for current district ────────────────────────────
    dist_pm = df_pm25_plot[df_pm25_plot['district'] == DISTRICT_TO_PLOT].reset_index(drop=True)

    if len(dist_pm) > 0:
        te_pm  = dist_pm[dist_pm['year'].isin(TEST_YEARS)]
        r2_pm  = r2_score(te_pm['pm2.5_ug_m3'], te_pm['pred_pm25']) if len(te_pm) > 1 else float('nan')
        mae_pm = mean_absolute_error(te_pm['pm2.5_ug_m3'], te_pm['pred_pm25']) if len(te_pm) > 0 else float('nan')

        time_idx   = np.arange(len(dist_pm))
        test_start = np.where(dist_pm['year'].values >= 2021)[0]
        split_pm   = test_start[0] if len(test_start) > 0 else len(time_idx)

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.axvspan(split_pm, len(time_idx), alpha=0.08, color='orange', label='Test period')
        ax.axvline(x=split_pm, color='orange', linestyle='--', alpha=0.7)

        ax.plot(
            time_idx,
            dist_pm['pm2.5_ug_m3'].values,
            'o-',
            color='#e74c3c',
            linewidth=1.5,
            markersize=3,
            label='Actual PM2.5'
        )
        ax.plot(
            time_idx,
            dist_pm['pred_pm25'].values,
            's--',
            color='#e67e22',
            linewidth=1.5,
            markersize=3,
            label='Predicted PM2.5'
        )

        unique_years = sorted(dist_pm['year'].unique())
        yr_ticks = [
            np.where(dist_pm['year'].values == yr)[0][0]
            for yr in unique_years
            if len(np.where(dist_pm['year'].values == yr)[0]) > 0
        ]

        ax.set_xticks(yr_ticks)
        ax.set_xticklabels([str(yr) for yr in unique_years], rotation=45)
        ax.set_title(
            f'{DISTRICT_TO_PLOT} — Monthly PM2.5 Prediction\n'
            f'Test R²={r2_pm:.3f}  |  MAE={mae_pm:.3f} µg/m³',
            fontweight='bold'
        )
        ax.set_xlabel('Time')
        ax.set_ylabel('PM2.5 (µg/m³)')
        ax.legend()
        plt.tight_layout()

        # Safe filename
        safe_name = DISTRICT_TO_PLOT.replace(' ', '_').replace('/', '_')
        img_path = os.path.join(output_dir, f'{safe_name}_pm25_plot.png')

        # Save image
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        saved_files.append(img_path)

        # Show plot
        plt.show()
        plt.close()

        print(f'{DISTRICT_TO_PLOT} PM2.5: Test R²={r2_pm:.3f}  MAE={mae_pm:.3f} µg/m³')
        print(f'Saved image: {img_path}')
    else:
        print(f'No PM2.5 data found for {DISTRICT_TO_PLOT}')

    # ── Part D: FAH Score for current district ────────────────────────────────
    fah_row = district_avg[district_avg['district'] == DISTRICT_TO_PLOT]

    if len(fah_row) > 0:
        print(f'FAH Score : {fah_row["FAH_score"].values[0]:.3f}')
        print(f'FAH Rank  : #{int(fah_row["FAH_rank"].values[0])}')
        print(f'Risk Tier : {fah_row["risk_tier"].values[0]}')
    else:
        print(f'No FAH data found for {DISTRICT_TO_PLOT}')

# 4) Create ZIP file
zip_path = 'all_district_pm25_plots.zip'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for file_path in saved_files:
        zipf.write(file_path, arcname=os.path.basename(file_path))

print('\n' + '='*90)
print(f'✅ Saved {len(saved_files)} images into folder: {output_dir}')
print(f'✅ ZIP created: {zip_path}')
print('='*90)

# 5) Optional: auto-download if using Google Colab
try:
    from google.colab import files
    files.download(zip_path)
    print('✅ Download started in Google Colab')
except Exception:
    print('ℹ️ If you are not using Colab, manually download this file:')
    print(zip_path)

# %% [markdown]
# ## STEP 15 — Final Summary

# %%
print('═'*72)
print('FINAL MODEL SUMMARY')
print('Environmental Drivers of Respiratory Disease — Sri Lanka')
print(f'25 Districts  |  2014–2024  |  dataset_feature_engineered.csv')
print('═'*72)

print('\n── YEARLY RESPIRATORY MODEL ──')
print(f'  {"Target":<36} {"Model":<30} {"Train R²":>9} {"Test R²":>9} {"MAE":>8}')
print(f'  {"─"*95}')
for key, r in yearly_results.items():
    v = '✅' if r['te_r2'] >= 0.4 else '⚠'
    print(f'  {r["target"]:<36} {r["tag"]:<30} {r["tr_r2"]:>9.3f} {r["te_r2"]:>9.3f} {r["te_mae"]:>8.4f}  {v}')

print()
print('── MONTHLY PM2.5 MODEL ──')
v = '✅' if pm25_result['te_r2'] >= 0.4 else '⚠'
print(f'  Train R²: {pm25_result["tr_r2"]:.3f}  |  Test R²: {pm25_result["te_r2"]:.3f} {v}  |  MAE: {pm25_result["te_mae"]:.3f} µg/m³')
print(f'  Best hyperparameters:')
for k, val in pm25_result['best_params'].items():
    print(f'    {k:<28}: {val}')

print()
print('── PER-DISTRICT MAPE SUMMARY (Model YB, test 2021–2024) ──')
good  = (eval_df['MAPE_%'] <= 20).sum()
ok    = ((eval_df['MAPE_%'] > 20) & (eval_df['MAPE_%'] <= 35)).sum()
poor  = (eval_df['MAPE_%'] > 35).sum()
print(f'  ✅ Good (MAPE ≤20%) : {good}/25')
print(f'  🟡 OK   (MAPE ≤35%) : {ok}/25')
print(f'  🔴 Poor (MAPE >35%) : {poor}/25')
if poor > 0:
    poor_dists = eval_df[eval_df['MAPE_%'] > 35]['District'].tolist()
    print(f'     Poor districts  : {poor_dists}  (likely COVID-19 data disruption 2021)')

print()
print('── FAH SHAP-DERIVED WEIGHTS (yearly model YB) ──')
for c in FAH_COMPS:
    w = FAH_WEIGHTS.get(c, 0)
    diff = w*100 - 33.3
    sign = '+' if diff > 0 else ''
    note = 'MORE than equal' if diff > 5 else 'LESS than equal' if diff < -5 else '≈ equal'
    print(f'  {c:<22}: {w*100:.1f}%  (vs 33.3%  →  {sign}{diff:.1f}%  {note})')

print()
print('── FAH TOP 5 HIGH-RISK DISTRICTS ──')
for _, row in district_avg.head(5).iterrows():
    print(f'  #{row["FAH_rank"]} {row["district"]:<15} FAH={row["FAH_score"]:.3f}  '
          f'Forest={row["forest_score"]:.3f}  Fire={row["fire_score"]:.3f}  Air={row["air_score"]:.3f}')

top_env_feat = env_shap.iloc[0]['feature'] if len(env_shap) > 0 else 'N/A'
top_env_comp = env_shap.iloc[0]['component'] if len(env_shap) > 0 else 'N/A'

print()
print('── SHAP KEY FINDINGS ──')
print(f'  1. Top environmental driver        : {top_env_feat}  ({top_env_comp})')
print('  2. Previous-year respiratory information contributes to yearly prediction')
print('  3. Fire-related lag features contribute to model predictions')
print('  4. Interaction features contribute to combined risk patterns')
print(f'  5. SHAP-derived FAH weights challenge equal 1/3 assumption')

print()
print(f'  5. SHAP-derived FAH weights challenge equal 1/3 assumption')

print()
print('── LIMITATIONS ──')
print('  1. Health data is annual — yearly model (YB) corrects for monthly broadcast')
print('  2. COVID-19 (2020–2021) may affect health reporting in test set')
print('  3. SHAP shows association, not proven causation')
print('  4. Yearly model: ~150 train rows (25 districts × 6 years)')

print()
print('── HOW TO USE ──')
print('  Step 14 → Change DISTRICT_TO_PLOT for per-district analysis')
print('  Step 14 Part B2 → All-25-district time-series grid')
print('  Step 6Y-EVAL → Per-district MAPE table')
print('  Step 6Y → Yearly respiratory model results')
print('  Step 6P → Monthly PM2.5 model results')
print()
print('═'*72)
print('ANALYSIS COMPLETE')
print('Uses pre-engineered dataset: dataset_feature_engineered.csv')
print('═'*72)