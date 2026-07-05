"""
================================================================================
 COMPREHENSIVE RESEARCH ANALYSIS — FINAL (DATA-DRIVEN)
 Sri Lanka Forest Loss, Air Quality & Respiratory Health (2014-2024)
 25 Districts | 11 Years | Monthly Resolution
================================================================================
"""
import warnings, os, glob
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from huggingface_hub import hf_hub_download
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import RidgeCV

# ─────────────────────────────────────────────────────────────────────────────
# 0.  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
HF_REPO_ID = "shazan18/environmental-drivers-respiratory-disease-sri-lanka"
HF_FILENAME = "sri_lanka_environmental_respiratory_panel_2014_2024.csv"
LOCAL_DATA_PATH = "Data/final_cleaned_dataset.csv"
BASE_OUT  = "outputs_final_v2/research_analysis"

DIRS = ["EDA","Insights_Overall","Insights_Districts",
        "Insights_Correlations","Insights_Seasonal","Insights_Health",
        "Insights_Environment"]
for d in [BASE_OUT] + [f"{BASE_OUT}/{x}" for x in DIRS]:
    os.makedirs(d, exist_ok=True)

MONTH_ORDER = ["January","February","March","April","May","June",
               "July","August","September","October","November","December"]
SEASON_MAP  = {
    "January":"NE Monsoon","February":"NE Monsoon","December":"NE Monsoon",
    "March":"Inter-monsoon 1","April":"Inter-monsoon 1",
    "May":"SW Monsoon","June":"SW Monsoon","July":"SW Monsoon","August":"SW Monsoon",
    "September":"Inter-monsoon 2","October":"Inter-monsoon 2","November":"Inter-monsoon 2",
}
CMAP_COOL = "coolwarm"
CMAP_SEQ  = "YlOrRd"

plt.rcParams.update({
    "figure.dpi":130, "savefig.dpi":150,
    "font.family":"DejaVu Sans","font.size":10,
    "axes.titlesize":12,"axes.labelsize":10,
    "legend.fontsize":9,"xtick.labelsize":8,"ytick.labelsize":8,
})

def savefig(fig, path):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD & CLEAN
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  1. LOADING DATA")
print("="*70)

try:
    DATA_PATH = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_FILENAME,
        repo_type="dataset",
    )
    print(f"  Loaded raw data from Hugging Face: {HF_REPO_ID}/{HF_FILENAME}")
except Exception as exc:
    if os.path.exists(LOCAL_DATA_PATH):
        DATA_PATH = LOCAL_DATA_PATH
        print(f"  Hugging Face download failed ({exc}); falling back to local file: {LOCAL_DATA_PATH}")
    else:
        raise

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
df.rename(columns={
    "gain_per_year_constant_by_year(2000-2012)": "tree_cover_gain_ha",
    "Gross Emissions_yr-1":  "gross_emissions_yr",
    "Gross_C_Removals_yr-1": "gross_removals_yr",
    "Net_C_Flux_yr-1":       "net_c_flux_yr",
    "Bronchitis_emphysema_and_other_chronic_obstructive_pulmonary_disease_live_discharges_(j40-j44)": "bronchitis_live_discharges",
    "Bronchitis_emphysema_and_other_chronic_obstructive_pulmonary_disease_deaths_(j40-j44)": "bronchitis_deaths",
    "Asthma_live_discharges_(j45-j46)": "asthma_live_discharges",
    "Asthma_deaths_(j45-j46)": "asthma_deaths",
}, inplace=True)
df.rename(columns={df.columns[0]: "province"}, inplace=True)

df["month"]     = pd.Categorical(df["month"], categories=MONTH_ORDER, ordered=True)
df["month_num"] = df["month"].cat.codes + 1
df["season"]    = df["month"].astype(str).map(SEASON_MAP)
df.sort_values(["district","year","month_num"], inplace=True)
df.reset_index(drop=True, inplace=True)

DISTRICTS = sorted(df["district"].unique())
YEARS     = sorted(df["year"].unique())
print(f"  Shape: {df.shape} | Districts: {len(DISTRICTS)} | Years: {YEARS[0]}–{YEARS[-1]}")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  2. FEATURE ENGINEERING")
print("="*70)

fe = df.copy()

# Demographic
fe["pop_density_per_ha"]    = fe["total_population_1k"] * 1000 / fe["area_ha"]
fe["gender_ratio"]          = fe["male_population_1k"] / fe["female_population_1k"]

# Health burden (per 1 000 population)
fe["bronchitis_rate_per_1k"]  = fe["bronchitis_live_discharges"] / fe["total_population_1k"]
fe["asthma_rate_per_1k"]      = fe["asthma_live_discharges"]     / fe["total_population_1k"]
fe["bronchitis_death_rate"]   = fe["bronchitis_deaths"]  / (fe["bronchitis_live_discharges"] + 1e-9)
fe["asthma_death_rate"]       = fe["asthma_deaths"]      / (fe["asthma_live_discharges"]     + 1e-9)
fe["total_resp_discharges"]   = fe["bronchitis_live_discharges"] + fe["asthma_live_discharges"]
fe["total_resp_deaths"]       = fe["bronchitis_deaths"]  + fe["asthma_deaths"]
fe["total_resp_rate_per_1k"]  = fe["total_resp_discharges"] / fe["total_population_1k"]
fe["combined_cfr"]            = fe["total_resp_deaths"]  / (fe["total_resp_discharges"] + 1e-9)

# Forest / carbon
fe["forest_cover_pct"]        = fe["ExtentIn_2010"] / fe["area_ha"] * 100
fe["tc_loss_pct_of_extent"]   = fe["tc_loss_ha"]    / (fe["ExtentIn_2010"] + 1e-9) * 100
fe["tc_loss_per_km2"]         = fe["tc_loss_ha"]    / (fe["area_ha"] / 100)
fe["carbon_loss_per_ha"]      = fe["carbon_gross_emissions_yearly"] / (fe["tc_loss_ha"] + 1e-9)
fe["carbon_efficiency"]       = fe["gross_removals_yr"] / (fe["gross_emissions_yr"].abs() + 1e-9)
fe["net_flux_per_ha"]         = fe["net_c_flux_yr"]  / fe["area_ha"]
fe["is_carbon_sink"]          = (fe["net_c_flux_yr"] < 0).astype(int)
fe["carbon_sink_strength"]    = -fe["net_c_flux_yr"] / (fe["area_ha"] + 1e-9)

# Air quality composite (MinMax scaled then averaged)
aq_cols = ["pm2.5_ug_m3","so2_ug_m3","no2_ug_m3"]
fe_aq   = MinMaxScaler().fit_transform(fe[aq_cols])
fe["pollution_index"]         = fe_aq.mean(axis=1)
fe["pm25_so2_ratio"]          = fe["pm2.5_ug_m3"] / (fe["so2_ug_m3"] + 1e-12)
fe["total_pollutant_load"]    = fe["pm2.5_ug_m3"] + fe["so2_ug_m3"] + fe["no2_ug_m3"]

# Fire
fe["fire_active"]             = (fe["frp_mean"] > 0).astype(int)
fe["fire_intensity_class"]    = pd.cut(fe["frp_mean"],
    bins=[-1,0,5,20,100,np.inf],
    labels=["None","Low","Medium","High","Extreme"])
fe["frp_normalized"]          = fe["frp_mean"] / (fe["area_ha"] / 1e5)
fe["fire_heat_diff"]          = fe["brightness"] - fe["bright_t31"]
fe["is_forest_fire"]          = fe["type"].str.contains("forest", case=False, na=False).astype(int)
fe["fire_smoke_proxy"]        = fe["frp_total"] * fe["is_forest_fire"]

# Vegetation
fe["vegetation_stress"]       = (fe["vim_anomaly"] < 0).astype(int)
fe["ndvi_range"]              = fe["vim_max"] - fe["vim_min"]
fe["veg_deficit"]             = fe["vim_climatology"] - fe["vim"]
fe["below_climatology"]       = (fe["vim"] < fe["vim_climatology"]).astype(int)

# Cross-domain interactions
fe["fire_pollution_interaction"]  = fe["fire_active"] * fe["pollution_index"]
fe["deforest_pollution_index"]    = fe["tc_loss_pct_of_extent"] * fe["pollution_index"]
fe["forest_health_index"]         = fe["forest_cover_pct"] * (1 - fe["tc_loss_pct_of_extent"]/100)
fe["pollution_health_burden"]     = fe["pollution_index"] * fe["total_resp_rate_per_1k"]
fe["fire_health_burden"]          = fe["fire_smoke_proxy"] * fe["total_resp_rate_per_1k"]
fe["veg_pollution_interaction"]   = fe["vim"] * fe["pollution_index"]
fe["deforest_health_interaction"] = fe["tc_loss_ha"] * fe["total_resp_rate_per_1k"]

# Lag / rolling features
fe.sort_values(["district","year","month_num"], inplace=True)
lag_cols = ["pm2.5_ug_m3","so2_ug_m3","no2_ug_m3",
            "total_resp_rate_per_1k","vim","frp_mean","tc_loss_ha"]
for col in lag_cols:
    fe[f"{col}_lag1m"]  = fe.groupby("district")[col].shift(1)
    fe[f"{col}_lag3m"]  = fe.groupby("district")[col].shift(3)
    fe[f"{col}_roll3m"] = fe.groupby("district")[col].transform(lambda x: x.rolling(3,min_periods=1).mean())
    fe[f"{col}_roll6m"] = fe.groupby("district")[col].transform(lambda x: x.rolling(6,min_periods=1).mean())
    fe[f"{col}_yoy"]    = fe.groupby(["district","month_num"])[col].diff()

# Annual district aggregates
annual = fe.groupby(["district","province","year"]).agg(
    tc_loss_ha              =("tc_loss_ha","mean"),
    carbon_gross_emis       =("carbon_gross_emissions_yearly","mean"),
    pm25_annual             =("pm2.5_ug_m3","mean"),
    so2_annual              =("so2_ug_m3","mean"),
    no2_annual              =("no2_ug_m3","mean"),
    frp_annual              =("frp_mean","mean"),
    vim_annual              =("vim","mean"),
    vim_anomaly_annual      =("vim_anomaly","mean"),
    pollution_index_annual  =("pollution_index","mean"),
    forest_fire_months      =("is_forest_fire","sum"),
    bronchitis_rate         =("bronchitis_rate_per_1k","mean"),
    asthma_rate             =("asthma_rate_per_1k","mean"),
    total_resp_rate         =("total_resp_rate_per_1k","mean"),
    bronchitis_deaths       =("bronchitis_deaths","sum"),
    asthma_deaths           =("asthma_deaths","sum"),
    combined_cfr            =("combined_cfr","mean"),
    pop_density             =("pop_density_per_ha","mean"),
    forest_cover_pct        =("forest_cover_pct","first"),
    tc_loss_pct             =("tc_loss_pct_of_extent","mean"),
    is_carbon_sink          =("is_carbon_sink","mean"),
    fire_active_months      =("fire_active","sum"),
    pollution_health_burden =("pollution_health_burden","mean"),
    total_resp_discharges   =("total_resp_discharges","sum"),
    total_resp_deaths       =("total_resp_deaths","sum"),
).reset_index()

ann_avg = annual.groupby("district")[
    ["tc_loss_pct","total_resp_rate","pm25_annual","pop_density",
     "forest_cover_pct","asthma_rate","bronchitis_rate","no2_annual",
     "so2_annual","pollution_index_annual","vim_annual","frp_annual",
     "forest_fire_months","fire_active_months","is_carbon_sink"]
].mean().reset_index()

print(f"  Feature engineering complete: {fe.shape[1]} columns ({fe.shape[1]-len(df.columns)} new)")



# ─────────────────────────────────────────────────────────────────────────────
# 3.  EDA PLOTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  3. EDA PLOTS")
print("="*70)

# ── .1  Distributions ────────────────────────────────────────────────────────
print("  [3.1] Distributions...")
dist_vars = ["pm2.5_ug_m3","so2_ug_m3","no2_ug_m3","pollution_index",
             "tc_loss_ha","forest_cover_pct","vim","vim_anomaly",
             "frp_mean","total_resp_rate_per_1k","bronchitis_rate_per_1k",
             "asthma_rate_per_1k","carbon_gross_emissions_yearly","pop_density_per_ha"]
from scipy import stats as scipy_stats
fig, axes = plt.subplots(4, 4, figsize=(20, 16))
for i, col in enumerate(dist_vars):
    ax = axes.flat[i]
    data = fe[col].dropna()
    ax.hist(data, bins=40, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax2 = ax.twinx(); ax2.set_yticks([])
    kde_x = np.linspace(data.min(), data.max(), 200)
    ax2.plot(kde_x, scipy_stats.gaussian_kde(data)(kde_x), color="crimson", lw=1.5)
    ax.set_title(col.replace("_"," "), fontsize=9)
    ax.text(0.98, 0.95, f"Skew={data.skew():.2f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=7, color="gray")
fig.suptitle("Distribution of Key Variables", fontsize=14, fontweight="bold", y=1.01)
savefig(fig, f"{BASE_OUT}/EDA/distributions_key_variables.png")

# ── 3.2  Correlation heatmaps ─────────────────────────────────────────────────
print("  [3.2] Correlation heatmaps...")
corr_groups = {
    "AirQuality_Health": ["pm2.5_ug_m3","so2_ug_m3","no2_ug_m3","pollution_index",
                          "bronchitis_rate_per_1k","asthma_rate_per_1k",
                          "total_resp_rate_per_1k","combined_cfr"],
    "Forest_Environment": ["tc_loss_ha","forest_cover_pct","carbon_gross_emissions_yearly",
                           "vim","vim_anomaly","frp_mean","fire_smoke_proxy","net_flux_per_ha"],
    "CrossDomain_Full":   ["pm2.5_ug_m3","so2_ug_m3","no2_ug_m3","pollution_index",
                           "tc_loss_ha","forest_cover_pct","vim","vim_anomaly","frp_mean",
                           "bronchitis_rate_per_1k","asthma_rate_per_1k",
                           "total_resp_rate_per_1k","combined_cfr","pop_density_per_ha",
                           "carbon_gross_emissions_yearly","deforest_pollution_index"],
}
for name, cols in corr_groups.items():
    avail = [c for c in cols if c in fe.columns]
    corr_mat = fe[avail].corr()
    fig, ax = plt.subplots(figsize=(max(10, len(avail)*0.8), max(8, len(avail)*0.7)))
    mask = np.triu(np.ones_like(corr_mat, dtype=bool))
    sns.heatmap(corr_mat, mask=mask, annot=True, fmt=".2f", cmap=CMAP_COOL,
                center=0, vmin=-1, vmax=1, linewidths=0.4, annot_kws={"size":7}, ax=ax)
    ax.set_title(f"Correlation – {name.replace('_',' ')}", fontweight="bold")
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    savefig(fig, f"{BASE_OUT}/EDA/correlation_{name}.png")

# ── 3.3  Spearman heatmap: predictors → health ────────────────────────────────
print("  [3.3] Spearman heatmap...")
health_targets = ["bronchitis_rate_per_1k","asthma_rate_per_1k",
                  "total_resp_rate_per_1k","combined_cfr"]
predictor_cols = ["pm2.5_ug_m3","so2_ug_m3","no2_ug_m3","pollution_index",
                  "tc_loss_ha","tc_loss_pct_of_extent","vim","vim_anomaly",
                  "frp_mean","fire_smoke_proxy","carbon_gross_emissions_yearly",
                  "forest_cover_pct","pop_density_per_ha","deforest_pollution_index",
                  "fire_pollution_interaction","pollution_health_burden"]
sp_rows = []
for tgt in health_targets:
    for pred in predictor_cols:
        tmp = fe[[tgt, pred]].dropna()
        if len(tmp) > 10:
            r, p = spearmanr(tmp[tgt], tmp[pred])
            sp_rows.append({"target":tgt,"predictor":pred,"rho":r,"p_value":p})
sp_df     = pd.DataFrame(sp_rows)
sp_pivot  = sp_df.pivot(index="predictor", columns="target", values="rho")
fig, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(sp_pivot, annot=True, fmt=".2f", cmap=CMAP_COOL,
            center=0, vmin=-1, vmax=1, linewidths=0.5, annot_kws={"size":9}, ax=ax)
ax.set_title("Spearman Correlations: Predictors vs Health Outcomes", fontweight="bold")
plt.xticks(rotation=20, ha="right"); plt.yticks(rotation=0)
savefig(fig, f"{BASE_OUT}/EDA/spearman_heatmap_predictors_vs_health.png")
sp_df.to_csv(f"{BASE_OUT}/EDA/spearman_correlations.csv", index=False)

# ── 3.4  Box plots: pollution by district ─────────────────────────────────────
print("  [3.4] Pollution boxplots by district...")
for pol, label, color in [("pm2.5_ug_m3","PM₂.₅ (µg/m³)","#E74C3C"),
                           ("so2_ug_m3","SO₂ (µg/m³)","#8E44AD"),
                           ("no2_ug_m3","NO₂ (µg/m³)","#2980B9")]:
    fig, ax = plt.subplots(figsize=(18, 6))
    order = fe.groupby("district")[pol].median().sort_values().index.tolist()
    sns.boxplot(data=fe, x="district", y=pol, order=order,
                color=color, ax=ax, linewidth=0.8, fliersize=2)
    ax.set_title(f"{label} by District (sorted by median)", fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    savefig(fig, f"{BASE_OUT}/EDA/boxplot_{pol}_by_district.png")

# ── 3.5  Seasonal monthly patterns ───────────────────────────────────────────
print("  [3.5] Seasonal patterns...")
seasonal_cols = ["pm2.5_ug_m3","so2_ug_m3","no2_ug_m3","vim","frp_mean",
                 "total_resp_rate_per_1k","bronchitis_rate_per_1k","asthma_rate_per_1k"]
fig, axes = plt.subplots(2, 4, figsize=(22, 10))
for i, col in enumerate(seasonal_cols):
    ax = axes.flat[i]
    monthly = fe.groupby("month")[col].agg(["mean","sem"]).reindex(MONTH_ORDER)
    ax.bar(range(12), monthly["mean"], color=plt.cm.tab20(np.linspace(0,1,12)),
           alpha=0.85, edgecolor="white")
    ax.errorbar(range(12), monthly["mean"], yerr=monthly["sem"]*1.96,
                fmt="none", color="black", capsize=3, lw=1)
    ax.set_xticks(range(12))
    ax.set_xticklabels([m[:3] for m in MONTH_ORDER], rotation=45, ha="right")
    ax.set_title(col.replace("_"," "), fontsize=9)
fig.suptitle("Average Monthly Patterns — All Districts, 2014–2024", fontsize=13, fontweight="bold")
savefig(fig, f"{BASE_OUT}/EDA/seasonal_monthly_patterns.png")

# ── 3.6  Violin by season ─────────────────────────────────────────────────────
print("  [3.6] Violin by season...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
season_order = ["NE Monsoon","Inter-monsoon 1","SW Monsoon","Inter-monsoon 2"]
for ax, col in zip(axes, ["total_resp_rate_per_1k","bronchitis_rate_per_1k","asthma_rate_per_1k"]):
    sns.violinplot(data=fe, x="season", y=col, order=season_order,
                   palette="Set2", inner="quartile", ax=ax)
    ax.set_title(col.replace("_"," "), fontsize=9)
    ax.tick_params(axis="x", rotation=20)
fig.suptitle("Respiratory Health by Season", fontsize=13, fontweight="bold")
savefig(fig, f"{BASE_OUT}/EDA/violin_health_by_season.png")

# ── 3.7  PCA ─────────────────────────────────────────────────────────────────
print("  [3.7] PCA...")
pca_cols = [c for c in ["pollution_index","fire_pollution_interaction",
    "deforest_pollution_index","pollution_health_burden","forest_health_index",
    "carbon_sink_strength","total_resp_rate_per_1k","vim","frp_mean",
    "pop_density_per_ha","tc_loss_pct_of_extent"] if c in fe.columns]
pca_data   = fe[pca_cols + ["district","province"]].dropna()
scaled     = StandardScaler().fit_transform(pca_data[pca_cols])
pca_model  = PCA(n_components=min(6, len(pca_cols)))
pcs        = pca_model.fit_transform(scaled)
var_exp    = pca_model.explained_variance_ratio_
provinces  = pca_data["province"].unique()

fig, axes = plt.subplots(1, 3, figsize=(20, 6),constrained_layout=True)
axes[0].bar(range(1, len(var_exp)+1), var_exp*100, color="#3498DB", edgecolor="white")
axes[0].plot(range(1, len(var_exp)+1), np.cumsum(var_exp)*100, "ro-", label="Cumulative")
axes[0].set_xlabel("Principal Component"); axes[0].set_ylabel("Variance Explained (%)")
axes[0].set_title("PCA Scree Plot"); axes[0].legend()
prov_palette = {p: plt.cm.tab10(i/10) for i, p in enumerate(provinces)}
for prov in provinces:
    mask = pca_data["province"].values == prov
    axes[1].scatter(pcs[mask, 0], pcs[mask, 1], label=prov, alpha=0.4, s=15,
                    color=prov_palette[prov])
axes[1].set_xlabel(f"PC1 ({var_exp[0]*100:.1f}%)"); axes[1].set_ylabel(f"PC2 ({var_exp[1]*100:.1f}%)")
axes[1].set_title("PC1 vs PC2 by Province"); axes[1].legend(fontsize=7)
loadings = pd.DataFrame(pca_model.components_[:2].T, index=pca_cols, columns=["PC1","PC2"])
loadings.sort_values("PC1").plot(kind="barh", ax=axes[2], color=["#E74C3C","#3498DB"])
axes[2].set_title("PCA Loadings (PC1 & PC2)"); axes[2].axvline(0, color="black", lw=0.8, ls="--")
fig.suptitle("Principal Component Analysis", fontsize=13, fontweight="bold")
savefig(fig, f"{BASE_OUT}/EDA/pca_analysis.png")

# ── 3.8  K-Means clustering ───────────────────────────────────────────────────
print("  [3.8] K-Means clustering...")
cluster_cols = ["pm25_annual","so2_annual","no2_annual","vim_annual",
                "total_resp_rate","forest_cover_pct","tc_loss_pct","frp_annual","pop_density"]
dist_annual  = annual.groupby("district")[cluster_cols].mean().reset_index()
clust_scaled = StandardScaler().fit_transform(dist_annual[cluster_cols])
inertias     = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(clust_scaled).inertia_
                for k in range(2,8)]
best_k = 4
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
dist_annual["cluster"] = kmeans.fit_predict(clust_scaled).astype(str)
pca2     = PCA(n_components=2)
pca2_pts = pca2.fit_transform(clust_scaled)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].plot(range(2,8), inertias, "bo-"); axes[0].axvline(best_k, ls="--", color="red", lw=1)
axes[0].set_xlabel("k"); axes[0].set_ylabel("Inertia"); axes[0].set_title("Elbow Method")
cpal = ["#E74C3C","#3498DB","#2ECC71","#F39C12"]
for c_id, grp in dist_annual.groupby("cluster"):
    idx = grp.index
    axes[1].scatter(pca2_pts[idx,0], pca2_pts[idx,1],
                    label=f"Cluster {c_id}", s=120, color=cpal[int(c_id)],
                    alpha=0.85, edgecolors="white")
    for _, row in grp.iterrows():
        axes[1].annotate(row["district"], (pca2_pts[row.name,0], pca2_pts[row.name,1]),
                         fontsize=6.5, ha="center", va="bottom")
axes[1].set_title(f"District Clusters (k={best_k}) — PCA View"); axes[1].legend()
fig.suptitle("K-Means District Clustering — Environmental & Health Profiles",
             fontsize=12, fontweight="bold")
savefig(fig, f"{BASE_OUT}/EDA/kmeans_district_clustering.png")
dist_annual.to_csv(f"{BASE_OUT}/EDA/district_cluster_assignments.csv", index=False)

# ── 3.9  Year-over-year trends ────────────────────────────────────────────────
print("  [3.9] YoY national trends...")
yr_overall = annual.groupby("year").agg(
    pm25=("pm25_annual","mean"), so2=("so2_annual","mean"),
    no2=("no2_annual","mean"),  vim=("vim_annual","mean"),
    resp_rate=("total_resp_rate","mean"), frp=("frp_annual","mean"),
    tc_loss=("tc_loss_ha","sum"), carbon=("carbon_gross_emis","sum"),
).reset_index()
fig, axes = plt.subplots(2, 4, figsize=(22, 10))
pairs = [("pm25","PM₂.₅ (µg/m³)","#E74C3C"),("so2","SO₂ (µg/m³)","#8E44AD"),
         ("no2","NO₂ (µg/m³)","#2980B9"),("vim","NDVI","#27AE60"),
         ("resp_rate","Resp. Rate per 1k","#C0392B"),("frp","FRP Mean (MW)","#E67E22"),
         ("tc_loss","Tree Cover Loss (ha)","#6C3483"),
         ("carbon","Carbon Emissions (Mg CO₂e)","#17202A")]
for ax, (col, lbl, clr) in zip(axes.flatten(), pairs):
    ax.plot(yr_overall["year"], yr_overall[col], "o-", color=clr, lw=2, ms=6)
    z = np.polyfit(yr_overall["year"], yr_overall[col], 1)
    ax.plot(yr_overall["year"], np.poly1d(z)(yr_overall["year"]),
            "--", color="gray", lw=1.2, label=f"slope={z[0]:.2f}")
    ax.set_title(lbl, fontsize=9); ax.legend(fontsize=7); ax.grid(alpha=0.25)
fig.suptitle("Year-Over-Year National Trends", fontsize=13, fontweight="bold")
savefig(fig, f"{BASE_OUT}/EDA/yoy_national_trends.png")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  OVERALL INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  4. OVERALL INSIGHTS")
print("="*70)

# ── 4.1  Province pollution & health trends ───────────────────────────────────
print("  [4.1] Province trends...")
prov_yr = annual.groupby(["province","year"])[
    ["pm25_annual","so2_annual","no2_annual","total_resp_rate"]].mean().reset_index()
provinces_list = sorted(annual["province"].unique())
prov_colors = {p: plt.cm.tab10(i/len(provinces_list)) for i, p in enumerate(provinces_list)}
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
for ax, (col, lbl) in zip(axes.flatten(), [
        ("pm25_annual","PM₂.₅ (µg/m³)"),("so2_annual","SO₂ (µg/m³)"),
        ("no2_annual","NO₂ (µg/m³)"),("total_resp_rate","Respiratory Rate per 1k")]):
    for prov in provinces_list:
        sub = prov_yr[prov_yr["province"]==prov]
        ax.plot(sub["year"], sub[col], "o-", label=prov, color=prov_colors[prov], lw=1.8, ms=5)
    ax.set_title(lbl, fontweight="bold"); ax.legend(fontsize=7); ax.grid(alpha=0.25)
fig.suptitle("Province-Level Trends: Pollution & Respiratory Health", fontsize=14, fontweight="bold")
savefig(fig, f"{BASE_OUT}/Insights_Overall/province_pollution_health_trends.png")

# ── 4.2  Scatter: predictors vs respiratory rate ──────────────────────────────
print("  [4.2] Scatter predictors vs resp rate...")
nat_monthly = fe.groupby(["year","month"])[
    ["pollution_index","total_resp_rate_per_1k","pm2.5_ug_m3",
     "so2_ug_m3","no2_ug_m3","vim","frp_mean"]].mean().reset_index()
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
predictors_sc = [("pollution_index","Composite Pollution Index","#E74C3C"),
                 ("pm2.5_ug_m3","PM₂.₅ (µg/m³)","#C0392B"),
                 ("so2_ug_m3","SO₂ (µg/m³)","#8E44AD"),
                 ("no2_ug_m3","NO₂ (µg/m³)","#2980B9"),
                 ("vim","NDVI","#27AE60"),
                 ("frp_mean","FRP Mean (MW)","#E67E22")]
for ax, (xcol, xlabel, clr) in zip(axes.flatten(), predictors_sc):
    x = nat_monthly[xcol].values; y = nat_monthly["total_resp_rate_per_1k"].values
    mask = ~(np.isnan(x) | np.isnan(y))
    ax.scatter(x[mask], y[mask], alpha=0.5, color=clr, s=30, edgecolors="white")
    z = np.polyfit(x[mask], y[mask], 1)
    ax.plot(np.linspace(x[mask].min(),x[mask].max(),100),
            np.poly1d(z)(np.linspace(x[mask].min(),x[mask].max(),100)), "k--", lw=1.5)
    r, p = pearsonr(x[mask], y[mask])
    ax.set_xlabel(xlabel); ax.set_ylabel("Resp. Rate per 1k")
    ax.set_title(f"Pearson r={r:.3f}, p={p:.4f}", fontsize=9); ax.grid(alpha=0.2)
fig.suptitle("National Monthly: Predictors vs Total Respiratory Rate", fontsize=13, fontweight="bold")
savefig(fig, f"{BASE_OUT}/Insights_Overall/scatter_predictors_vs_resp_rate.png")

# ── 4.3  Deforestation–health bubble ─────────────────────────────────────────
print("  [4.3] Deforestation–health bubble...")
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
sc = axes[0].scatter(ann_avg["tc_loss_pct"], ann_avg["total_resp_rate"],
                     s=ann_avg["pop_density"]*5000, c=ann_avg["pm25_annual"],
                     cmap="YlOrRd", alpha=0.8, edgecolors="gray", lw=0.5)
plt.colorbar(sc, ax=axes[0], label="PM₂.₅ (µg/m³)")
for _, row in ann_avg.iterrows():
    axes[0].annotate(row["district"], (row["tc_loss_pct"], row["total_resp_rate"]),
                     fontsize=6.5, ha="center", va="bottom")
axes[0].set_xlabel("Annual Tree Cover Loss (% of 2010 extent)")
axes[0].set_ylabel("Respiratory Rate per 1k")
axes[0].set_title("TC Loss vs Respiratory Rate\n(bubble=pop density, colour=PM₂.₅)", fontsize=9)
sc2 = axes[1].scatter(ann_avg["forest_cover_pct"], ann_avg["total_resp_rate"],
                      s=ann_avg["pop_density"]*5000, c=ann_avg["no2_annual"],
                      cmap="Blues_r", alpha=0.8, edgecolors="gray", lw=0.5)
plt.colorbar(sc2, ax=axes[1], label="NO₂ (µg/m³)")
for _, row in ann_avg.iterrows():
    axes[1].annotate(row["district"], (row["forest_cover_pct"], row["total_resp_rate"]),
                     fontsize=6.5, ha="center", va="bottom")
axes[1].set_xlabel("Forest Cover (%)"); axes[1].set_ylabel("Respiratory Rate per 1k")
axes[1].set_title("Forest Cover vs Respiratory Rate\n(bubble=pop density, colour=NO₂)", fontsize=9)
fig.suptitle("Deforestation, Forest Cover & Respiratory Health", fontsize=13, fontweight="bold")
savefig(fig, f"{BASE_OUT}/Insights_Overall/bubble_deforestation_health.png")

# ── 4.4  District × Year heatmaps ─────────────────────────────────────────────
print("  [4.4] Heatmap district × year...")
hm_cols = [("pm25_annual","PM₂.₅"),("so2_annual","SO₂"),("no2_annual","NO₂"),
           ("total_resp_rate","Resp. Rate"),("tc_loss_pct","TC Loss %"),
           ("vim_annual","NDVI"),("frp_annual","FRP"),("forest_fire_months","Fire Months")]
fig = plt.figure(figsize=(24, 40))
gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)
for idx, (col, title) in enumerate(hm_cols):
    ax = fig.add_subplot(gs[idx//2, idx%2])
    pivot = annual.pivot_table(index="district", columns="year", values=col, aggfunc="mean")
    sns.heatmap(pivot, ax=ax, cmap=CMAP_SEQ, annot=True, fmt=".1f",
                annot_kws={"size":6.5}, linewidths=0.3, linecolor="white",
                cbar_kws={"shrink":0.7})
    ax.set_title(f"{title} by District & Year", fontweight="bold", fontsize=10)
    ax.set_ylabel(""); ax.set_xlabel("")
fig.suptitle("District × Year Heatmaps — All Key Metrics", fontsize=14, fontweight="bold", y=1.005)
savefig(fig, f"{BASE_OUT}/Insights_Overall/heatmap_district_year_all_metrics.png")

# ── 4.5  Top-10 rankings ─────────────────────────────────────────────────────
print("  [4.5] District rankings...")
rank_metrics = {
    "Highest PM₂.₅":     ("pm25_annual", True),
    "Highest Resp. Rate": ("total_resp_rate", True),
    "Most Forest Loss":   ("tc_loss_pct", True),
    "Most Fire Activity": ("fire_active_months", True),
    "Best Forest Cover":  ("forest_cover_pct", False),
    "Lowest Pollution Index": ("pollution_index_annual", False),
}
fig, axes = plt.subplots(2, 3, figsize=(20, 14))
for ax, (title, (col, desc)) in zip(axes.flatten(), rank_metrics.items()):
    vals = annual.groupby("district")[col].mean().reset_index()
    vals = vals.sort_values(col, ascending=not desc).head(10)
    colors = plt.cm.RdYlGn(np.linspace(0,1,10)) if not desc else plt.cm.RdYlGn(np.linspace(1,0,10))
    bars = ax.barh(range(10), vals[col].values, color=colors, edgecolor="white")
    ax.set_yticks(range(10)); ax.set_yticklabels(vals["district"].values, fontsize=8)
    ax.set_title(title, fontweight="bold"); ax.invert_yaxis()
    for bar, v in zip(bars, vals[col].values):
        ax.text(bar.get_width()*0.01, bar.get_y()+bar.get_height()/2,
                f"{v:.2f}", va="center", fontsize=7.5)
fig.suptitle("District Rankings — Key Metrics (2014–2024 Averages)", fontsize=13, fontweight="bold")
savefig(fig, f"{BASE_OUT}/Insights_Overall/top_district_rankings.png")

# ── 4.6  Fire–pollution–health triangle ───────────────────────────────────────
print("  [4.6] Fire–pollution–health triangle...")
ann_d = annual.groupby("district")[["frp_annual","pm25_annual","total_resp_rate",
                                     "forest_fire_months","vim_annual","no2_annual"]].mean().reset_index()
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, (xcol, ycol, ccol, xl, yl, cl) in zip(axes, [
        ("frp_annual","pm25_annual","total_resp_rate","FRP Mean (MW)","PM₂.₅","Resp. Rate"),
        ("forest_fire_months","total_resp_rate","pm25_annual","Forest Fire Months","Resp. Rate/1k","PM₂.₅"),
        ("vim_annual","total_resp_rate","no2_annual","NDVI","Resp. Rate/1k","NO₂")]):
    sc = ax.scatter(ann_d[xcol], ann_d[ycol], c=ann_d[ccol], cmap="YlOrRd",
                    s=120, alpha=0.85, edgecolors="gray", lw=0.5)
    plt.colorbar(sc, ax=ax, label=cl)
    for _, row in ann_d.iterrows():
        ax.annotate(row["district"][:6], (row[xcol], row[ycol]), fontsize=6, ha="center", va="bottom")
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(f"{xl} vs {yl}", fontsize=9)
fig.suptitle("Fire, Pollution & Health Triangle Analysis", fontsize=13, fontweight="bold")
savefig(fig, f"{BASE_OUT}/Insights_Overall/fire_pollution_health_triangle.png")

# ── 4.7  Carbon flux ─────────────────────────────────────────────────────────
print("  [4.7] Carbon flux...")
carbon_d = annual.groupby("district")[["is_carbon_sink","carbon_gross_emis",
                                        "total_resp_rate","pm25_annual"]].mean().reset_index()
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
sink_pct = carbon_d["is_carbon_sink"] * 100
clr = ["#27AE60" if v >= 50 else "#E74C3C" for v in sink_pct]
axes[0].barh(carbon_d["district"], sink_pct, color=clr, edgecolor="white")
axes[0].axvline(50, ls="--", color="black", lw=1)
axes[0].set_xlabel("% Years as Carbon Sink"); axes[0].set_title("Carbon Sink Status by District", fontweight="bold")
sc = axes[1].scatter(carbon_d["carbon_gross_emis"], carbon_d["total_resp_rate"],
                     c=carbon_d["pm25_annual"], cmap="Reds", s=120, alpha=0.85, edgecolors="gray")
plt.colorbar(sc, ax=axes[1], label="PM₂.₅")
for _, row in carbon_d.iterrows():
    axes[1].annotate(row["district"][:6], (row["carbon_gross_emis"], row["total_resp_rate"]),
                     fontsize=6, ha="center", va="bottom")
axes[1].set_xlabel("Carbon Emissions (Mg CO₂e/yr)"); axes[1].set_ylabel("Resp. Rate per 1k")
axes[1].set_title("Carbon Emissions vs Health Burden", fontweight="bold")
yr_carbon = annual.groupby("year")["carbon_gross_emis"].sum().reset_index()
axes[2].bar(yr_carbon["year"], yr_carbon["carbon_gross_emis"]/1e6, color="#6C3483", alpha=0.8)
axes[2].set_xlabel("Year"); axes[2].set_ylabel("Carbon Emissions (×10⁶ Mg CO₂e)")
axes[2].set_title("National Annual Carbon Emissions", fontweight="bold")
fig.suptitle("Carbon Flux & Respiratory Health Insights", fontsize=13, fontweight="bold")
savefig(fig, f"{BASE_OUT}/Insights_Overall/carbon_flux_analysis.png")

# ── 4.8  Seasonal radar charts ────────────────────────────────────────────────
print("  [4.8] Radar charts...")
def radar_chart(ax, values, labels, title, color):
    N = len(labels); angles = [n/N*2*np.pi for n in range(N)]; angles += angles[:1]
    v = np.array(values); v = (v - v.min()) / (v.max() - v.min() + 1e-9)
    v = np.concatenate([v, [v[0]]])
    ax.plot(angles, v, "o-", color=color, lw=2); ax.fill(angles, v, alpha=0.2, color=color)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(0,1); ax.set_yticks([]); ax.set_title(title, fontsize=9, fontweight="bold", pad=10)

monthly_nat = fe.groupby("month")[["pm2.5_ug_m3","so2_ug_m3","no2_ug_m3","frp_mean",
                                    "vim","total_resp_rate_per_1k",
                                    "vim_anomaly","brightness"]].mean().reindex(MONTH_ORDER)
mn_labels   = [m[:3] for m in MONTH_ORDER]
fig, axes   = plt.subplots(2, 4, figsize=(20, 10), subplot_kw=dict(projection="polar"))
configs     = [("pm2.5_ug_m3","PM₂.₅","#E74C3C"),("so2_ug_m3","SO₂","#8E44AD"),
               ("no2_ug_m3","NO₂","#2980B9"),("frp_mean","FRP","#E67E22"),
               ("vim","NDVI","#27AE60"),("total_resp_rate_per_1k","Resp. Rate","#C0392B"),
               ("vim_anomaly","NDVI Anomaly","#16A085"),("brightness","Fire Brightness","#F39C12")]
for ax, (col, title, clr) in zip(axes.flatten(), configs):
    radar_chart(ax, monthly_nat[col].values, mn_labels, title, clr)
fig.suptitle("Seasonal Radar Charts — National Monthly Patterns", fontsize=13, fontweight="bold", y=1.02)
savefig(fig, f"{BASE_OUT}/Insights_Seasonal/radar_seasonal_all_metrics.png")

# ── 4.9  Province stacked health burden ───────────────────────────────────────
print("  [4.9] Province health burden...")
prov_health = annual.groupby("province")[["total_resp_discharges","total_resp_deaths"]].mean().reset_index()
fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(prov_health))
ax.bar(x, prov_health["total_resp_discharges"], 0.5, label="Live Discharges", color="#3498DB", alpha=0.85)
ax.bar(x, prov_health["total_resp_deaths"], 0.5,
       bottom=prov_health["total_resp_discharges"], label="Deaths", color="#E74C3C", alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(prov_health["province"], rotation=25, ha="right")
ax.set_ylabel("Annual Average Cases"); ax.legend()
ax.set_title("Respiratory Disease Burden by Province (Avg Annual)", fontweight="bold")
savefig(fig, f"{BASE_OUT}/Insights_Overall/province_health_burden_stacked.png")

# ── 4.10  Pollution trend with CI ─────────────────────────────────────────────
print("  [4.10] Pollution trend with CI...")
nat_yr_poll = fe.groupby("year")[["pm2.5_ug_m3","so2_ug_m3","no2_ug_m3"]].agg(["mean","sem"]).reset_index()
fig, ax = plt.subplots(figsize=(14, 6))
for col, lbl, clr in [("pm2.5_ug_m3","PM₂.₅","#E74C3C"),
                       ("so2_ug_m3","SO₂","#8E44AD"),
                       ("no2_ug_m3","NO₂","#2980B9")]:
    means = nat_yr_poll[(col,"mean")]; sems = nat_yr_poll[(col,"sem")]*1.96
    yrs   = nat_yr_poll["year"]
    ax.plot(yrs, means, "o-", label=lbl, color=clr, lw=2, ms=6)
    ax.fill_between(yrs, means-sems, means+sems, alpha=0.15, color=clr)
ax.set_xlabel("Year"); ax.set_ylabel("Concentration (µg/m³)")
ax.set_title("National Annual Pollution Trends (Mean ± 95% CI)", fontweight="bold")
ax.legend(); ax.grid(alpha=0.2)
savefig(fig, f"{BASE_OUT}/Insights_Overall/national_pollution_trends_ci.png")

# ── 4.11  Ridge feature importance ────────────────────────────────────────────
print("  [4.11] Ridge feature importance...")
model_features = [c for c in ["pm2.5_ug_m3","so2_ug_m3","no2_ug_m3","vim","vim_anomaly",
    "frp_mean","tc_loss_ha","forest_cover_pct","pop_density_per_ha",
    "fire_smoke_proxy","pollution_index","deforest_pollution_index"] if c in fe.columns]
fig, axes = plt.subplots(1, 3, figsize=(18, 8))
for ax, tgt in zip(axes, ["total_resp_rate_per_1k","bronchitis_rate_per_1k","asthma_rate_per_1k"]):
    df_model = fe[model_features + [tgt]].dropna()
    X = StandardScaler().fit_transform(df_model[model_features])
    y = df_model[tgt].values
    model = RidgeCV(alphas=[0.01,0.1,1,10,100]).fit(X, y)
    coef_df = pd.Series(model.coef_, index=model_features).sort_values()
    colors  = ["#E74C3C" if v > 0 else "#27AE60" for v in coef_df.values]
    coef_df.plot(kind="barh", ax=ax, color=colors, alpha=0.85)
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_title(f"RidgeCV → {tgt.replace('_',' ')}", fontsize=9)
    ax.set_xlabel("Standardised Coefficient")
fig.suptitle("Linear Feature Importance (RidgeCV) for Respiratory Outcomes",
             fontsize=13, fontweight="bold")
savefig(fig, f"{BASE_OUT}/Insights_Overall/ridge_feature_importance.png")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  PER-DISTRICT DASHBOARDS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  5. PER-DISTRICT DASHBOARDS (25 districts)")
print("="*70)

# ── 5.1  Individual district dashboard ────────────────────────────────────────
print("  [5.1] District dashboards...")
for dist in DISTRICTS:
    sub     = fe[fe["district"] == dist].copy()
    ann_sub = annual[annual["district"] == dist].copy()
    sub_m   = sub.groupby(["year","month"])[
        ["pm2.5_ug_m3","so2_ug_m3","no2_ug_m3",
         "total_resp_rate_per_1k","vim","frp_mean","vim_anomaly"]
    ].mean().reset_index()
    sub_m["date_num"] = sub_m["year"] + (sub_m["month"].cat.codes + 0.5) / 12

    fig = plt.figure(figsize=(22, 22))
    gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.5, wspace=0.4)
    fig.suptitle(f"District Dashboard — {dist}", fontsize=16, fontweight="bold", y=1.01)

    # Air quality time series
    ax0 = fig.add_subplot(gs[0, :2])
    ax0.plot(sub_m["date_num"], sub_m["pm2.5_ug_m3"], label="PM₂.₅", color="#E74C3C", lw=1.3)
    ax0b = ax0.twinx()
    ax0b.plot(sub_m["date_num"], sub_m["so2_ug_m3"], label="SO₂", color="#8E44AD", lw=1.2, ls="--")
    ax0b.plot(sub_m["date_num"], sub_m["no2_ug_m3"], label="NO₂", color="#2980B9", lw=1.2, ls=":")
    ax0.set_title("Air Pollutants Over Time"); ax0.set_ylabel("PM₂.₅ (µg/m³)")
    ax0b.set_ylabel("SO₂ / NO₂ (µg/m³)")
    lines1, lbl1 = ax0.get_legend_handles_labels()
    lines2, lbl2 = ax0b.get_legend_handles_labels()
    ax0.legend(lines1+lines2, lbl1+lbl2, fontsize=7); ax0.grid(alpha=0.2)

    # Respiratory health trend
    ax1 = fig.add_subplot(gs[0, 2])
    yr_h = ann_sub.groupby("year")[["bronchitis_rate","asthma_rate","total_resp_rate"]].mean().reset_index()
    ax1.plot(yr_h["year"], yr_h["bronchitis_rate"], "s-", label="Bronchitis", color="#E67E22")
    ax1.plot(yr_h["year"], yr_h["asthma_rate"],     "o-", label="Asthma",    color="#9B59B6")
    ax1.plot(yr_h["year"], yr_h["total_resp_rate"], "^-", label="Total",     color="#C0392B", lw=2)
    ax1.set_title("Health Rates per 1k"); ax1.legend(fontsize=7); ax1.grid(alpha=0.2)

    # NDVI time series + anomaly
    ax2 = fig.add_subplot(gs[1, :2])
    ax2.plot(sub_m["date_num"], sub_m["vim"], label="NDVI", color="#27AE60", lw=1.3)
    ax2.axhline(sub_m["vim"].mean(), ls="--", color="gray", lw=1, label="Mean NDVI")
    ax2b = ax2.twinx()
    ax2b.fill_between(sub_m["date_num"], 0, sub_m["vim_anomaly"],
                      where=sub_m["vim_anomaly"]>0, color="#27AE60", alpha=0.3)
    ax2b.fill_between(sub_m["date_num"], sub_m["vim_anomaly"], 0,
                      where=sub_m["vim_anomaly"]<0, color="#E74C3C", alpha=0.3)
    ax2.set_title("Vegetation Index & Anomaly"); ax2.set_ylabel("NDVI"); ax2b.set_ylabel("Anomaly")
    ax2.legend(fontsize=7); ax2.grid(alpha=0.2)

    # Monthly FRP
    ax3 = fig.add_subplot(gs[1, 2])
    mon_frp = sub[sub["frp_mean"]>0].groupby("month")["frp_mean"].mean().reindex(MONTH_ORDER)
    ax3.bar(range(12), mon_frp.fillna(0), color="#E67E22", alpha=0.85)
    ax3.set_xticks(range(12)); ax3.set_xticklabels([m[:3] for m in MONTH_ORDER], rotation=45, ha="right")
    ax3.set_title("Avg FRP by Month (Active Fire Months Only)"); ax3.set_ylabel("FRP (MW)")

    # Monthly pollution pattern (normalized)
    ax4 = fig.add_subplot(gs[2, 0])
    m_poll = sub.groupby("month")[["pm2.5_ug_m3","so2_ug_m3","no2_ug_m3"]].mean().reindex(MONTH_ORDER)
    m_poll_norm = (m_poll - m_poll.min()) / (m_poll.max() - m_poll.min() + 1e-9)
    m_poll_norm.plot(ax=ax4, marker="o", colormap="Set1", lw=1.5, ms=5)
    ax4.set_xticks(range(12)); ax4.set_xticklabels([m[:3] for m in MONTH_ORDER], rotation=45, ha="right")
    ax4.set_title("Monthly Pollution Patterns (Normalised)"); ax4.legend(fontsize=7); ax4.grid(alpha=0.2)

    # Tree cover loss
    ax5 = fig.add_subplot(gs[2, 1])
    yr_loss = ann_sub[["year","tc_loss_ha"]].drop_duplicates()
    ax5.bar(yr_loss["year"], yr_loss["tc_loss_ha"], color="#6C3483", alpha=0.8, edgecolor="white")
    ax5.set_title("Annual Tree Cover Loss (ha)"); ax5.grid(alpha=0.2, axis="y")

    # Carbon emissions
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.bar(ann_sub["year"], ann_sub["carbon_gross_emis"]/1e3, color="#1ABC9C", alpha=0.8, edgecolor="white")
    ax6.set_title("Annual Carbon Emissions (×10³ Mg CO₂e)"); ax6.grid(alpha=0.2, axis="y")

    # Spearman: predictors vs resp rate
    ax7 = fig.add_subplot(gs[3, :2])
    local_corrs = {}
    for pc in ["pm2.5_ug_m3","so2_ug_m3","no2_ug_m3","vim","frp_mean",
               "tc_loss_ha","vim_anomaly","pollution_index","fire_smoke_proxy"]:
        if pc in sub.columns:
            tmp = sub[["total_resp_rate_per_1k", pc]].dropna()
            if len(tmp) > 10:
                r, p = spearmanr(tmp["total_resp_rate_per_1k"], tmp[pc])
                local_corrs[pc] = r
    if local_corrs:
        corr_s = pd.Series(local_corrs).sort_values()
        ax7.barh(corr_s.index, corr_s.values,
                 color=["#E74C3C" if v>0 else "#27AE60" for v in corr_s.values], alpha=0.85)
        ax7.axvline(0, color="black", lw=0.8); ax7.set_xlim(-1,1)
    ax7.set_title("Spearman ρ: Predictors vs Respiratory Rate"); ax7.set_xlabel("ρ")

    # Seasonal health boxplot
    ax8 = fig.add_subplot(gs[3, 2])
    s_order = ["NE Monsoon","Inter-monsoon 1","SW Monsoon","Inter-monsoon 2"]
    sub_s   = sub[sub["season"].isin(s_order)]
    if len(sub_s) > 0:
        sns.boxplot(data=sub_s, x="season", y="total_resp_rate_per_1k",
                    order=s_order, palette="Set3", ax=ax8)
    ax8.set_title("Resp. Rate by Season"); ax8.set_xlabel("")
    ax8.tick_params(axis="x", rotation=20)

    savefig(fig, f"{BASE_OUT}/Insights_Districts/{dist.replace(' ','_')}_dashboard.png")

# ── 5.2  Facet: pollution time series all districts ───────────────────────────
print("  [5.2] Facet pollution plots...")
col_map = {"pm2.5_ug_m3":"pm25_annual","so2_ug_m3":"so2_annual","no2_ug_m3":"no2_annual"}
for pol, lbl, clr in [("pm2.5_ug_m3","PM₂.₅ (µg/m³)","#E74C3C"),
                       ("so2_ug_m3","SO₂ (µg/m³)","#8E44AD"),
                       ("no2_ug_m3","NO₂ (µg/m³)","#2980B9")]:
    ycol = col_map[pol]
    fig, axes = plt.subplots(5, 5, figsize=(24, 18), sharey=False)
    for i, dist in enumerate(DISTRICTS):
        ax = axes.flat[i]; sub = annual[annual["district"]==dist]
        ax.plot(sub["year"], sub[ycol], "o-", color=clr, lw=1.5, ms=4)
        z = np.polyfit(sub["year"], sub[ycol], 1)
        ax.plot(sub["year"], np.poly1d(z)(sub["year"]), "--", color="gray", lw=1)
        ax.set_title(dist, fontsize=8, fontweight="bold"); ax.grid(alpha=0.2)
        ax.text(0.98, 0.04, "↑" if z[0]>0 else "↓", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=10, color="red" if z[0]>0 else "green")
    fig.suptitle(f"{lbl} Trends — All 25 Districts", fontsize=14, fontweight="bold")
    savefig(fig, f"{BASE_OUT}/Insights_Districts/facet_{pol}_all_districts.png")

# ── 5.3  Facet: health rates all districts ────────────────────────────────────
print("  [5.3] Facet health rates...")
fig, axes = plt.subplots(5, 5, figsize=(24, 18), sharey=False)
for i, dist in enumerate(DISTRICTS):
    ax = axes.flat[i]; sub = annual[annual["district"]==dist]
    ax.plot(sub["year"], sub["bronchitis_rate"], "s-", color="#E67E22", label="Bronch.", lw=1.3, ms=4)
    ax.plot(sub["year"], sub["asthma_rate"],     "o-", color="#9B59B6", label="Asthma",  lw=1.3, ms=4)
    ax.set_title(dist, fontsize=8, fontweight="bold"); ax.grid(alpha=0.2)
    if i == 0: ax.legend(fontsize=6)
fig.suptitle("Bronchitis & Asthma Rates per 1k — All 25 Districts", fontsize=14, fontweight="bold")
savefig(fig, f"{BASE_OUT}/Insights_Districts/facet_health_rates_all_districts.png")

# ── 5.4  Facet: tree cover loss all districts ─────────────────────────────────
print("  [5.4] Facet tree loss...")
fig, axes = plt.subplots(5, 5, figsize=(24, 18), sharey=False)
for i, dist in enumerate(DISTRICTS):
    ax = axes.flat[i]; sub = annual[annual["district"]==dist].drop_duplicates("year")
    ax.bar(sub["year"], sub["tc_loss_ha"], color="#6C3483", alpha=0.8, edgecolor="white")
    ax.set_title(dist, fontsize=8, fontweight="bold"); ax.grid(alpha=0.2, axis="y")
    ax.tick_params(axis="x", labelsize=6, rotation=45)
fig.suptitle("Annual Tree Cover Loss (ha) — All 25 Districts", fontsize=14, fontweight="bold")
savefig(fig, f"{BASE_OUT}/Insights_Districts/facet_tree_loss_all_districts.png")

# ── 5.5  Per-district correlation heatmaps ────────────────────────────────────
print("  [5.5] District correlation heatmaps...")
corr_vars = ["pm2.5_ug_m3","so2_ug_m3","no2_ug_m3","vim","frp_mean","tc_loss_ha",
             "total_resp_rate_per_1k","bronchitis_rate_per_1k","asthma_rate_per_1k"]
fig, axes = plt.subplots(5, 5, figsize=(25, 25))
for i, dist in enumerate(DISTRICTS):
    ax = axes.flat[i]
    sub = fe[fe["district"]==dist][corr_vars].dropna()
    if len(sub) > 5:
        sns.heatmap(sub.corr(), ax=ax, cmap=CMAP_COOL, center=0, vmin=-1, vmax=1,
                    annot=True, fmt=".1f", annot_kws={"size":5},
                    linewidths=0.3, cbar=False)
    ax.set_title(dist, fontsize=8, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=5)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=5)
for j in range(len(DISTRICTS), len(axes.flat)):
    axes.flat[j].set_visible(False)
fig.suptitle("Per-District Correlation Matrices", fontsize=14, fontweight="bold", y=1.01)
savefig(fig, f"{BASE_OUT}/Insights_Correlations/per_district_correlation_heatmaps.png")

# ── 5.6  Seasonal health heatmap per district ─────────────────────────────────
print("  [5.6] Seasonal health heatmaps...")
fig, axes = plt.subplots(5, 5, figsize=(28, 24))
for i, dist in enumerate(DISTRICTS):
    ax = axes.flat[i]; sub = fe[fe["district"]==dist]
    pivot = sub.groupby(["year","month"])["total_resp_rate_per_1k"].mean().unstack()
    pivot = pivot.reindex(columns=MONTH_ORDER)
    sns.heatmap(pivot, ax=ax, cmap="YlOrRd", annot=False, linewidths=0.2,
                cbar_kws={"shrink":0.6}, linecolor="white")
    ax.set_title(dist, fontsize=8, fontweight="bold")
    ax.set_xticklabels([m[:3] for m in MONTH_ORDER], rotation=45, ha="right", fontsize=5)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=6)
for j in range(len(DISTRICTS), len(axes.flat)):
    axes.flat[j].set_visible(False)
fig.suptitle("Respiratory Rate per 1k — Month × Year by District", fontsize=14, fontweight="bold")
savefig(fig, f"{BASE_OUT}/Insights_Health/seasonal_health_heatmap_by_district.png")

# ── 5.7  Lag correlation: PM2.5 → resp rate ───────────────────────────────────
print("  [5.7] Lag correlation analysis...")
lag_results = []
for dist in DISTRICTS:
    sub = fe[fe["district"]==dist].sort_values(["year","month_num"])
    for lag, xcol in [(0,"pm2.5_ug_m3"),(1,"pm2.5_ug_m3_lag1m"),(3,"pm2.5_ug_m3_lag3m")]:
        if xcol in sub.columns:
            tmp = sub[["total_resp_rate_per_1k", xcol]].dropna()
            if len(tmp) > 10:
                r, p = spearmanr(tmp["total_resp_rate_per_1k"], tmp[xcol])
                lag_results.append({"district":dist,"lag_months":lag,"rho":r,"p_value":p})
lag_df    = pd.DataFrame(lag_results)
lag_pivot = lag_df.pivot(index="district", columns="lag_months", values="rho")
fig, ax   = plt.subplots(figsize=(10, 10))
sns.heatmap(lag_pivot, annot=True, fmt=".2f", cmap=CMAP_COOL,
            center=0, vmin=-1, vmax=1, linewidths=0.4, annot_kws={"size":9}, ax=ax)
ax.set_title("Spearman ρ: PM₂.₅ → Resp. Rate at Different Lags by District", fontweight="bold")
ax.set_xlabel("Lag (months)"); ax.set_ylabel("District")
savefig(fig, f"{BASE_OUT}/Insights_Correlations/lag_correlation_pm25_resp_by_district.png")
lag_df.to_csv(f"{BASE_OUT}/Insights_Correlations/lag_correlation_results.csv", index=False)

# ── 5.8  Fire analysis by district ────────────────────────────────────────────
print("  [5.8] Fire analysis...")
fire_d = fe[fe["frp_mean"]>0].groupby("district").agg(
    fire_months=("frp_mean","count"),
    avg_frp=("frp_mean","mean"),
    total_frp=("frp_total","sum"),
    forest_fire_pct=("is_forest_fire","mean"),
).reset_index()
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
o1 = fire_d.sort_values("fire_months", ascending=True)
axes[0,0].barh(o1["district"], o1["fire_months"], color="#E67E22", alpha=0.85)
axes[0,0].set_title("Active Fire Months by District"); axes[0,0].set_xlabel("Months (FRP > 0)")
o2 = fire_d.sort_values("avg_frp", ascending=True)
axes[0,1].barh(o2["district"], o2["avg_frp"], color="#C0392B", alpha=0.85)
axes[0,1].set_title("Average FRP (MW)"); axes[0,1].set_xlabel("FRP (MW)")
sc = axes[1,0].scatter(fire_d["avg_frp"], fire_d["forest_fire_pct"]*100,
                       s=fire_d["fire_months"]*3, c=fire_d["total_frp"],
                       cmap="YlOrRd", alpha=0.85, edgecolors="gray")
plt.colorbar(sc, ax=axes[1,0], label="Total FRP (MW)")
for _, row in fire_d.iterrows():
    axes[1,0].annotate(row["district"][:6], (row["avg_frp"], row["forest_fire_pct"]*100), fontsize=6)
axes[1,0].set_xlabel("Avg FRP (MW)"); axes[1,0].set_ylabel("Forest Fire Months (%)")
axes[1,0].set_title("FRP vs Forest Fire Proportion\n(bubble = fire months)")
mfh = fire_d.merge(ann_avg[["district","total_resp_rate","pm25_annual"]], on="district")
sc2 = axes[1,1].scatter(mfh["total_frp"], mfh["total_resp_rate"], s=100,
                         c=mfh["pm25_annual"], cmap="Reds", alpha=0.85, edgecolors="gray")
plt.colorbar(sc2, ax=axes[1,1], label="PM₂.₅")
for _, row in mfh.iterrows():
    axes[1,1].annotate(row["district"][:6], (row["total_frp"], row["total_resp_rate"]),
                       fontsize=6, ha="center", va="bottom")
axes[1,1].set_xlabel("Total FRP (MW)"); axes[1,1].set_ylabel("Avg Resp Rate/1k")
axes[1,1].set_title("Total Fire Intensity vs Respiratory Health")
fig.suptitle("Fire Analysis by District", fontsize=13, fontweight="bold")
savefig(fig, f"{BASE_OUT}/Insights_Environment/fire_analysis_by_district.png")

# ── 5.9  Vegetation analysis ──────────────────────────────────────────────────
print("  [5.9] Vegetation analysis...")
veg_d = annual.groupby("district")[["vim_annual","vim_anomaly_annual","total_resp_rate",
                                     "pm25_annual","tc_loss_pct","forest_cover_pct"]].mean().reset_index()
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
pairs_veg = [
    ("vim_annual","total_resp_rate","NDVI","Resp. Rate per 1k"),
    ("vim_anomaly_annual","total_resp_rate","NDVI Anomaly","Resp. Rate per 1k"),
    ("vim_annual","pm25_annual","NDVI","PM₂.₅ (µg/m³)"),
    ("vim_anomaly_annual","pm25_annual","NDVI Anomaly","PM₂.₅ (µg/m³)"),
    ("forest_cover_pct","vim_annual","Forest Cover (%)","NDVI"),
    ("tc_loss_pct","vim_anomaly_annual","TC Loss (%)","NDVI Anomaly"),
]
for ax, (xc, yc, xl, yl) in zip(axes.flatten(), pairs_veg):
    sc = ax.scatter(veg_d[xc], veg_d[yc], c=veg_d["total_resp_rate"],
                    cmap="YlOrRd", s=120, alpha=0.85, edgecolors="gray", lw=0.5)
    plt.colorbar(sc, ax=ax, label="Resp. Rate")
    for _, row in veg_d.iterrows():
        ax.annotate(row["district"][:5], (row[xc], row[yc]), fontsize=6, ha="center", va="bottom")
    z = np.polyfit(veg_d[xc], veg_d[yc], 1)
    xr = np.linspace(veg_d[xc].min(), veg_d[xc].max(), 100)
    ax.plot(xr, np.poly1d(z)(xr), "k--", lw=1.2)
    r, p = spearmanr(veg_d[xc], veg_d[yc])
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(f"ρ={r:.3f}, p={p:.4f}", fontsize=9)
fig.suptitle("Vegetation × Pollution × Health by District", fontsize=13, fontweight="bold")
savefig(fig, f"{BASE_OUT}/Insights_Environment/vegetation_pollution_health.png")

# ── 5.10  CFR analysis ────────────────────────────────────────────────────────
print("  [5.10] CFR analysis...")
cfr_d  = annual.groupby("district")[["combined_cfr","pm25_annual","forest_cover_pct",
                                      "pop_density","no2_annual","total_resp_rate"]].mean().reset_index()
cfr_yr = annual.groupby(["district","year"])["combined_cfr"].mean().reset_index()
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
o_cfr = cfr_d.sort_values("combined_cfr", ascending=True)
axes[0,0].barh(o_cfr["district"], o_cfr["combined_cfr"],
               color=plt.cm.RdYlGn_r(np.linspace(0,1,len(DISTRICTS))), alpha=0.85)
axes[0,0].set_title("Combined CFR by District", fontweight="bold"); axes[0,0].set_xlabel("CFR")
for dist in DISTRICTS[:8]:
    sub_cfr = cfr_yr[cfr_yr["district"]==dist]
    axes[0,1].plot(sub_cfr["year"], sub_cfr["combined_cfr"], label=dist[:10], alpha=0.8, lw=1.3)
axes[0,1].legend(fontsize=6, ncol=2); axes[0,1].grid(alpha=0.2)
axes[0,1].set_title("CFR Trend — First 8 Districts Alphabetically")
sc = axes[1,0].scatter(cfr_d["pm25_annual"], cfr_d["combined_cfr"],
                        s=cfr_d["pop_density"]*5000, c=cfr_d["no2_annual"],
                        cmap="Blues", alpha=0.85, edgecolors="gray")
plt.colorbar(sc, ax=axes[1,0], label="NO₂")
for _, row in cfr_d.iterrows():
    axes[1,0].annotate(row["district"][:6], (row["pm25_annual"], row["combined_cfr"]),
                       fontsize=6, ha="center", va="bottom")
axes[1,0].set_xlabel("PM₂.₅ (µg/m³)"); axes[1,0].set_ylabel("Combined CFR")
axes[1,0].set_title("PM₂.₅ vs CFR\n(bubble=pop density, colour=NO₂)")
sc2 = axes[1,1].scatter(cfr_d["forest_cover_pct"], cfr_d["combined_cfr"],
                         s=120, c=cfr_d["total_resp_rate"], cmap="Reds", alpha=0.85, edgecolors="gray")
plt.colorbar(sc2, ax=axes[1,1], label="Resp. Rate")
for _, row in cfr_d.iterrows():
    axes[1,1].annotate(row["district"][:6], (row["forest_cover_pct"], row["combined_cfr"]),
                       fontsize=6, ha="center", va="bottom")
z = np.polyfit(cfr_d["forest_cover_pct"], cfr_d["combined_cfr"], 1)
xr = np.linspace(cfr_d["forest_cover_pct"].min(), cfr_d["forest_cover_pct"].max(), 100)
axes[1,1].plot(xr, np.poly1d(z)(xr), "k--", lw=1.2)
axes[1,1].set_xlabel("Forest Cover (%)"); axes[1,1].set_ylabel("Combined CFR")
axes[1,1].set_title("Forest Cover vs CFR")
fig.suptitle("Case Fatality Rate (CFR) Analysis", fontsize=13, fontweight="bold")
savefig(fig, f"{BASE_OUT}/Insights_Health/cfr_analysis.png")

# ── 5.11  Population-normalised health burden ─────────────────────────────────
print("  [5.11] Health burden rankings...")
fig, axes = plt.subplots(1, 3, figsize=(18, 8))
for ax, (col, lbl, clr) in zip(axes, [("bronchitis_rate","Bronchitis Rate/1k","#E67E22"),
                                        ("asthma_rate","Asthma Rate/1k","#9B59B6"),
                                        ("total_resp_rate","Total Resp Rate/1k","#C0392B")]):
    d = annual.groupby("district")[col].mean().sort_values(ascending=True).reset_index()
    ax.barh(d["district"], d[col], color=clr, alpha=0.85, edgecolor="white")
    avg = d[col].mean()
    ax.axvline(avg, ls="--", color="black", lw=1.2, label=f"Avg={avg:.2f}")
    ax.set_title(lbl, fontweight="bold"); ax.legend(fontsize=8)
fig.suptitle("Population-Normalised Respiratory Burden — All Districts", fontsize=13, fontweight="bold")
savefig(fig, f"{BASE_OUT}/Insights_Health/population_normalized_health_burden.png")

# ── 5.12  Cross-pollutant vs health scatter ───────────────────────────────────
print("  [5.12] Cross-pollutant health scatter...")
fig, axes = plt.subplots(3, 3, figsize=(18, 16))
poll_health_pairs = [
    ("pm2.5_ug_m3","bronchitis_rate_per_1k","PM₂.₅","Bronchitis"),
    ("pm2.5_ug_m3","asthma_rate_per_1k","PM₂.₅","Asthma"),
    ("pm2.5_ug_m3","total_resp_rate_per_1k","PM₂.₅","Total Resp"),
    ("so2_ug_m3","bronchitis_rate_per_1k","SO₂","Bronchitis"),
    ("so2_ug_m3","asthma_rate_per_1k","SO₂","Asthma"),
    ("so2_ug_m3","total_resp_rate_per_1k","SO₂","Total Resp"),
    ("no2_ug_m3","bronchitis_rate_per_1k","NO₂","Bronchitis"),
    ("no2_ug_m3","asthma_rate_per_1k","NO₂","Asthma"),
    ("no2_ug_m3","total_resp_rate_per_1k","NO₂","Total Resp"),
]
for ax, (xc, yc, xl, yl) in zip(axes.flatten(), poll_health_pairs):
    for j, dist in enumerate(DISTRICTS):
        sub = fe[fe["district"]==dist][[xc,yc]].dropna()
        if len(sub) > 5:
            ax.scatter(sub[xc], sub[yc], alpha=0.3, s=12, color=plt.cm.tab20(j/len(DISTRICTS)))
    tmp = fe[[xc,yc]].dropna()
    z = np.polyfit(tmp[xc], tmp[yc], 1)
    xr = np.linspace(tmp[xc].min(), tmp[xc].max(), 100)
    ax.plot(xr, np.poly1d(z)(xr), "k-", lw=1.5)
    r, p = spearmanr(tmp[xc], tmp[yc])
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    ax.set_title(f"{xl} vs {yl} — ρ={r:.3f}, p={p:.4f}", fontsize=9)
fig.suptitle("Pollutant vs Health Outcomes — All Districts Combined", fontsize=13, fontweight="bold")
savefig(fig, f"{BASE_OUT}/Insights_Correlations/crossdistrict_pollutant_health_scatter.png")

# ── Save engineered dataset ────────────────────────────────────────────────────
fe.to_csv(f"{BASE_OUT}/dataset_feature_engineered.csv", index=False)
sp_df.to_csv(f"{BASE_OUT}/EDA/spearman_correlations.csv", index=False)

# ── Final count ───────────────────────────────────────────────────────────────
all_pngs = glob.glob(f"{BASE_OUT}/**/*.png", recursive=True)
all_csvs = glob.glob(f"{BASE_OUT}/**/*.csv", recursive=True)
print("\n" + "="*70)
print(f"  ✅ COMPLETE — {len(all_pngs)} plots  +  {len(all_csvs)} CSV files")
print(f"  📁 Output: {BASE_OUT}/")
for d in DIRS:
    count = len(glob.glob(f"{BASE_OUT}/{d}/*"))
    print(f"     {d:30s}: {count} files")
print("="*70)
