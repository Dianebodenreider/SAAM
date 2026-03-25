"""
SAAM Project 2026 - Groupe J
Région : North America (AMER) + Europe (EUR) | Scope CO2 : Scope 1 + Scope 2
"""

import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# PARAMÈTRES
# ──────────────────────────────────────────────────────────────
FILE_PATH        = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SAAM_Project_Group_J.xlsx")
REGIONS          = ["AMER", "EUR"]
STALE_THRESHOLD  = 0.50
MIN_OBS_YEARS    = 3
ESTIMATION_YEARS = 10
LOW_PRICE_THRES  = 0.5
START_ALLOC_YEAR = 2013

print("=" * 60)
print("  SAAM — Chargement & Nettoyage des données")
print("=" * 60)

# ──────────────────────────────────────────────────────────────
# ÉTAPE 1 : Chargement des feuilles
# ──────────────────────────────────────────────────────────────
print("\n[1/7] Chargement des données...")

static = pd.read_excel(FILE_PATH, sheet_name="Static_2025")
static.columns = ["ISIN", "NAME", "Country", "Region"]
static = static.dropna(subset=["ISIN"])

def load_ts(sheet_name):
    df = pd.read_excel(FILE_PATH, sheet_name=sheet_name)
    df = df[df.iloc[:, 1] != "$$ER: E100,INVALID CODE OR EXPRESSION ENTERED"]
    df = df.dropna(subset=[df.columns[1]])
    df = df.rename(columns={df.columns[0]: "NAME", df.columns[1]: "ISIN"})
    df = df.set_index("ISIN").drop(columns=["NAME"], errors="ignore")
    df = df.loc[:, df.columns.notna()]
    return df

ri_m  = load_ts("DS_RI_T_USD_M_2025")
ri_y  = load_ts("DS_RI_T_USD_Y_2025")
mv_m  = load_ts("DS_MV_T_USD_M_2025")
mv_y  = load_ts("DS_MV_T_USD_Y_2025")
rev_y = load_ts("DS_REV_Y_2025")
sc1   = load_ts("Scope1")
sc2   = load_ts("Scope2")

rf_raw = pd.read_excel(FILE_PATH, sheet_name="RF_Research_Data_Factors", names=["Date", "RF"])
rf_raw = rf_raw.dropna(subset=["Date"])
rf_raw["Date"] = pd.to_datetime(rf_raw["Date"].astype(str), format="%Y%m")
rf_raw = rf_raw.set_index("Date")["RF"] / 100

print(f"  RI mensuel : {ri_m.shape} | MV mensuel : {mv_m.shape} | RF : {len(rf_raw)} mois")

# ──────────────────────────────────────────────────────────────
# ÉTAPE 2 : Supprimer firmes sans aucune donnée de prix
# ──────────────────────────────────────────────────────────────
print("\n[2/7] Suppression des firmes sans données de prix...")
n0 = len(ri_m)
ri_m = ri_m[ri_m.notna().any(axis=1)]
print(f"  {n0 - len(ri_m)} supprimées ({n0} → {len(ri_m)})")
valid_isins = ri_m.index

# ──────────────────────────────────────────────────────────────
# ÉTAPE 3 : Filtrage AMER + EUR
# ──────────────────────────────────────────────────────────────
print(f"\n[3/7] Filtrage sur les régions {REGIONS}...")
region_isins   = set(static.loc[static["Region"].isin(REGIONS), "ISIN"].dropna())
universe_isins = list(region_isins & set(valid_isins))
for r in REGIONS:
    print(f"  '{r}' : {len(static.loc[static['Region']==r,'ISIN'].dropna())} firmes")
print(f"  Total AMER+EUR avec données RI : {len(universe_isins)}")

def filt(df):
    return df[df.index.isin(universe_isins)]

ri_m  = filt(ri_m);  ri_y  = filt(ri_y)
mv_m  = filt(mv_m);  mv_y  = filt(mv_y)
rev_y = filt(rev_y); sc1   = filt(sc1);  sc2 = filt(sc2)

# ──────────────────────────────────────────────────────────────
# ÉTAPE 4 : Délistements → retour de -100%
# ──────────────────────────────────────────────────────────────
print("\n[4/7] Traitement des délistements...")
ri_m = ri_m.sort_index(axis=1)
ri_m_clean = ri_m.copy()
ret_m = ri_m_clean.pct_change(axis=1)

n_delisted = 0
for isin in ri_m.index:
    row  = ri_m.loc[isin]
    last = row.last_valid_index()
    if last is not None and last != row.index[-1]:
        cols = ret_m.columns.tolist()
        if last in cols:
            idx = cols.index(last)
            if idx + 1 < len(cols):
                ret_m.at[isin, cols[idx + 1]] = -1.0
                n_delisted += 1
print(f"  {n_delisted} firmes avec retour -100% appliqué")

# ──────────────────────────────────────────────────────────────
# ÉTAPE 5 : Prix très bas (< 0.5) → NaN
# ──────────────────────────────────────────────────────────────
print("\n[5/7] Prix très bas traités comme NaN...")
n_low = (ri_m_clean < LOW_PRICE_THRES).sum().sum()
ri_m_clean[ri_m_clean < LOW_PRICE_THRES] = np.nan
ret_m = ri_m_clean.pct_change(axis=1)
print(f"  {n_low} valeurs remplacées, rendements recalculés")

# ──────────────────────────────────────────────────────────────
# ÉTAPE 6 : Forward-fill CO2 & Revenus + CO2 Scope 1+2
# ──────────────────────────────────────────────────────────────
print("\n[6/7] Forward-fill + calcul CO2 total (Sc1+Sc2)...")
sc1_filled   = sc1.ffill(axis=1).apply(pd.to_numeric, errors="coerce")
sc2_filled   = sc2.ffill(axis=1).apply(pd.to_numeric, errors="coerce")
rev_y_filled = rev_y.ffill(axis=1).apply(pd.to_numeric, errors="coerce")

sc1_a, sc2_a = sc1_filled.align(sc2_filled, join="outer", axis=1)
co2_total = sc1_a.add(sc2_a, fill_value=0)
co2_total[sc1_a.isna() & sc2_a.isna()] = np.nan
print(f"  CO2 total (Sc1+Sc2) : {co2_total.shape}")

# ──────────────────────────────────────────────────────────────
# ÉTAPE 7 : Univers d'investissement par année
# ──────────────────────────────────────────────────────────────
print("\n[7/7] Construction de l'univers par année...")

def get_stale_firms(ret_df, year):
    end   = pd.Timestamp(f"{year}-12-31")
    start = pd.Timestamp(f"{year - ESTIMATION_YEARS}-12-31")
    cols  = [c for c in ret_df.columns if start < c <= end]
    if not cols:
        return []
    zero_prop = (ret_df[cols] == 0).sum(axis=1) / ret_df[cols].notna().sum(axis=1)
    return zero_prop[zero_prop > STALE_THRESHOLD].index.tolist()

def get_investment_set(year):
    dec_cols = [c for c in ri_m_clean.columns if c.year == year and c.month == 12]
    if dec_cols:
        has_price = ri_m_clean[dec_cols[0]].notna()
        eligible  = has_price[has_price].index.tolist()
    else:
        eligible = ri_m_clean.index.tolist()

    end   = pd.Timestamp(f"{year}-12-31")
    start = pd.Timestamp(f"{year - ESTIMATION_YEARS}-12-31")
    w     = [c for c in ret_m.columns if start < c <= end]
    if w:
        obs      = ret_m.loc[eligible, w].notna().sum(axis=1)
        eligible = obs[obs >= MIN_OBS_YEARS * 12].index.tolist()

    stale    = get_stale_firms(ret_m.loc[eligible], year)
    eligible = [i for i in eligible if i not in stale]

    co2_cols = [c for c in co2_total.columns if c == year]
    if co2_cols:
        has_co2  = co2_total.loc[co2_total.index.isin(eligible), co2_cols[0]].notna()
        eligible = has_co2[has_co2].index.tolist()

    return eligible

investment_sets = {}
print(f"\n  {'Année':<8} {'N firmes':>10}")
print("  " + "-" * 20)
for y in range(START_ALLOC_YEAR, 2025):
    investment_sets[y] = get_investment_set(y)
    print(f"  {y:<8} {len(investment_sets[y]):>10}")

print("\n✓ Nettoyage terminé. Prêt pour la Partie I.")