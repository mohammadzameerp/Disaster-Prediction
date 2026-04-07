import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from pandas.api.types import is_datetime64_any_dtype, is_bool_dtype, is_numeric_dtype, is_string_dtype, is_object_dtype, is_categorical_dtype

def load_dataset(path):
    ext = path.lower()
    if ext.endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_excel(path)

def find_target_column(df):
    cols = {c: c.lower().replace(" ", "").replace("-", "") for c in df.columns}
    desired = ["disastertype", "disastersubtype", "disastersubgroup", "disastergroup"]
    for want in desired:
        for original, norm in cols.items():
            if norm == want:
                return original
    return None

def summarize_dataset(df):
    desc = df.describe(include="all").fillna(0).to_dict()
    missing = df.isna().sum().to_dict()
    columns = df.columns.tolist()
    rows = int(df.shape[0])
    return {"summary": desc, "missing": missing, "columns": columns, "rows": rows}

def basic_plots(df):
    plots = {}
    ycol = None
    for c in ["Year", "year"]:
        if c in df.columns:
            ycol = c
            break
    tcol = find_target_column(df)
    if ycol and tcol:
        g = df.groupby([ycol, tcol]).size().reset_index(name="count")
        plots["disaster_by_year"] = {
            "x": g[ycol].astype(str).tolist(),
            "y": g["count"].tolist(),
            "label": g[tcol].astype(str).tolist()
        }
    if tcol:
        c = df[tcol].value_counts()
        plots["type_distribution"] = {"labels": c.index.astype(str).tolist(), "values": c.values.tolist()}
    return plots

def preview_html(df, n=10):
    return df.head(n).to_html(classes="table table-dark table-striped", index=False)

def flood_insights(df):
    out = {}
    cols = {c.lower(): c for c in df.columns}
    lat = None; lon = None
    for k in ["latitude","lat"]:
        if k in cols: lat = cols[k]; break
    for k in ["longitude","lon","long"]:
        if k in cols: lon = cols[k]; break
    rain_col = None
    for k in ["rainfall","precipitation","rain"]:
        if k in cols: rain_col = cols[k]; break
    if lat and lon:
        out["geo"] = {
            "lat": df[lat].dropna().astype(float).tolist()[:1000],
            "lon": df[lon].dropna().astype(float).tolist()[:1000]
        }
    if rain_col:
        s = pd.to_numeric(df[rain_col], errors="coerce").dropna()
        hist = np.histogram(s, bins=10)
        out["rain_hist"] = {"bins": hist[1].tolist(), "counts": hist[0].tolist()}
        out["rain_summary"] = {"mean": float(s.mean()), "max": float(s.max()), "min": float(s.min())}
    return out

def preprocess_dataset(df):
    df = df.copy()
    target_col = find_target_column(df)
    if target_col is None:
        fallback = None
        cols_norm = {c.lower().replace(" ", "").replace("-", ""): c for c in df.columns}
        for key in ["disastersubtype", "disastersubgroup", "disastergroup"]:
            if key in cols_norm:
                fallback = cols_norm[key]
                break
        if fallback is None:
            # DEFAULT TO THE LAST COLUMN IF NONE MATCH
            fallback = df.columns[-1]
        df["__target__"] = df[fallback].astype(str)
        target_col = "__target__"
    y = df[target_col].astype(str)
    X = df.drop(columns=[target_col])

    # Expand datetime columns into numeric parts and drop originals
    dt_new_cols = []
    for col in list(X.columns):
        if is_datetime64_any_dtype(X[col]):
            dt = pd.to_datetime(X[col], errors="coerce")
            X[f"{col}_year"] = dt.dt.year
            X[f"{col}_month"] = dt.dt.month
            X[f"{col}_day"] = dt.dt.day
            dt_new_cols.extend([f"{col}_year", f"{col}_month", f"{col}_day"])
            X = X.drop(columns=[col])

    # Attempt to parse obvious date-like strings
    for col in list(X.columns):
        if is_object_dtype(X[col]) or is_string_dtype(X[col]) or is_categorical_dtype(X[col]):
            parsed = pd.to_datetime(X[col], errors="coerce")
            if parsed.notna().sum() > 0:
                X[f"{col}_year"] = parsed.dt.year
                X[f"{col}_month"] = parsed.dt.month
                X[f"{col}_day"] = parsed.dt.day
                dt_new_cols.extend([f"{col}_year", f"{col}_month", f"{col}_day"])
                # Keep original for label encoding, but fill NaNs first
                mode = X[col].mode().iloc[0] if X[col].mode().shape[0] else ""
                X[col] = X[col].fillna(mode)

    # Store default values for missing columns during prediction
    default_values = {}
    
    # Fill missing values and type-normalize
    for col in X.columns:
        if is_bool_dtype(X[col]):
            default_val = False
            X[col] = X[col].fillna(default_val).astype(int)
            default_values[col] = 0  # encoded as 0
        elif is_object_dtype(X[col]) or is_string_dtype(X[col]) or is_categorical_dtype(X[col]):
            mode = X[col].mode().iloc[0] if X[col].mode().shape[0] else ""
            default_values[col] = mode  # store mode before encoding
            X[col] = X[col].fillna(mode)
        elif is_numeric_dtype(X[col]):
            med = pd.to_numeric(X[col], errors="coerce").median()
            if pd.isna(med):
                med = 0.0
            default_values[col] = float(med)
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(med)
        else:
            # Fallback: convert to string then label encode later
            default_val = ""
            default_values[col] = default_val
            X[col] = X[col].astype(str)
    
    encoders = {}
    for col in X.columns:
        if is_object_dtype(X[col]) or is_string_dtype(X[col]) or is_categorical_dtype(X[col]):
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
            # Update default value to encoded version
            try:
                default_values[col] = le.transform([str(default_values[col])])[0]
            except:
                default_values[col] = 0
    
    y_le = LabelEncoder()
    y = y_le.fit_transform(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)
    counts = np.bincount(y)
    min_count = int(np.min(counts)) if counts.size > 0 else 0
    if min_count <= 1:
        X_res, y_res = X_scaled, y
    else:
        k = max(1, min(5, min_count - 1))
        smote = SMOTE(k_neighbors=k)
        X_res, y_res = smote.fit_resample(X_scaled, y)
    # Use stratify only when all classes have >=2 samples
    counts_res = np.bincount(y_res)
    use_strat = counts_res.size > 0 and int(np.min(counts_res)) >= 2
    if use_strat:
        X_train, X_test, y_train, y_test = train_test_split(
            X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_res, y_res, test_size=0.2, random_state=42
        )
    artifacts = {
        "encoders": encoders,
        "y_encoder": y_le,
        "scaler": scaler,
        "columns": X.columns.tolist(),
        "default_values": default_values
    }
    return X_train, X_test, y_train, y_test, artifacts
