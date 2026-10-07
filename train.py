"""Freight rate model. Run: python train.py  (writes validation_predictions.csv + december_predictions.csv)"""
import numpy as np, pandas as pd, lightgbm as lgb

tr = pd.read_csv("train-test.csv", parse_dates=["date"])
va = pd.read_csv("validation.csv", parse_dates=["date"])
dec = pd.read_csv("december-chart-inputs.csv", parse_dates=["date"])

# Daily market context from feature columns only (no labels) -> usable for December too.
ctx = pd.concat([tr, va]).groupby("date")[["market_index", "quote_signal"]].mean().add_prefix("day_")
ctx["day_mi_7d"] = ctx.day_market_index.rolling(7, center=True, min_periods=1).mean()
# ponytail: December has no market_index; borrow the validation-file daily mean for that date.
dec_pick = dec.merge(va.drop_duplicates(), how="left", on=["pickup", "delivery"], suffixes=("", "_v"))
coords = pd.concat([tr, va]).drop_duplicates(["pickup", "delivery"])[
    ["pickup", "delivery", "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]]
dec = dec.merge(coords, on=["pickup", "delivery"], how="left")
dec["quote_signal"] = 2.05; dec["market_index"] = np.nan

CATS = ["pickup", "delivery", "equipment"]

def hav(a, b, c, d):
    a, b, c, d = map(np.radians, (a, b, c, d))
    x = np.sin((c - a) / 2) ** 2 + np.cos(a) * np.cos(c) * np.sin((d - b) / 2) ** 2
    return 2 * 3958.8 * np.arcsin(np.sqrt(x))

def feats(df, cat_types):
    X = pd.DataFrame(index=df.index)
    X["dist"] = df.distance
    X["log_dist"] = np.log(df.distance)
    hv = hav(df.pickup_lat, df.pickup_lon, df.delivery_lat, df.delivery_lon)
    X["dist_ratio"] = df.distance / hv                  # >>1.5 flags corrupted distance
    X["weight"] = df.weight.abs()                       # negative weights = sign errors
    X["w_neg"] = (df.weight < 0).astype(float)          # NaN stays NaN; LightGBM handles it
    X["mi"] = df.market_index                           # NaN stays NaN
    X["qs"] = df.quote_signal
    X["qs_dev"] = (df.quote_signal - 2.05).abs()        # U-shaped relation to rate
    X = X.join(df[["date"]].merge(ctx, left_on="date", right_index=True, how="left").drop(columns="date"))
    X["mi"] = X.mi.fillna(X.day_market_index)
    X["doy"] = df.date.dt.dayofyear; X["dow"] = df.date.dt.dayofweek
    for c in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]:
        X[c] = df[c]
    for c in CATS:
        X[c] = pd.Categorical(df[c], categories=cat_types[c])
    return X

cat_types = {c: sorted(set(tr[c]) | set(va[c]) | set(dec[c])) for c in CATS}
y = np.log(tr.posted_rate)
P = dict(objective="huber", alpha=0.3, learning_rate=0.03, n_estimators=1500, num_leaves=63,
         min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
         cat_smooth=20, verbose=-1)

def fit(Xa, ya, n=P["n_estimators"]):
    return lgb.LGBMRegressor(**{**P, "n_estimators": n}).fit(Xa, ya)

def mape(y_true, y_pred): return np.mean(np.abs(np.exp(y_pred) - np.exp(y_true)) / np.exp(y_true))

if __name__ == "__main__":
    X = feats(tr, cat_types)
    # Forward-chaining time folds (mirrors the Nov-Dec validation set being strictly later).
    for cut_a, cut_b in [("2025-07-01", "2025-08-31"), ("2025-09-01", "2025-10-31")]:
        a, b = tr.date < cut_a, (tr.date >= cut_a) & (tr.date <= cut_b)
        m = fit(X[a], y[a])
        p = m.predict(X[b])
        print(f"train<{cut_a} -> {cut_a}..{cut_b}: MAPE={mape(y[b], p):.4f} "
              f"MAE=${np.mean(np.abs(np.exp(p) - np.exp(y[b]))):.0f}")
    m = fit(X, y)
    Xv = feats(va, cat_types)
    pd.DataFrame({"load_id": va.load_id, "predicted_rate": np.exp(m.predict(Xv)).round(2)}) \
        .to_csv("validation_predictions.csv", index=False)
    d = dec.copy(); d["distance"] = d.distance.astype(float); d["weight"] = d.weight.astype(float)
    d["predicted_rate"] = np.exp(m.predict(feats(d, cat_types))).round(2)
    d[["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]] \
        .assign(date=lambda x: x.date.dt.strftime("%Y-%m-%d")).to_csv("december_predictions.csv", index=False)
    imp = pd.Series(m.feature_importances_, X.columns).sort_values(ascending=False)
    print(imp.head(10))
