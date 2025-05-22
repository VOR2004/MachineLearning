import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

DATA_PATH = Path("AmesHousing.csv")
df = pd.read_csv(DATA_PATH)

numeric_cols = df.select_dtypes(include=["number"]).columns
categorical_cols = df.select_dtypes(exclude=["number"]).columns

def delete_correlated(data: pd.DataFrame, threshold: float = 0.90):

    corr_matrix = data[numeric_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones_like(corr_matrix, dtype=bool), k=1))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    cleaned = data.drop(columns=to_drop)
    return cleaned, to_drop

df, dropped_cols = delete_correlated(df, threshold=0.90)
print(f"Удалено столбцов: {dropped_cols}")

numeric_cols = [c for c in numeric_cols if c not in dropped_cols]
categorical_cols = [c for c in categorical_cols if c in df.columns]

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ],
    remainder="drop",
)

processed_array = preprocessor.fit_transform(df)

encoded_cat_cols = list(
    preprocessor.named_transformers_["cat"]["encoder"].get_feature_names_out(categorical_cols)
)
processed_columns = numeric_cols + encoded_cat_cols
df_processed = pd.DataFrame(processed_array, columns=processed_columns)

OUTPUT_PATH = Path("clean_data.csv")
df_processed.to_csv(OUTPUT_PATH, index=False)

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

target = df["SalePrice"].values

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(df_processed)

x = X_reduced[:, 0]
y = X_reduced[:, 1]
z = target

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

scatter = ax.scatter(x, y, z, c=z, cmap='viridis', alpha=0.8)
ax.set_xlabel("Reduced X")
ax.set_ylabel("Reduced Y")
ax.set_zlabel("SalePrice")
fig.colorbar(scatter, ax=ax, label="SalePrice")
plt.tight_layout()
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

X = df_processed.values
y = df["SalePrice"].values

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"RMSE: {rmse:,.2f}")

from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

alphas = np.logspace(-2, 4, 20)
rmse_list = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha, random_state=42)
    ridge.fit(x_train, y_train)
    y_pred = ridge.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_list.append(rmse)

plt.figure(figsize=(8, 5))
plt.plot(alphas, rmse_list, marker='o')
plt.xscale('log')
plt.xlabel("Коэффициент регуляризации")
plt.ylabel("RMSE")
plt.title("Зависимость ошибки от коэффициента регуляризации")
plt.grid(True)
plt.tight_layout()
plt.show()

from sklearn.linear_model import Lasso

lasso = Lasso(alpha=1.0, max_iter=10000, random_state=42)
lasso.fit(x_train, y_train)

coefficients = lasso.coef_
feature_names = df_processed.columns

coef_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
})

non_zero = coef_df[coef_df["Coefficient"] != 0]
non_zero_sorted = non_zero.reindex(non_zero["Coefficient"].abs().sort_values(ascending=False).index)

most_important = non_zero_sorted.iloc[0]
print(f"Наиболее влияющий признак: {most_important['Feature']}")
print(f"коэффициент: {most_important['Coefficient']:.2f}")


