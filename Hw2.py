import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')
from statsmodels.stats.outliers_influence import variance_inflation_factor

df = pd.read_csv('AmesHousing.csv')
df = df.fillna(df.median(numeric_only=True))

for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.drop(['SalePrice'], axis=1))

X_df = pd.DataFrame(X_scaled, columns=df.drop(['SalePrice'], axis=1).columns)
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X_df.values, i) for i in range(X_df.shape[1])]
vif["features"] = X_df.columns
features_to_keep = vif[vif["VIF Factor"] < 10]["features"]
X_final = X_df[features_to_keep]


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_final)

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca[:, 0], X_pca[:, 1], df['SalePrice'], c=df['SalePrice'], cmap='viridis', s=20)
ax.set_xlabel('PCA1')
ax.set_ylabel('PCA2')
ax.set_zlabel('SalePrice')
plt.title('3D график')
plt.show()


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_final, df['SalePrice'], test_size=0.2, random_state=42
)


from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import numpy as np

alphas = np.logspace(-3, 2, 50)
rmse_list = []

for alpha in alphas:
    model = Lasso(alpha=alpha, max_iter=10000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_list.append(rmse)

plt.figure(figsize=(8,5))
plt.plot(alphas, rmse_list)
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('RMSE')
plt.show()


best_alpha = alphas[np.argmin(rmse_list)]
lasso = Lasso(alpha=best_alpha, max_iter=10000)
lasso.fit(X_train, y_train)

y_pred = lasso.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Итоговый RMSE: {rmse}')

feature_importance = pd.Series(lasso.coef_, index=X_final.columns)
top_feature = feature_importance.abs().idxmax()
print(f'Наиболее важный признак: {top_feature}')
