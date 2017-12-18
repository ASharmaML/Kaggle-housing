import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import skew
from sklearn import kernel_ridge as kr
from sklearn.linear_model import Lasso

import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score

#Load the data
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
train = train[train.GrLivArea < 4000]

all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))


# Tried dropping columns
del all_data["MSSubClass" ]
del all_data["LandSlope" ]
#del all_data["OverallCond" ] valuable feature
#del all_data["BsmtHalfBath" ]
#del all_data["BsmtFullBath" ] valuable feature
del all_data["MoSold" ]
#del all_data["YrSold" ]
del all_data["MiscVal" ]
#del all_data["KitchenAbvGr"]
#del all_data["TotalBsmtSF"]
del all_data["Fence"]
#del all_data["SaleType"]
#del all_data["SaleCondition"]
#del all_data["Functional"]
#del all_data["WoodDeckSF"]
#del all_data["KitchenAbvGr"]
#del all_data["TotalBsmtSF"]
del all_data["RoofStyle"]
del all_data["BsmtFinType2"]
#del all_data["PoolQC"]
del all_data["MiscFeature"]
del all_data["Alley"]
del all_data["FireplaceQu"]
del all_data["LotFrontage"]
#del all_data["MasVnrArea"]
#del all_data["MasVnrType"]
#del all_data["Exterior1st"]
#del all_data["Utilities"]
#del all_data["LotConfig"]
#del all_data["LandContour"]
#del all_data["TotalBsmtSF"]
#del all_data["GarageYrBlt"]

# all_data["lat"] = all_data.Neighborhood.replace({'Blmngtn' : 42.062806,
#                                                'Blueste' : 42.009408,
#                                                 'BrDale' : 42.052500,
#                                                 'BrkSide': 42.033590,
#                                                 'ClearCr': 42.025425,
#                                                 'CollgCr': 42.021051,
#                                                 'Crawfor': 42.025949,
#                                                 'Edwards': 42.022800,
#                                                 'Gilbert': 42.027885,
#                                                 'GrnHill': 42.000854,
#                                                 'IDOTRR' : 42.019208,
#                                                 'Landmrk': 42.044777,
#                                                 'MeadowV': 41.991866,
#                                                 'Mitchel': 42.031307,
#                                                 'NAmes'  : 42.042966,
#                                                 'NoRidge': 42.050307,
#                                                 'NPkVill': 42.050207,
#                                                 'NridgHt': 42.060356,
#                                                 'NWAmes' : 42.051321,
#                                                 'OldTown': 42.028863,
#                                                 'SWISU'  : 42.017578,
#                                                 'Sawyer' : 42.033611,
#                                                 'SawyerW': 42.035540,
#                                                 'Somerst': 42.052191,
#                                                 'StoneBr': 42.060752,
#                                                 'Timber' : 41.998132,
#                                                 'Veenker': 42.040106})
#
# all_data["lon"] = all_data.Neighborhood.replace({'Blmngtn' : -93.639963,
#                                                'Blueste' : -93.645543,
#                                                 'BrDale' : -93.628821,
#                                                 'BrkSide': -93.627552,
#                                                 'ClearCr': -93.675741,
#                                                 'CollgCr': -93.685643,
#                                                 'Crawfor': -93.620215,
#                                                 'Edwards': -93.663040,
#                                                 'Gilbert': -93.615692,
#                                                 'GrnHill': -93.643377,
#                                                 'IDOTRR' : -93.623401,
#                                                 'Landmrk': -93.646239,
#                                                 'MeadowV': -93.602441,
#                                                 'Mitchel': -93.626967,
#                                                 'NAmes'  : -93.613556,
#                                                 'NoRidge': -93.656045,
#                                                 'NPkVill': -93.625827,
#                                                 'NridgHt': -93.657107,
#                                                 'NWAmes' : -93.633798,
#                                                 'OldTown': -93.615497,
#                                                 'SWISU'  : -93.651283,
#                                                 'Sawyer' : -93.669348,
#                                                 'SawyerW': -93.685131,
#                                                 'Somerst': -93.643479,
#                                                 'StoneBr': -93.628955,
#                                                 'Timber' : -93.648335,
#                                                 'Veenker': -93.657032})

#take log of skewed features
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.5] + skewed_feats[skewed_feats < -0.5]
skewed_feats = skewed_feats.index

# replace string values with dummy numerical values
all_data = pd.get_dummies(all_data)
all_data = all_data.fillna(all_data.mean())





all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

# Scales all numerical features to improve score
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(all_data[numeric_feats])

scaled = scaler.transform(all_data[numeric_feats])
for i, col in enumerate(numeric_feats):
    all_data[col] = scaled[:, i]

scaled = scaler.transform(all_data[numeric_feats])
for i, col in enumerate(numeric_feats):
    all_data[col] = scaled[:, i]

# Isolate x training data and test data
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = np.log1p(train.SalePrice)

# Use to evaluate cross validation performance of a model
def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)

# Parameter search for Ridge Regression (L2 regularisation)
alphas = np.arange(1, 51, 10)
cv_ridge = [rmse_cv(kr.KernelRidge(alpha = alpha)).mean()
            for alpha in alphas]

# Parameter search for lasso (L1 regularisation)
alphas2 = np.arange(0.0004, 0.00042, 0.00001)
cv_lasso = [rmse_cv(Lasso(alpha = alpha)).mean()
             for alpha in alphas2]

model_xgb = xgb.XGBRegressor(
                colsample_bytree=0.2,
                 gamma=0.0,
                 learning_rate=0.05,
                 max_depth=6,
                 min_child_weight=1.5,
                 n_estimators=7200,
                 reg_alpha=0.9,
                 reg_lambda=0.6,
                 subsample=0.2,
                 seed=42,
                 silent=1)
#cv_xgb = rmse_cv(model_xgb).mean()

print(min(cv_ridge))
print(min(cv_lasso))


cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Validation - Just Do It")
plt.xlabel("alpha")
plt.ylabel("rmse")
plt.show()

cv_lasso = pd.Series(cv_lasso, index = alphas2)
cv_lasso.plot(title = "Validation - Just Do It")
plt.xlabel("alpha")
plt.ylabel("rmse")

# Comment this code out to suppress plot. Plot seems to prevent rest of code running. Uncomment to optimise parameters
# visually.
plt.show()


# Two alphas observed from plot of parameter against performance
alpha = 11
alpha2 = 0.0004

#Encode the species as a number (0-99)
y_train = np.log1p(train.SalePrice)

#Set the parameters on the linear regressor and then run the algorithm
#kr.KernelRidge can be switched with lasso, and alpha with alpha2 to see what the Lasso does

regr = kr.KernelRidge(alpha=alpha)
regr.fit(X_train,y_train)

regr_lasso = Lasso(alpha=alpha2)
regr_lasso.fit(X_train,y_train)


y_test_lasso = regr_lasso.predict(X_test)
y_test_lasso = np.exp(y_test_lasso) - 1

regr_xgb = xgb.XGBRegressor(
                colsample_bytree=0.2,
                 gamma=0.0,
                 learning_rate=0.05,
                 max_depth=6,
                 min_child_weight=1.5,
                 n_estimators=7200,
                 reg_alpha=0.9,
                 reg_lambda=0.6,
                 subsample=0.2,
                 seed=42,
                 silent=1)

regr_xgb.fit(X_train,y_train)


y_test_xgb = regr_xgb.predict(X_test)
y_test_xgb = np.exp(y_test_xgb) - 1

# predicting on training set to evaluate types of errors with lasso and ridge
y_train_predict = regr.predict(X_train)
y_train_predict = np.exp(y_train_predict) - 1
y_train_predict[y_train_predict > 300000] = y_train_predict[y_train_predict > 300000] * 1
y_train_predict[y_train_predict > 450000] = y_train_predict[y_train_predict > 450000] * 1.06
y_train_predict_lasso = regr_xgb.predict(X_train)
y_train_predict_lasso = np.exp(y_train_predict_lasso) - 1
y_train_predict_lasso[y_train_predict_lasso > 300000] = y_train_predict_lasso[y_train_predict_lasso > 300000] * 1
y_train_predict_lasso[y_train_predict_lasso > 450000] = y_train_predict_lasso[y_train_predict_lasso > 450000] * 1

ridge = pd.DataFrame({"ridge train": y_train_predict,"actual": train.SalePrice})
ridge.plot(x = "ridge train", y = "actual", kind = "scatter",s=0.1)
x = y_train_predict_lasso
y = train.SalePrice
plt.scatter(x, y,color='red',s=0.1)
xx = range(0,700000,1)
yy = range(0,700000,1)
plt.plot(xx, yy)
plt.show()

# Predict the house price for each validation data
y_test = regr.predict(X_test)
y_test = np.exp(y_test) - 1
y_test[y_test > 300000] = y_test[y_test > 300000] * 1
y_test[y_test > 450000] = y_test[y_test > 450000] * 1.06

y_test_xgb[y_test_xgb > 300000] = y_test_xgb[y_test_xgb > 300000] * 1
y_test_xgb[y_test_xgb > 450000] = y_test_xgb[y_test_xgb > 450000] * 1.06
y_test_xgb[y_test_xgb > 545000] = y_test_xgb[y_test_xgb > 545000] + 250000

lasso_ridge = pd.DataFrame({"ridge": y_test,"xgb": y_test_xgb})
lasso_ridge.plot(x = "ridge", y = "xgb", kind = "scatter")
plt.show()
test_ids = test.pop('Id')

# Comment out last line if you do not want to change the sample submission file.
# Rename submission file name to avoid confusion
submission = pd.DataFrame(y_test*1 + y_test_xgb, index=test_ids)
submission = submission.rename(columns={0: 'SalePrice'})
submission.to_csv('./submission_krr_regressor.csv')
