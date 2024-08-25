# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import dill to save the model
import dill


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, RidgeCV, LassoCV, ElasticNetCV, SGDRegressor
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, VotingRegressor, StackingRegressor
from xgboost import XGBRegressor,XGBRFRegressor

# =============================================================================
# from sklearn.neural_network import MLPRegressor
# =============================================================================

import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD, AdamW, RMSprop
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
# Load the data from dill db

#dill.load_session('notebook_env.db')


# %% [markdown]
# ##### Data Description
# 
# Introduction About the Data : The dataset The goal is to predict price of given diamond (Regression Analysis).
# 
# There are 10 independent variables (including id):
# 
# id : unique identifier of each diamond
# 
# carat : Carat (ct.) refers to the unique unit of weight measurement used exclusively to weigh gemstones and diamonds.
# 
# cut : Quality of Diamond Cut
# 
# color : Color of Diamond
# 
# clarity : Diamond clarity is a measure of the purity and rarity of the stone, graded by the visibility of these characteristics under 10-power magnification.
# 
# depth : The depth of diamond is its height (in millimeters) measured from the culet (bottom tip) to the table (flat, top surface)
# 
# table : A diamond's table is the facet which can be seen when the stone is viewed face up.
# 
# x : Diamond X dimension
# 
# y : Diamond Y dimension
# 
# x : Diamond Z dimension
# 
# Target variable:
# 
# price: Price of the given Diamond.
# 

# %%
data = pd.read_csv('train.csv')
data.head()


# %%
data.info()

# %%
data.isna().sum()

# %%
data.describe()

# %%
data.cut.value_counts()

# %%
data.color.value_counts()

# %%
data.clarity.value_counts()

# %%
data.columns

# %%
data.drop('id', axis=1, inplace=True)

# %%
data

# %%
data.duplicated().sum()

# %% [markdown]
# #### EDA

# %%
num_cols = data.select_dtypes(include=np.number).columns
cat_cols = data.select_dtypes(exclude=np.number).columns
print('Numerical Columns:', num_cols)
print('Categorical Columns:', cat_cols)


# %%
for col in cat_cols:
    print(col, data[col].unique())

# %%
for col in cat_cols:
    sns.countplot(x=data[col],palette='Set3', hue=data['cut'])
    plt.show()

# %%
for col in num_cols:
    sns.histplot(data[col], kde=True,color='red')
    plt.show()

# %%
# Correlation Matrix

plt.figure(figsize=(10, 6))
sns.heatmap(data[num_cols].corr(), annot=True, cmap='magma')
plt.show()


# %%
# Outliers

for col in num_cols:
    sns.boxplot(data[col])
    plt.show()

# %%
data.head(10)

# %% [markdown]
# ##### Feature Engineering

# %%
#Encoding

cut_map = {'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5}
clarity_map = {'I1': 1, 'SI2': 2, 'SI1': 3, 'VS2': 4, 'VS1': 5, 'VVS2': 6, 'VVS1': 7, 'IF': 8}


data['cut'] = data['cut'].map(cut_map)
data['clarity'] = data['clarity'].map(clarity_map)

data.head()


# %%
color_map = {'J': 1, 'I': 2, 'H': 3, 'G': 4, 'F': 5, 'E': 6, 'D': 7}

data['color'] = data['color'].map(color_map)

data.head()


data['volume'] = data['x'] * data['y'] * data['z']
data['surface_area'] = 2 * (data['x'] * data['y'] + data['x'] * data['z'] + data['y'] * data['z'])

data.drop(['x', 'y', 'z'], axis=1, inplace=True)



# %%
X = data.drop('price', axis=1)
y = data['price']



# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



# # %%
# lr = LinearRegression()
# lr.fit(X_train, y_train)
# y_pred = lr.predict(X_test)

# print('R2 Score:', r2_score(y_test, y_pred))
# print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
# print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))

# print('Cross Validation Score:', cross_val_score(lr, X_train, y_train, cv=5).mean())

# # %%
# # get all parameters of the model
# lr.get_params()


# # %%
# sgd_pipe = Pipeline([ ('sgd', SGDRegressor())])

# param_grid = {
#     'sgd__alpha': [0.0001, 0.001,],
#     'sgd__max_iter': [1000, 2000, 3000,4000],
#     'sgd__penalty': ['l1', 'l2'],
#     'sgd__learning_rate': ['adaptive'],
#     'sgd__eta0': [0.01, 0.1,],
#     'sgd__tol': [1e-3, 1e-4, 1e-5]



# }

# sgd_grid = GridSearchCV(sgd_pipe, param_grid, cv=5, n_jobs=-1,verbose=2)
# sgd_grid.fit(X_train, y_train)
# y_pred = sgd_grid.predict(X_test)

# print('R2 Score:', r2_score(y_test, y_pred))
# print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
# print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
# print('best parameters:', sgd_grid.best_params_)




# # %%
# sgd_grid.best_estimator_.get_params()

# # %%
# # ridge cv

# ridge_pipe = Pipeline([('ridge', RidgeCV())])

# param_grid = {
#     'ridge__alphas': [(0.1, 1.0, 10.0), (0.1, 0.5, 1.0), (0.1, 0.2, 0.3, 0.4, 0.5),(0.001, 0.01, 0.1),  (0.0001, 0.001, 0.01),(0.0003)],
# }

# ridge_grid = GridSearchCV(ridge_pipe, param_grid, cv=5, n_jobs=-1, verbose=2)
# ridge_grid.fit(X_train, y_train)
# y_pred = ridge_grid.predict(X_test)

# print('R2 Score:', r2_score(y_test, y_pred))
# print('Mean Squared Error:', mean_squared_error(y_test, y_pred))

# print('best parameters:', ridge_grid.best_params_)
# ridge_grid.best_estimator_.get_params()



# # %%
# # lasso cv

# lasso_pipe = Pipeline([('lasso', LassoCV())])

# param_grid = {
#     'lasso__alphas': [(0.1, 1.0, 10.0), (0.1, 0.5, 1.0), (0.1, 0.2, 0.3, 0.4, 0.5),(0.0001, 0.001, 0.01),(0.0003)],
# }

# lasso_grid = GridSearchCV(lasso_pipe, param_grid, cv=5, n_jobs=-1, verbose=2)

# lasso_grid.fit(X_train, y_train)

# y_pred = lasso_grid.predict(X_test)

# print('R2 Score:', r2_score(y_test, y_pred))

# print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
# print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
# print('best parameters:', lasso_grid.best_params_)

# lasso_grid.best_estimator_.get_params()



# # %%
# # elastic net cv

# elastic_pipe = Pipeline([('elastic', ElasticNetCV())])

# param_grid = {
#     'elastic__alphas': [(0.0001, 0.001, 0.01)],
#     'elastic__l1_ratio': [0.5],
#     'elastic__max_iter': [8000],
# }

# elastic_grid = GridSearchCV(elastic_pipe, param_grid, cv=5, n_jobs=-1, verbose=2)

# elastic_grid.fit(X_train, y_train)

# y_pred = elastic_grid.predict(X_test)

# print('R2 Score:', r2_score(y_test, y_pred))
# print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
# print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))

# print('best parameters:', elastic_grid.best_params_)

# elastic_grid.best_estimator_.get_params()









# # %%

# # SVR model
# # lets do polynomialisation with ridge regression

# from sklearn.preprocessing import PolynomialFeatures

# poly_ridge_pipe = Pipeline([('poly', PolynomialFeatures()), ('ridge', Ridge())])

# param_grid = {
#     'poly__degree': [2, 3],
#     'ridge__alpha': [0.1, 1.0, 5.0,7.0, 10.0,20.0,22.0,25.0,27.0],
    
# }

# poly_ridge_grid = GridSearchCV(poly_ridge_pipe, param_grid, cv=5, n_jobs=-1, verbose=2)

# poly_ridge_grid.fit(X_train, y_train)

# y_pred = poly_ridge_grid.predict(X_test)

# print('R2 Score:', r2_score(y_test, y_pred))
# print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
# print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))

# print('best parameters:', poly_ridge_grid.best_params_)

# poly_ridge_grid.best_estimator_.get_params()





# # %%
# # polynomial features with lasso regression

# poly_lasso_pipe = Pipeline([('poly', PolynomialFeatures()), ('lasso', Lasso())])

# param_grid = {
#     'poly__degree': [2, 3],
#     'lasso__alpha': [0.1, 1.0, 5.0,7.0, 10.0,20.0,22.0,25.0,27.0],
    
# }

# poly_lasso_grid = GridSearchCV(poly_lasso_pipe, param_grid, cv=5, n_jobs=-1, verbose=2)

# poly_lasso_grid.fit(X_train, y_train)

# y_pred = poly_lasso_grid.predict(X_test)

# print('R2 Score:', r2_score(y_test, y_pred))
# print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
# print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
# print('best parameters:', poly_lasso_grid.best_params_)
# poly_lasso_grid.best_estimator_.get_params()


# # %%

# ## Decision Tree Regressor

# dec_pipe = Pipeline([('dec', DecisionTreeRegressor())])

# param_grid = {
#     'dec__max_depth': [5, 10,12],
#     'dec__min_samples_split': [10,30,40,50],
#     'dec__min_samples_leaf': [1, 2, 4],
#     'dec__ccp_alpha': [0.0, 0.01, 0.05,0.1]

#     }

# dec_grid = GridSearchCV(dec_pipe, param_grid, cv=4, n_jobs=-1, verbose=2)

# dec_grid.fit(X_train, y_train)

# y_pred = dec_grid.predict(X_test)

# print('R2 Score:', r2_score(y_test, y_pred))

# print('Mean Squared Error:', mean_squared_error(y_test, y_pred))

# print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))

# print('best parameters:', dec_grid.best_params_)

# dec_grid.best_estimator_.get_params()



# # %%
# # set the best params with polynomial features and apply the model

# poly_dec_pipe = Pipeline([('poly', PolynomialFeatures()), ('dec', DecisionTreeRegressor())])

# param_grid = {
#     'poly__degree': [2, 3],
#     'dec__max_depth': [5, 10, 15],
#     'dec__min_samples_split': [20, 30],
#     'dec__min_samples_leaf': [1, 2, 4],
#     'dec__ccp_alpha': [0.0, 0.01, 0.05, 0.1]

# }

# poly_dec_grid = GridSearchCV(poly_dec_pipe, param_grid, cv=3, n_jobs=-1, verbose=2)

# poly_dec_grid.fit(X_train, y_train)

# y_pred = poly_dec_grid.predict(X_test)

# print('R2 Score:', r2_score(y_test, y_pred))

# print('Mean Squared Error:', mean_squared_error(y_test, y_pred))

# print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))

# print('best parameters:', poly_dec_grid.best_params_)

# poly_dec_grid.best_estimator_.get_params()



# # %% [markdown]
# # ##### Ensemble Models

# # %%
# rf_pipe = Pipeline([('rf', RandomForestRegressor())])

# param_grid = {
#     'rf__n_estimators': [200],
#     'rf__max_depth': [10, 15],
#     'rf__min_samples_split': [15,20],
#     'rf__min_samples_leaf': [ 2 ],
    
#     'rf__ccp_alpha': [0.0, 0.02]

# }

# rf_grid = GridSearchCV(rf_pipe, param_grid, cv=3, n_jobs=-1, verbose=2)

# rf_grid.fit(X_train, y_train)

# y_pred = rf_grid.predict(X_test)

# print('R2 Score:', r2_score(y_test, y_pred))

# print('Mean Squared Error:', mean_squared_error(y_test, y_pred))

# print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))

# print('best parameters:', rf_grid.best_params_)

# rf_grid.best_estimator_.get_params()

# # %%
# rf_grid.best_estimator_[0].get_params()
# #print r2 score, mean squared error, mean absolute error, best parameters and best estimator parameters
# print('R2 Score:', r2_score(y_test, y_pred))

# # %%
# ## Gradient Boosting Regressor

# gb_pipe = Pipeline([('gb', GradientBoostingRegressor())])

# param_grid = {
#     'gb__n_estimators': [150],
#     'gb__learning_rate': [0.01,0.1,1],
#     'gb__max_depth': [3, 5],
#     'gb__min_samples_split': [2,4,6],
#     'gb__min_samples_leaf': [2, 4],
#     # 'gb__ccp_alpha': [0.0, 0.02]
# }

# gb_grid = GridSearchCV(gb_pipe, param_grid, cv=3, n_jobs=-1, verbose=2)

# gb_grid.fit(X_train, y_train)

# y_pred = gb_grid.predict(X_test)

# print('R2 Score:', r2_score(y_test, y_pred))

# print('Mean Squared Error:', mean_squared_error(y_test, y_pred))

# print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))

# print('best parameters:', gb_grid.best_params_)

# gb_grid.best_estimator_.get_params()


# # %%

# # XGBoost Regressor

xgb_pipe = Pipeline([('xgb', XGBRegressor())])

param_grid = {
    'xgb__n_estimators': [150,200],
    'xgb__learning_rate': [0.1,0.3],
    'xgb__max_depth': [3, 5,7,10],
    'xgb__min_child_weight': [1, 3, 5,7],
    'xgb__lambda': [0.01, 0.1, 1],
    'xgb__gamma': [0, 0.1, 0.2, 0.3],
  
}

xgb_grid = GridSearchCV(xgb_pipe, param_grid, cv=4, n_jobs=-1, verbose=2)

xgb_grid.fit(X_train, y_train)

y_pred = xgb_grid.predict(X_test)

print('R2 Score:', r2_score(y_test, y_pred))

print('Mean Squared Error:', mean_squared_error(y_test, y_pred))

print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))

print('best parameters:', xgb_grid.best_params_)

xgb_grid.best_estimator_.get_params()



## Neural Network




# =============================================================================
# model = Sequential([
#     Dense(64, input_dim=X_train.shape[1], activation='relu'),
#     Dense(128, activation='relu'),
#     Dense(64, activation='relu'),
#     Dense(1)  # Output layer for regression
# ])
# 
# 
# model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])
# 
# history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)
# 
# 
# y_pred = model.predict(X_test)
# 
# # Calculate R² score
# r2 = r2_score(y_test, y_pred)
# print(f"R² Score: {r2}")
# 
# =============================================================================

print(np.isnan(X_train).any(), np.isnan(y_train).any())
print(np.isinf(X_train).any(), np.isinf(y_train).any())



# =============================================================================
# from tensorflow.keras.optimizers import SGD
# =============================================================================

# =============================================================================
# # Define the model with regularization and batch normalization
# model = Sequential([
#     Dense(64, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=l2(0.0001)),
# 
#     
#     Dense(128, activation='relu', kernel_regularizer=l2(0.0001)),
# 
#     
#     Dense(64, activation='relu', kernel_regularizer=l2(0.0001)),
#     
#     Dense(1)  # Output layer for regression
# ])
# 
# # Compile the model with Gradient Descent optimizer
# model.compile(optimizer=AdamW(learning_rate=0.001,weight_decay=0.0001),  # Set learning rate for gradient descent
#               loss='mean_squared_error', 
#               metrics=['mae'])
# 
# from keras.callbacks import EarlyStopping
# from keras.callbacks import ReduceLROnPlateau
# 
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1, callbacks=[reduce_lr,early_stopping])
# 
# # Predict and calculate the R² score
# y_pred = model.predict(X_test)
# 
# # Calculate R² score using Scikit-learn's r2_score function
# from sklearn.metrics import r2_score
# r2 = r2_score(y_test, y_pred)
# print(f"R² Score: {r2}")
# 
# 
# 
# import matplotlib.pyplot as plt
# 
# plt.plot(history.history['loss'], label='Train Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
# 
# plt.plot(history.history['mae'], label='Train MAE')
# plt.plot(history.history['val_mae'], label='Validation MAE')
# plt.xlabel('Epoch')
# plt.ylabel('MAE')
# plt.legend()
# plt.show()
# 
#     
# ##->>> achieved 0.979 on this
# 
# 
# =============================================================================



# =============================================================================
# 
# model = Sequential([
#     Dense(128, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=l2(0.0001)),
#     
#     
#     Dense(256, activation='relu', kernel_regularizer=l2(0.0001)),
#    
#     
#     Dense(128, activation='relu', kernel_regularizer=l2(0.0001)),
#     
#     
#     Dense(1)  # Output layer for regression
# ])
# 
# # Compile the model with Gradient Descent optimizer
# model.compile(optimizer=AdamW(learning_rate=0.001,weight_decay=0.0001),  # Set learning rate for gradient descent
#               loss='mean_squared_error', 
#               metrics=['mae'])
# 
#
# 
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1, callbacks=[reduce_lr,early_stopping])
# 
# # Predict and calculate the R² score
# y_pred = model.predict(X_test)
# 
# # Calculate R² score using Scikit-learn's r2_score function
# from sklearn.metrics import r2_score
# r2 = r2_score(y_test, y_pred)
# print(f"R² Score: {r2}")
# 
# 
# 
# import matplotlib.pyplot as plt
# 
# plt.plot(history.history['loss'], label='Train Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
# 
# plt.plot(history.history['mae'], label='Train MAE')
# plt.plot(history.history['val_mae'], label='Validation MAE')
# plt.xlabel('Epoch')
# plt.ylabel('MAE')
# plt.legend()
# plt.show()
# 

# =============================================================================
# ---> achieved 0.9790
# =============================================================================
# =============================================================================
    

# Now stacking xgboost and NNs



# =============================================================================

# model = Sequential([
#     Dense(128, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=l2(0.0001)),
#     
#     
#     Dense(256, activation='relu', kernel_regularizer=l2(0.0001)),
#    
#     
#     Dense(128, activation='relu', kernel_regularizer=l2(0.0001)),
#     
#     
#     Dense(1)  # Output layer for regression
# ])
# 
# # Compile the model with Gradient Descent optimizer
# model.compile(optimizer=AdamW(learning_rate=0.001,weight_decay=0.0001),  # Set learning rate for gradient descent
#               loss='mean_squared_error', 
#               metrics=['mae'])
# 
# from keras.callbacks import EarlyStopping
# from keras.callbacks import ReduceLROnPlateau
# 
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1, callbacks=[reduce_lr,early_stopping])
# 
# # Predict and calculate the R² score
# y_pred = model.predict(X_test)
# 
# # Calculate R² score using Scikit-learn's r2_score function
# from sklearn.metrics import r2_score
# r2 = r2_score(y_test, y_pred)
# print(f"R² Score: {r2}")
# 
# 
# 
# import matplotlib.pyplot as plt
# 
# plt.plot(history.history['loss'], label='Train Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
# 
# plt.plot(history.history['mae'], label='Train MAE')
# plt.plot(history.history['val_mae'], label='Validation MAE')
# plt.xlabel('Epoch')
# plt.ylabel('MAE')
# plt.legend()
# plt.show()
# 

# =============================================================================
# =============================================================================
import xgboost as xgb


nn_model = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=l2(0.0001)),
    Dense(256, activation='relu', kernel_regularizer=l2(0.0001)),
    Dense(128, activation='relu', kernel_regularizer=l2(0.0001)),
    Dense(1)  # Output layer for regression
])

nn_model.compile(optimizer=AdamW(learning_rate=0.001, weight_decay=0.0001),
                 loss='mean_squared_error',
                 metrics=['mae'])

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = nn_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1, callbacks=[reduce_lr, early_stopping])

# Predict using neural network
nn_train_preds = nn_model.predict(X_train)
nn_valid_preds = nn_model.predict(X_test)

# 2. **Train XGBoost**

xgboost_model = xgb.XGBRegressor(learning_rate=0.01, n_estimators=1000, max_depth=5, alpha=10, gamma=0.1)
xgboost_model.fit(X_train, y_train)

# Predict using XGBoost
xgboost_train_preds = xgboost_model.predict(X_train)
xgboost_valid_preds = xgboost_model.predict(X_test)

# 3. **Combine Predictions**

# Stack predictions from both models
train_meta_features = np.column_stack((xgboost_train_preds, nn_train_preds.flatten()))
valid_meta_features = np.column_stack((xgboost_valid_preds, nn_valid_preds.flatten()))

# Train meta-learner
meta_learner = Ridge()
meta_learner.fit(train_meta_features, y_train)

# Predict and evaluate
meta_predictions = meta_learner.predict(valid_meta_features)
r2 = r2_score(y_test, meta_predictions)
print(f"Stacked Model R² Score: {r2}")

# Plot training history for Neural Network
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.show()




def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

print(rmse(y_test,meta_predictions))










# =============================================================================
# =============================================================================








### Trying again by PCA features

# =============================================================================
# X_pca = data.drop('price', axis=1)
# y_pca = data['price']
# 
# from sklearn.decomposition import PCA
# features = ['carat', 'volume', 'surface_area']
# X_pca_features = X_pca[features]
# X_pca_features
# 
# 
# 
# pca = PCA(n_components=1)
# X_pca_var = pca.fit_transform(X_pca)
# 
# X_pca_df = pd.DataFrame(X_pca_var, columns=['PCA1'])
# 
# 
# X_pca_df
# 
# 
# X_reduced = X_pca.drop(['carat', 'volume', 'surface_area'], axis=1)
# 
# X_final = pd.concat([X_reduced, X_pca_df], axis=1)
# 
# X_final
# 
# 
# sns.heatmap(X_final.corr(), annot=True, cmap='magma')
# 
# 
# 
# X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_final, y_pca, test_size=0.25, random_state=42)
# 
# scaler = StandardScaler()
# 
# X_train_pca = scaler.fit_transform(X_train_pca)
# 
# X_test_pca = scaler.transform(X_test_pca)
# 
# 
# =============================================================================





# =============================================================================
# 
# 
# nn_model_2 = Sequential([
#     Dense(128, input_dim=X_train_pca.shape[1], activation='relu', kernel_regularizer=l2(0.0001)),
#     Dense(256, activation='relu', kernel_regularizer=l2(0.0001)),
#     Dense(128, activation='relu', kernel_regularizer=l2(0.0001)),
#     Dense(1)  # Output layer for regression
# ])
# 
# nn_model_2.compile(optimizer=AdamW(learning_rate=0.001, weight_decay=0.0001),
#                  loss='mean_squared_error',
#                  metrics=['mae'])
# 
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# 
# history_2 = nn_model_2.fit(X_train_pca, y_train_pca, epochs=100, batch_size=32, validation_split=0.2, verbose=1, callbacks=[reduce_lr, early_stopping])
# 
# # Predict using neural network
# nn_train_preds_2 = nn_model_2.predict(X_train_pca)
# nn_valid_preds_2 = nn_model_2.predict(X_test_pca)
# 
# # 2. **Train XGBoost**
# 
# xgboost_model_2 = xgb.XGBRegressor(learning_rate=0.01, n_estimators=1000, max_depth=5, alpha=10, gamma=0.1)
# xgboost_model_2.fit(X_train_pca, y_train_pca)
# 
# # Predict using XGBoost
# xgboost_train_preds_2 = xgboost_model_2.predict(X_train_pca)
# xgboost_valid_preds_2 = xgboost_model_2.predict(X_test_pca)
# 
# # 3. **Combine Predictions**
# 
# # Stack predictions from both models
# train_meta_features_2 = np.column_stack((xgboost_train_preds_2, nn_train_preds_2.flatten()))
# valid_meta_features_2 = np.column_stack((xgboost_valid_preds_2, nn_valid_preds_2.flatten()))
# 
# # Train meta-learner
# meta_learner_2 = Ridge()
# meta_learner_2.fit(train_meta_features_2, y_train_pca)
# 
# # Predict and evaluate
# meta_predictions_2 = meta_learner_2.predict(valid_meta_features_2)
# new_r2 = r2_score(y_test_pca, meta_predictions_2)
# print(f"Stacked Model R² Score: {new_r2}")
# 
# # Plot training history for Neural Network
# plt.plot(history_2.history['loss'], label='Train Loss')
# plt.plot(history_2.history['val_loss'], label='Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
# 
# plt.plot(history_2.history['mae'], label='Train MAE')
# plt.plot(history_2.history['val_mae'], label='Validation MAE')
# plt.xlabel('Epoch')
# plt.ylabel('MAE')
# plt.legend()
# plt.show()
# 
# 
# 
# 
# def rmse(y_true, y_pred):
#     return np.sqrt(mean_squared_error(y_true, y_pred))
# 
# print(rmse(y_test_pca,meta_predictions_2))
# 
# 
# =============================================================================


