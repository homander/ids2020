import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model, model_selection
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Fit the model with/without splitting into separate training and testing sets
# Based on this comparison there seems to be a risk of overfitting

#############
# Load data #
#############

df = pd.read_csv('processed-data/df_cleaned_filtered.csv', index_col=0)
X_skills = pd.read_csv('processed-data/2020-X.csv', index_col=0)

# Look at a subset of the data: pick only data scientist jobs
# (because the Theil-Sen estimator seems really slow with larger datasets)
# (and can't use the regular OLS estimator which is fast but tends to break down)
mask = df['Job Type'] == 'data scientist'
df = df[mask]
X_skills = X_skills[mask]

print('df.shape:', df.shape)
print('X_skills.shape:', X_skills.shape)

###############################
# Encode categorical features #
###############################

df_categ = df[['Job Type', 'Location City', 'Company Name']]
enc = OneHotEncoder(drop='first').fit(df_categ)
X_categ = pd.DataFrame(enc.transform(df_categ).toarray(), index=df_categ.index)

##############################################
# Modeling: salary ~ skills + other features #
##############################################

y = df['Avg Salary']
X = pd.concat([X_categ, X_skills], axis=1)

plt.figure(figsize=(10, 3.5))

# Without train/test split
reg = linear_model.LinearRegression().fit(X, y)
r2 = reg.score(X, y)
print('When no splitting')
print('  Estimator: OLS')
print('  R^2:', r2)
print('  Intercept:', reg.intercept_)

plt.subplot(1, 2, 1)
plt.scatter(reg.predict(X), y)
plt.title(f'No train/test split (OLS, R2={r2:0.2f})')
plt.xlabel('Predicted salary')
plt.ylabel('Actual salary')
plt.tight_layout()

# With train/test split
# (in this case, we use Theil-Sen estimator because OLS seems to produce huge outliers)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y)
reg = linear_model.TheilSenRegressor().fit(X_train, y_train)
r2 = reg.score(X_test, y_test)
print('When split')
print('  Estimator: Theil-Sen')
print('  R^2:', r2)
print('  Intercept:', reg.intercept_)

plt.subplot(1, 2, 2)
plt.scatter(reg.predict(X_test), y_test)
plt.title(f'With train/test split (Theil-Sen, R2={r2:0.2f})')
plt.xlabel('Predicted salary')
plt.ylabel('Actual salary')
plt.tight_layout()

plt.show()
