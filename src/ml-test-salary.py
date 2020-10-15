import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model, model_selection, feature_selection

#############
# Load data #
#############

df = pd.read_csv('processed-data/df_cleaned_filtered.csv', index_col=0)
X = pd.read_csv('processed-data/2020-X.csv', index_col=0)

print('df.shape:', df.shape)
print('X.shape:', X.shape)

# FOR TESTING: CHOOSE ONLY THE FIRST 1000 JOBS
df = df[:1000]
X = X[:1000]

print(X)

#############################
# Modeling: salary ~ skills #
#############################

y = df['Avg Salary']
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y)

#reg = linear_model.LinearRegression().fit(X_train, y_train)
#reg = linear_model.RANSACRegressor().fit(X_train, y_train)
reg = linear_model.TheilSenRegressor().fit(X_train, y_train)
print('R^2:', reg.score(X_test, y_test))
print('Intercept:', reg.intercept_)
print('Coefficients:', reg.coef_, sep='\n')
#print(reg.estimator_.coef_)

p_values = pd.DataFrame()
p_values['Skill'] = X.columns
p_values['p'] = feature_selection.f_regression(X_test, y_test)[1]
p_values['F'] = feature_selection.f_regression(X_test, y_test)[0]
p_values['coef'] = reg.coef_

p_values.to_csv('processed-data/p-values.csv')
p_values.sort_values(by='p', inplace=True)
p_values.to_csv('processed-data/p-values-sorted.csv')
print('p-values:', p_values, sep='\n')

plt.scatter(reg.predict(X_test), y_test)
plt.show()
