import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model, model_selection, feature_selection

# This line takes a while to process
import feature_count_extractor as f

print('Imports complete...')

#############
# Load data #
#############

df = pd.read_csv('processed-data/df_cleaned.csv', index_col=0)

# FOR TESTING: CHOOSE ONLY THE FIRST 1000 JOBS
df = df[:1000]

# model education
ed_vocab = f.load_skills('resources/dict_education.txt')
ed_vectorizer = f.create_lemmatizing_vectorizer(ed_vocab, 2, 2, f.stopwords_list)
ed_skills = ed_vectorizer.get_feature_names()

# model experience
exp_vocab = f.load_skills('resources/dict_experience.txt')
exp_vectorizer = f.create_lemmatizing_vectorizer(exp_vocab, 3, 3, f.stopwords_list)
exp_skills = exp_vectorizer.get_feature_names()

# extract datastore skills
db_vocab = f.load_skills('resources/dict_skills_datastores.txt')
db_vectorizer = f.create_nonlemma_vectorizer(db_vocab, 1, 2, f.stopwords_list)
db_skills = db_vectorizer.get_feature_names()

# model cloud providers
cp_vocab = f.load_skills('resources/dict_skills_cloudproviders.txt')
cp_vectorizer = f.create_lemmatizing_vectorizer(cp_vocab, 1, 3, f.stopwords_list)
cp_skills = cp_vectorizer.get_feature_names()

# model data formats
df_vocab = f.load_skills('resources/dict_skills_dataformats.txt')
df_vectorizer = f.create_lemmatizing_vectorizer(df_vocab, 1, 1, f.stopwords_list)
df_skills = df_vectorizer.get_feature_names()

# model data pipelines
dp_vocab = f.load_skills('resources/dict_skills_datapipelines.txt')
dp_vectorizer = f.create_lemmatizing_vectorizer(dp_vocab, 1, 2, f.stopwords_list)
dp_skills = dp_vectorizer.get_feature_names()

# model general data analytics skills / keywords
ga_vocab = f.load_skills('resources/dict_skills_generalanalytics.txt')
ga_vectorizer = f.create_lemmatizing_vectorizer(ga_vocab, 1, 2, f.stopwords_list)
ga_skills = ga_vectorizer.get_feature_names()

# model general miscellaneous keywords
gm_vocab = f.load_skills('resources/dict_skills_generalmisc.txt')
gm_vectorizer = f.create_lemmatizing_vectorizer(gm_vocab, 1, 2, f.stopwords_list)
gm_skills = gm_vectorizer.get_feature_names()

# model programming languages
pl_vocab = f.load_skills('resources/dict_skills_programminglanguages.txt')
pl_vectorizer = f.create_lemmatizing_vectorizer(pl_vocab, 1, 1, f.stopwords_list)
pl_skills = pl_vectorizer.get_feature_names()

#############
# Compute X #
#############

desc = f.descriptions_per_type[2][:1000]
ed_count_vector = ed_vectorizer.fit_transform(desc)
exp_count_vector = exp_vectorizer.fit_transform(desc)
db_count_vector = db_vectorizer.fit_transform(desc)
cp_count_vector = cp_vectorizer.fit_transform(desc)
df_count_vector = df_vectorizer.fit_transform(desc)
dp_count_vector = dp_vectorizer.fit_transform(desc)
ga_count_vector = ga_vectorizer.fit_transform(desc)
gm_count_vector = gm_vectorizer.fit_transform(desc)
pl_count_vector = pl_vectorizer.fit_transform(desc)

ed_X = pd.DataFrame(ed_count_vector.toarray(), columns=ed_skills)
exp_X = pd.DataFrame(exp_count_vector.toarray(), columns=exp_skills)
db_X = pd.DataFrame(db_count_vector.toarray(), columns=db_skills)
cp_X = pd.DataFrame(cp_count_vector.toarray(), columns=cp_skills)
df_X = pd.DataFrame(df_count_vector.toarray(), columns=df_skills)
dp_X = pd.DataFrame(dp_count_vector.toarray(), columns=dp_skills)
ga_X = pd.DataFrame(ga_count_vector.toarray(), columns=ga_skills)
gm_X = pd.DataFrame(gm_count_vector.toarray(), columns=gm_skills)
pl_X = pd.DataFrame(pl_count_vector.toarray(), columns=pl_skills)

X = pd.concat([ed_X, exp_X, db_X, cp_X, df_X, dp_X, ga_X, gm_X, pl_X], axis=1)

# Convert the counts to binary yes/no values
X[X > 0] = 1

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

#print(y)
#print(df)
