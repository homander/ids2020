import matplotlib.pyplot as plt
import seaborn as sns
from nltk import WordNetLemmatizer, RegexpTokenizer, SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm

# Import preprocessed CSV file
df = pd.read_csv('processed-data/df_cleaned_filtered.csv', index_col=0)
df_X = pd.read_csv('processed-data/2020-X.csv', index_col=0)

# If you want to build a separate model for each job type,
# then set this to True
if False:
    mask = df['Job Type'] == 'data scientist'
    df = df[mask]
    df_X = df_X[mask]

#############################
# ADDITIONAL DATA WRANGLING #
#############################

# Industry, Sector, and Location processing: convert categorical features to integers
le = LabelEncoder()
le.fit(df['Industry'])
df['Industry_le'] = le.fit_transform(df['Industry'])
le.fit(df['Sector'])
df['Sector_le'] = le.fit_transform(df['Sector'])
le.fit(df['Location City'])
df['Location City_le'] = le.fit_transform(df['Location City'])
le.fit(df['Location State'])
df['Location State_le'] = le.fit_transform(df['Location State'])
le.fit(df['Job Type'])
df['Job Type_le'] = le.fit_transform(df['Job Type'])
le.fit(df['Company Name'])
df['Company Name_le'] = le.fit_transform(df['Company Name'])
le.fit(df['Type of ownership'])
df['Type of ownership_le'] = le.fit_transform(df['Type of ownership'])

onehotencoder = OneHotEncoder()

cols = ['Avg Salary', 'Company Name_le', 'Location City_le', 'Job Type_le', 'Industry_le', 'Location State_le',
        'Type of ownership_le', 'Rating']
hm = sns.heatmap(df[cols].corr(), cbar=True, annot=True)
plt.title('Features correlation', fontweight="bold")
plt.savefig("visualisation/features_correlation.png")

###########
# ML PART #
###########

# Check R squared correlation coefficient between job type and average salary
X = df['Job Type_le'].values.reshape(-1, 1)
X = pd.DataFrame(onehotencoder.fit_transform(X).toarray())
y = df['Avg Salary'].values.reshape(-1, 1)
y_model = LinearRegression().fit(X, y)
print('Correlations')
print(f"R squared correlation coefficient between job type and salary: {y_model.score(X, y)}")
# print(sm.OLS(y, sm.add_constant(X)).fit().summary())

# Check R squared correlation coefficient between city and average salary
X = df['Location City_le'].values.reshape(-1, 1)
X = pd.DataFrame(onehotencoder.fit_transform(X).toarray())
y = df['Avg Salary'].values.reshape(-1, 1)
y_model = LinearRegression().fit(X, y)
print(f"R squared correlation coefficient between city and salary: {y_model.score(X, y)}")
# print(sm.OLS(y, sm.add_constant(X)).fit().summary())

# Check R squared correlation coefficient between Company's rating and average salary
X = df['Rating'].values.reshape(-1, 1)
X = pd.DataFrame(onehotencoder.fit_transform(X).toarray())
y = df['Avg Salary'].values.reshape(-1, 1)
y_model = LinearRegression().fit(X, y)
print(f"R squared correlation coefficient between Company's rating and salary: {y_model.score(X, y)}")
# print(sm.OLS(y, sm.add_constant(X)).fit().summary())

# Check R squared correlation coefficient between Type of ownership and average salary
X = df['Type of ownership_le'].values.reshape(-1, 1)
X = pd.DataFrame(onehotencoder.fit_transform(X).toarray())
y = df['Avg Salary'].values.reshape(-1, 1)
y_model = LinearRegression().fit(X, y)
print(f"R squared correlation coefficient between type of ownership and salary: {y_model.score(X, y)}")
# print(sm.OLS(y, sm.add_constant(X)).fit().summary())

# Check R squared correlation coefficient between industry and average salary
X = df['Industry_le'].values.reshape(-1, 1)
X = pd.DataFrame(onehotencoder.fit_transform(X).toarray())
y = df['Avg Salary'].values.reshape(-1, 1)
y_model = LinearRegression().fit(X, y)
print(f"R squared correlation coefficient between industry and salary: {y_model.score(X, y)}")
# print(sm.OLS(y, sm.add_constant(X)).fit().summary())

# Check R squared correlation coefficient between sector and average salary
X = df['Sector_le'].values.reshape(-1, 1)
X = pd.DataFrame(onehotencoder.fit_transform(X).toarray())
y = df['Avg Salary'].values.reshape(-1, 1)
y_model = LinearRegression().fit(X, y)
print(f"R squared correlation coefficient between sector and salary: {y_model.score(X, y)}")
# print(sm.OLS(y, sm.add_constant(X)).fit().summary())

# Check R squared correlation coefficient between company name and average salary
X = df['Company Name_le'].values.reshape(-1, 1)
X = pd.DataFrame(onehotencoder.fit_transform(X).toarray())
y = df['Avg Salary'].values.reshape(-1, 1)
y_model = LinearRegression().fit(X, y)
print(f"R squared correlation coefficient between company name and salary: {y_model.score(X, y)}")
# print(sm.OLS(y, sm.add_constant(X)).fit().summary())

# Check R squared correlation coefficient between state and average salary
X = df['Location State_le'].values.reshape(-1, 1)
X = pd.DataFrame(onehotencoder.fit_transform(X).toarray())
y = df['Avg Salary'].values.reshape(-1, 1)
y_model = LinearRegression().fit(X, y)
print(f"R squared correlation coefficient between state and salary: {y_model.score(X, y)}")
# print(sm.OLS(y, sm.add_constant(X)).fit().summary())

# Check R squared correlation coefficient between city+job+company and average salary
X = df[['Job Type_le', 'Location City_le', 'Company Name_le']]
X = pd.DataFrame(onehotencoder.fit_transform(X).toarray())
y = df['Avg Salary'].values.reshape(-1, 1)
y_model = LinearRegression().fit(X, y)
print(f"R squared correlation coefficient between job+city+company and salary: {y_model.score(X, y)}")
# print(sm.OLS(y, sm.add_constant(X)).fit().summary())


# Check R squared correlation coefficient between state+city+job+industry+company and average salary
X = df[['Job Type_le', 'Location State_le', 'Location City_le', 'Company Name_le']]
X = pd.DataFrame(onehotencoder.fit_transform(X).toarray())
y = df['Avg Salary'].values.reshape(-1, 1)
y_model = LinearRegression().fit(X, y)
print(f"R squared correlation coefficient between job+state+city+company and salary: {y_model.score(X, y)}")
# print(sm.OLS(y, sm.add_constant(X)).fit().summary())

###########################
# ADD SKILLS TO THE MODEL #
###########################

# Check R squared correlation coefficient between skills and average salary
X = df_X
y = df['Avg Salary'].values.reshape(-1, 1)
y_model = LinearRegression().fit(X, y)
print(f"R squared correlation coefficient between skills and salary: {y_model.score(X, y)}")
# print(sm.OLS(y, sm.add_constant(X)).fit().summary())

# Check R squared correlation coefficient between city+job+company+skills and average salary
X = df[['Job Type_le', 'Location City_le', 'Company Name_le']]
X = pd.DataFrame(onehotencoder.fit_transform(X).toarray())
X_concat = pd.concat([X, df_X], axis=1, sort=False)
y = df['Avg Salary'].values.reshape(-1, 1)
y_model = LinearRegression().fit(X_concat, y)
print(f"R squared correlation coefficient between job+city+company+skills and salary: {y_model.score(X_concat, y)}")
# print(sm.OLS(y, sm.add_constant(X_concat)).fit().summary())

##########################################
# ADDITIONAL WRANGLING (JOB DESCRIPTION) #
##########################################

# In this section, we double-check our results by trying an alternative approach
# so that instead of predefined skill vocabularies we use a large set of n-grams

# Tokenizing words
# print('Tokenizing...', end=' ')
tokenizer = RegexpTokenizer(r'\S+')
df['Skills'] = df['Job Description'].apply(lambda x: tokenizer.tokenize(x.lower()))
# print('Tokenizing OK')

# Words Lemmatization
# print('Lemmatization...', end=' ')
lemmatizer = WordNetLemmatizer()


def word_lemmatizer(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text])


df['Skills_lem'] = df['Skills'].apply(lambda x: word_lemmatizer(x))
# print('Lemmatization OK')

# Stemming the words
# print('Stemming...', end=' ')
stemmer = SnowballStemmer('english')


def word_stemmer(text):
    return " ".join([stemmer.stem(word) for word in text])


df['Skills_stem'] = df['Skills'].apply(lambda x: word_stemmer(x))
# print('Steming OK')

# Vectorizing skills descriptions with ngrams 2 and max_features 3000
# print('Vectorizing...', end=' ')
vector = TfidfVectorizer(ngram_range=(2, 2), analyzer='word', max_features=3000)
X_lem = pd.DataFrame(vector.fit_transform(df["Skills_lem"]).toarray())
# X_stem = pd.DataFrame(vector.fit_transform(df["Skills_stem"]).toarray())
# print('Vectorizing OK')

# Check R squared correlation coefficient between job description and average salary
X = X_lem
y = df['Avg Salary'].values.reshape(-1, 1)
y_model = LinearRegression().fit(X, y)
print(f"R squared correlation coefficient between job description and salary: {y_model.score(X, y)}")
# print(sm.OLS(y, sm.add_constant(X)).fit().summary())

# Check R squared correlation coefficient between city+job+company+job_description and average salary
X = df[['Job Type_le', 'Location City_le', 'Company Name_le']]
X = pd.DataFrame(onehotencoder.fit_transform(X).toarray())
X_concat = pd.concat([X, X_lem], axis=1, sort=False)
y = df['Avg Salary'].values.reshape(-1, 1)
y_model = LinearRegression().fit(X_concat, y)
print(f"R squared correlation coefficient between job+city+company+job_description and salary (ngram=2, feat=3000): {y_model.score(X_concat, y)}")
# print(sm.OLS(y, sm.add_constant(X_concat)).fit().summary())

plt.show()
