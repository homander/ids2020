from nltk import WordNetLemmatizer, RegexpTokenizer, SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Import preprocessed CSV file
df = pd.read_csv('processed-data/df_cleaned_for_visualisation.csv', index_col=0)

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
df['Location_City_le'] = le.fit_transform(df['Location City'])
le.fit(df['Location State'])
df['Location_State_le'] = le.fit_transform(df['Location State'])
le.fit(df['Job Type'])
df['Job Type_le'] = le.fit_transform(df['Job Type'])
le.fit(df['Company Name'])
df['Company Name_le'] = le.fit_transform(df['Company Name'])
le.fit(df['Type of ownership'])
df['Type of ownership_le'] = le.fit_transform(df['Type of ownership'])

onehotencoder = OneHotEncoder()


###########
# ML PART #
###########

# Check R squared correlation coefficient between Company's rating and average salary
X = df['Rating'].values.reshape(-1, 1)
X = pd.DataFrame(onehotencoder.fit_transform(X).toarray())
y = df['Avg Salary'].values.reshape(-1, 1)
y_model = LinearRegression().fit(X, y)
print('Correlations')
print(f"The R squared correlation coefficient between Company's rating and salary is: {y_model.score(X, y)}")

# Check R squared correlation coefficient between Company's rating and average salary
X = df['Type of ownership_le'].values.reshape(-1, 1)
X = pd.DataFrame(onehotencoder.fit_transform(X).toarray())
y = df['Avg Salary'].values.reshape(-1, 1)
y_model = LinearRegression().fit(X, y)
print(f"The R squared correlation coefficient between type of ownership and salary is: {y_model.score(X, y)}")

# Check R squared correlation coefficient between industry and average salary
X = df['Industry_le'].values.reshape(-1, 1)
X = pd.DataFrame(onehotencoder.fit_transform(X).toarray())
y = df['Avg Salary'].values.reshape(-1, 1)
y_model = LinearRegression().fit(X, y)
print(f"The R squared correlation coefficient between industry and salary is: {y_model.score(X, y)}")

# Check R squared correlation coefficient between sector and average salary
X = df['Sector_le'].values.reshape(-1, 1)
X = pd.DataFrame(onehotencoder.fit_transform(X).toarray())
y = df['Avg Salary'].values.reshape(-1, 1)
y_model = LinearRegression().fit(X, y)
print(f"The R squared correlation coefficient between sector and salary is: {y_model.score(X, y)}")

# Check R squared correlation coefficient between company name and average salary
X = df['Company Name_le'].values.reshape(-1, 1)
X = pd.DataFrame(onehotencoder.fit_transform(X).toarray())
y = df['Avg Salary'].values.reshape(-1, 1)
y_model = LinearRegression().fit(X, y)
print(f"The R squared correlation coefficient between company name and salary is: {y_model.score(X, y)}")

# Check R squared correlation coefficient between city and average salary
X = df['Location_City_le'].values.reshape(-1, 1)
X = pd.DataFrame(onehotencoder.fit_transform(X).toarray())
y = df['Avg Salary'].values.reshape(-1, 1)
y_model = LinearRegression().fit(X, y)
print(f"The R squared correlation coefficient between city and salary is: {y_model.score(X, y)}")

# Check R squared correlation coefficient between state and average salary
X = df['Location_State_le'].values.reshape(-1, 1)
X = pd.DataFrame(onehotencoder.fit_transform(X).toarray())
y = df['Avg Salary'].values.reshape(-1, 1)
y_model = LinearRegression().fit(X, y)
print(f"The R squared correlation coefficient between state and salary is: {y_model.score(X, y)}")

# Check R squared correlation coefficient between job type and average salary
X = df['Job Type_le'].values.reshape(-1, 1)
X = pd.DataFrame(onehotencoder.fit_transform(X).toarray())
y = df['Avg Salary'].values.reshape(-1, 1)
y_model = LinearRegression().fit(X, y)
print(f"The R squared correlation coefficient between job type and salary is: {y_model.score(X, y)}")

# Check R squared correlation coefficient between city+job+company and average salary
X = df[['Job Type_le', 'Location_City_le', 'Company Name_le']]
X = pd.DataFrame(onehotencoder.fit_transform(X).toarray())
y = df['Avg Salary'].values.reshape(-1, 1)
y_model = LinearRegression().fit(X, y)
print(f"The R squared correlation coefficient between job+city+company and salary is: {y_model.score(X, y)}")

# Check R squared correlation coefficient between state+city+job+industry+company and average salary
X = df[['Job Type_le', 'Location_State_le', 'Location_City_le', 'Company Name_le']]
X = pd.DataFrame(onehotencoder.fit_transform(X).toarray())
y = df['Avg Salary'].values.reshape(-1, 1)
y_model = LinearRegression().fit(X, y)
print(f"The R squared correlation coefficient between job+state+city+company and salary is: {y_model.score(X, y)}")

########################################
# SKILLS REQUIRED ADDITIONAL WRANGLING #
########################################

# Tokenizing words
print('Tokenizing...', end=' ')
tokenizer = RegexpTokenizer(r'\S+')
df['Skills'] = df['Skills required'].apply(lambda x: tokenizer.tokenize(x.lower()))
print('Tokenizing OK')

'''
# Removing stopwords
print('Removing stopwords - takes around 20min...', end=' ')
def remove_stopwords(text):
    return [word for word in text if word not in stopwords.words('english')]
df['Skills'] = df['Skills'].apply(lambda x: remove_stopwords(x))
print('Removing stopwords OK')
'''

# Words Lemmatization
print('Lemmatizing...', end=' ')
lemmatizer = WordNetLemmatizer()
def word_lemmatizer(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text])
df['Skills_lem'] = df['Skills'].apply(lambda x: word_lemmatizer(x))
print('Lemmatizing OK')

# Stemming the words
print('Stemming...', end=' ')
stemmer = SnowballStemmer('english')
def word_stemmer(text):
    return " ".join([stemmer.stem(word) for word in text])
df['Skills_stem'] = df['Skills'].apply(lambda x: word_stemmer(x))
print('Steming OK')

# Vectorizing skills descriptions with ngrams 3 and max_features 1000
print('Vectorizing...', end=' ')
vector = TfidfVectorizer(ngram_range=(2, 2), analyzer='word', max_features=2500)
X_lem = pd.DataFrame(vector.fit_transform(df["Skills_lem"]).toarray())
X_stem = pd.DataFrame(vector.fit_transform(df["Skills_stem"]).toarray())
print('Vectorizing OK')

# Check R squared correlation coefficient between city+job+company+description and average salary
X = df[['Job Type_le', 'Location_City_le', 'Company Name_le']]
X = pd.DataFrame(onehotencoder.fit_transform(X).toarray())
X_concat = pd.concat([X, X_lem], axis=1, sort=False)
y = df['Avg Salary'].values.reshape(-1, 1)
y_model = LinearRegression().fit(X_concat, y)
print(f"The R squared corr. job+city+company+description(lem) and salary is (ngram=2, feat=2500): {y_model.score(X_concat, y)}")

# Check R squared correlation coefficient between city+job+company+description and average salary
X = df[['Job Type_le', 'Location_City_le', 'Company Name_le']]
X = pd.DataFrame(onehotencoder.fit_transform(X).toarray())
X_concat = pd.concat([X, X_stem], axis=1, sort=False)
y = df['Avg Salary'].values.reshape(-1, 1)
y_model = LinearRegression().fit(X_concat, y)
print(f"The R squared corr. job+city+company+description(stem) and salary is (ngram=2, feat=2500): {y_model.score(X_concat, y)}")

# Vectorizing skills descriptions with ngrams 3 and max_features 1000
print('Vectorizing...', end=' ')
vector = TfidfVectorizer(ngram_range=(2, 2), analyzer='word', max_features=3000)
X_lem = pd.DataFrame(vector.fit_transform(df["Skills_lem"]).toarray())
X_stem = pd.DataFrame(vector.fit_transform(df["Skills_stem"]).toarray())
print('Vectorizing OK')

# Check R squared correlation coefficient between city+job+company+description and average salary
X = df[['Job Type_le', 'Location_City_le', 'Company Name_le']]
X = pd.DataFrame(onehotencoder.fit_transform(X).toarray())
X_concat = pd.concat([X, X_lem], axis=1, sort=False)
y = df['Avg Salary'].values.reshape(-1, 1)
y_model = LinearRegression().fit(X_concat, y)
print(f"The R squared corr. job+city+company+description(lem) and salary is (ngram=2, feat=3000): {y_model.score(X_concat, y)}")

# Check R squared correlation coefficient between city+job+company+description and average salary
X = df[['Job Type_le', 'Location_City_le', 'Company Name_le']]
X = pd.DataFrame(onehotencoder.fit_transform(X).toarray())
X_concat = pd.concat([X, X_stem], axis=1, sort=False)
y = df['Avg Salary'].values.reshape(-1, 1)
y_model = LinearRegression().fit(X_concat, y)
print(f"The R squared corr. job+city+company+description(stem) and salary is (ngram=2, feat=3000): {y_model.score(X_concat, y)}")

# Vectorizing skills descriptions with ngrams 3 and max_features 1000
print('Vectorizing...', end=' ')
vector = TfidfVectorizer(ngram_range=(3, 3), analyzer='word', max_features=3000)
X_lem = pd.DataFrame(vector.fit_transform(df["Skills_lem"]).toarray())
X_stem = pd.DataFrame(vector.fit_transform(df["Skills_stem"]).toarray())
print('Vectorizing OK')

# Check R squared correlation coefficient between city+job+company+description and average salary
X = df[['Job Type_le', 'Location_City_le', 'Company Name_le']]
X = pd.DataFrame(onehotencoder.fit_transform(X).toarray())
X_concat = pd.concat([X, X_lem], axis=1, sort=False)
y = df['Avg Salary'].values.reshape(-1, 1)
y_model = LinearRegression().fit(X_concat, y)
print(f"The R squared corr. job+city+company+description(lem) and salary is (ngram=3, feat=3000): {y_model.score(X_concat, y)}")

# Check R squared correlation coefficient between city+job+company+description and average salary
X = df[['Job Type_le', 'Location_City_le', 'Company Name_le']]
X = pd.DataFrame(onehotencoder.fit_transform(X).toarray())
X_concat = pd.concat([X, X_stem], axis=1, sort=False)
y = df['Avg Salary'].values.reshape(-1, 1)
y_model = LinearRegression().fit(X_concat, y)
print(f"The R squared corr. job+city+company+description(stem) and salary is (ngram=3, feat=3000): {y_model.score(X_concat, y)}")
