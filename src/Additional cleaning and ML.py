import pandas as pd
import nltk
from nltk import SnowballStemmer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Import preprocessed CSV file
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('processed-data/df_cleaned.csv', index_col=0)

############################
# ADDITIONAL DATA CLEANING #
############################

# remove rating from Company name
df['Company Name'] = df["Company Name"].str.partition("\n")

# Separate City and State location
df['Location City'] = df['Location'].str.split(",", expand=True)[0]
df['Location State'] = df['Location'].str.split(",", expand=True)[1]
df['Headquarters City'] = df['Headquarters'].str.split(",", expand=True)[0]
df['Headquarters State'] = df['Headquarters'].str.split(",", expand=True)[1]

# Industry, Sector, and Location processing: convert categorical features to integers
le = LabelEncoder()
le.fit(df['Industry'])
df['Industry_le'] = le.transform(df['Industry'])
le.fit(df['Sector'])
df['Sector_le'] = le.transform(df['Sector'])
le.fit(df['Location City'])
df['Location_City_le'] = le.transform(df['Location City'])



#########################################
# VISUALIZATION PART (will be extended) #
#########################################

# Explore Company's rating and average salary correlation
plt.scatter(df['Rating'], df['Avg Salary'])
plt.xlabel("Avg Salary")
plt.ylabel("Company's Rating")

# Show salary distribution
f, axes = plt.subplots(1, 2, figsize=(8, 4))
sns.distplot(df['Min Salary'], color='r', ax=axes[0])
sns.distplot(df['Max Salary'], ax=axes[1])

# Most popular job titles
plt.subplots(figsize=(10,10))
sns.barplot(x=df['Job Title'].value_counts()[0:20].index, y=df['Job Title'].value_counts()[0:20])
plt.xlabel('Job Title',fontsize=10)
plt.ylabel('Job Count',fontsize=10)
plt.xticks(rotation=20)
plt.yticks(fontsize=10)
plt.title('Top 20 Job Title Counts')

plt.show()

################
# CORRELATIONS #
################

# Check R squared correlation coefficient between Company's rating and average salary
X = df['Rating'].values.reshape(-1, 1)
y = df['Avg Salary'].values.reshape(-1, 1)
y_model = LinearRegression().fit(X, y)
print('Correlations')
print(f"The R squared correlation coefficient between Company's rating and salary is: {y_model.score(X, y)}")

# Check R squared correlation coefficient between industry and average salary
X = df['Industry_le'].values.reshape(-1, 1)
y = df['Max Salary'].values.reshape(-1, 1)
y_model = LinearRegression().fit(X, y)
print(f"The R squared correlation coefficient between industry and salary is: {y_model.score(X, y)}")

# Check R squared correlation coefficient between sector and average salary
X = df['Sector_le'].values.reshape(-1, 1)
y = df['Max Salary'].values.reshape(-1, 1)
y_model = LinearRegression().fit(X, y)
print(f"The R squared correlation coefficient between sector and salary is: {y_model.score(X, y)}")
print()


#######################################################
# MODELS BASED ON OTHER THAN JOB DESCRIPTION FEATURES #
#######################################################

X = df[['Industry_le', 'Sector_le', 'Location_City_le', 'Rating']]
y = df['Avg Salary']
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=0.8)

# GaussianNB
model = GaussianNB()
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)
print('Models implementing:')
print('GaussianNB (industry, sector, location, rating VS. avg salary):', accuracy_score(ytest, y_model))

# RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)
print('RandomForestClassifier (industry, sector, location, rating VS. avg salary, n_estimators=100):', accuracy_score(ytest, y_model))

# MultinomialNB
model = MultinomialNB()
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)
print('MultinomialNB (industry, sector, location, rating VS. avg salary):', accuracy_score(ytest, y_model))

# SVC
model = SVC(kernel='rbf')
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)
print('SVC (industry, sector, location, rating VS. avg salary, rbf):', accuracy_score(ytest, y_model))
print()

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

# Lemmatizing the words
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

# Vectorizing skills descriptions
print('Vectorizing...', end=' ')
vector = TfidfVectorizer(ngram_range=(3, 3), analyzer='word', max_features=1000)
X_lem = vector.fit_transform(df["Skills_lem"]).toarray()
X_stem = vector.fit_transform(df["Skills_stem"]).toarray()
y = df['Avg Salary']
print('Vectorizing OK')

# Spliting to train and test sets
print('Spliting to train and test sets...', end=' ')
X_lem_train, X_lem_test, ytrain, ytest = train_test_split(X_lem, y, train_size=0.8)
X_stem_train, X_stem_test = train_test_split(X_stem, train_size=0.8)
print('Spliting OK')
print()

print('Models for text features implementing (5 models for lemmatized and stemmed words separately), takes appx.10min...')
model = GaussianNB()
model.fit(X_lem_train, ytrain)
y_model = model.predict(X_lem_test)
print('GaussianNB(lem):', accuracy_score(ytest, y_model))
model.fit(X_stem_train, ytrain)
y_model = model.predict(X_stem_test)
print('GaussianNB(stem):', accuracy_score(ytest, y_model))

model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(X_lem_train, ytrain)
y_model = model.predict(X_lem_test)
print('RandomForestClassifier (lem, n_estimators=100):', accuracy_score(ytest, y_model))
model.fit(X_stem_train, ytrain)
y_model = model.predict(X_stem_test)
print('RandomForestClassifier (stem, n_estimators=100):', accuracy_score(ytest, y_model))

model = DecisionTreeClassifier()
model.fit(X_lem_train, ytrain)
y_model = model.predict(X_lem_test)
print('DecisionTreeClassifier(lem):', accuracy_score(ytest, y_model))
model.fit(X_stem_train, ytrain)
y_model = model.predict(X_stem_test)
print('DecisionTreeClassifier(stem):', accuracy_score(ytest, y_model))

model = MultinomialNB()
model.fit(X_lem_train, ytrain)
y_model = model.predict(X_lem_test)
print('MultinomialNB(lem):', accuracy_score(ytest, y_model))
model.fit(X_stem_train, ytrain)
y_model = model.predict(X_stem_test)
print('MultinomialNB(stem):', accuracy_score(ytest, y_model))

model = SVC(kernel='rbf', C=1E10)
model.fit(X_lem_train, ytrain)
y_model = model.predict(X_lem_test)
print('SVC (lem, rbf):', accuracy_score(ytest, y_model))
print()

# Drop processed columns
df.drop('Location', axis=1, inplace=True)
df.drop('Headquarters', axis=1, inplace=True)

print('Now saving modified dataset to files')
df.to_csv('processed-data/df_prepared_for_further_ML.csv')
df.to_excel('processed-data/df_prepared_for_further_ML.xlsx')
