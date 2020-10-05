import pandas as pd
import re
import nltk
from nltk import SnowballStemmer
# nltk.download('stopwords')
# nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder

# Import preprocessed CSV file
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

# Drop processed columns and rows with NAN
df = df.dropna(axis=0, how='any').reset_index(drop=True)
df.drop('Location', axis=1, inplace=True)
df.drop('Headquarters', axis=1, inplace=True)

# Clean Job Title
df['Job Title'] = df['Job Title'].apply(lambda x: re.sub(r'(?i)(sr\.)|sr|lead', 'Senior', x))
df['Job Title'] = df['Job Title'].apply(lambda x: re.sub(r'(?i)(jr\.)|jr', 'Junior', x))

# Industry, Sector, City, Job Type, and Ownership processing: convert categorical features to integers
le = LabelEncoder()
le.fit(df['Industry'])
df['Industry_le'] = le.transform(df['Industry'])
le.fit(df['Sector'])
df['Sector_le'] = le.transform(df['Sector'])
le.fit(df['Location City'])
df['Location_City_le'] = le.transform(df['Location City'])
le.fit(df['Type of ownership'])
df['Type of ownership_le'] = le.transform(df['Type of ownership'])
le.fit(df['Company Name'])
df['Company Name_le'] = le.transform(df['Company Name'])
le.fit(df['Job Type'])
df['Job Type_le'] = le.transform(df['Job Type'])
le.fit(df['Location State'])
df['Location State_le'] = le.transform(df['Location State'])

############################################
# ADDITIONAL WRANGLING OF SKILLS REQUIRED  #
############################################

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

'''
# Vectorizing skills descriptions
print('Vectorizing...', end=' ')
vector = TfidfVectorizer(ngram_range=(3, 3), analyzer='word', max_features=1000)
X_lem = vector.fit_transform(df["Skills_lem"]).toarray()
X_stem = vector.fit_transform(df["Skills_stem"]).toarray()
y = df['Avg Salary']
print('Vectorizing OK')
'''

# Save to files for ML
print('Now saving modified dataset to files')
df.to_csv('processed-data/df_prepared_for_visualisation.csv')
df.to_excel('processed-data/df_prepared_for_visualisation.xlsx')
