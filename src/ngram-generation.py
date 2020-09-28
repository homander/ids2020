# script to get ngrams from Glassdoor job descriptions for skills analysis

import pandas as pd
import string
import nltk

all_columns = pd.read_csv('processed-data\df_cleaned.csv')

jds = pd.DataFrame(all_columns['Skills required'])

# rename column
jds.rename(columns={'Skills required': 'description'}, inplace=True)
jds['description'] = jds['description'].str.lower()

# remove punctuation
jds["description"] = jds['description'].str.replace(r'[^\w\s]+', ' ')


def tokenize(row):
    reviewText = row['description']
    tokenizedText = nltk.word_tokenize(reviewText)
    return tokenizedText


jds['description_tokens'] = jds.apply(tokenize, axis=1)

# remove stop words from tokens
stopwords_file = open("resources/stopwords.txt", "r")
try:
    content = stopwords_file.read()
    stopwords = content.split(",")
finally:
    stopwords_file.close()

sw_set = set(stopwords)


def process_stopwords(row):
    text = row['description_tokens']
    cleared_row = [w for w in text if not w in sw_set]
    return cleared_row


jds['description_sw'] = jds.apply(process_stopwords, axis=1)

temp = jds['description_sw'].apply(pd.Series)
word_list = temp.stack().to_list()

# produce 4-, 3-, 2- and 1-grams
four_gram = (pd.Series(nltk.ngrams(word_list, 4)).value_counts())[:2000]
four_gram_df = pd.DataFrame(four_gram)
four_gram_df.to_csv('processed-data/gd-jobdescription-four-gram-counts.csv')
four_gram_df.to_excel('processed-data/gd-jobdescription-four-gram-counts.xlsx')

three_gram = (pd.Series(nltk.ngrams(word_list, 3)).value_counts())[:2000]
three_gram_df = pd.DataFrame(three_gram)
three_gram_df.to_csv('processed-data/gd-jobdescription-three-gram-counts.csv')
three_gram_df.to_excel('processed-data/gd-jobdescription-three-gram-counts.xlsx')

two_gram = (pd.Series(nltk.ngrams(word_list, 2)).value_counts())[:2000]
two_gram_df = pd.DataFrame(two_gram)
two_gram_df.to_csv('processed-data/gd-jobdescription-two-gram-counts.csv')
two_gram_df.to_excel('processed-data/gd-jobdescription-two-gram-counts.xlsx')

one_gram = pd.Series(nltk.ngrams(word_list, 1)).value_counts()
one_gram_df = pd.DataFrame(one_gram)
one_gram_df.to_csv('processed-data/gd-jobdescription-one-gram-counts.csv')
one_gram_df.to_excel('processed-data/gd-jobdescription-one-gram-counts.xlsx')