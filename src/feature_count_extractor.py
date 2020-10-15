import sys
import nltk
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer

nltk.download('wordnet')
wordnet_lemmatizer = WordNetLemmatizer()


# a function to lemmatize words, e.g. 'masters degree' --> 'master degree'
def lemmatizing_preprocessor(text):
    text = text.lower()
    text = text.replace('\'', '')  # e.g. master's --> masters, for lemmatizing
    punctuations = "?:!.,;"
    sentence_words = nltk.word_tokenize(text)
    for word in sentence_words:
        if word in punctuations:
            sentence_words.remove(word)

    text = ' '.join([wordnet_lemmatizer.lemmatize(word) for word in sentence_words])
    return text


def load_skills(path):
    file = open(path, "r")
    try:
        contents = file.read()
        skills_dict = ast.literal_eval(contents)
    finally:
        file.close()

    return skills_dict


def load_stopwords_list():
    stopwords_file = open("resources/stopwords.txt", "r")
    try:
        content = stopwords_file.read()
        stopwords = content.split(",")
    finally:
        stopwords_file.close()

    return stopwords


# create a non-lemmatizing count vectorizer with a given vocabulary and n-gram range, uses default 'word' analyzer
def create_nonlemma_vectorizer(vocab, g_min, g_max, stopwords_list):
    vectorizer = CountVectorizer(ngram_range=(g_min, g_max), vocabulary=vocab, analyzer='word',
                                 stop_words=stopwords_list)
    return vectorizer


# create a lemmatizing count vectorizer with a given vocabulary and n-gram range
# usese tokenizer that keeps numbers as well
def create_lemmatizing_vectorizer(vocab, g_min, g_max, stopwords_list):
    lemma_vectorizer = CountVectorizer(ngram_range=(g_min, g_max), vocabulary=vocab,
                                       tokenizer=TreebankWordTokenizer().tokenize, stop_words=stopwords_list)
    return lemma_vectorizer


# count skills with given vectorizer and data (list of documents)
# produces a pandas dataframe with counts
def count_skills(vectorizer, doc_list):
    count_vector = vectorizer.fit_transform(doc_list)
    counts = count_vector.getnnz(axis=0)
    feature_names = vectorizer.get_feature_names()
    results = pd.DataFrame({'Skill': feature_names, 'Count': counts})
    results['Frequency'] = results['Count'] / len(doc_list)

    return results


# count skills with given vectorizer and data (list of documents, one for each job type)
# produces a pandas dataframe with counts
def summarize_counts(vectorizer, doc_lists):
    counts = []
    for doc_list in doc_lists:
        count_vector = vectorizer.fit_transform(doc_list)
        counts.append(count_vector.getnnz(axis=0))

    feature_names = vectorizer.get_feature_names()
    results = pd.DataFrame(
        {'Skill': feature_names, 'Analyst Count': counts[0], 'Engineer Count': counts[1], 'Scientist Count': counts[2]})
    results['Analyst Frequency'] = round((results['Analyst Count'] / len(doc_lists[0])) * 100, 2)
    results['Engineer Frequency'] = round((results['Engineer Count'] / len(doc_lists[1])) * 100, 2)
    results['Scientist Frequency'] = round((results['Scientist Count'] / len(doc_lists[2])) * 100, 2)

    return results


if len(sys.argv) != 3:
    print('Wrong number of arguments.')
    exit()

file_name = sys.argv[1]
year = sys.argv[2]

# load data
jobs = pd.read_csv('processed-data/' + file_name)

# lemmatize job descriptions to get base form for some counts
jobs['description_lemmatized'] = jobs['Job Description'].apply(lemmatizing_preprocessor)

analyst_listings = jobs[jobs['Job Type'] == 'data analyst']
engineer_listings = jobs[jobs['Job Type'] == 'data engineer']
scientist_listings = jobs[jobs['Job Type'] == 'data scientist']

descriptions_per_type = []
descriptions_per_type.append(analyst_listings['Job Description'].to_list())
descriptions_per_type.append(engineer_listings['Job Description'].to_list())
descriptions_per_type.append(scientist_listings['Job Description'].to_list())

lemmatized_descriptions_per_type = []
lemmatized_descriptions_per_type.append(analyst_listings['description_lemmatized'].to_list())
lemmatized_descriptions_per_type.append(engineer_listings['description_lemmatized'].to_list())
lemmatized_descriptions_per_type.append(scientist_listings['description_lemmatized'].to_list())

stopwords_list = load_stopwords_list()

# extract education
ed_vocab = load_skills('resources/dict_education.txt')
ed_vectorizer = create_lemmatizing_vectorizer(ed_vocab, 2, 2, stopwords_list)
ed_results = summarize_counts(ed_vectorizer, lemmatized_descriptions_per_type)
ed_results.to_csv('processed-data/' + year + '-counts-education.csv')

# extract experience
exp_vocab = load_skills('resources/dict_experience.txt')
exp_vectorizer = create_lemmatizing_vectorizer(exp_vocab, 3, 3, stopwords_list)
exp_results = summarize_counts(exp_vectorizer, lemmatized_descriptions_per_type)
exp_results.to_csv('processed-data/' + year + '-counts-experience.csv')

# extract datastore skills
db_vocab = load_skills('resources/dict_skills_datastores.txt')
db_vectorizer = create_nonlemma_vectorizer(db_vocab, 1, 2, stopwords_list)
db_results = summarize_counts(db_vectorizer, descriptions_per_type)
db_results.to_csv('processed-data/' + year + '-counts-skill-datastores.csv')

# extract cloud providers
cp_vocab = load_skills('resources/dict_skills_cloudproviders.txt')
cp_vectorizer = create_lemmatizing_vectorizer(cp_vocab, 1, 3, stopwords_list)
cp_results = summarize_counts(cp_vectorizer, descriptions_per_type)
cp_results.to_csv('processed-data/' + year + '-counts-skill-cloudproviders.csv')

# extract data formats
df_vocab = load_skills('resources/dict_skills_dataformats.txt')
df_vectorizer = create_lemmatizing_vectorizer(df_vocab, 1, 1, stopwords_list)
df_results = summarize_counts(df_vectorizer, descriptions_per_type)
df_results.to_csv('processed-data/' + year + '-counts-skill-dataformats.csv')

# extract data pipelines
dp_vocab = load_skills('resources/dict_skills_datapipelines.txt')
dp_vectorizer = create_lemmatizing_vectorizer(dp_vocab, 1, 2, stopwords_list)
dp_results = summarize_counts(dp_vectorizer, descriptions_per_type)
dp_results.to_csv('processed-data/' + year + '-counts-skill-datapipelines.csv')

# extract general data analytics skills / keywords
ga_vocab = load_skills('resources/dict_skills_generalanalytics.txt')
ga_vectorizer = create_lemmatizing_vectorizer(ga_vocab, 1, 2, stopwords_list)
ga_results = summarize_counts(ga_vectorizer, descriptions_per_type)
ga_results.to_csv('processed-data/' + year + '-counts-skill-generalanalytics.csv')

# extract general miscellaneous keywords
gm_vocab = load_skills('resources/dict_skills_generalmisc.txt')
gm_vectorizer = create_lemmatizing_vectorizer(gm_vocab, 1, 2, stopwords_list)
gm_results = summarize_counts(gm_vectorizer, lemmatized_descriptions_per_type)
gm_results.to_csv('processed-data/' + year + '-counts-skill-generalmisc.csv')

# extract programming languages
pl_vocab = load_skills('resources/dict_skills_programminglanguages.txt')
pl_vectorizer = create_lemmatizing_vectorizer(pl_vocab, 1, 1, stopwords_list)
pl_results = summarize_counts(pl_vectorizer, descriptions_per_type)
pl_results.to_csv('processed-data/' + year + '-counts-skill-programminglanguages.csv')

# extract devops related skills
do_vocab = load_skills('resources/dict_skills_devops.txt')
do_vectorizer = create_lemmatizing_vectorizer(do_vocab, 1, 1, stopwords_list)
do_results = summarize_counts(do_vectorizer, descriptions_per_type)
do_results.to_csv('processed-data/' + year + '-counts-skill-devops.csv')

############################
# Compute X (for modeling) #
############################

desc = jobs['Job Description']
desc_lem = jobs['description_lemmatized']

ed_count_vector = ed_vectorizer.fit_transform(desc_lem)
exp_count_vector = exp_vectorizer.fit_transform(desc_lem)
db_count_vector = db_vectorizer.fit_transform(desc)
cp_count_vector = cp_vectorizer.fit_transform(desc)
df_count_vector = df_vectorizer.fit_transform(desc)
dp_count_vector = dp_vectorizer.fit_transform(desc)
ga_count_vector = ga_vectorizer.fit_transform(desc)
gm_count_vector = gm_vectorizer.fit_transform(desc)
pl_count_vector = pl_vectorizer.fit_transform(desc)
do_count_vector = do_vectorizer.fit_transform(desc)

ed_X = pd.DataFrame(ed_count_vector.toarray(), columns=ed_vectorizer.get_feature_names())
exp_X = pd.DataFrame(exp_count_vector.toarray(), columns=exp_vectorizer.get_feature_names())
db_X = pd.DataFrame(db_count_vector.toarray(), columns=db_vectorizer.get_feature_names())
cp_X = pd.DataFrame(cp_count_vector.toarray(), columns=cp_vectorizer.get_feature_names())
df_X = pd.DataFrame(df_count_vector.toarray(), columns=df_vectorizer.get_feature_names())
dp_X = pd.DataFrame(dp_count_vector.toarray(), columns=dp_vectorizer.get_feature_names())
ga_X = pd.DataFrame(ga_count_vector.toarray(), columns=ga_vectorizer.get_feature_names())
gm_X = pd.DataFrame(gm_count_vector.toarray(), columns=gm_vectorizer.get_feature_names())
pl_X = pd.DataFrame(pl_count_vector.toarray(), columns=pl_vectorizer.get_feature_names())
do_X = pd.DataFrame(do_count_vector.toarray(), columns=do_vectorizer.get_feature_names())

X = pd.concat([ed_X, exp_X, db_X, cp_X, df_X, dp_X, ga_X, gm_X, pl_X, do_X], axis=1)

# Convert the counts to binary yes/no values
X[X > 0] = 1

X.to_csv(f'processed-data/{year}-X.csv')
