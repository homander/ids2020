#!/bin/sh
rm -f ./processed-data/*.csv
rm -f ./processed-data/*.xlsx
python src/salary-survey-transformation.py
python src/us-job-postings-transformation.py
python src/preprocessing_concatenation.py
python src/data_cleaning.py
python src/job-filtering.py df_cleaned_filtered.csv
python src/job-filtering.py data-scientist-job-postings-from-us-2019-preprocessed.csv
python src/feature_count_extractor.py data-scientist-job-postings-from-us-2019-preprocessed_filtered.csv 2019
python src/feature_count_extractor.py df_cleaned_filtered.csv 2020
python src/ngram-generation.py
python src/visualisation.py
python src/ML.py