#!/bin/sh
rm -f ./processed-data/*.csv
rm -f ./processed-data/*.xlsx
python src/preprocessing_concatenation.py
python src/salary-survey-transformation.py
python src/cleaning_salary.py
python src/jobdescription.py
python src/ngram-generation.py