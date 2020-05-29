import requests
import csv
import pandas as pd
from tqdm import tqdm

SENTILECTO_URL = "http://dev.natural.do/api/v0.9/deep-structure/%s/%d"

def generate_url(text, apikey):
    return SENTILECTO_URL % (text, apikey)



INPUT_CORPORA = "basic-syntax-tests/tests/benchmark_ds_v0.1.csv"

corpora_dataframe = pd.read_csv(INPUT_CORPORA)

processed = []
testtype = None
N = len(corpora_dataframe.index)

for index, row in tqdm(corpora_dataframe.iterrows()):
    url_request = generate_url(text=row['input'], apikey=123456)
    r = requests.get(url_request)
    row_dict = row.to_dict()
    testtype = row_dict['type']
    del row_dict['input']
    del row_dict['type']
    processed_row = {'response': r.json()}
    processed_row.update(row_dict)
    processed.append(processed_row)

resuts = {'type':
        'task'
        'N'
        'true+'
        'false-'
        'false+'}


subject_recognition = {'true+': 0,
        'false-': 0,
        'false+': 0
        }
od_recognition = {'true+': 0,
        'false-': 0,
        'false+': 0
        }
verb_recognition = {'true+': 0,
        'false-': 0,
        'false+': 0
        }

for test in processed: 
    slots = test['response']['2-Sentences'][0]\
            ['2-Sentence-Clauses'][0]['2-Deep-Structure']\
            ['1-SVO-Slots']
    if test['expected subject'] == slots['0-Subject']:
        subject_recognition['true+'] = subject_recognition.get('true+', 0) + 1
    elif test['expected subject'] == slots.items():
        subject_recognition['false+'] = subject_recognition.get('false+', 0) + 1
        print('FN found. Expected: %s, got: %s' % (test['expected subject'], ', '.join([ x for x in slots.values() if x])))
    else:
        subject_recognition['false-'] = subject_recognition.get('false-', 0) + 1
        print('FN found. Expected: %s, got: %s' % (test['expected subject'], ', '.join([ x for x in slots.values() if x])))


sr_precision = subject_recognition['true+'] / float(subject_recognition['true+'] + subject_recognition['false+'])
sr_recall = subject_recognition['true+'] / float(subject_recognition['true+'] + subject_recognition['false-'])

sr_f1 = 2 * sr_precision * sr_recall /(sr_precision + sr_recall)


# | corpus | test type | N | true+ | false- | false+ | precision | recall | F-score |
# |------------------- |--------------- |-------- |-------- |-------- |-------- |------------ |--------- |--------- |

print("""Corpus: %s
    test: %s
    N: %d
    tp: %d
    fn: %d
    fp: %d
    precision: %.2f
    recall: %.2f
    F-score: %.2f""" % (INPUT_CORPORA, testtype, N, subject_recognition['true+'],  subject_recognition['false+'], subject_recognition['false-'], sr_precision, sr_recall, sr_f1))


    # assert test['expected principal verb'] == test['response']['2-Sentences'][0]['2-Sentence-Clauses'][0]['2-Deep-Structure']['1-SVO-Slots']['1-Verb']
    # assert test['expected do'] == test['response']['2-Sentences'][0]['2-Sentence-Clauses'][0]['2-Deep-Structure']['1-SVO-Slots']['2-Direct-Object']

