#!/usr/bin/env python3

# ibm imports
import json
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 \
    import Features, CategoriesOptions

import requests
import csv
import pandas as pd
from tqdm import tqdm



class Classificator(object):
    """Base class for the Classificators"""
    def __init__(self):
        super(Classificator, self).__init__()

class Sentilecto(Classificator):
    """Class to call Sentilecto API"""
    def __init__(self):
        super(Sentilecto, self).__init__()
        self.SENTILECTO_URL = "http://dev.natural.do/api/v0.9/deep-structure/%s/%d"
        self.apikey = 123456

    def generate_url(self, text):
        return self.SENTILECTO_URL % (text, self.apikey)

    def process_text(self, row):
        url_request = self.generate_url(text=row['input'])
        r = requests.get(url_request)
        row_dict = row.to_dict()
        testtype = row_dict['type']
        del row_dict['input']
        del row_dict['type']

        processed_row = {'response': r.json()}
        processed_row.update(row_dict)

        return(processed_row)

    def test_subject_classification(self, got, expected): 
        # hardcoded for the first sentence / clause
        res = None
        slots = got['2-Sentences'][0]\
                ['2-Sentence-Clauses'][0]['2-Deep-Structure']\
                ['1-SVO-Slots']
        if expected == slots['0-Subject']:
            res = 'tp'
        elif expected in slots.items():
            res = 'fp'
            print('FP found. Expected: %s, got: %s' % (expected, ', '.join([ x for x in slots.values() if x])))
        else:
            res = 'fn'
            print('FN found. Expected: %s, got: %s' % (expected, ', '.join([ x for x in slots.values() if x])))

        return res

class InputDatasetIterator(object):
    """Class for iterating the csv dataset InputDatasetIterator"""
    def __init__(self):
        super(InputDatasetIterator, self).__init__()
        self.INPUT_CORPORA = "basic-syntax-tests/tests/benchmark_ds_v0.1.csv"

        self.corpora_dataframe = pd.read_csv(self.INPUT_CORPORA)

        self.iterator = self.corpora_dataframe.iterrows()

        processed = []
        self.testtype = 'prueba'
        self.N = len(self.corpora_dataframe.index)

    def length(self):
        return self.N

    def __iter__(self):
        return self#.iterator

    def __next__(self): 
        index, row = self.iterator.__next__()
        return row

class OutputProcessor(object):
    """Class to store the outputs of the processors"""
    def __init__(self):
        super(OutputProcessor, self).__init__()
        self.processed = []
        self.subject_recognition = {'fp': 0,
                'tp':0,
                'fn':0}
                

    def add_output(self, output):
        self.processed.append(output) 

    def analyze_outputs(self):
        for processor, test in self.processed: 
            res = processor.test_subject_classification(\
                    test['response'], test['expected subject']) 
            self.subject_recognition[res] = self.subject_recognition[res] + 1

            
if __name__ == '__main__':
    ds = InputDatasetIterator()
    output_processor = OutputProcessor()
    sentilecto_processor = Sentilecto()
    for row in ds:
        sentilecto_processed = sentilecto_processor.process_text(row)
        output_processor.add_output((sentilecto_processor, sentilecto_processed))

    output_processor.analyze_outputs()
    sr_precision = output_processor.subject_recognition['tp'] / float(output_processor.subject_recognition['tp'] + output_processor.subject_recognition['fp'])
    sr_recall = output_processor.subject_recognition['tp'] / float(output_processor.subject_recognition['tp'] + output_processor.subject_recognition['fn'])

    sr_f1 = 2 * sr_precision * sr_recall /(sr_precision + sr_recall)

    print("""Corpus: %s
test: %s
N: %d
tp: %d
fn: %d
fp: %d
precision: %.2f
recall: %.2f
F-score: %.2f""" % (ds.INPUT_CORPORA, ds.testtype, ds.length(),
    output_processor.subject_recognition['tp'],
    output_processor.subject_recognition['fp'],
    output_processor.subject_recognition['fn'], sr_precision,
    sr_recall, sr_f1))


