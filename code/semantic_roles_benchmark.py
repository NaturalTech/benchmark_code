#!/usr/bin/env python3

from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types
from google.oauth2 import service_account
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1  import Features, CategoriesOptions
from ibm_watson.natural_language_understanding_v1 import SemanticRolesOptions
from tqdm import tqdm
import csv
import json
import pandas as pd
import requests

class Classificator(object):
    """Base class for the Classificators"""
    def __init__(self):
        super(Classificator, self).__init__()

    def test_classification(self, got, expected): 
        # hardcoded for the first sentence / clause
        res = None
        tp_s = got['test_subject'] == expected['expected_subject']
        fp_s = got['test_subject'] != expected['expected_subject'] and got['test_subject'] in [expected['expected_object'], expected['expected_verb']]
        fn_s = got['test_subject'] != expected['expected_subject'] and (not got['test_subject'] in [expected['expected_object'], expected['expected_verb']])

        assert (tp_s + fp_s + fn_s) == 1, 'multiple conditions met error'

        res_func = lambda tp, fp: 'tp' if tp else ('fp' if fp else 'fn')

        res_s = res_func(tp_s, fp_s)

        tp_o = got['test_object'] == expected['expected_object']
        fp_o = got['test_object'] != expected['expected_object'] and got['test_object'] in [expected['expected_subject'], expected['expected_verb']]
        fn_o = got['test_object'] != expected['expected_object'] and (not got['test_object'] in [expected['expected_subject'], expected['expected_verb']])
        assert (tp_o + fp_o + fn_o) == 1, 'multiple conditions met error' 
        res_o = res_func(tp_o, fp_o)

        tp_v = got['test_verb'] == expected['expected_verb']
        fp_v = got['test_verb'] != expected['expected_verb'] and got['test_verb'] in [expected['expected_subject'], expected['expected_object']]
        fn_v = got['test_verb'] != expected['expected_verb'] and (not got['test_verb'] in [expected['expected_subject'], expected['expected_object']])

        assert (tp_v + fp_v + fn_v) == 1, 'multiple conditions met error' 
        res_v = res_func(tp_v, fp_v)

        res = {'result_subject': res_s, 'result_object': res_o, 'result_verb': res_v}
        return res

class Google(Classificator):
    """clase  for Google classificator"""
    name = 'Google'
    def __init__(self):
        super(Google, self).__init__() 
        # Instantiates a client
        self.client = language.LanguageServiceClient()


    def process_text(self, row):
        # The text to analyze
        text = row['input']
        document = types.Document(
            content=text,
            type=enums.Document.Type.PLAIN_TEXT)

        ret = {}
        ret['text'] = text
        ret['test_subject'] = None
        ret['test_object'] = None
        ret['test_verb'] = None

	# Detects the sentiment of the text
        response = self.client.analyze_syntax(document) 
        for token in response.tokens:
            dependency_edge = token.dependency_edge
            if 'root' == enums.DependencyEdge.Label(dependency_edge.label).\
                    name.lower():
                ret['test_verb'] = token.lemma
            elif 'nsubj' in enums.DependencyEdge.Label(dependency_edge.label).\
                    name.lower():
                # could be nsubj or nsubjpassive
                ret['test_subject'] = token.lemma
            elif enums.DependencyEdge.Label(dependency_edge.label).name.lower() in ['dobj', 'iobj']: 
                ret['test_object'] = token.lemma
            
            debug_response = ['lema: %s, label: %s' % (token.lemma,
                enums.DependencyEdge.Label(dependency_edge.label).name.lower())
                for token in response.tokens]
            ret['debug_response'] = debug_response
        return ret


class Watson(Classificator):
    """Class to call Watson api"""
    name = 'Watson'
    def __init__(self):
        super(Watson, self).__init__()
        
        
        self.apikey='WTc7S5wG9IYhOK_Qe8TYl_mgKpzhb_5Gk7ejCk-vtYVS'
        self.url = 'https://api.us-south.natural-language-understanding.watson.cloud.ibm.com/instances/c205be7e-27ed-4e7d-8c95-c9aa4e6bf9e0'

        self.authenticator = IAMAuthenticator(self.apikey)
        self.processor = NaturalLanguageUnderstandingV1(\
                version='2019-07-12', authenticator=self.authenticator)
        self.processor.set_service_url(self.url)
        self.errors = []

    def process_text(self, row):
        'returns text, s, o, v. expected s, expected o, expected v'
        text = row['input']
        response = self.processor.analyze(
            text=text,
            features=Features(semantic_roles=SemanticRolesOptions())).get_result()

        ret = {}
        ret['text'] = text
        ret['test_subject'] = None
        ret['test_object'] = None
        ret['test_verb'] = None
        try:
            ret['test_subject'] = response['semantic_roles'][0]['subject']['text']
        except Exception as e:
            print(e)
        try:
            ret['test_object'] = response['semantic_roles'][0]['object']['text']
        except Exception as e:
            print(e)
        try:
            ret['test_verb'] = response['semantic_roles'][0]['action']['text']
        except Exception as e:
            print(e)

        ret['debug_response'] = json.dumps(response)

        return(ret)


class Sentilecto(Classificator):
    """Class to call Sentilecto API"""
    name = 'Sentilecto'
    def __init__(self):
        super(Sentilecto, self).__init__()
        # TODO: migrate to config file
        self.SENTILECTO_URL = "http://dev.natural.do/api/v0.9/deep-structure/%s/%d"
        self.apikey = 123456


    def generate_url(self, text):
        return self.SENTILECTO_URL % (text, self.apikey)

    def process_text(self, row):
        url_request = self.generate_url(text=row['input'])
        response = requests.get(url_request)
        text = row['input']

        ret = {}
        ret['text'] = text
        ret['test_subject'] = None
        ret['test_object'] = None
        ret['test_verb'] = None

        slots = response.json()['2-Sentences'][0]\
            ['2-Sentence-Clauses'][0]['2-Deep-Structure']\
            ['1-SVO-Slots']
        try:
            ret['test_subject'] = slots['0-Subject']
        except Exception as e:
            print(e)
        try:
            obj= slots["2-Equative"] + slots["2-Phrasal-Verb-Complement"] + slots["2-Direct-Object"]
            ret['test_object'] = obj
        except Exception as e:
            print(e)
        try:
            ret['test_verb'] = slots['1-Verb']
        except Exception as e:
            print(e)
        return(ret)


class InputDatasetIterator(object):
    """Class for iterating the csv dataset InputDatasetIterator"""
    def __init__(self):
        super(InputDatasetIterator, self).__init__()
        # TODO: Move to config file
        self.INPUT_CORPORA = "/home/danito/proj/naturaltech/benchmark-repo/data/benchmark_ds_v0.1.csv"

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
        self.subject_recognition = {}
        for subclass in Classificator.__subclasses__():
            self.subject_recognition[subclass.name] = {'tp': 0, 'fp': 0, 'fn': 0} 

    def add_output(self, output):
        self.processed.append(output) 

    def analyze_outputs(self):
        self.results = []
        for processor, test, expected in self.processed: 
            res = processor.test_classification(test, expected)
            res_d = {'processor': processor.name}
            res_d.update(test)
            res_d.update(expected)
            res_d.update(res)
            self.results.append(res_d)
            
if __name__ == '__main__':
    ds = InputDatasetIterator()
    output_processor = OutputProcessor()
    sentilecto_processor = Sentilecto()
    watson_processor = Watson()
    google_processor = Google()
    for row in ds:
        sentilecto_processed = sentilecto_processor.process_text(row)
        watson_processed = watson_processor.process_text(row)
        google_processed = google_processor.process_text(row)
        output_processor.add_output((sentilecto_processor, sentilecto_processed, row.to_dict()))
        output_processor.add_output((watson_processor, watson_processed, row.to_dict()))
        output_processor.add_output((google_processor, google_processed, row.to_dict()))

    output_processor.analyze_outputs()
    df = pd.DataFrame(output_processor.results)
    # sr_precision = output_processor.subject_recognition['tp'] / \
    #         float(output_processor.subject_recognition['tp'] + \
    #         output_processor.subject_recognition['fp'])
    # sr_recall = output_processor.subject_recognition['tp'] / \
    #         float(output_processor.subject_recognition['tp'] + \
    #         output_processor.subject_recognition['fn'])

    # sr_f1 = 2 * sr_precision * sr_recall /(sr_precision + sr_recall)

    print(output_processor)
    __import__('ipdb').set_trace()
    results_df = pd.DataFrame(output_processor.results)
    results_df.to_csv('results.csv')
    results_df['id'] = results_df.index
    wide_res_df = pd.wide_to_long(results_df, stubnames=['result'], i=['id'], j='resulting', sep='_',suffix='\\w+')

    res_count_df = wide_res_df.groupby(['resulting', 'result', 'processor']).agg({'text':['count']}) 
    res_count_df = res_count_df.reset_index()
    res_count_df['count'] = res_count_df.text.reset_index()['count']

    del res_count_df['text']

    print(res_count_df)


    # print("""Corpus: %s
# test: %s
# N: %d
# tp: %d
# fn: %d
# fp: %d
# precision: %.2f
# recall: %.2f
# F-score: %.2f""" % (ds.INPUT_CORPORA, ds.testtype, ds.length(),
    # output_processor.subject_recognition['tp'],
    # output_processor.subject_recognition['fp'],
    # output_processor.subject_recognition['fn'], sr_precision,
    # sr_recall, sr_f1))

