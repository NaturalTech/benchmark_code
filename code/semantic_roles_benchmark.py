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
import spacy

class Classificator(object):
    """Base class for the Classificators"""
    def __init__(self):
        super(Classificator, self).__init__()

    def test_true_positive(self, obtained, expected):
        # Is equal, the lower is equal or the lower is inside the expected. Obviously the expected could be 'in' the obtained
        return obtained == expected or \
                (obtained  != "None"  and expected  != "None"  and
                        ((obtained.lower() == expected.lower()) or
                (obtained.lower() in expected.lower())))

    def test_true_negative(self, obtained, expected):
        return obtained == expected and obtained == 'None'

    def test_false_positive(self, obtained, expected):
        return (not self.test_true_positive(obtained, expected)) and (obtained  != "None" )

    def test_false_negative(self, obtained, expected):
        return (obtained == "None" and obtained != expected) 

    def test_classification(self, obtained, expected): 
        # hardcoded for the first sentence / clause

        # TP: condicion in string
        # FP: condicion not in string
        # FN: no hubo deteccion (es None y el expected es distinto de None)
        res = None
        res_func = lambda res_arr: ['tp', 'tn', 'fp', 'fn'][res_arr.index(True)]

        tp_s = self.test_true_positive(obtained['test_subject'], expected['expected_subject'])
        tn_s = self.test_true_negative(obtained['test_subject'], expected['expected_subject'])
        fp_s = self.test_false_positive(obtained['test_subject'], expected['expected_subject'])
        fn_s = self.test_false_negative(obtained['test_subject'], expected['expected_subject']) 
        if (tp_s + tn_s + fp_s + fn_s) > 1:#, 'multiple conditions met error' 
            print("ERROR: Multiple classifaction error")
        res_s = res_func([tp_s, tn_s, fp_s, fn_s])

        tp_o = self.test_true_positive(obtained['test_object'], expected['expected_object'])
        tn_o = self.test_true_negative(obtained['test_object'], expected['expected_object'])
        fp_o = self.test_false_positive(obtained['test_object'], expected['expected_object'])
        fn_o = self.test_false_negative(obtained['test_object'], expected['expected_object']) 
        if (tp_o + + tn_o + fp_o + fn_o) > 1:
            print("ERROR: Multiple classifaction error")
        res_o = res_func([tp_o, tn_o, fp_o, fn_o])

        tp_v = self.test_true_positive(obtained['test_verb'], expected['expected_verb'])
        tn_v = self.test_true_negative(obtained['test_verb'], expected['expected_verb'])
        fp_v = self.test_false_positive(obtained['test_verb'], expected['expected_verb'])
        fn_v = self.test_false_negative(obtained['test_verb'], expected['expected_verb']) 
        if (tp_v + + tn_v + fp_v + fn_v) > 1:#, 'multiple conditions met error' 
            print("ERROR: Multiple classifaction error")
        res_v = res_func([tp_v, tn_v, fp_v, fn_v])

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
                enums.DependencyEdge.Label(token.dependency_edge.label).name.lower())
                for token in response.tokens]
            ret['debug_response'] = json.dumps(debug_response)

        for k, v in ret.items():
            if v is None or v.strip()== "":
                ret[k] = "None"
        return ret


class Watson(Classificator):
    """Class to call Watson api"""
    name = 'Watson'
    def __init__(self, lang = 'es'):
        super(Watson, self).__init__()
        
        self.lemmatizer = spacy.load(lang)
        
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
            print('Identified subject is empty for text: "%s", processor: "%s"' % (text, self.name))
        try:
            ret['test_object'] = response['semantic_roles'][0]['object']['text']
        except Exception as e:
            print('Identified object is empty for text: "%s", processor: "%s"' % (text, self.name))
        try:
            ret['test_verb'] = response['semantic_roles'][0]['action']['text']
            tokens = self.lemmatizer(ret['test_verb']) 
            for tok in tokens:
                if tok.pos_ in ['VERB', 'PROPN']:
                    ret['test_verb'] = tok.lemma_
        except Exception as e:
            print('Identified verb is empty for text: "%s", processor: "%s"' % (text, self.name))

        ret['debug_response'] = json.dumps(response['semantic_roles'])

        for k, v in ret.items():
            if v is None or v.strip()== "":
                ret[k] = "None"

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

        # TODO: fixme, this is taking only first sentence and clause.
        json_obj = response.json()
        try:
            slots = json_obj['2-Sentences'][0]\
                ['2-Sentence-Clauses'][0]['2-Deep-Structure']\
                ['1-SVO-Slots']
        except Exception as e: 
            print('Identification for text: "%s", processor: "%s" is invalid: out %s' % (text, self.name, json.dumps(json_obj)))
            return ret

        try:
            ret['test_subject'] = slots['0-Subject']
        except Exception as e:
            print('Identified subject is empty for text: "%s", processor: "%s"' % (text, self.name))
        try:
            obj= slots["2-Phrasal-Verb-Complement"] + slots["2-Direct-Object"] # slots["2-Equative"] + 
            ret['test_object'] = obj
        except Exception as e:
            print('Identified object is empty for text: "%s", processor: "%s"' % (text, self.name))
        try:
            ret['test_verb'] = slots['1-Verb']
        except Exception as e:
            print('Identified verb is empty for text: "%s", processor: "%s"' % (text, self.name))

        ret['debug_response'] = json.dumps(json_obj['2-Sentences'])
        for k, v in ret.items():
            if v is None or v.strip()== "":
                ret[k] = "None"
        return(ret)


class InputDatasetIterator(object):
    """Class for iterating the csv dataset InputDatasetIterator"""
    def __init__(self):
        super(InputDatasetIterator, self).__init__()
        # TODO: Move to config file
        self.INPUT_CORPORA = "/home/danito/proj/naturaltech/benchmark-repo/data/test_subject.csv"

        self.corpora_dataframe = pd.read_csv(self.INPUT_CORPORA, encoding='latin8')

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

    def get_index_and_default_to_zero(self, key, df):
        ret = 0
        try:
            ret = df.set_index('result').loc[[key], ['count']]['count']
        except KeyError:
            pass
        return float(ret) 

    def get_group_metrics(self): 
        results_df = pd.DataFrame(self.results)
        results_df.to_csv('results.csv', sep=';')
        results_df['id'] = results_df.index
        wide_res_df = pd.wide_to_long(results_df, stubnames=['result'],
                i=['id'], j='test', sep='_',suffix='\\w+')

        res_count_df = wide_res_df.groupby(['test', 'result',
            'processor']).agg({'text':['count']})
        res_count_df = res_count_df.reset_index()
        res_count_df['count'] = res_count_df.text.reset_index()['count']

        del res_count_df['text'] 

        tp_acum = {}
        for i in res_count_df.groupby(['test', 'processor']):
            test = i[0][0]
            processor = i[0][1]
            fp = self.get_index_and_default_to_zero('fp', i[1])
            fn = self.get_index_and_default_to_zero('fn', i[1])
            tp = self.get_index_and_default_to_zero('tp', i[1])
            tn = self.get_index_and_default_to_zero('tn', i[1])
            tp_acum[processor] = tp_acum.get(processor, 0) + tp
            N = fp + fn + tp + tn
            # assert N == ds.N
            accuracy = tp / float(N)
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall)

            print("""Results: test: '%s', processor = '%s', N: %d, tp: %d, tn: %d, fp: %d, fn: %d,
                        precision: %.2f, recall: %.2f, F-score: %.2f""" % ( test, processor, N,
                            tp, tn, fp, fn, precision, recall, f1))

        for k, v in tp_acum.items():
            print("Total accuracy: processor = '%s', Accuracy = %.4f" % (k, v/(3*N)))

            
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
    output_processor.get_group_metrics()
