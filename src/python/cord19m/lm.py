# For Extractive and Abstractive Summary

# pip install bert-extractive-summarizer, torch
from summarizer import Summarizer
from transformers import *
import os
from spacy.lang.en import English


class SentenceHandler(object):
    """ 
    Sentence tokenizer
    """
    def __init__(self):
        self.nlp = English()
        self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))

    def process(self, body: str, min_length: int = 40, max_length: int = 600):
        """
        Processes the content sentences.

        :param body: The raw string body to process
        :param min_length: Minimum length that the sentences must be
        :param max_length: Max length that the sentences mus fall under
        :return: Returns a list of sentences.
        """
        doc = self.nlp(body)
        return [
            c.string.strip() for c in doc.sents
            if max_length > len(c.string.strip()) > min_length
        ]

    def __call__(self, body: str, min_length: int = 40, max_length: int = 600):
        return self.process(body, min_length, max_length)


class LanguageModel():
    """docstring for LanguageModel"""
    MODELS = {
        'bert-base-uncased': (BertModel, BertTokenizer),
        'bert-large-uncased': (BertModel, BertTokenizer),
        'xlnet-base-cased': (XLNetModel, XLNetTokenizer),
        'xlm-mlm-enfr-1024': (XLMModel, XLMTokenizer),
        'distilbert-base-uncased': (DistilBertModel, DistilBertTokenizer),
        'albert-base-v1': (AlbertModel, AlbertTokenizer),
        'albert-large-v1': (AlbertModel, AlbertTokenizer)
    }

    def __init__(self, model_name: str, cache_dir: str = None):

        ## saved language model directory
        if cache_dir:
            cache_dir = os.path.join(cache_dir, model_name)
            os.makedirs(cache_dir, exist_ok=True)

        self.sentence_handler = SentenceHandler()

        base_model, base_tokenizer = self.MODELS.get(model_name, (None, None))
        if base_model:
            self.model = base_model.from_pretrained(base_model,
                                                    output_hidden_states=True,
                                                    cache_dir=cache_dir)
            self.tokenizer = base_model.from_pretrained(base_tokenizer,
                                                        cache_dir=cache_dir)
            self.summmarizer = Summarizer(
                custom_model=self.model,
                custom_tokenizer=self.tokenizer,
                sentence_handler=self.sentence_handler)

    def get_sentences_embeddings(self,
                                 text: str,
                                 min_length: int = 60,
                                 max_length: int = 600,
                                 hidden: int = -2,
                                 reduce_option: str = 'mean'):
        """
        :param min_length: Minimum length of sentence candidates
        :param max_length: Maximum length of sentence candidates
        :param hidden: This signifies which layer of the BERT model you would like to use as embeddings.
        :param reduce_option: Given the output of the bert model, this param determines how you want to reduce results.
        """

        sentences = self.sentence_handler(text, min_length, max_length)
        return self.model(sentences, self.hidden, self.reduce_option)

    def summarize(self,
                  text,
                  ratio: float = 0.2,
                  min_length: int = 60,
                  max_length: int = 600,
                  use_first: bool = True):
        """
        :param ratio: Ratio of sentences to use
        """

        return self.summmarizer(text,
                                ratio=ratio,
                                min_length=min_length,
                                max_length=max_length,
                                use_first=use_first)
