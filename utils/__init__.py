import sys
import spacy
import nltk
from spacy.util import is_package

def verify_resources():
    missing = []

    # spacy checker
    spacy_models = ["en_core_web_trf"]
    for model in spacy_models:
        if not is_package(model):
            missing.append(model)

    # nltk checker
    nltk_resources = [
        ('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng'),
        ('corpora/omw-1.4', 'omw-1.4'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('corpora/stopwords', 'stopwords'),
        ('corpora/wordnet', 'wordnet')
    ]
    
    for path, name in nltk_resources:
        try:
            nltk.data.find(path)
        except LookupError:
            missing.append(name)

    if missing:
        print("Missing resources for preprocessing: " + missing)
        sys.exit(1)
