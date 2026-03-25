import ast
import nltk
import re
import spacy
from utils.constants import SECTION_MAPPINGS, ENTITY_MASKS
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nlp = spacy.load("en_core_web_trf") # might take a while


def preprocess_article_section(entry, mappings = SECTION_MAPPINGS):
    """
    Preprocesses an article section (category) of a news article.

    Standardises article sections using a predefined set of mappings.
    Maps rare article categories or those that have no defined mapping to "other".
    """
    if entry == '':
        return entry
    
    if isinstance(entry, list):
        items = entry
    elif isinstance(entry, str) and entry.startswith('['):
        try:
            items = ast.literal_eval(entry)
        except:
            items = [entry]
    else:
        items = [entry]

    # map to standard value or "other" if not in mappings
    standardized = [mappings.get(str(item).lower(), "other") for item in items]

    # remove duplicates
    standardized = list(set(standardized))
    
    return ", ".join(standardized)


def preprocess_description(text: str, masks: list = ENTITY_MASKS):
    """
    Preprocessor for the description of a news article for transformer models.

    Masks named entities such as names of locations and people to prevent data leakage.
    Replaces named entities with placeholder tokens like [PERSON].
    """

    # remove starting lines like "NEW YORK—" because this pattern indicates the Onion
    text = re.sub(r'^[A-Z\s,.]+—', '', text).strip()

    # masker
    doc = nlp(text)
    sentences = list(doc.sents)
    processed_text = " ".join([s.text for s in sentences])
    for ent in reversed(doc.ents):
        if ent.label_ in masks and ent.end <= sentences[-1].end:
            processed_text = (processed_text[:ent.start_char] + f"[{ent.label_}]" + processed_text[ent.end_char:])

    # remove extra whitespace
    processed_text = re.sub(r'\s+', ' ', processed_text).strip()

    return processed_text


def preprocess_for_bow(text: str, remove_punctuation: bool = True,
               lemmatize: bool = True, remove_stopwords: bool = True):
    """
    Standard preprocessing function for use with Bag-of-Words based models, i.e.
    CountVectorizer/TfidfVectorizer with MultinomialNB or LogisticRegression.
    """

    # lowercase
    text = text.lower()

    if remove_punctuation:
        # remove punctuation (anything that is not word or whitespace)
        text = re.sub(r"[^\w\s]", " ", text)

    # tokenize
    tokens = word_tokenize(text)

    # lemmatize, may take a while
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        def get_wordnet_pos(word):
            """Map POS tag to first character lemmatize() accepts"""
            tag = nltk.pos_tag([word])[0][1][0].upper()
            tag_dict = {"J": wordnet.ADJ,
                        "N": wordnet.NOUN,
                        "V": wordnet.VERB,
                        "R": wordnet.ADV}
            return tag_dict.get(tag, wordnet.NOUN)
        tokens = [lemmatizer.lemmatize(t, get_wordnet_pos(t)) for t in tokens]

    # remove stopwords except some hardcoded negations
    if remove_stopwords:
        _negations  = {
          'not', 'no', 'nor', 'never', 'neither', "n't", 'nothing', 'nobody', 'nowhere',
        }
        _stopwords = set(stopwords.words('english')) - _negations
        tokens = [w for w in tokens if w not in _stopwords]

    return " ".join(tokens)
