import nltk
import re
import spacy
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nlp = spacy.load("en_core_web_trf")

DEFAULT_MASKS = ["PERSON", "ORG", "GPE", "NORP", "PRODUCT", "DATE", "MONEY"]

def mask(text: str, masks: list = DEFAULT_MASKS):
    """
    Masks named entities such as names of locations and people to prevent data leakage.
    Replaces named entities with placeholder tokens like [PERSON].

    Preferred for transformer-based models like RoBERTa.
    """

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

if __name__ == "__main__":
    test_text = "PROVIDENCE, RI - In spite of The Onion's best efforts to brave the ongoing winter storm " + \
        "and freezing temperatures, the inclement weather currently affecting the Northeast has left " + \
        "Providence-area liar Tim Carlson unable to commute to his office, the habitual deceiver " + \
        "reported to his colleagues today. They haven't been able to."
    print(mask(test_text))