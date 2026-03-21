import nltk
import re
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('averaged_perceptron_tagger_eng')
nltk.download('omw-1.4')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess(text: str, remove_punctuation: bool = True,
               lemmatize: bool = True, remove_stopwords: bool = True):
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
