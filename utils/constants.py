# masks for named entity recognition

# list of all possible tag options for reference, from https://github.com/explosion/spaCy/discussions/9147

# PERSON:      People, including fictional.
# NORP:        Nationalities or religious or political groups.
# FAC:         Buildings, airports, highways, bridges, etc.
# ORG:         Companies, agencies, institutions, etc.
# GPE:         Countries, cities, states.
# LOC:         Non-GPE locations, mountain ranges, bodies of water.
# PRODUCT:     Objects, vehicles, foods, etc. (Not services.)
# EVENT:       Named hurricanes, battles, wars, sports events, etc.
# WORK_OF_ART: Titles of books, songs, etc.
# LAW:         Named documents made into laws.
# LANGUAGE:    Any named language.
# DATE:        Absolute or relative dates or periods.
# TIME:        Times smaller than a day.
# PERCENT:     Percentage, including ”%“.
# MONEY:       Monetary values, including unit.
# QUANTITY:    Measurements, as of weight or distance.
# ORDINAL:     “first”, “second”, etc.
# CARDINAL:    Numerals that do not fall under another type.

ENTITY_MASKS = ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC"]

# mappings for article section standardisation
SECTION_MAPPINGS = {
    "news": "news",
    "news in brief": "news",
    "local": "news",
    "news in photos": "news",
    "lifestyle": "lifestyle",
    "world": "world",
    "work": "work",
    "animals": "animals",
    "food": "lifestyle",
    "family": "family",
    "death": "health",
    "kids": "family",
    "economy": "money",
    "robert mueller": "politics",
    "celebrities": "people",
    "people": "people",
    "movies": "entertainment",
    "government": "politics",
    "brett kavanaugh": "politics",
    "friends": "relationships",
    "science technology": "technology",
    "technology": "technology",
    "tv": "entertainment",
    "health": "health",
    "relationships": "relationships",
    "education": "education",
    "entertainment": "entertainment",
    "politics": "politics",
    "business": "business",
    "sports": "sports",
    "science": "science",
    "books": "books",
    "money": "money",
    "travel": "travel",
    "religion": "religion",
    "weddings": "relationships",
    "crime": "crime",
    "college": "education",
    "home": "family",
    "parents": "family",
    "media": "entertainment",
    "environment": "environment",
    "women": "women",
    "teen": "family",
    "parenting": "family",
    "celebrity": "people",
    "home & living": "family",
    "divorce": "relationships",
    "opinion": "opinion",
    "good news": "news",
    "arts & culture": "arts and culture",
    "food & drink": "food and drink",
    "latino voices": "voices",
    "u.s. news": "news",
    "worldpost": "world",
    "the world post": "world",
    "the worldpost": "world",
    "culture & arts": "arts and culture",
    "tech": "technology",
    "style": "style",
    "style & beauty": "style",
    "healthy living": "health",
    "weird news": "news",
    "world news": "news",
    "wellness": "lifestyle",
    "black voices": "voices",
    "comedy": "entertainment",
    "queer voices": "voices",
    "workplace": "lifestyle",
    "donald trump": "politics",
    "patriotism": "politics",
    "animals": "nature",
    "human interest": "lifestyle",
    "taste": "lifestyle",
    "impact": "impact",
    "green": "nature",

    # TAGS TO BE FILTERED OUT ENTIRELY
    "sponsored post": "", # appears only in onion
    "unsponsored": "", # appears only in onion
    "www.theonion.com": "",
    "entertainment.theonion.com": "",
    "local.theonion.com": "",
    "ogn": "",
    "onion social": "",
    "vol 55 issue 14": "",
    "huffpost live": "",
    "huffpost personal": "",
    "own": "",
    "huffington post": "",
    "post 50": "",
}
