import os
import pandas as pd 
import dill
from sklearn.preprocessing import LabelEncoder
import re
import nltk
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords


nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')

def label_status(status):
    values = {'single': 0, 'seeing someone': 1, 'married': 2, 'available': 3}
    return values[status]

def label_drinks(drinks):
    values = {'socially': 0, 'often': 1, 'rarely': 2, 'very often': 3, 'not at all': 4, 'desperately': 5}
    return values[drinks]

def label_drugs(drugs):
    values = {'never': 0, 'sometimes': 1, 'often': 2}
    return values[drugs]

def label_smokes(smokes):
    values = {'no': 0, 'sometimes': 1, 'yes': 2, 'when drinking': 3, 'trying to quit': 4}
    return values[smokes]

def label_new_lang(new_lang):
    values = {'interested': 0, 'not interested': 1, 'somewhat interested': 2}
    return values[new_lang]

def label_drop_out(drop_out):
    values = {'no': 0, 'yes': 1}
    return values[drop_out]

def label_location_pref(loc_pref):
    values = {'same state': 0, 'anywhere': 1, 'same city': 2}
    return values[loc_pref]

def one_hot_encode_languages(languages, all_languages):
    lang_dict = {}
    for lang in all_languages:
        lang_dict[lang] = languages.apply(lambda x: 1 if lang in x.split(', ') else 0)
    
    return pd.DataFrame(lang_dict)

def save_dictionary(dictionary, dict_name):
    file_path = os.path.join('models',f'{dict_name}.pkl')
    with open(file_path, 'wb') as file:
        dill.dump(dictionary, file)
        
def load_dictionary(dict_name):
    file_path = os.path.join('models',f'{dict_name}.pkl')
    with open(file_path, 'rb') as file:
        dictionary = dill.load(file)
    return dictionary

def tokenize(text):
    stwds = stopwords.words('english')
    lemma = WordNetLemmatizer()
    text = re.sub('[^a-zA-Z]',' ',text)
    text = text.lower()
    text = word_tokenize(text)
    text = [lemma.lemmatize(word) for word in text if word not in stwds]
    text = ' '.join(text)
    return text

def is_match_possible(sex_i, pref_i, sex_j, pref_j):

    if pref_i == 'gay':
        return sex_i == sex_j and pref_j in ['gay', 'bisexual']
    elif pref_i == 'straight':
        return sex_i != sex_j and pref_j in ['straight', 'bisexual']
    elif pref_i == 'bisexual':
        return (pref_j == 'straight' and sex_i != sex_j) or (pref_j == 'gay' and sex_i == sex_j) or (pref_j == 'bisexual')
    else:
        return False


def save_model(model_name, model):
    file_path = os.path.join('models',f'{model_name}.pkl')
    with open(file_path, 'wb') as file_path:
        dill.dump(model, file_path)

def load_model(model_name):
    file_path = os.path.join('models',f'{model_name}.pkl')
    with open(file_path, 'rb') as file:
        model = dill.load(file)
    return model

def get_mappings(le):
    return dict(zip(le.classes_, le.transform(le.classes_)))