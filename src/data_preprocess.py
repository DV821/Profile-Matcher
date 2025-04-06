import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from utils import *


if __name__ == "__main__":
    data_path = os.path.join('data', 'data.csv')
    df = pd.read_csv(data_path)


    df['status'] = df['status'].apply(label_status) 
    df['drinks'] = df['drinks'].apply(label_drinks) 
    df['drugs'] = df['drugs'].apply(label_drugs) 
    df['smokes'] = df['smokes'].apply(label_smokes) 
    df['new_languages'] = df['new_languages'].apply(label_new_lang) #
    df['dropped_out'] = df['dropped_out'].apply(label_drop_out) 
    df['location_preference'] = df['location_preference'].apply(label_location_pref) 

    le_job = LabelEncoder()
    df['job'] = le_job.fit_transform(df['job'])
    save_model('le_job',le_job)

    le_loc = LabelEncoder()
    df['location'] = le_loc.fit_transform(df['location'])
    save_model('le_loc',le_loc)

    le_pets = LabelEncoder() #
    df['pets'] = le_pets.fit_transform(df['pets'])
    save_model('le_pets',le_pets)
        
    le_body_prof = LabelEncoder() #
    df['body_profile'] = le_body_prof.fit_transform(df['body_profile'])
    save_model('le_body_prof',le_body_prof)
        
    le_interests = LabelEncoder()
    df['interests'] = le_interests.fit_transform(df['interests'])
    save_model('le_interests',le_interests)
        
    le_othr_interests = LabelEncoder() #
    df['other_interests'] = le_othr_interests.fit_transform(df['other_interests'])
    save_model('le_othr_interests',le_othr_interests)
        
    langs = []
    for i in range(len(df)):
        langs.extend(df.language[i].split(', '))
    all_langs = list(set(langs))
    
    all_langs = {}
    for lang in langs:
        all_langs[lang] = 0

    save_dictionary(all_langs, 'all_langs')

    lang_pd = one_hot_encode_languages(df['language'], all_langs)

    df = pd.concat((df,lang_pd), axis = 1)
    
    user_id = pd.DataFrame(df['user_id']) 

    bio = pd.DataFrame(df['bio']) 

    lang = pd.DataFrame(df['language']) 
    
    orientation = df[['sex','orientation']]

    df = df.drop(['user_id','username','bio','language','sex','orientation'],axis=1) 

    df.to_csv(os.path.join('data', 'features.csv'), index=False)
    user_id.to_csv(os.path.join('data', 'id.csv'), index=False)
    bio.to_csv(os.path.join('data', 'bios.csv'), index=False)
    orientation.to_csv(os.path.join('data', 'preference.csv'), index=False)