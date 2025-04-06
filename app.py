from flask import Flask, render_template_string, request, redirect, url_for, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.utils import *
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

le_job = load_model('le_job')
le_job = get_mappings(le_job)

le_loc = load_model('le_loc')
le_loc = get_mappings(le_loc)

le_pets = load_model('le_pets')
le_pets = get_mappings(le_pets)
    
le_body_prof = load_model('le_body_prof')
le_body_prof = get_mappings(le_body_prof)
        
le_interests = load_model('le_interests')
le_interests = get_mappings(le_interests)
        
le_othr_interests = load_model('le_othr_interests')
le_othr_interests = get_mappings(le_othr_interests)

all_languages = load_dictionary('all_langs')

vectorizer = load_model('count_vectorizer')
scaler = load_model('scaler')

profile1 = {}
feats1 = []
profile2 = {}
feats2 = []

def encode_langs(lang_list):
    encoded_langs = all_languages.copy()
    if len(lang_list) != 0:
        for lang in lang_list:
            encoded_langs[lang] = 1
    return list(encoded_langs.values())
        


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/user1', methods = ['GET', 'POST'])
def user1_profile():
    global profile1, feats1
    if request.method == 'GET':
        return render_template('user1.html',
                                jobs = le_job.keys(),
                                locations = le_loc.keys(),
                                pets = le_pets.keys(),
                                body_prof = le_body_prof.keys(),
                                interests = le_interests.keys(),
                                othr_interests = le_othr_interests.keys(),
                                all_langs = all_languages.keys())
    
    else:    
        age = request.form.get('age')
        status = request.form.get('status')
        profile1['sex']  = request.form.get('sex')
        profile1['pref'] = request.form.get('orientation')
        drinks = request.form.get('drinks')
        drugs = request.form.get('drugs')
        height = float(request.form.get('height'))
        job = int(le_job[request.form.get('job')])
        location = int(le_loc[request.form.get('location')])
        pets = int(le_pets[request.form.get('pets')])
        smokes = request.form.get('smokes')
        language = request.form.getlist('language')
        new_language = request.form.get('new_languages')
        body_prof = int(le_body_prof[request.form.get('body_prof')])
        education_level = request.form.get('education_level')
        dropped_out = request.form.get('dropped_out') 
        bio = request.form.get('bio')
        interests = int(le_interests[request.form.get('interests')])
        othr_interests = int(le_othr_interests[request.form.get('othr_interests')])
        location_preference = request.form.get('location_preference')
        
        print(job)
        
        feats1 = [age,status,drinks,drugs,height,job,location,pets,smokes,
                    new_language,body_prof,education_level,dropped_out,
                    interests,othr_interests,location_preference]
        
        language = encode_langs(language) 
        feats1.extend(language)
        
        
        if bio is None:
            bio = ''
        bio = vectorizer.transform([tokenize(bio)]).toarray()[0]
        bio = bio.tolist()
        
        feats1.extend(bio)
        feats1 = scaler.transform([feats1])
        
        return redirect(url_for('user2_profile'))


@app.route('/user2', methods = ['GET', 'POST'])
def user2_profile():
    global profile2, feats2
    if request.method == 'GET':
        return render_template('user2.html',
                                jobs = le_job.keys(),
                                locations = le_loc.keys(),
                                pets = le_pets.keys(),
                                body_prof = le_body_prof.keys(),
                                interests = le_interests.keys(),
                                othr_interests = le_othr_interests.keys(),
                                all_langs = all_languages.keys())
    
    else:
        age = request.form.get('age')
        status = request.form.get('status')
        profile2['sex']  = request.form.get('sex')
        profile2['pref'] = request.form.get('orientation')
        drinks = request.form.get('drinks')
        drugs = request.form.get('drugs')
        height = float(request.form.get('height'))
        job = int(le_job[request.form.get('job')])
        location = int(le_loc[request.form.get('location')])
        pets = int(le_pets[request.form.get('pets')])
        smokes = request.form.get('smokes')
        language = request.form.getlist('language')
        new_language = request.form.get('new_languages')
        body_prof = int(le_body_prof[request.form.get('body_prof')])
        education_level = request.form.get('education_level')
        dropped_out = request.form.get('dropped_out') 
        bio = request.form.get('bio')
        interests = int(le_interests[request.form.get('interests')])
        othr_interests = int(le_othr_interests[request.form.get('othr_interests')])
        location_preference = request.form.get('location_preference')
        
        print(job)
        
        feats2 = [age,status,drinks,drugs,height,job,location,pets,smokes,
                    new_language,body_prof,education_level,dropped_out,
                    interests,othr_interests,location_preference]
        
        language = encode_langs(language)
        feats2.extend(language)
        
        
        if bio is None:
            bio = ''
        bio = vectorizer.transform([tokenize(bio)]).toarray()[0]
        bio = bio.tolist()
        
        feats2.extend(bio)
        feats2 = scaler.transform([feats2])
        
        return redirect(url_for('submit_profile'))

@app.route('/final', methods=['GET','POST'])
def submit_profile():
    
    print(profile1['sex'], profile1['pref'], profile2['sex'], profile2['pref'])

    percentage = 0
    if is_match_possible(profile1['sex'], profile1['pref'], profile2['sex'], profile2['pref']):
        percentage = np.round(cosine_similarity(feats1, feats2)[0][0] * 100).astype('int')
    
    print('Percentage : ', percentage)

    return render_template('final.html', percentage = percentage)

@app.route('/reset_user1')
def reset_user1():
    print('Back to User 1')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug = True)