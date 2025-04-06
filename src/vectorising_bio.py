import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from utils import *


if __name__ == "__main__":
    data_path = os.path.join('data', 'bios.csv')
    bio = pd.read_csv(data_path)
    
    corpus = bio.bio.apply(tokenize)
    
    vectorizer = CountVectorizer()
    
    X = vectorizer.fit_transform(corpus).toarray().astype('int16')
    
    X = pd.DataFrame(X,columns=vectorizer.get_feature_names_out())
    
    save_model('vectorizer', vectorizer)
    
    features = pd.read_csv(os.path.join('data', 'features.csv'))
    
    refined_data = pd.concat((features,X),axis=1)
    feats = refined_data.columns
    
    scaler = MinMaxScaler()
    
    refined_data = pd.DataFrame(scaler.fit_transform(refined_data), columns=feats)
    
    refined_data.to_csv(os.path.join('data', 'refined_data.csv'), index=False)
    
    save_model('scaler',scaler)