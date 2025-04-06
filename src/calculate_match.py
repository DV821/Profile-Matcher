import os 
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from utils import *

if __name__ == "__main__":
    user_id = pd.read_csv(os.path.join('data', 'id.csv'))
    refined_data = pd.read_csv(os.path.join('data', 'refined_data.csv'))
    pref = pd.read_csv(os.path.join('data', 'preference.csv'))
    
    X = refined_data.values
    
    match = [] 
    for i in range(len(refined_data)):
        sex_i = pref.loc[i, 'sex'] 
        pref_i = pref.loc[i, 'orientation'] 
        l = []
        for j in range(len(refined_data)):
            sex_j = pref.loc[j, 'sex']
            pref_j = pref.loc[j, 'orientation'] 
            if i != j:
                if is_match_possible(sex_i, pref_i, sex_j, pref_j):
                    l.append(cosine_similarity([X[i]], [X[j]])[0][0] * 100)
                else:
                    l.append(0)
            else:
                l.append(0)
        match.append(l)
        
    sub = pd.concat((user_id,pd.DataFrame(match,columns=user_id.user_id)),axis=1)
    
    sub.to_csv(os.path.join('data', 'sub.csv'), index=False)