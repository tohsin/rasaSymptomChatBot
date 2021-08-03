
import logging
import pandas as pd
import numpy as np
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from autocorrect import Speller
import pickle

filename = 'actions/disease_pred_model.sav'

nlp = spacy.load('en_core_web_md')
I = "actions/p.csv"
desc="actions/symptom_Description.csv"
prec="actions/symptom_precaution.csv"
description_df=pd.read_csv(desc)
precaution_df=pd.read_csv(prec)
df =  pd.read_csv(I) 

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    filename='actions/logging.log',
    filemode='a',
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.DEBUG
)

def encode_symptom(symptom):
    '''
    Convert symptom string to vector using spacy

    :param symptom:
    :return: 256-D vector
    '''

    # logging.info(f"Encoding symptom {symptom}")
    encoded_symptom = nlp(symptom).vector.tolist()
    # logging.info(f"Encoded vector {encoded_symptom}")

    return encoded_symptom
def return_responce(symptoms):
    flag=False
    solution=[]
    weight_list=[]
    count=0
    cache={}
    encoded_symptoms = [encode_symptom(symptom) for symptom in symptoms]
    for encoded_symptom in  encoded_symptoms:
        high=0
        found=""
        val=0
        index=0
        for i in range(len(df)):#loop through each entry
            load=df.loc[i, "Symptom"]#select symptom
            if load=="scurring":
                load="scurry"
            load=load.replace("_", " ")
            encoded_data_symptom=encode_symptom(load)
            diff=cosine_similarity(np.array(encoded_symptom).reshape(1, -1),
                                                          np.array(encoded_data_symptom).reshape(1, -1))[0]
            cache[load]=diff
            if  diff>0.85:
                count+=1
            if diff>high:
                high=diff
                found=load
                val=diff
                index=i
        if(high!=0):
            simp=df.loc[index, "weight"]
            weight_list.append(simp)
            solution.append(found)
        elif(high==0):
            return ("I am not sure i know what you are talking about \n kindly check your spellings and try again ")
        elif(high<0.85 and high>0):
            flag=True
            simp=df.loc[index, "weight"]
            weight_list.append(simp)
            solution.append(found)
        
    print(weight_list)
    remainder=17-len(weight_list)
    print(remainder)
    other=remainder*[0]
    print(len(other))
    psy=[weight_list + other]   
    print(len(psy))  
    model = pickle.load(open(filename, 'rb'))
    pred = model.predict(psy)  
    print(description_df.loc[description_df['Disease'] ==pred[0]])
    logging.info(f"row found{description_df.loc[description_df['Disease'] ==pred[0]]}")
    descto
    for i in range(len(precaution_df)):#loop through each entry
            if pred[0]==df.loc[i, "Disease"]:
               print("disease found returning description")

    if(not flag):
        return("Based on your symptoms it looks like you could have {}".format(pred[0]))
    else:
        return ("Unable to find a diagnosis, the closest match was{}".format(pred[0]))

