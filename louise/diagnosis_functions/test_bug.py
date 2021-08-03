import logging
import pandas as pd
import numpy as np
import spacy
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load('en_core_web_md')
df =  pd.read_csv("sym.csv")
df_ = pd.read_pickle("symptoms.pkl")
# print("show his own dataset")
# print(df_.head)
# print("show my own dataset")
# print(df.head())
# print(len(df.columns))
# logging config
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    filename='logging.log',
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

    logging.info(f"Encoding symptom {symptom}")
    encoded_symptom = nlp(symptom).vector.tolist()

    return encoded_symptom


def create_illness_vector(encoded_symptoms):
    '''
    Compares the list of encoded symptoms to a list of encoded symptoms. Any symptom above threshold (0.85) will be
    flagged.

    :param encoded_symptoms: A list of encoded symptoms
    :return: A single vector flagging each symptoms appearence in the u
    ser message (based on vector similarity)
    '''

  
    # df['symptom_flagged'] = 0
    # print(df.head())
    # # df['symptom_vector']=[0]
    # cache={}
    # similarity={}
    # chosen=[]

    # for i in range(len(df)):
    #     load=df.loc[i, "Symptom"]
    #     if load=="scurring":
    #         load="scurry"
    #     orig=load
    #     load=load.replace("_", " ")
    #     print(load)
    #     cache[orig]=encode_symptom(load)
    # print(cache)
    # for encoded_symptom in encoded_symptoms:
    #     for key in cache:
    #         diff=cosine_similarity( np.array(encoded_symptom).reshape( 1,-1),np.array(cache[key]).reshape(1,-1))[0]
    #         similarity[key]=diff
    #         if diff>threshold:
    #             print(key)
    #             chosen.append(key)
  



    threshold = 0.85
    df['symptom_flagged'] = 0
    for encoded_symptom in encoded_symptoms:

        df['similarity'] = list(cosine_similarity(np.array(encoded_symptom).reshape(1, -1),
                                                           np.array(list(df['symptom_vector'])))[0])
        print(df.head())
        df.loc[df['similarity'] > threshold, 'symptom_flagged'] = 1

        number_of_symptoms_flagged = len(symptoms_df.loc[df['similarity'] > threshold, 'symptom_flagged'])

        logging.info(f"Flagged {number_of_symptoms_flagged} potential symptom matches")

    return list(df['symptom_flagged'])

def get_diagnosis(illness_vector):
    '''
    Compares the symptoms vector to our diagnosis df and generate the diagnosis (if one exists)

    :param illness_vector:
    :return: A string containing the diagnosis based off of illness vector similarity
    '''

    threshold = 0.5

    diagnosis_df['similarity'] = list(cosine_similarity(np.array(illness_vector).reshape(1, -1),
                                                        np.array(list(diagnosis_df['illness_vector'])))[0])

    # If there is an illness (or multiple illnesses)
    if len(diagnosis_df.loc[diagnosis_df['similarity'] > threshold]) > 0:
        illness = (
            diagnosis_df
            .sort_values(by='similarity', ascending=False)['illness']
            .iloc[0]
        )

        logging.info(f"Diagnosing user with {illness}")
        diagnosis_string = f"Based on your symptoms it looks like you could have {illness}"

    else:
        closest_match = (
            diagnosis_df
            .sort_values(by='similarity', ascending=False)[['illness', 'similarity']]
            .head(1)
        )
        logging.info(f"Unable to find a diagnosis, the closest match was {closest_match['illness'].iloc[0]} "
                     f"at {closest_match['similarity'].iloc[0]}")
        diagnosis_string = "Unfortunately I am unable to diagnose you based on the symptoms you provided"

    return diagnosis_string
def SVM(Symptom1,Symptom2):
    psymptoms = [Symptom1,Symptom2]
    loc = location.get()
    a = np.array(df1["Symptom"])
    b = np.array(df1["weight"])
    for j in range(len(psymptoms)):
        for k in range(len(a)):
            if psymptoms[j]==a[k]:
                psymptoms[j]=b[k]

    nulls = [0,0,0,0,0,0,0,0,0,0,0,0]
    psy = [psymptoms + nulls]

    pred2 = model.predict(psy)
    t3.delete("1.0", END)
    t3.insert(END, pred2[0])

    if(pred2[0]=="GERD"):
        z=urllib.request.urlopen('https://api.thingspeak.com/update?api_key=MP77HD9B13Z7N6BO&field1=1&field2=0&field3='+str(loc))
        z.read()
    if(pred2[0]=="Hepatitis C"):
        r=urllib.request.urlopen('https://api.thingspeak.com/update?api_key=MP77HD9B13Z7N6BO&field1=0&field2=1&field3='+str(loc))
        r.read()

symptoms=["stomach pain","vomiting"]
encoded_symptoms = [encode_symptom(symptom) for symptom in symptoms]
# print(encoded_symptoms)
illness_vector = create_illness_vector(encoded_symptoms)

        # perform diagnosis
# diagnosis_string = get_diagnosis(illness_vector)

