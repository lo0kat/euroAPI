import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from pydantic import BaseModel
from sklearn.metrics import accuracy_score
import pickle
import os.path

class Draw(BaseModel):
    N1:int
    N2:int
    N3:int
    N4:int
    N5:int
    E1:int
    E2:int

def drawToArray(draw:Draw):
    return [draw.N1,draw.N2,draw.N3,draw.N4,draw.N5,draw.E1,draw.E2]

"""
Imports CSV into a dataFrame and returns it
"""
def CSVtoDataFrame(path:str,separator:str = ";")->pd.DataFrame :
    return pd.read_csv(path,sep=separator)

"""
Saves DataFrame in a CSV under /app/data
"""
def DataFrametoCSV(data,name):
    data.to_csv("../data/"+name+".csv", index = False, header = True)

"""

"""
def generateLosingDraw(path:str):
    data = CSVtoDataFrame(path)
    data['estGagnant'] = [True]*len(data)
    for i in range(len(data)) :
        date = data.iloc[i]['Date']
        gain = data.iloc[i]['Gain']
        for j in range(4):
            random_5 = np.random.randint(low=1, high=50, size=5)
            while len(random_5) != len(np.unique(random_5)):
                random_5 = np.random.randint(low=1, high=50, size=5)
        
            random_2 = np.random.randint(low=1, high=12, size=2)
            while len(random_2) != len(np.unique(random_2)):
                random_2 = np.random.randint(low=1, high=12, size=2)

            data = data.append({
                'Date' : date,
                'N1' : random_5[0],
                'N2' : random_5[1],
                'N3' : random_5[2],
                'N4' : random_5[3],
                'N5' : random_5[4],
                'E1' : random_2[0],
                'E2' : random_2[1],
                'Winner' : 0,
                'estGagnant' : False,
                'Gain' : gain,
            }, ignore_index=True)
    data = data.sort_values(by='Date',ascending=True)
    return data
 

def getXandY(data):
    Y = data[['estGagnant']]
    X = data[['N1', 'N2', 'N3', 'N4', 'N5', 'E1', 'E2']]
    return (X,Y)

"""
Takes a dataframe, splits it into sets for training / testing and returns the result 
"""
def split_train_test(X,Y):  
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    return (X_train,Y_train,X_test,Y_test)


"""
Uses the random Forest algorithm to predict which draw has the highest chance of winning
"""
def random_Forest(X_train,Y_train,X_test,Y_test,maximum_depth=2):
    randomForest = RandomForestClassifier(max_depth=maximum_depth)
    #Training
    randomForest.fit(X_train,np.ravel(Y_train))

    #Prediction
    pred        = randomForest.predict(X_test)
    pred_proba  = randomForest.predict_proba(X_test)
    score       = randomForest.score(X_test, np.ravel(Y_test))
    return (pred,pred_proba,score)

def predict_model(model,X_test,Y_test):
    #Prediction
    pred        = model.predict(X_test)
    pred_proba  = model.predict_proba(X_test)
    score       = model.score(X_test, np.ravel(Y_test))
    return (pred,pred_proba,score) 

def build_res_df(pred,pred_proba,X_test,Y_test) :
    df_res = pd.concat([X_test, Y_test], axis=1)
    df_res['estGagnant_pred'] = pred
    df_res['probaDeGagner'] = pred_proba[:,1]
    df_res = df_res.sort_values(by='probaDeGagner', ascending=False)
    df_res = df_res.set_index(np.arange(len(df_res)))
    return df_res

# Extraction de la combinaison gagnante
def get_winner(df, method) :   
    df['method'] = method     
    winner = df.head(1) 
    return winner.to_dict(orient="records")[0]

def predict_value(value,model):
    proba_perte,proba = model.predict_proba(value)[0]
    return dict ({
        "tirage" : value,
        "Proba gain": proba,
        "Proba perte": proba_perte,
    })


def serialize_model(model) : 
    output = open('app/data/model.pkl', 'wb')
    pickle.dump(model, output)
    output.close()

def load_model():
    f = open('app/data/model.pkl', 'rb')
    model = pickle.load(f)
    f.close()
    return model

def get_metrics(model,X_test,Y_test):
    pred    = model.predict(X_test)
    score   = accuracy_score(Y_test, pred)
    params  = model.get_params()
    name    = type(model).__name__
    return {
        'name' : name,
        'score' : score, 
        'params' : params
    }


generatedData = CSVtoDataFrame("app/data/Completed_EuroMillions.csv",",")
(X,Y) = getXandY(generatedData)
trainingTestSet = split_train_test(X,Y)

if not(os.path.isfile("app/data/model.pkl")):
    modelAI = RandomForestClassifier(max_depth=2)
    modelAI.fit(trainingTestSet[0],np.ravel(trainingTestSet[1]))
    serialize_model(modelAI)
else :
    modelAI = load_model()

trainingRes = predict_model(modelAI,trainingTestSet[2],np.ravel(trainingTestSet[3]))   
