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

def CSVtoDataFrame(path:str,separator:str = ";")->pd.DataFrame : 
    """Imports CSV into a dataFrame and returns it

    Args: None

    Returns : dataFrame
    """
    return pd.read_csv(path,sep=separator)

def DataFrametoCSV(data,name):
    """Saves DataFrame in a CSV under /app/data

    Args: dataframe, name of the saved file (string)

    Returns : Nothing
    """
    data.to_csv("../data/"+name+".csv", index = False, header = True)

def generateLosingDraw(path:str):
    """Given the CSV at the path with only winning tickets, generates 4 losing tickets per winning tickets and return the result as dataframe

    Args: path (string) of the loaded CSV

    Returns : dataframe containing the winning tickets and randomly generated loosing tickets 
    """
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
    """Select meaningful fields to define X and Y for our model 

    Args: dataframe that has generated losing draws

    Returns : X(=[N1,N2,N3,N4,N5,E1,E2]) and Y(=1 if winning draw 0 else)
    """
    Y = data[['estGagnant']]
    X = data[['N1', 'N2', 'N3', 'N4', 'N5', 'E1', 'E2']]
    return (X,Y)

def split_train_test(X,Y): 
    """Takes a dataframe, splits it into sets for training / testing and returns the result 

    Args: X and Y

    Returns : X_train,Y_train,X_test,Y_test the separated pieces of data for training and testing the model
    """
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    return (X_train,Y_train,X_test,Y_test)

def predict_model(model,X_test,Y_test):
    """Uses a model to predict the chances of winning for each draw in the test Sample

    Args: model (random forest in our program), X_test : Input test sample, Y_test : output test sample

    Returns : pred : list of predictions for X_test, pred_proba : list of probabilities for guessed predictions,score : model accuracy
    """
    pred        = model.predict(X_test)
    pred_proba  = model.predict_proba(X_test)
    score       = model.score(X_test, np.ravel(Y_test))
    return (pred,pred_proba,score) 

def build_res_df(pred,pred_proba,X_test,Y_test) :
    """Builds a modified dataframe that adds probability columns for each winning draws according to our model

    Args: model (random forest in our program), X_test : Input test sample, Y_test : output test sample

    Returns : pred : list of predictions for X_test, pred_proba : list of probabilities for guessed predictions,score : model accuracy
    """
    df_res = pd.concat([X_test, Y_test], axis=1)
    df_res['estGagnant_pred'] = pred
    df_res['probaDeGagner'] = pred_proba[:,1]
    df_res = df_res.sort_values(by='probaDeGagner', ascending=False)
    df_res = df_res.set_index(np.arange(len(df_res)))
    return df_res

def get_winner(df, method) :
    """ From the previous generated dataframe, extract the winning draw (highest probability)

    Args: dataframe with probabilities of winning, model type (RandomForestClassifier by default) 

    Returns : dictionnary describing the winning the draw
    """   
    df['method'] = method     
    winner = df.head(1) 
    return winner.to_dict(orient="records")[0]

def predict_value(value,model):
    """ Given a draw, predict its probability of winning with our model

    Args: Draw as an array, AI model used (Random Forest in this program) 

    Returns : dictionnary describing the winning the draw
    """ 
    proba_perte,proba = model.predict_proba(value)[0]
    return dict ({
        "tirage" : value,
        "Proba gain": proba,
        "Proba perte": proba_perte,
    })

def serialize_model(model) : 
    """ Save the model in a pickel file

    Args: model : AI model used (Random Forest in this program)
   
    Returns : Nothing
    """ 
    output = open('app/data/model.pkl', 'wb')
    pickle.dump(model, output)
    output.close()

def load_model():
    """ Load the saved model from the pickel file

    Args: None
   
    Returns : model : AI model used (Random Forest in this program)
    """
    f = open('app/data/model.pkl', 'rb')
    model = pickle.load(f)
    f.close()
    return model

def get_metrics(model,X_test,Y_test):
    """ Get the metrics from a model

    Args: AI model used, X_test : inputs for test sample, Y_test : outputs for test sample
   
    Returns : { Name,Score,Params } of the loaded model
    """
    pred    = model.predict(X_test)
    score   = accuracy_score(Y_test, pred)
    params  = model.get_params()
    name    = type(model).__name__
    return {
        'name' : name,
        'score' : score, 
        'params' : params
    }

# Import data from CSV and split training and testing samples
generatedData = CSVtoDataFrame("app/data/Completed_EuroMillions.csv",",")
(X,Y) = getXandY(generatedData)
trainingTestSet = split_train_test(X,Y)

# Either load pickle file with model if it exists or create it, train the model and save it to the pickle file
if not(os.path.isfile("app/data/model.pkl")):
    modelAI = RandomForestClassifier(max_depth=2)
    modelAI.fit(trainingTestSet[0],np.ravel(trainingTestSet[1]))
    serialize_model(modelAI)
else :
    modelAI = load_model()
trainingRes = predict_model(modelAI,trainingTestSet[2],np.ravel(trainingTestSet[3]))   
