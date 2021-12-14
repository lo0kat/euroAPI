from fastapi import APIRouter
from pydantic import BaseModel
from ..model import dataModel

completedData = dataModel.CSVtoDataFrame("~/Microservices/euroAPI/app/data/Completed_EuroMillions.csv",",")
train_test = dataModel.split_train_test(completedData)
forest = dataModel.random_Forest(*train_test)
winner = dataModel.get_winner(dataModel.build_res_df(*forest,train_test[2],train_test[3]),'RandomForestClassifier')

class Draw(BaseModel):
    Date:str
    N1:int
    N2:int
    N3:int
    N4:int
    N5:int
    N6:int
    E1:int
    E2:int
    Winner:int
    Gain : int
    

router = APIRouter(
     prefix="/predict",
    tags=["predict"],
)

@router.get("/")
async def read_predict():
    return winner.values.tolist()


@router.post("/")
async def read_predict(draw:Draw):
    return {"Model":model}