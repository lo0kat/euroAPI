from fastapi import APIRouter
from ..model import dataModel
import numpy as np

router = APIRouter(
     prefix="/model",
    tags=["model"],
)

@router.get("")
async def read_model():
    loaded_model = dataModel.load_model()
    return dataModel.get_metrics(loaded_model,dataModel.trainingTestSet[2],dataModel.trainingTestSet[3])


@router.put("")
async def read_model(draw: dataModel.Draw):
    return {"Draw":draw}

@router.post("/retrain")
async def read_model():
    dataModel.trainingTestSet = dataModel.split_train_test(dataModel.X,dataModel.Y)
    dataModel.modelAI.fit(dataModel.trainingTestSet[0],np.ravel(dataModel.trainingTestSet[1]))
    dataModel.serialize_model(dataModel.modelAI)
    
    return "Le modèle a été réentrainé !"