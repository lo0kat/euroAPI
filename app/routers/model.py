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
    dataModel.X = dataModel.X.append({
            "N1" : draw.N1,
            "N2" : draw.N2,
            "N3" : draw.N3,
            "N4" : draw.N4,
            "N5" : draw.N5,
            "E1" : draw.E1,
            "E2" : draw.E2,
        }, ignore_index=True)
    dataModel.Y = dataModel.Y.append({
            "estGagnant" : 1
        }, ignore_index=True)

    return {"Draw":draw}

@router.post("/retrain")
async def read_model():
    dataModel.trainingTestSet = dataModel.split_train_test(dataModel.X,dataModel.Y)
    dataModel.modelAI.fit(dataModel.trainingTestSet[0],np.ravel(dataModel.trainingTestSet[1]))
    dataModel.serialize_model(dataModel.modelAI)
    return "Le modèle a été réentrainé !"