from fastapi import APIRouter
from ..model import dataModel

router = APIRouter(
     prefix="/model",
    tags=["model"],
)

completedData = dataModel.CSVtoDataFrame("app/data/Completed_EuroMillions.csv",",")
train_test = dataModel.split_train_test(completedData)


@router.get("")
async def read_model():
    loaded_model = dataModel.load_model()
    return dataModel.get_metrics(loaded_model,train_test[2],train_test[3])


@router.put("")
async def read_model(model: dataModel.Model):
    return {"Model":model}

@router.post("/retrain")
async def read_model():
    return "Model is being retrained"