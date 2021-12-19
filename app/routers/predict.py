from fastapi import APIRouter
from ..model import dataModel


router = APIRouter(
     prefix="/predict",
    tags=["predict"],
)

@router.get("")
async def read_predict():
    winner = dataModel.get_winner(dataModel.build_res_df(dataModel.trainingRes[0],dataModel.trainingRes[1],dataModel.trainingTestSet[2],dataModel.trainingTestSet[3]),'RandomForestClassifier')
    return winner


@router.post("")
async def read_predict(draw:dataModel.Draw):
    return dataModel.predict_value([dataModel.drawToArray(draw)],dataModel.modelAI)