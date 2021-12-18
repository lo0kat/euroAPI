from fastapi import APIRouter
from ..model import dataModel

completedData = dataModel.CSVtoDataFrame("app/data/Completed_EuroMillions.csv",",")
train_test = dataModel.split_train_test(completedData)
forest = dataModel.random_Forest(*train_test)

router = APIRouter(
     prefix="/predict",
    tags=["predict"],
)

@router.get("")
async def read_predict():
    winner = dataModel.get_winner(dataModel.build_res_df(forest[0],forest[1],train_test[2],train_test[3]),'RandomForestClassifier')
    return winner


@router.post("")
async def read_predict(draw:dataModel.Draw):
    loaded_model = dataModel.load_model()
    return dataModel.predict_value([dataModel.drawToArray(draw)],loaded_model)