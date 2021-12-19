from fastapi import APIRouter
from ..model import dataModel

router = APIRouter(
     prefix="/predict",
    tags=["predict"],
)

@router.get("")
async def generate_winning_draw():
    """Generates a draw that has a very high chance of winning according to the loaded model

    Args: None

    Returns : the winning draw
    """
    winner = dataModel.get_winner(dataModel.build_res_df(dataModel.trainingRes[0],dataModel.trainingRes[1],dataModel.trainingTestSet[2],dataModel.trainingTestSet[3]),'RandomForestClassifier')
    return winner

@router.post("")
async def predict_draw(draw:dataModel.Draw):
    """ Predicts if a given draw will win according to the model and gives the probability of such event happening

    Args: Draw

    Returns : Both Losing and winning probability of this draw
    """
    return dataModel.predict_value([dataModel.drawToArray(draw)],dataModel.modelAI)