from fastapi import APIRouter
from pydantic import BaseModel


class Model(BaseModel):
    name : str

    

router = APIRouter(
     prefix="/predict",
    tags=["predict"],
)

@router.get("/")
async def read_predict():
    return "Prediction"


@router.post("/")
async def read_predict(model:Model):
    return {"Model":model}