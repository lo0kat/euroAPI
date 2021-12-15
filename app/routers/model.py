from fastapi import APIRouter
from pydantic import BaseModel

class Model(BaseModel):
    name : str


router = APIRouter(
     prefix="/model",
    tags=["model"],
)

@router.get("")
async def read_model(model : Model):
    return {"Model":model}


@router.put("")
async def read_model(model:Model):
    return {"Model":model}

@router.post("/retrain")
async def read_model():
    return "Model is being retrained"