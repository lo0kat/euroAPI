from fastapi import APIRouter
from pydantic import BaseModel


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
     prefix="/model",
    tags=["model"],
)

@router.get("/")
async def read_model():
    return {"name":"Random Forest"}


@router.put("/")
async def read_model(draw:Draw):
    return {"Draw":draw}

@router.post("/retrain")
async def read_model():
    return "Model is being retrained"