from typing import Optional
from enum import Enum
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ModelName(str,Enum):
    alexnet="alexnet"
    resnet="resnet"
    lenet="lenet"

class Item(BaseModel) :
    nom : str
    description : Optional[str] = None
    price : float
    tax : Optional[float] = None

@app.post("/items/")
async def create_item(item : Item):
    return item

@app.get("/models/{model_name}")
async def get_model(model_name : ModelName):
    if model_name == ModelName.alexnet:
        return {"model_name" : model_name,"message": "DeepL for the win !"}
    if model_name == ModelName.lenet:
        return {"model_name" : model_name,"message": "LeCNN all the images!"}
    return {"model_name" : model_name,"message": "Have some residuals"}

@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
async def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}
