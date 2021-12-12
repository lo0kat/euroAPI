from fastapi import FastAPI
from .routers import predict,model

app = FastAPI()

app.include_router(predict.router,prefix="/api")
app.include_router(model.router,prefix="/api")

@app.get("/")
async def root():
    return {"message": "Welcome to the main page!"}
