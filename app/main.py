from fastapi import FastAPI
from app.train import train_router
from app.predict import predict_router

app = FastAPI(title="ML Model API", version="1.0")

app.include_router(train_router, prefix="/train", tags=["training"])
app.include_router(predict_router, prefix="/predict", tags=["prediction"])

@app.get("/")
def health():
    return {"status": "ok"}
