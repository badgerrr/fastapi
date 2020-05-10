from fastapi import FastAPI
from forecast import get_baseline

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}

@app.get("/models/{model_id}")
def read_model(model_id: int, q: str = None, m: str = None):
    return {"model_id": model_id, "q": q, "m": m}

@app.get("/forecast/{seasonality}")
def run_forecast(seasonality: bool):
    import pickle
    pickle.dump(baseline, open('baseline.pkl', 'wb'))
    return get_baseline(seasonality)
