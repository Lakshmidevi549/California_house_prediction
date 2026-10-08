from pathlib import Path

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

BASE_DIR = Path(__file__).resolve().parent

# Change these to match your actual model filename.
MODEL_FILENAMES = (
    "california.joblib",
    "california_model.joblib",
    "california_model.pkl",
)

app = FastAPI(title="California Home Value Predictor")

app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static",
)

templates = Jinja2Templates(directory=BASE_DIR / "templates")


def load_classifier():
    model_path = next(
        (BASE_DIR / filename for filename in MODEL_FILENAMES
         if (BASE_DIR / filename).is_file()),
        None,
    )

    if model_path is None:
        raise FileNotFoundError(
            "Model file not found. Put your model beside app.py and "
            f"add its filename to MODEL_FILENAMES: {MODEL_FILENAMES}"
        )

    loaded = joblib.load(model_path)

    if hasattr(loaded, "predict"):
        return loaded

    if isinstance(loaded, dict):
        model = next(
            (value for value in loaded.values() if hasattr(value, "predict")),
            None,
        )
        if model is not None:
            return model

        raise ValueError(
            f"No object with predict() found in {model_path.name}. "
            f"Dictionary keys: {list(loaded.keys())}"
        )

    raise TypeError(
        f"Unsupported model object in {model_path.name}: {type(loaded)}"
    )


classifier = load_classifier()


@app.get("/", response_class=HTMLResponse)
def main_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request},
    )


@app.get("/predict")
def predict(
    MedInc: float,
    HouseAge: float,
    AveRooms: float,
    Population: float,
    AveOccup: float,
    Latitude: float,
):
    input_data = [[
        MedInc,
        HouseAge,
        AveRooms,
        Population,
        AveOccup,
        Latitude,
    ]]

    try:
        prediction = classifier.predict(input_data)
        return {"prediction": float(prediction[0])}
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {exc}",
        ) from exc


@app.post("/predict_file")
def predict_file(file: UploadFile = File(...)):
    if not file.filename or Path(file.filename).suffix.lower() != ".csv":
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file.",
        )

    try:
        df_test = pd.read_csv(file.file)
        predictions = classifier.predict(df_test)
        return {"predictions": predictions.tolist()}
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not process CSV: {exc}",
        ) from exc


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)