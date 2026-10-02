from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Literal, Annotated
import joblib
import pandas as pd
from pathlib import Path


# ============================================================
# Load ML Model
# ============================================================

with open('car_predictor.pkl','rb') as f:
    ml_model = joblib.load(f)


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="Car Price Prediction API",
    description="Machine Learning API for Car Price Category Prediction",
    version="1.0.0"
)


# ============================================================
# Home
# ============================================================

# @app.get("/")
# def home():
#     return {
#         "message": "Car Price Prediction API is running",
#         "status": "success"
#     }


# ============================================================
# Pydantic Model
# ============================================================

class UserInput(BaseModel):

    Brand: Annotated[
        Literal[
            "Hyundai",
            "Volkswagen",
            "Toyota",
            "Honda",
            "Maruti",
            "Mahindra",
            "Tata",
            "Kia"
        ],
        Field(description="Car Brand name")
    ]

    Car_Age: Annotated[
        int,
        Field(
            description="Enter your car age",
            gt=0,
            lt=15
        )
    ]

    Kilometers_Driven: Annotated[
        int,
        Field(description="How much car drove")
    ]

    Engine_CC: Annotated[
        int,
        Field(description="How much is Engine power")
    ]

    Mileage_KMPL: Annotated[
        float,
        Field(description="Enter car mileage in KMPL")
    ]

    Fuel_Type: Annotated[
        Literal[
            "Petrol",
            "Diesel",
            "CNG",
            "Electric"
        ],
        Field(description="Enter fuel type")
    ]

    Transmission: Annotated[
        Literal[
            "Manual",
            "Automatic"
        ],
        Field(description="Enter transmission type")
    ]

    Previous_Owners: Annotated[
        int,
        Field(description="Number of previous owners")
    ]

    Seats: Annotated[
        Literal[5, 7],
        Field(description="Enter number of seats")
    ]

    Location: Annotated[
        Literal[
            "Delhi",
            "Pune",
            "Kolkata",
            "Hyderabad",
            "Chennai",
            "Durgapur",
            "Bengaluru",
            "Mumbai"
        ],
        Field(description="Enter city name")
    ]


# ============================================================
# Prediction
# ============================================================

@app.post("/predict")
def predict_car_price(data: UserInput):

    input_data = pd.DataFrame([
        {
            "Brand": data.Brand,
            "Car_Age": data.Car_Age,
            "Kilometers_Driven": data.Kilometers_Driven,
            "Engine_CC": data.Engine_CC,
            "Mileage_KMPL": data.Mileage_KMPL,
            "Fuel_Type": data.Fuel_Type,
            "Transmission": data.Transmission,
            "Previous_Owners": data.Previous_Owners,
            "Seats": data.Seats,
            "Location": data.Location
        }
    ])

    prediction = ml_model.predict(input_data)[0]

    return {
        "predicted_category": str(prediction)
    }
