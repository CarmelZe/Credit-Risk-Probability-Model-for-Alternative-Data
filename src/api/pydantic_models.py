from pydantic import BaseModel
from typing import List

class PredictionInput(BaseModel):
    feature1: float
    feature2: float
    # Add all your model features here
    
class PredictionOutput(BaseModel):
    risk_probability: float
    risk_class: int