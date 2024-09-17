from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image, ImageDraw
import io
import psycopg2
import yaml
import numpy as np
from typing import List
from ultralytics import YOLO
from obj_detect import resize_images,load_yolo_model, load_rcnn_model, yolo_predict, rcnn_predict, rcnn_to_original_annot, yolo_to_original_annot, display_annotations
from typing import Dict

app = FastAPI()


# Load database configuration
def load_db_config(config_path: str) -> dict:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config['database']


db_config = load_db_config('config.yaml')


# Define a Pydantic model for the metrics
class Metric(BaseModel):
    image_name: str
    class_name: str
    confidence: float
    xmin: int
    ymin: int
    xmax: int
    ymax: int


def save_metrics_to_db(metrics: List[Metric], db_config: dict, model_type: str):
    try:
        conn = psycopg2.connect(
            dbname=db_config['dbname'],
            user=db_config['user'],
            password=db_config['password'],
            host=db_config['host'],
            port=db_config['port']
        )
        cursor = conn.cursor()

        if model_type == 'yolo':
            table_name = 'yolo_prediction_metrics'
        elif model_type == 'rcnn':
            table_name = 'rcnn_prediction_metrics'
        else:
            raise ValueError("Invalid model_type. Must be 'yolo' or 'rcnn'.")

        insert_query = f"""
        INSERT INTO {table_name} (image_name, class_name, confidence, xmin, ymin, xmax, ymax)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """

        for metric in metrics:
            cursor.execute(insert_query, (
                metric.image_name,
                metric.class_name,
                metric.confidence,
                metric.xmin,
                metric.ymin,
                metric.xmax,
                metric.ymax
            ))

        conn.commit()
        cursor.close()
        conn.close()
        print("Metrics saved to database successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

predictions_store: Dict[str, Dict] = {}  # Store predictions by file name

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Load models
        image_file = file


        #preprocessing
        preprocessed_file = resize_images(image_file, size=(320, 320))

        #model predction
        yolo_model_path = 'path_to_yolo_model.pt'
        fasterrcnn_model_path = 'path_to_fasterrcnn_model.pt'
        yolo_model = load_yolo_model(yolo_model_path)
        fasterrcnn_model = load_rcnn_model(fasterrcnn_model_path)

        # Perform predictions
        yolo_predictions = yolo_predict(yolo_model, preprocessed_file)
        fasterrcnn_predictions = rcnn_predict(fasterrcnn_model, preprocessed_file)

        # Define class names
        classes = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']

        # Convert predictions to metrics
        yolo_metrics = yolo_to_original_annot(file.filename, yolo_predictions, classes)
        fasterrcnn_metrics = rcnn_to_original_annot(file.filename, fasterrcnn_predictions, classes)

        # Save metrics to the database
        save_metrics_to_db(yolo_metrics + fasterrcnn_metrics, db_config)

        # Store predictions
        predictions_store[file.filename] = {
            "yolo": yolo_predictions,
            "fasterrcnn": fasterrcnn_predictions
        }

        return JSONResponse(content={"message": "Predictions and metrics saved successfully."}, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


@app.post("/visualize/")
async def visualize(file: UploadFile = File(...)):
    try:
        image_name = file

        # Retrieve stored predictions
        yolo_predictions = predictions_store[file.filename]["yolo"]
        fasterrcnn_predictions = predictions_store[file.filename]["fasterrcnn"]

        display_annotations(image_name, yolo_predictions)
        display_annotations(image_name, fasterrcnn_predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


#######################

'''      uvicorn main:app --host 0.0.0.0 --port 8000   '''
