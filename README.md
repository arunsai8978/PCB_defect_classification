# PCB_defect_classification

PCB Defect Classification
1. Project Title
PCB defect classification using API with YOLO and Faster R-CNN.

2. Description
This project detects and classifies defects in Printed Circuit Boards (PCBs) using object detection algorithms. It processes images and annotations, and highlights defects like missing holes, mouse bites, open circuits, and more using pre-trained models such as YOLO and Faster R-CNN.

3. Installation and Setup Instructions
Clone the repository:
bash
Copy code
git clone <repo-url>
Install dependencies:
bash
Copy code
pip install -r requirements.txt
Ensure that your dataset is organized into the correct subfolders.
4. Usage
Run the main script for training the model and visulaize the predictions.
The script will parse XML annotations, count image and annotation files, and display images with annotated defects. It will then undergo an image preprocessing pipeline, data shuffling, data splitting, and Stratified K-Fold cross-validation. The model will be trained with a grid search of hyperparameters, and predictions will be obtained from the model. These predictions will be converted back to their original form to draw bounding boxes for visualization
5. Libraries used: 

FastAPI: For building the web API.
Pydantic: For data validation.
Torch: For deep learning and model training.
Torchvision: For object detection models and image transformations.
PIL (Pillow): For image handling and drawing.
Ultralytics YOLO: For YOLO object detection.
OpenCV: For image processing and visualization.
Pandas: For data manipulation.
NumPy: For numerical operations.
Matplotlib: For plotting.
Scikit-learn: For cross-validation and hyperparameter tuning.
Psycopg2: For PostgreSQL database interaction.
YAML: For configuration file handling.
Shutil: For file operations.
XML.etree.ElementTree: For parsing XML files.
Pathlib: For filesystem paths.
JSON: For JSON handling.
Collections: For data structures like Counter.
OS: For operating system interactions.




# PCB_API

## start the API
uvicorn main:app --host 0.0.0.0 --port 8000
cmd/terminal: curl -X POST "http://localhost:8000/predict/" -F "file=@D:/JOB/PCB_DATASET/images/Missing_hole/01_missing_hole_01.jpg"

# Call backs for predctions and vaisualization
Prediction Endpoint:
Endpoint: /predict/
Method: POST
Input: Upload an image file (UploadFile)
Output: JSON response with prediction and metrics.
Visualization Endpoint:
Endpoint: /visualize/
Method: POST
Input: Upload an image file (UploadFile)
Output: Visualized annotations of defects on the image.
