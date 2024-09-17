import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import shutil
import random
import xml.etree.ElementTree as ET
import yaml
from pathlib import Path
from collections import Counter
import json
from sklearn.model_selection import ParameterGrid
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from ultralytics import YOLO
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.optim as optim




def main():
    dataset_dir = "D:\JOB\PCB_DATASET"


    def checking_images_labels(folder_path):
        # Get list of all files in the folder
        files = os.listdir(folder_path)

        # Count the number of files
        num_files = len(files)

        return num_files

    subfolders = ['Missing_hole', 'Mouse_bite', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper']
    labels = ['Missing_hole', 'Mouse_bite', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper']
    images_dir = os.path.join(dataset_dir, 'images')
    annot_dir = os.path.join(dataset_dir, 'Annotations')

    for subfolder in subfolders:
        images_path = os.path.join(images_dir, subfolder)
        annot_path = os.path.join(annot_dir, subfolder)
        print(f'{subfolder}:\t\ images :{checking_images_labels(images_path)} \t\
                annotations:{checking_images_labels(annot_path)} ')


    def parse_xml(xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        data = []

        filename = root.find('filename').text
        width = int(root.find('size/width').text)
        height = int(root.find('size/height').text)

        for obj in root.findall('object'):
            name = obj.find('name').text
            xmin = int(obj.find('bndbox/xmin').text)
            ymin = int(obj.find('bndbox/ymin').text)
            xmax = int(obj.find('bndbox/xmax').text)
            ymax = int(obj.find('bndbox/ymax').text)

            data.append({
                'filename': filename,
                'width': width,
                'height': height,
                'class': name,
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax
            })

        return data

    # List to store parsed data from all XML files
    all_data = []

    # Recursively traverse subdirectories
    for root, dirs, files in os.walk(annot_dir):
        for name in files:
            if name.endswith('.xml'):
                xml_path = os.path.join(root, name)
                all_data.extend(parse_xml(xml_path))

    annot_df = pd.DataFrame(all_data)

    def determine_subfolder(image_name):
        categories = {
            'missing': 'Missing_hole',
            'mouse': 'Mouse_bite',
            'open': 'Open_circuit',
            'short': 'Short',
            'spur': 'Spur',
            'spurious': 'Spurious_copper'
        }

        for key, value in categories.items():
            if key in image_name.split('_'):
                return value
        return None

    def display_annotations(image_name, images_directory, annotations_df, use_subfolder=False):
        # Construct the image file path
        if use_subfolder:
            folder = determine_subfolder(image_name)
            image_path = os.path.join(images_directory, folder, image_name)
        else:
            image_path = os.path.join(images_directory, image_name)

        # Load the image
        image = cv2.imread(image_path)

        # Retrieve relevant annotations
        image_annotations = annotations_df[annotations_df['filename'] == image_name]

        # Draw each annotation on the image
        for _, annotation in image_annotations.iterrows():
            x_min, y_min, x_max, y_max = annotation['xmin'], annotation['ymin'], annotation['xmax'], annotation['ymax']
            label = annotation['class']

            # Append confidence if available
            confidence_score = annotation.get('confidence')
            if confidence_score is not None:
                label += f" ({confidence_score:.2f})"

            # Set color and draw rectangle
            rectangle_color = (255, 255, 255)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), rectangle_color, 3)

            # Add background and text
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 1.2, 2)[0]
            cv2.rectangle(image, (x_min, y_min - text_size[1] - 5), (x_min + text_size[0], y_min - 1), rectangle_color,-1)
            cv2.putText(image, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_COMPLEX, 1,2 ,(0,0,0), 2)

        # Convert image to RGB for matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Display the image with annotations
        plt.figure(figsize=(18, 10))
        plt.imshow(image_rgb)
        plt.axis('off')
        plt.title('Annotations')
        plt.text(10, image_rgb.shape[0] + 100, f'Image: {image_name}',
                 color='black', fontsize=11, ha='left')
        plt.show()

        return image

    # Define the image file name
    file_name = '01_missing_hole_01.jpg'

    # Call the function to display annotations with subfolder organization
    display_annotations(file_name, images_dir, annot_df, use_subfolder=True)

    def resize_images(input_folder, output_folder, size):
        """
        Resize images in the input folder and save them to the output folder.

        Parameters:
        - input_folder (str): Path to the directory with the original images.
        - output_folder (str): Path to the directory where resized images will be saved.
        - size (tuple): Desired size for the images in (width, height).
        """
        os.makedirs(output_folder, exist_ok=True)  # Ensure output directory exists

        for directory, _, filenames in os.walk(input_folder):
            for filename in filenames:
                if filename.lower().endswith(('.jpg', '.png', '.jpeg')):  # Support common image formats
                    image_path = os.path.join(directory, filename)
                    image = cv2.imread(image_path)

                    if image is None:
                        print(f"Warning: Failed to load image {image_path}. Skipping...")
                        continue

                    # Resize the image
                    resized_image = cv2.resize(image, size)

                    # Apply noise reduction
                    denoised_image = cv2.GaussianBlur(resized_image, (5, 5), 0)

                    # Normalize the image
                    normalized_image = denoised_image / 255.0  # Scale to range [0, 1]

                    # Save the processed image
                    save_path = os.path.join(output_folder, filename)
                    cv2.imwrite(save_path,
                                (normalized_image * 255).astype(np.uint8))  # Convert back to uint8 for saving

    # Define directories for processed images
    processed_images_320 = os.path.join(dataset_dir, 'processed_320')
    processed_images_640 = os.path.join(dataset_dir, 'processed_640')

    # Process images with noise reduction and normalization
    resize_images(images_dir, processed_images_320, size=(320, 320))
    resize_images(images_dir, processed_images_640, size=(640, 640))

    def adjust_annotations(annotations_df, new_size):
        """
        Adjust the bounding box annotations according to the new image size.

        Parameters:
        - annotations_df (pd.DataFrame): DataFrame with the original annotations.
        - new_size (tuple): New size for scaling (width, height).

        Returns:
        - pd.DataFrame: DataFrame with updated annotations.
        """
        updated_annotations = []

        for _, row in annotations_df.iterrows():
            # Compute scaling factors
            width_scale = new_size[0] / row['width']
            height_scale = new_size[1] / row['height']

            # Scale bounding box coordinates
            scaled_xmin = int(row['xmin'] * width_scale)
            scaled_ymin = int(row['ymin'] * height_scale)
            scaled_xmax = int(row['xmax'] * width_scale)
            scaled_ymax = int(row['ymax'] * height_scale)

            # Append resized annotation
            updated_annotations.append({
                'filename': row['filename'],
                'width': new_size[0],
                'height': new_size[1],
                'class': row['class'],
                'xmin': scaled_xmin,
                'ymin': scaled_ymin,
                'xmax': scaled_xmax,
                'ymax': scaled_ymax
            })

        return pd.DataFrame(updated_annotations)

    # Define sizes for annotation scaling
    size_320 = (320, 320)
    size_640 = (640, 640)

    # Resize annotations for specified sizes
    annotations_resized_320 = adjust_annotations(annot_df, new_size=size_320)
    annotations_resized_640 = adjust_annotations(annot_df, new_size=size_640)

    def convert_annotations_to_yolo_format(df_annotations, class_list, target_dimensions):
        """
        Convert bounding box annotations from a DataFrame to YOLO format.

        Parameters:
        - df_annotations (pd.DataFrame): DataFrame containing bounding box annotations.
        - class_list (list): List of class names, used for mapping to class indices.
        - target_dimensions (tuple): Target dimensions for scaling (width, height).

        Returns:
        - list of tuples: Each tuple contains (filename, class_index, x_center, y_center, bbox_width, bbox_height).
        """
        yolo_format_labels = []

        for _, row in df_annotations.iterrows():
            image_filename = row['filename']
            img_width, img_height = row['width'], row['height']
            label_class = row['class']
            x_min, y_min, x_max, y_max = row['xmin'], row['ymin'], row['xmax'], row['ymax']

            # Calculate YOLO format coordinates
            x_center = (x_min + x_max) / (2 * img_width)
            y_center = (y_min + y_max) / (2 * img_height)
            bbox_width = (x_max - x_min) / img_width
            bbox_height = (y_max - y_min) / img_height

            # Find the index of the class
            class_id = class_list.index(label_class)

            # Store in YOLO format
            yolo_format_labels.append((image_filename, class_id, x_center, y_center, bbox_width, bbox_height))

        return yolo_format_labels

    def split_coco_directory(images_dir, annot_df, output_dir, train_split=0.85, val_split=0.10):
        """
        Split images and annotations into training, validation, and test sets, and save them in COCO format.

        Parameters:
        - images_dir (str): Directory containing images.
        - annot_df (pd.DataFrame): DataFrame with bounding box annotations.
        - output_dir (str): Directory where the split dataset will be saved.
        - train_split (float): Proportion of the dataset to use for training.
        - val_split (float): Proportion of the dataset to use for validation.
        """
        # Create output directories
        os.makedirs(os.path.join(output_dir, 'coco/images/train'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'coco/images/val'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'coco/images/test'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'coco/annotations'), exist_ok=True)

        # Group annotations by image filename
        image_annotations = {}
        for _, row in annot_df.iterrows():
            filename = row['filename']
            if filename not in image_annotations:
                image_annotations[filename] = []
            image_annotations[filename].append(row)

        # Shuffle image filenames
        image_filenames = list(image_annotations.keys())
        random.shuffle(image_filenames)

        # Split dataset
        num_images = len(image_filenames)
        num_train = int(num_images * train_split)
        num_val = int(num_images * val_split)

        train_filenames = image_filenames[:num_train]
        val_filenames = image_filenames[num_train:num_train + num_val]
        test_filenames = image_filenames[num_train + num_val:]

        # Function to copy images and collect annotations
        def prepare_data(filenames, dataset_name):
            annotations = []
            for filename in filenames:
                shutil.copy(os.path.join(images_dir, filename),
                            os.path.join(output_dir, f'coco/images/{dataset_name}/{filename}'))
                for annot in image_annotations[filename]:
                    annotations.append(annot)
            return annotations

        # Prepare data for each split
        train_annotations = prepare_data(train_filenames, 'train')
        val_annotations = prepare_data(val_filenames, 'val')
        test_annotations = prepare_data(test_filenames, 'test')

        # Helper function to convert annotations to COCO format
        def convert_to_coco_format(annotations, output_file):
            coco_data = {
                "images": [],
                "annotations": [],
                "categories": []
            }

            categories = list(set([ann['class'] for ann in annotations]))
            coco_data['categories'] = [{'id': idx, 'name': label} for idx, label in enumerate(categories)]

            img_id = 1
            ann_id = 1
            image_registry = {}

            for annot in annotations:
                filename = annot['filename']
                width, height = annot['width'], annot['height']
                label = annot['class']
                x_min, y_min, x_max, y_max = annot['xmin'], annot['ymin'], annot['xmax'], annot['ymax']

                if filename not in image_registry:
                    image_registry[filename] = img_id
                    coco_data['images'].append({
                        "id": img_id,
                        "file_name": filename,
                        "width": width,
                        "height": height
                    })
                    img_id += 1

                category_id = next(cat['id'] for cat in coco_data['categories'] if cat['name'] == label)

                coco_data['annotations'].append({
                    "id": ann_id,
                    "image_id": image_registry[filename],
                    "category_id": category_id,
                    "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                    "area": (x_max - x_min) * (y_max - y_min),
                    "iscrowd": 0
                })
                ann_id += 1

            with open(output_file, 'w') as f:
                json.dump(coco_data, f, indent=4)

        # Convert annotations to COCO format for each split
        convert_to_coco_format(train_annotations, os.path.join(output_dir, 'coco/annotations/instances_train.json'))
        convert_to_coco_format(val_annotations, os.path.join(output_dir, 'coco/annotations/instances_val.json'))
        convert_to_coco_format(test_annotations, os.path.join(output_dir, 'coco/annotations/instances_test.json'))


    split_coco_directory(annotations_resized_320, annotations_resized_320, processed_images_320)
    split_coco_directory(annotations_resized_640, annotations_resized_640, processed_images_640)

    def partition_data(images_path, annotations, destination_dir, train_ratio=0.85, validation_ratio=0.10):
        """
        Partition images and their corresponding labels into training, validation, and test sets.

        Parameters:
        - images_path (str): Directory where the original images are stored.
        - annotations (list): List of tuples where each tuple contains filename, class index, center x, center y, bbox width, and bbox height.
        - destination_dir (str): Directory where the split data will be saved.
        - train_ratio (float): Ratio of images used for training.
        - validation_ratio (float): Ratio of images used for validation.
        """
        # Create necessary directories
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(destination_dir, f'images/{split}'), exist_ok=True)
            os.makedirs(os.path.join(destination_dir, f'labels/{split}'), exist_ok=True)

        # Organize annotations by image filenames
        file_annotations = {}
        for entry in annotations:
            img_filename, class_id, center_x, center_y, width, height = entry
            if img_filename not in file_annotations:
                file_annotations[img_filename] = []
            file_annotations[img_filename].append(entry)

        # Shuffle image filenames for randomness
        filenames = list(file_annotations.keys())
        random.shuffle(filenames)

        # Determine split indices
        total_files = len(filenames)
        train_end = int(total_files * train_ratio)
        val_end = int(total_files * (train_ratio + validation_ratio))

        train_files = filenames[:train_end]
        val_files = filenames[train_end:val_end]
        test_files = filenames[val_end:]

        # Display split information
        print(f"Training set: {len(train_files)} images")
        print(f"Validation set: {len(val_files)} images")
        print(f"Test set: {len(test_files)} images")

        # Save split data
        for partition, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
            for file in files:
                # Save annotations
                with open(os.path.join(destination_dir, f'labels/{partition}/{os.path.splitext(file)[0]}.txt'),
                          'w') as label_file:
                    for entry in file_annotations[file]:
                        _, class_id, center_x, center_y, width, height = entry
                        label_file.write(f"{class_id} {center_x} {center_y} {width} {height}\n")
                # Copy image files
                shutil.copy(os.path.join(images_path, file),
                            os.path.join(destination_dir, f'images/{partition}/{file}'))

    # Define directories for resized images and output data
    resized_images_320 = os.path.join(dataset_dir, 'resized_images_320')
    resized_images_640 = os.path.join(dataset_dir, 'resized_images_640')
    output_data_320 = os.path.join(dataset_dir, 'data_output_320')
    output_data_640 = os.path.join(dataset_dir, 'data_output_640')

    # Resize images to the specified dimensions
    resize_images(images_dir, resized_images_320, target_size=(320, 320))
    resize_images(images_dir, resized_images_640, target_size=(640, 640))

    # Convert annotations to YOLO format for both image sizes
    yolo_annotations_320 = convert_annotations_to_yolo_format(annotations_resized_320, labels, target_size=(320, 320))
    yolo_annotations_640 = convert_annotations_to_yolo_format(annotations_resized_640, labels, target_size=(640, 640))

    # Partition the data and save the split images and labels for both sizes
    partition_data(resized_images_320, yolo_annotations_320, output_data_320)
    partition_data(resized_images_640, yolo_annotations_640, output_data_640)

    # Directories for different image sizes
    resized_images_320 = os.path.join(dataset_dir, 'resized_images_320')
    resized_images_640 = os.path.join(dataset_dir, 'resized_images_640')

    dataset_paths = ['processed_320', 'processed_640']
    cls_idx = list(range(len(labels)))

    for dataset_path in dataset_paths:
        full_path = Path(dataset_path)
        print(f"Checking dataset directory: {full_path}")

        labels_path = full_path / 'labels' / 'train'
        print(f"Labels directory: {labels_path}")

        if not labels_path.exists():
            print(f"Labels directory does not exist: {labels_path}")
            continue

        labels_files = sorted(labels_path.rglob("*.txt"))  # Adjust based on your actual file type
        print(f"Labels found: {labels_files}")

        if not labels_files:
            print("No labels found.")
            continue  # Skip this dataset if no labels found

        # Create DataFrame to hold label counts
        indx = [l.stem for l in labels_files]
        labels_df = pd.DataFrame([], columns=cls_idx, index=indx)
        for label_file in labels_files:
            lbl_counter = Counter()
            with open(label_file, 'r') as lf:
                lines = lf.readlines()
            for line in lines:
                lbl_counter[int(line.split(' ')[0])] += 1
            labels_df.loc[label_file.stem] = lbl_counter

        labels_df = labels_df.fillna(0.0)

        # Use Stratified K-Fold with 5 splits
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=20)
        X = labels_df.index
        y = labels_df.max(axis=1)  # Using the maximum class count for stratification

        skf_splits = list(skf.split(X, y))

        # Create directory to save split data
        save_path = full_path / '5fold_crossval'
        save_path.mkdir(parents=True, exist_ok=True)

        # Initialize fold index and data storage
        for fold_idx, (train_indices, val_indices) in enumerate(skf_splits):
            fold_name = f'fold_{fold_idx + 1}'
            split_dir = save_path / fold_name
            split_dir.mkdir(parents=True, exist_ok=True)
            (split_dir / 'train').mkdir(parents=True, exist_ok=True)
            (split_dir / 'train' / 'images').mkdir(parents=True, exist_ok=True)
            (split_dir / 'train' / 'labels').mkdir(parents=True, exist_ok=True)
            (split_dir / 'val').mkdir(parents=True, exist_ok=True)
            (split_dir / 'val' / 'images').mkdir(parents=True, exist_ok=True)
            (split_dir / 'val' / 'labels').mkdir(parents=True, exist_ok=True)

            # Create YAML file for dataset
            dataset_yaml = split_dir / f'{fold_name}_dataset.yaml'
            with open(dataset_yaml, 'w') as ds_y:
                yaml.safe_dump({
                    'path': split_dir.as_posix(),
                    'train': 'train',
                    'val': 'val',
                    'names': labels
                }, ds_y)

            # Create fold label distribution DataFrame
            fold_lbl_distrb = pd.DataFrame(index=[fold_name], columns=cls_idx)
            train_totals = labels_df.iloc[train_indices].sum()
            val_totals = labels_df.iloc[val_indices].sum()
            ratio = val_totals / (train_totals + 1E-7)
            fold_lbl_distrb.loc[fold_name] = ratio

            # Save fold label distribution to CSV
            fold_lbl_distrb.to_csv(save_path / f'kfold_label_distribution_{dataset_path}.csv')

            # Get all images
            images = sorted(full_path.rglob(f"images/train/*.jpg"))
            print(f"Images found: {images}")

            # Copy images and labels to corresponding directories
            for image in images:
                image_stem = image.stem
                if image_stem in labels_df.index:
                    split = 'train' if image_stem in labels_df.index[train_indices] else 'val'
                    img_to_path = split_dir / split / 'images'
                    lbl_to_path = split_dir / split / 'labels'

                    src_image_path = image
                    src_label_path = labels_path / f"{image_stem}.txt"

                    dest_image_path = img_to_path / image.name
                    dest_label_path = lbl_to_path / f"{image_stem}.txt"

                    img_to_path.mkdir(parents=True, exist_ok=True)
                    lbl_to_path.mkdir(parents=True, exist_ok=True)

                    if src_image_path.exists() and src_label_path.exists():
                        shutil.copy(src_image_path, dest_image_path)
                        shutil.copy(src_label_path, dest_label_path)
                    else:
                        print(f"Source file(s) missing: {src_image_path}, {src_label_path}")
                else:
                    print(f"Warning: No fold information for {image_stem}. Skipping.")

        # Save the fold data split to CSV
        folds_df = pd.DataFrame(index=labels_df.index, columns=['fold'])
        for fold_idx, (train_indices, val_indices) in enumerate(skf_splits):
            fold_name = f'fold_{fold_idx + 1}'
            folds_df.loc[labels_df.index[train_indices], 'fold'] = 'train'
            folds_df.loc[labels_df.index[val_indices], 'fold'] = 'val'
        folds_df.to_csv(save_path / f'kfold_datasplit_{dataset_path}.csv')

    # Define image sizes and corresponding model instances
    image_sizes = [320, 640]
    models = {size: YOLO('yolov8s.pt') for size in image_sizes}  # One model instance for each size
    results = {}

    models_dir = dataset_dir / 'models'  # Directory to save models

    # Function to get dataset YAML path for a given image size
    def get_dataset_yaml(size):
        return dataset_dir / f'output_{size}' / '1fold_crossval' / 'fold_1' / 'fold_1_dataset.yaml'

    # Function to save the trained YOLO model
    def save_model(model, size, models_dir):
        """
        Save the trained YOLO model to a specified directory.

        Parameters:
        - model: The YOLO model to be saved.
        - size: The image size (e.g., 320 or 640) to include in the model filename.
        - models_dir: The directory where the model will be saved.
        """
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / f'yolo_{size}.pt'
        model.save(model_path)
        print(f"Saved YOLO model for image size {size} to {model_path}")

    param_grid = {
        'batch': [16, 32],
        'epochs': [10, 20],
        'mixup': [0.1, 0.3],
        'imgsz': [320, 640]
    }

    # Create a grid of all possible hyperparameter combinations
    grid = ParameterGrid(param_grid)

    batch = 16
    project = 'pcb'
    epochs = 1
    save_period = 1
    verbose = True
    mixup = 0.3
    patience = 5  # Number of epochs to wait for improvement

    def train_and_evaluate(params, image_size):
        batch = params['batch']
        epochs = params['epochs']
        mixup = params['mixup']

        model = YOLO('yolov8s.pt')
        dataset_yaml = get_dataset_yaml(image_size)

        best_val_metric = 0
        patience_counter = 0

        for epoch in range(epochs):
            print(f"Training epoch {epoch + 1}/{epochs} for image size {image_size}")

            model.train(data=dataset_yaml,
                        epochs=1,  # Train for one epoch at a time to facilitate early stopping
                        batch=batch,
                        imgsz=image_size,
                        save_period=save_period,
                        verbose=verbose,
                        project=project,
                        mixup=mixup)

            if hasattr(model, 'metrics'):
                metrics_obj = model.metrics
                val_metric = metrics_obj.results_dict.get('metrics/mAP50(B)', 0)
                print(f"Validation mAP50 after epoch {epoch + 1}: {val_metric}")

                if val_metric > best_val_metric:
                    best_val_metric = val_metric
                    patience_counter = 0
                    save_model(model, image_size, models_dir)
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1} for image size {image_size} due to no improvement")
                    break

        return best_val_metric

    results = {}
    for image_size in image_sizes:
        print(f"\nStarting grid search for image size: {image_size}")
        for params in grid:
            print(f"Evaluating params: {params}")
            metric = train_and_evaluate(params, image_size)
            results[str(params)] = metric

    print("Grid search completed.")
    print(results)

    # Define the source and destination directories
    results_dir = Path('D:/JOB/PCB_DATASET/results')  # Adjust this path if necessary
    dest_results_dir = Path(dataset_dir) / 'results'

    # Check if the destination directory exists
    if dest_results_dir.exists():
        # Remove the existing directory and its contents
        shutil.rmtree(dest_results_dir)

    # Copy the source directory to the destination directory
    shutil.copytree(results_dir, dest_results_dir)

    # Load and process the results CSV
    results_df = pd.read_csv(dest_results_dir / 'results.csv')
    results_df.columns = results_df.columns.str.strip()
    results_df = results_df.apply(pd.to_numeric, errors='coerce').dropna()
    results_df.head()

    # Extract data for plotting
    epochs = results_df['epoch']
    train_box_loss = results_df['train/box_loss']
    val_box_loss = results_df['val/box_loss']
    train_cls_loss = results_df['train/cls_loss']
    val_cls_loss = results_df['val/cls_loss']
    train_dfl_loss = results_df['train/dfl_loss']
    val_dfl_loss = results_df['val/dfl_loss']

    # Create a figure with three subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Plot box loss
    axs[0].plot(epochs, train_box_loss, label='Train Box Loss', color='green')
    axs[0].plot(epochs, val_box_loss, label='Validation Box Loss', color='red')
    axs[0].set_title('Box Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].grid(True)

    # Plot class loss
    axs[1].plot(epochs, train_cls_loss, label='Train Cls Loss', color='green')
    axs[1].plot(epochs, val_cls_loss, label='Validation Cls Loss', color='red')
    axs[1].set_title('Class Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].legend()
    axs[1].grid(True)

    # Plot distribution focal loss
    axs[2].plot(epochs, train_dfl_loss, label='Train Dfl Loss', color='green')
    axs[2].plot(epochs, val_dfl_loss, label='Validation Dfl Loss', color='red')
    axs[2].set_title('Distribution Focal Loss')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('Loss')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()
#######################################################   Faster R CNN  ########################################################
    # Define the Dataset Class
    class PcbDataset(Dataset):
        def _init_(self, images_dir, annotations_file, transform=None):
            self.images_dir = images_dir
            self.transform = transform
            self.images = sorted(os.listdir(images_dir))
            self.annotations = self.load_annotations(annotations_file)
            assert len(self.images) == len(self.annotations), "Mismatch between number of images and annotations"

        def load_annotations(self, annotations_file):
            with open(annotations_file, 'r') as f:
                coco_data = json.load(f)

            annotations_dict = {}
            for ann in coco_data['annotations']:
                image_id = ann['image_id']
                if image_id not in annotations_dict:
                    annotations_dict[image_id] = []
                annotations_dict[image_id].append(ann)

            annotations = []
            for img in coco_data['images']:
                image_id = img['id']
                if image_id in annotations_dict:
                    boxes = []
                    labels = []
                    for ann in annotations_dict[image_id]:
                        xmin, ymin, width, height = ann['bbox']
                        boxes.append([xmin, ymin, xmin + width, ymin + height])
                        labels.append(ann['category_id'])
                    annotations.append({
                        'boxes': torch.tensor(boxes, dtype=torch.float32),
                        'labels': torch.tensor(labels, dtype=torch.int64),
                        'image_id': torch.tensor([image_id])
                    })
                else:
                    annotations.append({
                        'boxes': torch.tensor([], dtype=torch.float32),
                        'labels': torch.tensor([], dtype=torch.int64),
                        'image_id': torch.tensor([image_id])
                    })
            return annotations

        def _len_(self):
            return len(self.images)

        def _getitem_(self, idx):
            img_path = os.path.join(self.images_dir, self.images[idx])
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            annotation = self.annotations[idx]
            return img, annotation

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    def collate_fn(batch):
        return tuple(zip(*batch))

    train_images_dir = 'PCB_DATASET/output_320/images/train'
    train_annotations_file = 'PCB_DATASET/output_320/coco/annotations/instances_train.json'
    train_dataset = PcbDataset(train_images_dir, train_annotations_file, transform)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn)

    val_images_dir = 'PCB_DATASET/output_320/images/val'
    val_annotations_file = 'PCB_DATASET/output_320/coco/annotations/instances_val.json'
    val_dataset = PcbDataset(val_images_dir, val_annotations_file, transform)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # Define the Model
    def get_model(num_classes):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model

    num_classes = 7  # 6 defects + background
    model_rcnn = get_model(num_classes)

    # Training Loop
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model_rcnn.to(device)

    params = [p for p in model_rcnn.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    num_epochs = 10

    for epoch in range(num_epochs):
        model_rcnn.train()
        for images, targets in train_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model_rcnn(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {losses.item()}")


    ################################################################ predcitions #########################################

    # Define paths
    dest_results_dir = dataset_dir / 'results'
    best_model_path = dest_results_dir / 'weights' / 'best.pt'
    test_data_dir = dataset_dir / 'PCB_DATASET' /'output_640' /'images'/'val'
    predict_dir = dataset_dir /'predict'
    dest_predict_dir = dest_results_dir / 'predict'

    # Load YOLO model
    model = YOLO(best_model_path)

    # Run inference on the test dataset
    metrics = model(source=test_data_dir, imgsz=640, conf=0.25, save=True, save_txt=True, save_conf=True)
    print("Metrics:", metrics)

    # Copy prediction results to the destination directory
    shutil.copytree(predict_dir, dest_predict_dir)



    def read_yolo_labels_from_file(file_path):
        """
        Read YOLO formatted labels from a file.

        Parameters:
        - file_path: Path to the YOLO labels file.

        Returns:
        - List of YOLO labels.
        """
        with open(file_path, 'r') as file:
            labels = [list(map(float, line.strip().split())) for line in file]
        return labels

    def rcnn_to_original_annot(image_name, rcnn_predictions, annot_df, classes, threshold=0.5):
        """
        Convert Faster R-CNN predictions to original annotation format.

        Parameters:
        - image_name: Name of the image file.
        - rcnn_predictions: Predictions from Faster R-CNN for the image.
        - annot_df: DataFrame with original annotations.
        - classes: List of class names.
        - threshold: Confidence threshold to filter out weak predictions.

        Returns:
        - DataFrame with annotations in original format.
        """
        original_annot = []

        # Get original image size
        original_size = annot_df.loc[annot_df['filename'] == image_name, ['width', 'height']].iloc[0]
        original_width, original_height = original_size['width'], original_size['height']

        # Process predictions
        boxes = rcnn_predictions[0]['boxes'].cpu().numpy()
        labels = rcnn_predictions[0]['labels'].cpu().numpy()
        scores = rcnn_predictions[0]['scores'].cpu().numpy()

        for box, label, score in zip(boxes, labels, scores):
            if score > threshold:
                # Convert box coordinates to original image coordinates
                xmin, ymin, xmax, ymax = box

                original_annot.append({
                    'filename': image_name,
                    'width': int(original_width),
                    'height': int(original_height),
                    'class': classes[int(label)],
                    'xmin': int(xmin),
                    'ymin': int(ymin),
                    'xmax': int(xmax),
                    'ymax': int(ymax),
                    'confidence': float(score)
                })

        return pd.DataFrame(original_annot)

    # Example usage
    image_name = '12_spurious_copper_10.jpg'
    rcnn_predictions = model(image)  # Ensure `image` is prepared as shown previously

    # Load the annotation DataFrame (replace with actual DataFrame loading code)
    annot_df = pd.DataFrame()  # Example placeholder, replace with actual DataFrame

    classes = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']
    pred_annot_df_rcnn = rcnn_to_original_annot(image_name, rcnn_predictions, annot_df, classes)
    print(pred_annot_df_rcnn.head())

    # Display annotations (assuming `display_annotations` function is defined)
    display_annotations(image_name, images_dir, pred_annot_df_rcnn, is_subfolder=True)

    def yolo_to_original_annot(image_name, yolo_labels, annot_df, classes):
        """
        Convert YOLO labels to original annotation format.

        Parameters:
        - image_name: Name of the image file.
        - yolo_labels: YOLO labels for the image.
        - annot_df: DataFrame with original annotations.
        - classes: List of class names.

        Returns:
        - DataFrame with annotations in original format.
        """
        original_annot = []

        # Get original image size
        original_size = annot_df.loc[annot_df['filename'] == image_name, ['width', 'height']].iloc[0]
        original_width, original_height = original_size['width'], original_size['height']

        for yolo_label in yolo_labels:
            class_index, x_center, y_center, bbox_width, bbox_height, confidence = yolo_label

            # Convert YOLO coordinates to original image coordinates
            original_x_center = x_center * original_width
            original_y_center = y_center * original_height
            original_bbox_width = bbox_width * original_width
            original_bbox_height = bbox_height * original_height

            original_x_min = original_x_center - original_bbox_width / 2
            original_y_min = original_y_center - original_bbox_height / 2
            original_x_max = original_x_center + original_bbox_width / 2
            original_y_max = original_y_center + original_bbox_height / 2

            original_annot.append({
                'filename': image_name,
                'width': int(original_width),
                'height': int(original_height),
                'class': classes[int(class_index)],
                'xmin': int(original_x_min),
                'ymin': int(original_y_min),
                'xmax': int(original_x_max),
                'ymax': int(original_y_max),
                'confidence': confidence
            })

        return pd.DataFrame(original_annot)

    # Example usage
    image_name = '12_spurious_copper_10.jpg'
    yolo_labels_file_path = '/labels/12_spurious_copper_10.txt'
    yolo_labels = read_yolo_labels_from_file(yolo_labels_file_path)

    # Load the annotation DataFrame (replace with actual DataFrame loading code)
    annot_df = pd.DataFrame()  # Example placeholder, replace with actual DataFrame

    classes = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']
    pred_annot_df = yolo_to_original_annot(image_name, yolo_labels, annot_df, classes)
    print(pred_annot_df.head())

    display_annotations(image_name, images_dir, pred_annot_df, is_subfolder=True)

    def load_yolo_model(model_path):
        model = YOLO.load(model_path)
        return model

    def load_fasterrcnn_model(model_path):
        model = model_rcnn(pretrained=False)
        in_features = model.roi_heads.box_predictor.in_features
        num_classes = 7  # Adjust based on your number of classes
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        model.load_state_dict(torch.load(model_path))
        return model

    def evaluate_yolo(model, data_loader):
        all_predictions = []
        for images, _ in data_loader:
            predictions = model.predict(images)
            all_predictions.extend(predictions)
        return compute_map50(all_predictions)

    def evaluate_fasterrcnn(model, data_loader, device):
        model.to(device)
        model.eval()
        all_predictions = []
        all_targets = []
        with torch.no_grad():
            for images, targets in data_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                predictions = model(images)
                all_predictions.extend(predictions)
                all_targets.extend(targets)
        return compute_map50(all_predictions, all_targets)

    def compute_map50(predictions, targets=None):
        # Implement the logic to compute mAP50
        # For COCO format use COCOeval
        # This is a placeholder function and should be replaced with actual computation
        return np.random.rand()  # Replace with actual mAP50 computation

    def save_best_model(model, model_name, models_dir):
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / f'{model_name}.pt'
        torch.save(model.state_dict(), model_path)
        print(f"Saved {model_name} model to {model_path}")

    # Load models
    yolo_model_path = Path('yolo_model.pt')
    fasterrcnn_model_path = Path('fasterrcnn_model.pt')

    yolo_model = load_yolo_model(yolo_model_path)
    fasterrcnn_model = load_fasterrcnn_model(fasterrcnn_model_path)

    # Assuming data_loader is defined elsewhere
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    yolo_map50 = evaluate_yolo(yolo_model, pred_annot_df)
    fasterrcnn_map50 = evaluate_fasterrcnn(model_rcnn, pred_annot_df_rcnn, device)

    # Compare models
    if yolo_map50 > fasterrcnn_map50:
        save_best_model(yolo_model, 'best_yolo_model', models_dir)
    else:
        save_best_model(fasterrcnn_model, 'best_fasterrcnn_model', models_dir)

    print(f"YOLO mAP50: {yolo_map50}")
    print(f"Faster R-CNN mAP50: {fasterrcnn_map50}")

if __name__ == "__main__":
    main()