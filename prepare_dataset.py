# -*- coding: utf-8 -*-
"""Dataset Preparation for Oxford-IIIT Pet Dataset (YOLO Format)"""

import os
import requests
import tarfile
import xml.etree.ElementTree as ET
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import shutil

# Configuration
DATA_ROOT = './pet-dataset-yolo'
IMAGE_DIR = os.path.join(DATA_ROOT, 'images')
LABEL_DIR = os.path.join(DATA_ROOT, 'labels')

# URLs
DATA_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
ANNOTATION_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"

def download_and_extract(url, path):
    """Downloads and extracts a tar.gz file with a progress bar."""
    print(f"Downloading {os.path.basename(url)}...")
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

    temp_file = os.path.basename(url)
    with open(temp_file, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

    print(f"Extracting {temp_file} to {path}...")
    with tarfile.open(temp_file, 'r:gz') as tar:
        tar.extractall(path=path)
    os.remove(temp_file)
    print("Extraction complete.")

def convert_to_yolo_format(xml_file_path, label_file_path, class_to_index):
    """Converts a single Pascal VOC XML file to YOLO TXT format.
    For Oxford Pet Dataset, extracts breed from filename and uses bounding box from XML.
    """
    try:
        # Extract breed class from filename (e.g., "British_Shorthair_10.xml" -> "British_Shorthair")
        filename = os.path.basename(xml_file_path)
        parts = filename.replace('.xml', '').split('_')
        class_name = '_'.join(parts[:-1]) if len(parts) > 1 else parts[0]
        
        if class_name not in class_to_index:
            return False
            
        class_id = class_to_index[class_name]
        
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        # Get image dimensions
        size = root.find('size')
        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)

        yolo_lines = []

        # Check for bounding boxes in <object> tags
        objects = root.findall('object')
        
        if objects:
            # Use the first bounding box (typically one pet per image)
            obj = objects[0]
            
            # Get bounding box coordinates (xmin, ymin, xmax, ymax)
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            # Calculate normalized YOLO coordinates
            w = xmax - xmin
            h = ymax - ymin
            x_center = xmin + w / 2
            y_center = ymin + h / 2

            # Normalize
            x_center_norm = x_center / img_width
            y_center_norm = y_center / img_height
            w_norm = w / img_width
            h_norm = h / img_height

            yolo_lines.append(f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {w_norm:.6f} {h_norm:.6f}")
        else:
            # No bounding boxes - use whole image
            # Whole image bounding box (center at 0.5, 0.5, width and height = 1.0)
            yolo_lines.append(f"{class_id} 0.5 0.5 1.0 1.0")

        # Write to YOLO TXT file
        if yolo_lines:
            with open(label_file_path, 'w') as f:
                f.write('\n'.join(yolo_lines) + '\n')
            return True
        return False

    except Exception as e:
        print(f"Error converting {xml_file_path}: {e}")
        return False

def main():
    """Main function to prepare the Oxford-IIIT Pet Dataset."""
    
    # Create directories
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(LABEL_DIR, exist_ok=True)

    # Download and extract data
    temp_data = './temp_data'
    temp_annotations = './temp_annotations'
    os.makedirs(temp_data, exist_ok=True)
    os.makedirs(temp_annotations, exist_ok=True)

    download_and_extract(DATA_URL, temp_data)
    download_and_extract(ANNOTATION_URL, temp_annotations)

    # Extract class names from annotation files
    annotation_files = os.listdir(os.path.join(temp_annotations, 'annotations', 'xmls'))
    # Extract breed names: "British_Shorthair_10.xml" -> "British_Shorthair"
    # Remove the numeric suffix (last part after underscore) and .xml extension
    CLASSES = sorted(list(set(['_'.join(f.replace('.xml', '').split('_')[:-1]) for f in annotation_files if f.endswith('.xml')])))
    CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASSES)}
    NUM_CLASSES = len(CLASSES)

    print(f"\nFound {NUM_CLASSES} classes (breeds).")
    print("Starting annotation conversion...")

    # Convert annotations to YOLO format
    xml_dir = os.path.join(temp_annotations, 'annotations', 'xmls')
    labels_created = 0
    for xml_name in tqdm(os.listdir(xml_dir), desc="Converting Annotations"):
        if xml_name.endswith('.xml'):
            img_name = xml_name.replace('.xml', '')
            xml_path = os.path.join(xml_dir, xml_name)
            label_path = os.path.join(LABEL_DIR, f'{img_name}.txt')
            
            # Convert XML to YOLO format
            if convert_to_yolo_format(xml_path, label_path, CLASS_TO_INDEX):
                labels_created += 1
    
    print(f"\\n✅ Created {labels_created} label files")

    # Split data into Train/Val/Test sets
    print("\nSplitting data into Train/Val/Test sets...")
    image_files = [f for f in os.listdir(temp_data + '/images') if f.endswith('.jpg')]
    train_val_files, test_files = train_test_split(image_files, test_size=0.1, random_state=42)
    train_files, val_files = train_test_split(train_val_files, test_size=0.111, random_state=42)  # 0.111 * 0.9 = 0.1

    sets = {'train': train_files, 'val': val_files, 'test': test_files}

    # Create split directories
    for set_name in ['train', 'val', 'test']:
        os.makedirs(os.path.join(IMAGE_DIR, set_name), exist_ok=True)
        os.makedirs(os.path.join(LABEL_DIR, set_name), exist_ok=True)

    # Move files to respective splits
    for set_name, files in sets.items():
        for f in tqdm(files, desc=f"Moving {set_name}"):
            # Image
            src_img = os.path.join(temp_data, 'images', f)
            dst_img = os.path.join(IMAGE_DIR, set_name, f)
            if os.path.exists(src_img):
                shutil.move(src_img, dst_img)

            # Label (txt file, not xml)
            label_name = f.replace('.jpg', '.txt')
            src_label = os.path.join(LABEL_DIR, label_name)  # Labels are in root LABEL_DIR
            dst_label = os.path.join(LABEL_DIR, set_name, label_name)
            if os.path.exists(src_label):
                shutil.move(src_label, dst_label)
            else:
                # If no label file exists, this might be a background image
                pass

    # Create YAML configuration file
    FINAL_DATA_YAML_PATH = f'{DATA_ROOT}/pet_moe.yaml'
    
    with open(FINAL_DATA_YAML_PATH, 'w') as f:
        f.write(f"path: {os.path.abspath(DATA_ROOT)}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n")
        f.write(f"nc: {NUM_CLASSES}\n")
        f.write(f"names: {CLASSES}\n")

    # Cleanup temp directories
    print("\nCleaning up temporary files...")
    shutil.rmtree(temp_data, ignore_errors=True)
    shutil.rmtree(temp_annotations, ignore_errors=True)

    print("\n" + "="*50)
    print(f"✅ Dataset preparation complete.")
    print(f"Dataset ready at: {os.path.abspath(DATA_ROOT)}")
    print(f"Images: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
    print(f"Data YAML saved to: {os.path.abspath(FINAL_DATA_YAML_PATH)}")
    print(f"Number of classes: {NUM_CLASSES}")
    print("="*50)

if __name__ == "__main__":
    main()
