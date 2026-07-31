
# 🍅 Tomato Disease Detection using Deep Learning

## Project Overview

This project detects tomato leaf diseases using MobileNetV2 and Streamlit.

## Dataset

- Dataset: PlantVillage
- Classes: 10
- Total Images: 16,011

## Dataset Split

- Training: 11,203 images
- Validation: 2,402 images
- Testing: 2,406 images

## Model

- MobileNetV2
- Transfer Learning
- Image Size: 224x224
- Batch Size: 32
- Optimizer: Adam
- Learning Rate: 0.0001
- Epochs: 20

## Performance

- Training Accuracy: 91.73%
- Validation Accuracy: 91.59%
- Test Accuracy: 91.02%

## Technologies

- Python
- TensorFlow
- Keras
- Streamlit
- NumPy
- PIL
- Google Colab

## Files

- app.py
- best_tomato_model.keras
- class_names.json
- requirements.txt

## Run

pip install -r requirements.txt

streamlit run app.py
