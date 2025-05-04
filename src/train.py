# import sys
# import os
# sys.path.append(os.path.join(os.path.dirname(__file__)))

# from model import build_model
# from data_preprocessing import get_data_generators
# from evaluate import evaluate_model

# base_path = 'S:/SPIT/Experiments/Shreeya_Nemade/potato-disease-classification'
# dataset_path = os.path.join(base_path, 'dataset/Train') 
# saved_model_dir = os.path.join(base_path, 'saved_models')
# saved_model_path = os.path.join(saved_model_dir, 'potato_disease_model.h5')

# if not os.path.exists(dataset_path):
#     raise FileNotFoundError(f"❌ Dataset folder not found: {dataset_path}")
# print("✅ Dataset path verified:", dataset_path)

# train_gen, val_gen = get_data_generators(dataset_path)
# num_classes = len(train_gen.class_indices)

# model = build_model(input_shape=(128, 128, 3), num_classes=num_classes)

# print("🚀 Starting model training...")
# history = model.fit(train_gen, epochs=60, validation_data=val_gen)

# print("\n📈 Evaluating model on validation set...")
# evaluate_model(model, val_gen)

# os.makedirs(saved_model_dir, exist_ok=True)
# model.save(saved_model_path)
# print(f"✅ Model saved to: {saved_model_path}")


import sys
import os
import json
import numpy as np
from sklearn.metrics import confusion_matrix
sys.path.append(os.path.join(os.path.dirname(__file__)))

from model import build_model
from data_preprocessing import get_data_generators
from evaluate import evaluate_model

base_path = 'S:/SPIT/Experiments/Shreeya_Nemade/potato-disease-classification'
dataset_path = os.path.join(base_path, 'dataset/Train') 
saved_model_dir = os.path.join(base_path, 'saved_models')
saved_model_path = os.path.join(saved_model_dir, 'potato_disease_model.h5')

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"❌ Dataset folder not found: {dataset_path}")
print("✅ Dataset path verified:", dataset_path)

# Data Generators
train_gen, val_gen = get_data_generators(dataset_path)
num_classes = len(train_gen.class_indices)

# Build Model
model = build_model(input_shape=(128, 128, 3), num_classes=num_classes)

print("🚀 Starting model training...")
history = model.fit(train_gen, epochs=60, validation_data=val_gen)

# — SAVE TRAINING HISTORY —
history_path = os.path.join(saved_model_dir, "history.json")
with open(history_path, "w") as f:
    json.dump(history.history, f)
print(f"✅ Training history saved to {history_path}")

# — COMPUTE & SAVE CONFUSION MATRIX ON VALIDATION SET —
y_true = val_gen.classes
y_pred = np.argmax(model.predict(val_gen), axis=1)
cm = confusion_matrix(y_true, y_pred)
cm_path = os.path.join(saved_model_dir, "confusion_matrix.npy")
np.save(cm_path, cm)
print(f"✅ Confusion matrix saved to {cm_path}")

# — Evaluate Model on Validation Set —
print("\n📈 Evaluating model on validation set...")
evaluate_model(model, val_gen)

# — SAVE THE MODEL —
os.makedirs(saved_model_dir, exist_ok=True)
model.save(saved_model_path)
print(f"✅ Model saved to: {saved_model_path}")
