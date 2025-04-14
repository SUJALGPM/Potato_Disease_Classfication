# import os
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# # Define paths
# base_path = 'S:/SPIT/Experiments/Shreeya_Nemade/potato-disease-classification'  # Update base_path to the correct root path
# train_path = os.path.join(base_path, 'dataset/Train')  # Update paths based on the directory structure
# val_path = os.path.join(base_path, 'dataset/Valid')

# # 🔥 Save model path (make sure this is before using it)
# saved_model_dir = os.path.join(base_path, 'saved_models')  # Save model in the correct folder
# saved_model_path = os.path.join(saved_model_dir, 'potato_disease_model.h5')

# # Check if dataset directories exist
# if not os.path.exists(train_path):
#     raise FileNotFoundError(f"Training folder not found: {train_path}")
# if not os.path.exists(val_path):
#     raise FileNotFoundError(f"Validation folder not found: {val_path}")

# print("✅ Dataset folders verified.")
# print("📁 Train path:", train_path)
# print("📁 Valid path:", val_path)

# # Data preprocessing
# train_datagen = ImageDataGenerator(rescale=1./255)
# val_datagen = ImageDataGenerator(rescale=1./255)

# train = train_datagen.flow_from_directory(
#     train_path,
#     target_size=(128, 128),
#     batch_size=32,
#     class_mode='categorical'
# )

# val = val_datagen.flow_from_directory(
#     val_path,
#     target_size=(128, 128),
#     batch_size=32,
#     class_mode='categorical'
# )

# print(f"📊 Found {train.samples} training samples in {len(train.class_indices)} classes.")
# print(f"📊 Found {val.samples} validation samples.")

# # Build the model
# model = Sequential([
#     tf.keras.Input(shape=(128, 128, 3)),
#     Conv2D(32, (3, 3), activation='relu'),
#     MaxPooling2D(2, 2),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D(2, 2),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dense(len(train.class_indices), activation='softmax')
# ])

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# print("🚀 Starting model training...")
# model.fit(train, epochs=10, validation_data=val)

# # Save the trained model
# os.makedirs(saved_model_dir, exist_ok=True)
# model.save(saved_model_path)

# print(f"✅ Model saved to: {saved_model_path}")


# -----------------------------------------------------------------------------------------------------
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

from model import build_model
from data_preprocessing import get_data_generators
from evaluate import evaluate_model


# Paths
base_path = 'S:/SPIT/Experiments/Shreeya_Nemade/potato-disease-classification'
dataset_path = os.path.join(base_path, 'dataset/Train') 
saved_model_dir = os.path.join(base_path, 'saved_models')
saved_model_path = os.path.join(saved_model_dir, 'potato_disease_model.h5')

# Check path
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"❌ Dataset folder not found: {dataset_path}")

print("✅ Dataset path verified:", dataset_path)

# ✅ Get train and val generators with augmentation
train_gen, val_gen = get_data_generators(dataset_path)

# ✅ Build model
num_classes = len(train_gen.class_indices)
model = build_model(input_shape=(128, 128, 3), num_classes=num_classes)

# ✅ Train model
print("🚀 Starting model training...")
history = model.fit(train_gen, epochs=25, validation_data=val_gen)

# ✅ Evaluate model
print("\n📈 Evaluating model on validation set...")
evaluate_model(model, val_gen)

# ✅ Save model
os.makedirs(saved_model_dir, exist_ok=True)
model.save(saved_model_path)
print(f"✅ Model saved to: {saved_model_path}")
