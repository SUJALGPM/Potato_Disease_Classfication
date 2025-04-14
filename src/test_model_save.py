import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

# ✅ Step 1: Prepare directory
save_path = "saved_models"
if not os.path.exists(save_path):
    os.makedirs(save_path)

# ✅ Step 2: Load dummy dataset (MNIST)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# ✅ Step 3: Preprocess data
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# ✅ Step 4: Create simple model
model = Sequential([
    Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# ✅ Step 5: Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=2, validation_data=(x_test, y_test))

# ✅ Step 6: Save model
model_save_location = os.path.join(save_path, "potato_model.h5")
model.save(model_save_location)
print(f"✅ Model saved successfully at: {model_save_location}")
