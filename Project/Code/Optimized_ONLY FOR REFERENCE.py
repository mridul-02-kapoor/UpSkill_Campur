import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report

# Create directories
def create_directories():
    os.makedirs("~/.kaggle", exist_ok=True)
    os.system("cp kaggle.json ~/.kaggle/")
    os.system("chmod 600 ~/.kaggle/kaggle.json")
    os.system("kaggle datasets download -d ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes")
    with zipfile.ZipFile('/content/crop-and-weed-detection-data-with-bounding-boxes.zip', 'r') as zip_ref:
        zip_ref.extractall("/content/")
        
    os.makedirs("Cropped_data/Crop", exist_ok=True)
    os.makedirs("Cropped_data/Weed", exist_ok=True)

# Process bounding boxes and crop images
def process_bounding_boxes():
    info = pd.DataFrame(columns=["Name", "Class", "X", "Y", "Width", "Height"])
    path = "/content/agri_data/data/"
    
    for file in os.listdir(path):
        if file.split(".")[-1] == "txt":
            with open(path + file, "r") as f:
                for line in f.readlines():
                    data = line.split(" ")
                    name = file.split(".")[0]
                    clas = data[0]
                    x, y, w, h = map(float, data[1:])
                    
                    name = []
                    clas = []
                    x = []
                    y = []
                    w = []
                    h = []

    len(name), len(clas), len(x), len(y), len(w), len(h)
    info["Name"] = name
    info["Class"] = clas
    info["X"] = x
    info["Y"] = y
    info["Width"] = w
    info["Height"] = h
    
    for index in range(info.shape[0]):
        cropped_pic = crop_pic(info.iloc[index, 0], info.iloc[index, 2], info.iloc[index, 3], info.iloc[index, 4], info.iloc[index, 5])
        reduced_img = Image.fromarray(cropped_pic)
        reduced_img = reduced_img.resize((256, 256))
        
        if info.iloc[index, 1] == '0':
            reduced_img.save(f"Cropped_data/Crop/{index}.jpeg")
        else:
            reduced_img.save(f"Cropped_data/Weed/{index}.jpeg")

# Create CNN model
def create_cnn_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), padding='valid', activation='relu', input_shape=(256, 256, 3)))
    # Add more layers here
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train the model
def train_model(model, dataset, num_epochs=10):
    history = model.fit(dataset, epochs=num_epochs, batch_size=32)
    return history

# Test the model
def test_model(model):
    test_image_path_weed = "/content/Cropped_data/Weed/1029.jpeg"
    test_image_path_crop = "/content/Cropped_data/Crop/1034.jpeg"

    test_image_weed = plt.imread(test_image_path_weed)
    test_image_weed = process(test_image_weed, 0)[0]
    test_image_weed = np.array(test_image_weed).reshape((1, 256, 256, 3))

    test_image_crop = plt.imread(test_image_path_crop)
    test_image_crop = process(test_image_crop, 0)[0]
    test_image_crop = np.array(test_image_crop).reshape((1, 256, 256, 3))

    if model.predict(test_image_weed)[0] > 0.50:
        print("Weed")
    else:
        print("Crop")

    if model.predict(test_image_crop)[0] > 0.50:
        print("Weed")
    else:
        print("Crop")

# Plot model accuracy
def plot_accuracy(history):
    plt.plot(history.history['accuracy'], label='train')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')
    plt.legend()
    plt.show()

# Main function
def main():
    create_directories()
    process_bounding_boxes()

    dataset = image_dataset_from_directory("Cropped_data/", image_size=(256, 256))
    dataset = dataset.map(process)

    model = create_cnn_model()
    history = train_model(model, dataset, num_epochs=10)
    test_model(model)
    plot_accuracy(history)

if __name__ == "__main__":
    main()
