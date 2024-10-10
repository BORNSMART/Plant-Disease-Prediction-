import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

# Load the trained model
model = tf.keras.models.load_model('trained_model.keras')

# Define the path to the test image
image_path = r"C:\Users\Aditya Dubey\OneDrive\Desktop\Projects\Plant disease detection\archive\test\test\AppleCedarRust1.JPG"

# Read and preprocess the image
img = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
input_arr = tf.keras.preprocessing.image.img_to_array(img)
input_arr = np.expand_dims(input_arr, axis=0)  # Create a batch of size 1

# Make a prediction
predictions = model.predict(input_arr)
predicted_index = np.argmax(predictions, axis=1)[0]  # Get the index of the highest score

# Load the validation set to access class names
validation_set = tf.keras.utils.image_dataset_from_directory(
    r'C:\Users\Aditya Dubey\OneDrive\Desktop\Projects\Plant disease detection\archive\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\valid',
    labels="inferred",
    label_mode="categorical",
    image_size=(128, 128),
    batch_size=32
)

# Get the class names
class_names = validation_set.class_names
# print("Class names: ", class_names)

# Get the predicted disease name
model_prediction = class_names[predicted_index]
print("Predicted Disease: ", model_prediction)

# Display the image with the prediction
# Convert the image to a format suitable for displaying with Matplotlib
img_rgb = cv2.cvtColor(tf.keras.preprocessing.image.img_to_array(img), cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb / 255.0)  # Normalize the image to [0, 1] range
plt.title(f"Disease Name: {model_prediction}")
plt.axis('off')  # Turn off axis labels
plt.show()
