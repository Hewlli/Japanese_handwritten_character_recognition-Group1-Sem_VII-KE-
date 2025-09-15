import numpy as np
from tensorflow.keras.models import load_model
# Load test data
test_images = np.load("kanji_test_images.npz")['arr_0']
test_labels = np.load("kanji_test_labels.npz")['arr_0']

# Reshape and normalize
test_images = test_images.reshape(-1, 48, 48, 1)  # adjust if channels_first
test_images = test_images

# Load the trained model
model = load_model("kanji01.h5")

# Evaluate model on test data
loss, accuracy = model.evaluate(test_images, test_labels)
print(f"Test loss: {loss:.4f}")
print(f"Test accuracy: {accuracy:.4f}")

# Optional: Predict on first 5 test images
predictions = model.predict(test_images[5:15])
predicted_labels = np.argmax(predictions, axis=1)

# Assuming you have your label list imported or defined
from kanjijapanese import label

with open('results.txt', 'w', encoding='utf-8') as f:
    for i, pred in enumerate(predicted_labels, start=5):
        f.write(f"Test sample {i}: Predicted: {label[pred]}, Actual: {label[test_labels[i]]}\n")

