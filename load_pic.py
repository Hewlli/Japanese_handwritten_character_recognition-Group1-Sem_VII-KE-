import numpy as np
import matplotlib.pyplot as plt

# Load the data
train_images = np.load("k49-train-imgs.npz")['arr_0']
train_labels = np.load("k49-train-labels.npz")['arr_0']

# Reshape if needed (your script already does this, but for verification)
if len(train_images.shape) == 3:  # If not already (samples, 28, 28, 1)
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)

# Display the first 9 images
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(train_images[i].reshape(28, 28), cmap='gray')
    plt.title(f'Label: {train_labels[i]}')
    plt.axis('off')
plt.show()
