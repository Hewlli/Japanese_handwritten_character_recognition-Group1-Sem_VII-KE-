import skimage.transform
import numpy as np
from sklearn.model_selection import train_test_split

kanji = 879
rows = 48
cols = 48
batch_size = 1000  # memory-friendly batch size

# Load dataset
kan = np.load("kanji.npz")['arr_0'].reshape([-1, 127, 128]).astype(np.float32)

# Normalize in batches to avoid memory issues
for start in range(0, len(kan), batch_size):
    end = min(start + batch_size, len(kan))
    kan[start:end] = kan[start:end] / np.max(kan[start:end])
    print(f"Normalized {end}/{len(kan)} images")

# Preallocate array for resized images
train_images = np.zeros([kanji * 160, rows, cols], dtype=np.float32)

# Prepare labels
arr = np.arange(kanji)
train_labels = np.repeat(arr, 160)

# 4 characters were actually hiragana, so delete these 4 extras
for i in range((kanji + 4) * 160):
    idx = int(i / 160)
    if idx not in [88, 219, 349, 457]:
        if idx < 88:
            train_images[i] = skimage.transform.resize(kan[i], (rows, cols))
        elif idx < 219:
            train_images[i - 160] = skimage.transform.resize(kan[i], (rows, cols))
        elif idx < 349:
            train_images[i - 320] = skimage.transform.resize(kan[i], (rows, cols))
        elif idx < 457:
            train_images[i - 480] = skimage.transform.resize(kan[i], (rows, cols))
        else:  # idx > 457
            train_images[i - 640] = skimage.transform.resize(kan[i], (rows, cols))
    if i % 1000 == 0:
        print(f"Resized {i} images")

# Train/test split
train_images, test_images, train_labels, test_labels = train_test_split(
    train_images, train_labels, test_size=0.2
)

# Save as compressed npz
np.savez_compressed("kanji_train_images.npz", train_images)
np.savez_compressed("kanji_train_labels.npz", train_labels)
np.savez_compressed("kanji_test_images.npz", test_images)
np.savez_compressed("kanji_test_labels.npz", test_labels)

print("All files saved successfully!")
