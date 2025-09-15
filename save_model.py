from keras.models import load_model

model = load_model('kuzushiji.h5')  # If you have this file
model.save('my_model.keras')  # Save in native format


