from tensorflow import keras

model = keras.models.Sequential()
model.add(keras.layers.SeparableConv2D(filters=4, kernel_size=3, padding='same', input_shape=(24, 24, 3)))
#model.add(keras.layers.Conv2D(filters=4, kernel_size=3, padding='same', input_shape=(24, 24, 3)))
model.summary()