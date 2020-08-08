import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout, LeakyReLU, ReLU
from keras.layers import BatchNormalization
import keras
from keras.utils import to_categorical
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255)
train_generator = train_datagen.flow_from_directory(
    directory='data',
    target_size=(128, 128),
    batch_size= 32,
    class_mode='categorical'
    )
val_generator = train_datagen.flow_from_directory(
    directory='test',
    target_size=(128, 128),
    batch_size= 32,
    class_mode='categorical'
    )

from tensorflow.keras.applications import ResNet50
model = ResNet50(weights='imagenet',
    input_shape=(128, 128, 3),
    include_top=False)
for layer in model.layers:
    layer.trainable=False
x = Flatten()(model.output)
prediction = Dense(2, activation='softmax')(x)
final_model = keras.Model(inputs=model.input, outputs=prediction)

# final_model.summary()
final_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

final_model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10,
    validation_data=val_generator,
    validation_steps = len(val_generator),
    verbose=2
    )

final_model.save('final_model.h5')