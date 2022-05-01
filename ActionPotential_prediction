

#######################################################################################################################


#   According to researchers (https://doi.org/10.1038/s41598-021-87578-0), a very simple AI is able to accurately
#   predict the type of ion channel abnormality if properly trained
#   I have recreated this AI in the hopes of doing something similar for my own project.

#   Regardless, I hope you can make use of this. Hopefully I will share a pretrained model in the future.


#######################################################################################################################

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# To Avoid GPU errors
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

img_height = 128
img_width = 128
batch_size = 5

class CNNBlock(layers.Layer):
    def __init__(self, out_channels, kernel_size=3):
        super(CNNBlock, self).__init__()
        self.conv = layers.Conv2D(out_channels, kernel_size, padding="same")
        self.bn = layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.conv(input_tensor)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        return x


model = keras.Sequential(
    [CNNBlock(32), CNNBlock(130), layers.Flatten(), layers.Dense(10, activation= "softmax"),]
)

ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    "data/trace_collection/",
    labels="inferred",
    label_mode = "int",
    color_mode = "grayscale",
    batch_size = batch_size,
    image_size = (img_height, img_width),
    shuffle = True,
    seed = 12345,
    validation_split = 0.1,
    subset = "training",

)

ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
    "data/trace_collection/",
    labels="inferred",
    label_mode = "int",
    color_mode = "grayscale",
    batch_size = batch_size,
    image_size = (img_height, img_width),
    shuffle = True,
    seed = 12345,
    validation_split = 0.1,
    subset = "validation",

)

model.compile(
    loss=keras.losses.CategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"],
)

model.fit(ds_train, batch_size=32, epochs=2, verbose=2)
model.evaluate(ds_validation, batch_size=32, verbose=2)
print(model.summary())
#model.save("trace_prediction/")
