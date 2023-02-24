import tensorflow as tf
import pandas as pd

# ResNet-like
class ResNetLike():
    def __init__(self, train_directory='train/') -> None:
        self.directory = train_directory
        self.df = pd.read_csv(self.directory + 'train.csv')
        self.filenames  = self.df['filename'].values
        self.labels = self.df['sign'].values
        self.train_dset = tf.data.Dataset.from_tensor_slices((self.filenames, self.labels))
        self.train_dset = self.train_dset.map(self.read_image).batch(32)
        
    def read_image(self, img, label):
        image = tf.io.read_file(self.directory + img)
        image = tf.image.decode_image(image, channels=3, dtype=tf.float32)
        
        return image, label

    def build(self) -> None:
        inputs = tf.keras.Input(shape=(20, 20, 3), name='img')
        x = tf.keras.layers.Conv2D(20, 3, activation='relu')(inputs)
        x = tf.keras.layers.Conv2D(40, 3, activation='relu')(x)
        additional_1 = tf.keras.layers.MaxPooling2D(3)(x)
        x = tf.keras.layers.Conv2D(40, 3, activation='relu', padding='same')(additional_1)
        x = tf.keras.layers.Conv2D(40, 3, activation='relu', padding='same')(x)
        additional_2 = tf.keras.layers.add([x, additional_1])
        x = tf.keras.layers.Conv2D(40, 3, activation='relu', padding='same')(additional_2)
        x = tf.keras.layers.Conv2D(40, 3, activation='relu', padding='same')(x)
        additional_3 = tf.keras.layers.add([x, additional_2])
        x = tf.keras.layers.Conv2D(40, 3, activation='relu')(additional_3)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(160, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        
        self.model = tf.keras.Model(inputs, outputs, name='resnetlike')


    def run(self) -> None:
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        self.model.fit(self.train_dset, epochs=10)
