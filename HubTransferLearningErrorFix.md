# Intro
This file is to show an error I see when running tensorflow hub transfer learning and the method to fix it

# Example
https://www.tensorflow.org/tutorials/images/hub_with_keras

# Version info
python 3.6.9
tensorflow cpu 1.15.0-dev20190821

# Error

```python

steps_per_epoch = np.ceil(image_data.samples/image_data.batch_size)

batch_stats_callback = CollectBatchStats()

history = model.fit(image_data, epochs=2,
                    steps_per_epoch=steps_per_epoch,
#                     callbacks = [batch_stats_callback]
                   )

```

```
ValueError: `generator` yielded an element of shape (22, 224, 224, 3) where an element of shape (32, 224, 224, 3) was expected.


	 [[{{node PyFunc}}]] [Op:IteratorGetNextSync]
```

# Related Code
```python
data_root = tf.keras.utils.get_file(
  'flower_photos','https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
   untar=True)
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
image_data = image_generator.flow_from_directory(str(data_root), target_size=IMAGE_SHAPE)

feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2" #@param {type:"string"}
feature_extractor_layer.trainable = False

model = tf.keras.Sequential([
  feature_extractor_layer,
  layers.Dense(image_data.num_classes, activation='softmax')
])

model.compile(
  optimizer=tf.keras.optimizers.Adam(),
  loss='categorical_crossentropy',
  metrics=['acc'])
class CollectBatchStats(tf.keras.callbacks.Callback):
    def __init__(self):
        self.batch_losses = []
        self.batch_acc = []

    def on_train_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs['loss'])
        self.batch_acc.append(logs['acc'])
        self.model.reset_metrics()
steps_per_epoch = np.ceil(image_data.samples/image_data.batch_size)

batch_stats_callback = CollectBatchStats()

history = model.fit(image_data, epochs=2,
                    steps_per_epoch=steps_per_epoch,
#                     callbacks = [batch_stats_callback]
                   )
```

# Reason
The total number of pictures (3760) is not devisible by the batch size (32).
Therefore the last batch has a smaller size (22) instead of 32. 

# Solution:
Instead of using model.fit, use model.fit_generator, the error then disappeared.

# Yet to learn:
The difference between model.fit and model.fit_generator
