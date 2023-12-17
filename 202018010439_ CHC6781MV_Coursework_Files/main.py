import os
import tensorflow as tf

import numpy as np

data_dir = './train_data/train'
num_classes = 2
batch_size = 64
target_size = (224, 224)

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,  
    validation_split=0.15,
    # rotation_range=20,  
    # width_shift_range=0.1,  
    # height_shift_range=0.1,  
    # shear_range=0.2,  
    # zoom_range=0.2,  
    horizontal_flip=True  
)

train_dataset = train_datagen.flow_from_directory(
    data_dir,
    target_size=target_size,
    batch_size=batch_size,
    color_mode='rgb',
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_dataset = train_datagen.flow_from_directory(
    data_dir,
    target_size=target_size,
    batch_size=batch_size,
    color_mode='rgb',
    class_mode='categorical',
    subset='validation',
    shuffle=False
)



test_data_dir = './train_data/test'
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,  
    # rotation_range=20,  
    # width_shift_range=0.1,  
    # height_shift_range=0.1,  
    # shear_range=0.2,  
    # zoom_range=0.2,  
    horizontal_flip=True  
)
test_dataset = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=target_size,
    batch_size=batch_size,
    color_mode='rgb',
    class_mode='categorical',
    subset='validation',
    shuffle=False
)



# Define AttentionBlock in TensorFlow format
class AttentionBlock(tf.keras.layers.Layer):
    def __init__(self):
        super(AttentionBlock, self).__init__()
        self.conv = tf.keras.layers.Conv2D(64, (1, 1), activation='softmax')

    def call(self, inputs):
        attention_weights = self.conv(inputs)

        weighted_sum = attention_weights * inputs

        # output = tf.reduce_sum(weighted_sum, axis=[2, 3])
        return weighted_sum

# Define CNNWithAttention in TensorFlow format
class CNNWithAttention(tf.keras.Model):
    def __init__(self):
        super(CNNWithAttention, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')
        self.max_pool = tf.keras.layers.MaxPooling2D(pool_size=(4, 4), strides=2)
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')
        self.attention = AttentionBlock()
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(64, activation='relu')
        self.fc2 = tf.keras.layers.Dense(2, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = self.max_pool(x)
        attention_weights = self.attention(x)
        # x = x * tf.expand_dims(attention_weights, axis=-1)
        x = tf.reduce_sum(attention_weights, axis=[1, 2])
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# Define EnsembleCNN in TensorFlow format
class EnsembleCNN(tf.keras.Model):
    def __init__(self, num_networks=2):
        super(EnsembleCNN, self).__init__()
        self.num_networks = num_networks
        self.networks = [CNNWithAttention() for _ in range(num_networks)]

    def call(self, x):
        predictions = tf.zeros((tf.shape(x)[0], 2))
        for i in range(self.num_networks):
            predictions += self.networks[i](x)
        predictions /= self.num_networks
        return predictions

# flatten = tf.keras.layers.Flatten()(concat)
# outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(flatten)
# model = tf.keras.Model(inputs=inputs, outputs=outputs)

model = EnsembleCNN(num_networks=2)

# loss_fn = CategoricalCrossentropyWithLogitsLoss()
# accuracy_fn = CategoricalAccuracy()
def custom_sparse_categorical_crossentropy(y_true, y_pred):

    # print(y_true.shape)
    # print(y_pred.shape)
    
    # y_true = tf.one_hot(tf.cast(y_true, dtype=tf.int32), depth=tf.shape(y_pred)[-1])

    
    # y_pred = tf.reshape(y_pred, tf.shape(y_true))

    
    loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)

    
    return tf.reduce_mean(loss)



learning_rate = 0.0001
epochs = 10
optimizer = tf.keras.optimizers.Adam(learning_rate)

model.compile(optimizer = optimizer, loss = custom_sparse_categorical_crossentropy, metrics=['acc'])

checkpoint_path = 'best_model.h5'
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                         save_best_only=True,
                                                         save_weights_only=True,
                                                         monitor='acc',
                                                         mode='max',
                                                         verbose=1)


class LossAccuracyHistory(tf.keras.callbacks.Callback):
    def __init__(self):
        self.losses = []
        self.accuracies = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs['loss'])
        self.accuracies.append(logs['acc'])

history_callback = LossAccuracyHistory()

model.fit(train_dataset,
          epochs=epochs,
          validation_data=val_dataset,
          callbacks=[checkpoint_callback, history_callback])


import matplotlib.pyplot as plt

losses = history_callback.losses
accuracies = history_callback.accuracies

epochs_range = range(1, len(losses) + 1)


plt.subplot(2, 1, 1)
plt.plot(epochs_range, losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(epochs_range, accuracies, label='Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig("train.png")

plt.show()




from sklearn.metrics import roc_auc_score, recall_score, precision_score, confusion_matrix, f1_score, precision_recall_curve


y_pred_prob = []
batch_size = test_dataset.batch_size
num_samples = len(test_dataset.labels)  


for _ in range(int(np.ceil(num_samples / batch_size))):
    images, labels = next(test_dataset)
    predictions = model.predict(images)
    y_pred_prob.append(predictions)



y_pred_prob = tf.concat(y_pred_prob, axis=0)



y_pred_prob = tf.argmax(y_pred_prob, axis=1).numpy()
y_true = test_dataset.labels


roc_auc = roc_auc_score(y_true, y_pred_prob)
recall = recall_score(y_true, y_pred_prob)
precision = precision_score(y_true, y_pred_prob)
f1 = f1_score(y_true, y_pred_prob)

print(f'ROC-AUC Score: {roc_auc:.4f}')
print(f'Recall Score: {recall:.4f}')
print(f'Precision Score: {precision:.4f}')
print(f'F1 Score: {f1:.4f}')


confusion_mat = confusion_matrix(y_true, y_pred_prob)
print('Confusion Matrix:')
print(confusion_mat)


precision, recall, thresholds = precision_recall_curve(y_true, y_pred_prob)
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.savefig("Precision-Recall_Curve.png")
plt.show()