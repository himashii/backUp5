import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import pickle

root_dir = 'Data'
img_format = {'jpg', 'png', 'bmp'}
test_ratio = 0.2
validation_ratio = 0.2
image_dimension = (32, 32, 3)

data_list = os.listdir(root_dir)
print(data_list)
no_of_data_category = len(data_list)

images = []
image_index = []

for i in range(0, no_of_data_category):
    img_list = os.listdir(root_dir + '/' + str(i))
    for img in img_list:
        try:
            cur_image = cv2.imread(root_dir + '/' + str(i) + '/' + img)
            cur_image = cv2.resize(cur_image, (32, 32))
            images.append(cur_image)
            image_index.append(i)
        except:
            pass
    print(i, end=' ')
print(' ')

images = np.array(images)
image_index = np.array(image_index)

print(images.shape)
print(image_index.shape)

x_train, x_test, y_train, y_test = train_test_split(images, image_index, test_size=test_ratio)
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=validation_ratio)

print(len(x_test))
print(len(x_validation))
print(np.where(y_train == 0))

no_of_samples = []
for x in range(0, no_of_data_category):
    no_of_samples.append(len(np.where(y_train == x)[0]))

print(no_of_samples)

plt.figure(figsize=(10, 5))
plt.bar(range(0, no_of_data_category), no_of_samples)
plt.title("No of images for each category")
plt.xlabel("category id")
plt.ylabel("number of images")
plt.show()


def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img


x_train = np.array(list(map(preProcessing, x_train)))
x_test = np.array(list(map(preProcessing, x_test)))
x_validation = np.array(list(map(preProcessing, x_validation)))

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_validation = x_validation.reshape(x_validation.shape[0], x_validation.shape[1], x_validation.shape[2], 1)

data_gen = ImageDataGenerator(width_shift_range=0.1,
                              height_shift_range=0.1,
                              zoom_range=0.2,
                              shear_range=0.1,
                              rotation_range=10)

data_gen.fit(x_train)

y_train = to_categorical(y_train, no_of_data_category)
y_test = to_categorical(y_test, no_of_data_category)
y_validation = to_categorical(y_validation, no_of_data_category)


def cn_model():
    no_of_filters = 60
    size_of_filter_1 = (5, 5)
    size_of_filter_2 = (3, 3)
    size_of_pool = (2, 2)
    no_of_node = 500

    model = Sequential()
    model.add((Conv2D(no_of_filters, size_of_filter_1, input_shape=(image_dimension[0],
                                                                    image_dimension[1], 1), activation='relu')))
    model.add((Conv2D(no_of_filters, size_of_filter_1, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add((Conv2D(no_of_filters // 2, size_of_filter_2, activation='relu')))
    model.add((Conv2D(no_of_filters // 2, size_of_filter_2, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(no_of_node, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(no_of_data_category, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


model = cn_model()
print(model.summary())

batch_size_val = 50
epoch_val = 100
steps_per_epoch = len(x_train) // batch_size_val

history = model.fit_generator(data_gen.flow(x_train, y_train,
                                            batch_size=batch_size_val),
                              steps_per_epoch=steps_per_epoch,
                              epochs=epoch_val,
                              validation_data=(x_validation, y_validation),
                              shuffle=1)

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('epoch')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()

score = model.evaluate(x_test, y_test, verbose=0)
print('Test Score = ', score[0])
print('Test Accuracy =', score[1])

model.save('model.h5')
