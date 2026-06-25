

import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import os
from tensorflow.keras.utils import plot_model
import numpy as np
import cv2
import time
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold
from sklearn.metrics import  make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler ,MinMaxScaler
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.decomposition import PCA

Image_Size= 300
Batch_Size = 32
Channels=3
Epochs=50

dataset = tf.keras.preprocessing.image_dataset_from_directory(
      "/content/drive/MyDrive/real-and-fake-face-detection/real_and_fake_face/",
      shuffle=True,
      image_size = (Image_Size,Image_Size),
      batch_size=Batch_Size
  )

class_names = dataset.class_names

def splitting_dataset_tf(ds, train_split=0.7, val_split=0.15, test_split=0.15, shuffle=True, shuffle_size=10000):
    ds_size=len(ds)
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)
    train_size=int(train_split * ds_size)
    val_size= int(val_split * ds_size)
    train_ds= ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    return train_ds, val_ds, test_ds


train_ds, val_ds, test_ds=splitting_dataset_tf(dataset)


train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

resize_and_rescale = tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(Image_Size,Image_Size),
    layers.experimental.preprocessing.Rescaling(1.0/255)
])

data_aug = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2),
])

input_shape = (Batch_Size,Image_Size, Image_Size,Channels)

n_classes =  1

model = models.Sequential([
    resize_and_rescale,
    data_aug,
    layers.Conv2D(32, (3,3), activation='relu', input_shape = input_shape),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation = 'relu'),
    layers.Dense(n_classes, activation=  'sigmoid'),
])

model.build(input_shape=input_shape)

model.compile(
    optimizer='adam',
    #loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=['accuracy'])

model_path='/content/drive/MyDrive/ann/models/model1000epoch.h5'
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    print('Model Found')
model.summary()
plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=False)

model_checkpoint_path ='/content/model1.h5'

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=model_checkpoint_path,
    save_weights_only=False,  # ذخیره وزنه‌ها نیست بلکه کل مدل ذخیره می‌شود
    save_freq='epoch'  # ذخیره مدل به صورت دوره‌ای هر 20 اپوک
)

history = model.fit(
    train_ds,
    epochs=100,
    batch_size=Batch_Size,
    verbose=1,
    validation_data=val_ds,
    callbacks=[model_checkpoint_callback])

model.save('/content/drive/MyDrive/ann/models/model1000epoch.h5')

scores = model.evaluate(test_ds)

model = load_model('/content/drive/MyDrive/ann/models/model1000epoch.h5')
intermediate_layer_model = tf.keras.Model(inputs=model.input, outputs=model.layers[-3].output)
intermediate_layer_model.summary()
plot_model(intermediate_layer_model, to_file='model_architecture.png', show_shapes=True, show_layer_names=False)

model = load_model('/content/drive/MyDrive/ann/models/model1000epoch.h5')
intermediate_layer_model = tf.keras.Model(inputs=model.input, outputs=model.layers[-3].output)

# تابع برای بارگیری ویژگی‌های استخراج شده از مدل کانولوشنال
def extract_features(  image_paths):
    features_list = []
    for image_path in image_paths:
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(300, 300))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        features = intermediate_layer_model.predict(img_array)
        features_list.append(features.flatten())
    return np.array(features_list)

fake_data_dir="/content/drive/MyDrive/real-and-fake-face-detection/real_and_fake_face/training_fake"
real_data_dir="/content/drive/MyDrive/real-and-fake-face-detection/real_and_fake_face/training_real/"

real_images = [os.path.join(real_data_dir, img) for img in os.listdir(real_data_dir)]
fake_images = [os.path.join(fake_data_dir, img) for img in os.listdir(fake_data_dir)]


error=[]
features = []
start_time = time.time()

Xreal= extract_features(real_images)
y_real = np.ones(len(Xreal))
Xfake= extract_features(fake_images)
y_fake = np.zeros(len(Xfake) )
X_combined=np.concatenate((Xreal, Xfake), axis=0)
end_time = time.time()

# ترکیب داده‌های real و fake

y_combined = np.concatenate((y_real, y_fake), axis=0)
X_combined = X_combined.reshape(X_combined.shape[0], -1)
X_combined1=X_combined.copy()
extraction_time = end_time - start_time
print(extraction_time)
# تقلیل ابعاد با استفاده از PCA (به عنوان مثال، اندازه ویژگی‌ها را به 50 تغییر می‌دهیم)
np.save('/content/drive/MyDrive/ann/X_combined1000epoch.npy',X_combined)
np.save('/content/drive/MyDrive/ann/y_combined1000epoch.npy',y_combined)

#X_combined=np.load('/content/drive/MyDrive/ann/X_combined1000epoch.npy')
#y_combined=np.load('/content/drive/MyDrive/ann/y_combined1000epoch.npy')

def find_numComp_pca(dataset, percentage):
    # ساخت دیتافریم مثال
    data = dataset
    # اعمال PCA
    pca = PCA()
    pca.fit(data)
    num_components = 400
    # رسم نمودار شریب
    for i in range(1, len(pca.explained_variance_ratio_) + 1):

        if np.cumsum(pca.explained_variance_ratio_)[i] > percentage:
            num_components = i
            percentage = np.cumsum(pca.explained_variance_ratio_)[i]
            break
    print()
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('#component')


    plt.ylabel('var')
    plt.title(' Scree Plo')
    plt.text(0, 0.2, ' num_components : '+str(num_components), fontsize=12, ha='center')
    plt.axvline(x=num_components, color='r', linestyle='--')

    plt.axhline(y=percentage, color='r', linestyle='--')

    # اضافه کردن برچسب خط عمود

    plt.show()
    plt.savefig('PCA.png')
    return num_components
n_comp=find_numComp_pca(X_combined,0.85)

Xx = X_combined.copy()
y=y_combined.copy()
pca = PCA(n_components=n_comp )
Xx = pca.fit_transform(Xx)

def svm_pca(kernel, gamma):

    svm = SVC(kernel=kernel, C=1.0, gamma=gamma)  # تنظیمات مدل SVM

    scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    scores = cross_validate(svm, Xx, y, cv=5, scoring=scoring, return_train_score=False)

    mean_accuracy = scores['test_accuracy'].mean()
    mean_precision = scores['test_precision_macro'].mean()
    mean_recall = scores['test_recall_macro'].mean()
    mean_f1 = scores['test_f1_macro'].mean()

    print("Mean Accuracy:", mean_accuracy)
    print("Mean Precision:", mean_precision)
    print("Mean Recall:", mean_recall)
    print("Mean F1-Score:", mean_f1)

    X_train, X_test, y_train, y_test = train_test_split(Xx, y, test_size=0.2)

    svm.fit(X_train, y_train)
    y_scores = svm.decision_function(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_scores )
    auc = roc_auc_score(y_test, y_scores)

    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    # ماتریس درهم‌ریختگی
    cm = confusion_matrix(y_train, svm.predict(X_train))

    # رسم نمودار ماتریس درهم‌ریختگی با استفاده از heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative(fake)', 'Positive(real)'], yticklabels=['Negative(fake)', 'Positive(real)'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix  Train')
    plt.show()

    cm = confusion_matrix(y_test, svm.predict(X_test))

    # رسم نمودار ماتریس درهم‌ریختگی با استفاده از heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative(fake)', 'Positive(real)'], yticklabels=['Negative(fake)', 'Positive(real)'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix  Test')
    plt.show()


svm_pca('rbf','scale')