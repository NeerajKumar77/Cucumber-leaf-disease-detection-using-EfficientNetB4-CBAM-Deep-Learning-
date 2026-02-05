SOURCE CODE
# Cucumber Leaf Disease Classification
# EfficientNetB4 + CBAM + Augmentation (Colab Optimized)
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB4
import os
# 1. Dataset and parameters
dataset_dir = "/content/archive (4)/modified Image/Augmented Image/"
img_size = 224
batch_size = 16   # You can increase to 32 if Colab GPU memory allows
num_classes = 5
seed = 42
# 2. Data Augmentation & Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.15,  # 15% validation
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.15
)
train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=seed
)
val_generator = val_datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=seed
)
# 3. CBAM Attention Module
def cbam_block(input_feature, ratio=8):
    """CBAM Attention Module compatible with Keras Functional API"""
    channel = input_feature.shape[-1]
    # --- Channel Attention ---
    shared_dense_one = layers.Dense(channel//ratio, activation='relu', kernel_initializer='he_normal')
    shared_dense_two = layers.Dense(channel, kernel_initializer='he_normal')
    avg_pool = layers.GlobalAveragePooling2D()(input_feature)
    avg_pool = layers.Reshape((1,1,channel))(avg_pool)
    avg_pool = shared_dense_one(avg_pool)
    avg_pool = shared_dense_two(avg_pool)
    max_pool = layers.GlobalMaxPooling2D()(input_feature)
    max_pool = layers.Reshape((1,1,channel))(max_pool)
    max_pool = shared_dense_one(max_pool)
    max_pool = shared_dense_two(max_pool)
    channel_attention = layers.Add()([avg_pool, max_pool])
    channel_attention = layers.Activation('sigmoid')(channel_attention)
    x = layers.Multiply()([input_feature, channel_attention])
    # --- Spatial Attention ---
    avg_pool_spatial = layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(x)
    max_pool_spatial = layers.Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(x)
    concat = layers.Concatenate(axis=-1)([avg_pool_spatial, max_pool_spatial])
    spatial_attention = layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid', kernel_initializer='he_normal')(concat)
    out = layers.Multiply()([x, spatial_attention])
    return out
# 4. Model: EfficientNetB4 + CBAM
inputs = layers.Input(shape=(img_size, img_size, 3))
base_model = EfficientNetB4(include_top=False, input_tensor=inputs, weights='imagenet')
base_model.trainable = True  
x = base_model.output
x = cbam_block(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)
model = models.Model(inputs, outputs)
# 5. Compile Model
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
# 6. Callbacks
early_stop = callbacks.EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=4, min_lr=1e-6)
checkpoint = callbacks.ModelCheckpoint('/content/gdrive/MyDrive/cucumber_model_best.h5',
                                       monitor='val_accuracy',
                                       save_best_only=True)
# 7. Train Model (Stage 1: Frozen base)
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=40,
    callbacks=[early_stop, reduce_lr, checkpoint]
)
# 8. Fine-tune (Unfreeze base_model)
base_model.trainable = True
model.compile(optimizer=optimizers.Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
fine_tune_history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=40,
    callbacks=[early_stop, reduce_lr, checkpoint]
)
# 9. Evaluate on Validation
val_loss, val_acc = model.evaluate(val_generator)
print(f"Validation Accuracy: {val_acc*100:.2f}%")
# 10. Save Final Model
model.save('/content/gdrive/MyDrive/cucumber_leaf_disease_final.h5')
print("Model saved successfully to Google Drive.")
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    cohen_kappa_score, matthews_corrcoef
)
import matplotlib.pyplot as plt
import seaborn as sns

# Predictions
y_true = val_generator.classes
y_pred = model.predict(val_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

# 8 METRICS
acc = accuracy_score(y_true, y_pred_classes)
prec = precision_score(y_true, y_pred_classes, average='macro')
rec = recall_score(y_true, y_pred_classes, average='macro')
f1 = f1_score(y_true, y_pred_classes, average='macro')
auc = roc_auc_score(y_true, y_pred, multi_class='ovr')
kappa = cohen_kappa_score(y_true, y_pred_classes)                # ✅ Inter-rater agreement
mcc = matthews_corrcoef(y_true, y_pred_classes)                  # ✅ Balanced correlation metric
specificity = np.mean([cm[i, i] / (cm[i, :].sum() + 1e-7) for i in range(len(cm))]) if 'cm' in locals() else 0

# Confusion matrix and report
cm = confusion_matrix(y_true, y_pred_classes)
report = classification_report(y_true, y_pred_classes, target_names=val_generator.class_indices.keys())

print(" Model Performance Metrics")
print(f"1 Accuracy       : {acc:.4f}")
print(f"2 Precision      : {prec:.4f}")
print(f"3 Recall         : {rec:.4f}")
print(f"4 F1 Score       : {f1:.4f}")
print(f"5 ROC AUC Score  : {auc:.4f}")
print(f"6 Cohen Kappa    : {kappa:.4f}")
print(f"7 MCC Score      : {mcc:.4f}")
print(f"8 Specificity    : {specificity:.4f}")
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", report)

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

y_test_bin = label_binarize(y_true, classes=list(range(num_classes)))

plt.figure(figsize=(8,6))
for i in range(num_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{list(val_generator.class_indices.keys())[i]} (AUC={roc_auc:.2f})")
plt.plot([0,1],[0,1],'k--')
plt.title('ROC Curves')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

import random
from tensorflow.keras.preprocessing import image

plt.figure(figsize=(12,8))
for i in range(6):
    idx = random.randint(0, len(y_true)-1)
    img_path = val_generator.filepaths[idx]
    img = image.load_img(img_path, target_size=(224,224))
    plt.subplot(2,3,i+1)
    plt.imshow(img)
    plt.title(f"Actual: {list(val_generator.class_indices.keys())[y_true[idx]]}\nPred: {list(val_generator.class_indices.keys())[y_pred_classes[idx]]}")
    plt.axis('off')
plt.tight_layout()
plt.show()




READ ME - How to run the code

Cucumber Leaf Disease Classification
EfficientNetB4 + CBAM + Data Augmentation (Colab Optimized)

Overview
This project detects/categorizes cucumber leaf diseases using a deep neural network with EfficientNetB4 backbone and a CBAM (Convolutional Block Attention Module). The model is heavily augmented and optimized for training in Google Colab.

Setup Instructions
1. Environment
This project is intended for Google Colab with GPU.
Make sure the following libraries are available:
!pip install tensorflow numpy matplotlib seaborn scikit-learn

2. Dataset Preparation
Place your dataset in the following directory structure on Google Drive or Colab environment:
/content/archive (4)/modified Image/Augmented Image/
    ├── DownyMildew
    ├── Anthracnose
    ├── Gummy stem blight
    └── Bacterial wilt
    └── Fresh leaf

3. Colab & Drive Setup
Mount Google Drive if saving/reading models:
from google.colab import drive
drive.mount('/content/gdrive')
Ensure dataset_dir points to your image directory.

4. Running the Code
	- Import Modules and Set Parameters:
	  All necessary imports and main settings are included in the script.
	- Data Augmentation and Loading:
	  The script uses ImageDataGenerator with strong augmentation for training and simpler rescaling for validation.
	- Model Construction:
	  EfficientNetB4 (frozen, then unfrozen for fine-tuning)
	  CBAM block added after the backbone
	- Callbacks:
	  Includes early stopping, model checkpointing (saves best model), and learning rate reduction on plateaus.
	- Training Stages:
	  Stage 1: Train with frozen EfficientNetB4
	  Stage 2: Fine-tune (unfreeze) EfficientNetB4
	- Evaluation:
	  Evaluates on the validation set
	  Computes accuracy, precision, recall, F1-score, ROC-AUC, kappa, MCC, specificity
	  Plots learning curves, confusion matrix, ROC curves, and sample predictions
	- Saving Models:
	  Best weights: /content/gdrive/MyDrive/cucumber_model_best.h5
	  Final model: /content/gdrive/MyDrive/cucumber_leaf_disease_final.h5

5. Output and Results
Prints performance metrics and confusion matrix
Generates plots for:
   Training/validation accuracy and loss
   ROC curves for each class
   Visual samples of predictions versus actual class

6. Customization
You can change batch_size to 32 for faster training if GPU memory allows.
Adjust img_size or augmentation parameters as needed to suit your dataset.

Troubleshooting
OOM (out of memory): Lower the batch_size.
Slow Training: Use Colab’s high-RAM GPU runtime or make sure the dataset is not too large.
Poor Accuracy: Ensure class subdirectory names match your labels, and images are good quality.

