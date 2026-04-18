# =========================
# IMPORTS
# =========================
import cv2
import numpy as np
import time
import pyttsx3

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2

from textblob import TextBlob

# =========================
# TEXT TO SPEECH
# =========================
engine = pyttsx3.init()

# =========================
# MODEL BUILDING
# =========================
base_model = MobileNetV2(
    input_shape=(64,64,3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(24, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# DATA GENERATOR
# =========================
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    'C:/Users/surbh/s_dataset',
    target_size=(64,64),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    color_mode='rgb'
)

val_data = datagen.flow_from_directory(
    'C:/Users/surbh/s_dataset',
    target_size=(64,64),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    color_mode='rgb'
)

# =========================
# CALLBACK
# =========================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# =========================
# TRAINING
# =========================
model.fit(
    train_data,
    validation_data=val_data,
    epochs=50,
    callbacks=[early_stop]
)

# =========================
# SAVE MODEL
# =========================
model.save("sign_model.h5")

# =========================
# LOAD MODEL (for testing)
# =========================
model = load_model("sign_model.h5")

# =========================
# VIDEO + NLP + PREDICTION
# =========================
cap = cv2.VideoCapture("signv.mp4")

current_word = ""
sentence = ""
last_time = time.time()

labels = list(train_data.class_indices.keys())

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (64,64))
    img = img / 255.0
    img = np.reshape(img, (1,64,64,3))

    pred = model.predict(img)
    class_idx = np.argmax(pred)
    letter = labels[class_idx]

    # Build word
    if time.time() - last_time > 1:
        current_word += letter
        last_time = time.time()

    # NLP correction
    corrected = str(TextBlob(current_word).correct())

    cv2.putText(frame, f"Word: {corrected}", (10,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Sign Language Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        engine.say(corrected)
        engine.runAndWait()
        break

cap.release()
cv2.destroyAllWindows()
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

y_pred = model.predict(val_data)
y_pred_classes = np.argmax(y_pred, axis=1)

print(classification_report(val_data.classes, y_pred_classes))
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(val_data.classes, y_pred_classes)

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=False, cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()