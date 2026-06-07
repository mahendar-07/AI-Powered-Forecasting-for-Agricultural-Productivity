import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# ==========================
# Load Dataset
# ==========================
df = pd.read_csv("dataset/Crop_recommendation.csv")

# ==========================
# Features and Target
# ==========================
X = df.drop("label", axis=1)
y = df["label"]

# ==========================
# Encode Crop Labels
# ==========================
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("Number of Classes:", len(le.classes_))
print("Classes:", le.classes_)

# ==========================
# Train-Test Split
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.3,
    random_state=42
)

# ==========================
# Feature Scaling
# ==========================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nTraining Shape:", X_train_scaled.shape)
print("Testing Shape :", X_test_scaled.shape)

# ==========================
# Build Neural Network
# ==========================
model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),  # 7 features

    Dense(256, activation='tanh'),

    Dense(128, activation='tanh'),

    Dense(len(np.unique(y_encoded)), activation='softmax')
])

# ==========================
# Compile Model
# ==========================
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ==========================
# Train Model
# ==========================
history = model.fit(
    X_train_scaled,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# ==========================
# Evaluate Model
# ==========================
train_loss, train_acc = model.evaluate(
    X_train_scaled,
    y_train,
    verbose=0
)

test_loss, test_acc = model.evaluate(
    X_test_scaled,
    y_test,
    verbose=0
)

print("\n=========================")
print("Train Accuracy:", round(train_acc * 100, 2), "%")
print("Test Accuracy :", round(test_acc * 100, 2), "%")
print("=========================")

# ==========================
# Save Model
# ==========================
model.save("model/crop_model.keras")

# Save Scaler
with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Save Label Encoder
with open("model/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("\nModel Saved Successfully!")

print("\nSaved Files:")
print("✔ model/crop_model.keras")
print("✔ model/scaler.pkl")
print("✔ model/label_encoder.pkl")