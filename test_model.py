from tensorflow.keras.models import load_model
import pickle

model = load_model("model/crop_model.keras")

with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("model/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

print("Model Loaded Successfully!")
print("Classes:", le.classes_)