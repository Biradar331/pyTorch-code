import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from model import SimpleCNN
import translate

# Get class names from translate_dict (Italian animal names used in training)
classes = [k for k in translate.translate_dict if k not in ['dog', 'horse', 'elephant', 'butterfly', 'chicken', 'cat', 'cow','sheep', 'spider', 'squirrel']]

# Sort classes if needed to match the order used during training
classes.sort()

# Load model
model = SimpleCNN(num_classes=len(classes))
model.load_state_dict(torch.load("animal_cnn.pth", map_location=torch.device('cpu')))
model.eval()

st.title("Animal Image Recognition 🐾")
st.write("Upload an animal image to classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)
        predicted_name = classes[predicted.item()]
        english_name = translate.translate_dict.get(predicted_name, predicted_name)
        st.success(f"Prediction: **{english_name}**")