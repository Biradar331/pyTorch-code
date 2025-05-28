import torch
from torchvision import transforms
from PIL import Image
from model import SimpleCNN
import translate

classes = ['cat', 'dog', 'horse','elephant','butterfly','chicken','cow','sheep','squirrel','spider'] 

model = SimpleCNN(num_classes=len(classes))
model.load_state_dict(torch.load("animal_cnn.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

img = Image.open("D:\practice_projects\Tensorflow-Animal Prediction\Simple testing images\dog2.jpeg")
img = transform(img).unsqueeze(0)

with torch.no_grad():
    output = model(img)
    _, predicted = torch.max(output, 1)
    predicted_name = classes[predicted.item()]
    english_name = translate.translate_dict.get(predicted_name, predicted_name)
    print(f"Predicted: {english_name}")