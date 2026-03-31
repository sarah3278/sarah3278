
import pandas as pd
import numpy as np
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.models as models
import csv



transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

DATASET_PATH = "./Remote_Sensing_Data.v2i.yolov8/train/images"


model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 3)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Load dataset (optional training)
dataset = ImageFolder("./Remote_Sensing_Data.v2i.yolov8/", transform=transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

for epoch in range(1):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


df = pd.read_csv("./all countries global temperature.csv")
df = df[df['Country Name'] == 'India']

df_long = df.melt(
    id_vars=['Country Name'],
    value_vars=[str(year) for year in range(1970, 2022)],
    var_name='Year',
    value_name='Temperature'
)

df_long['Temperature'] = df_long['Temperature'].astype(float)
temp_data = df_long['Temperature'].values
max_val = np.max(temp_data)
temp_data = temp_data / max_val

def create_sequences(data, seq_length=5):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

X, y = create_sequences(temp_data)
X = torch.tensor(X).float().unsqueeze(-1)
y = torch.tensor(y).float()

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

model_lstm = LSTMModel()
optimizer_lstm = torch.optim.Adam(model_lstm.parameters(), lr=0.01)
criterion_lstm = nn.MSELoss()

for epoch in range(20):
    output = model_lstm(X)
    loss = criterion_lstm(output.squeeze(), y)

    optimizer_lstm.zero_grad()
    loss.backward()
    optimizer_lstm.step()

# Predict temperature
temp_pred = model_lstm(X[-1].unsqueeze(0)).item()




def process_image(image_path):
    img = Image.open(image_path)
    img = transform(img).unsqueeze(0).to(device)

    output = model(img)
    _, pred = torch.max(output, 1)

    # Normalize score (0–1)
    image_score = pred.item() / 2

    # Dummy deforestation score (can improve later)
    change_score = np.random.rand()

    # Temperature from LSTM
    temp_score = temp_pred

    # Fuzzy logic
    

    return image_score, change_score, temp_score

# =========================
# MAIN PIPELINE
# =========================
def run_pipeline():
    results = []

    for img_name in os.listdir(DATASET_PATH):
        img_path = os.path.join(DATASET_PATH, img_name)

        try:
            forest, defor, temp, fuzzy = process_image(img_path)

            results.append([img_name, forest, defor, temp, fuzzy])
            print("Processed:", img_name)

        except Exception as e:
            print("Error:", img_name, e)

    return results

# =========================
# SAVE RESULTS
# =========================
if __name__ == "__main__":
    results = run_pipeline()

    with open("results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "forest_score", "deforestation_score", "temperature_score", "fuzzy_score"])
        writer.writerows(results)

    print("✅ Results saved to results.csv")





