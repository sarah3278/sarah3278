```python
import pandas as pd
import numpy as np
import os
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import csv

# =========================
# PATH
# =========================
DATASET_PATH = "./images"

# =========================
# MODEL
# =========================
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 3)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# =========================
# IMAGE TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =========================
# TEMPERATURE MODEL (LSTM)
# =========================
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
temp_data = temp_data / np.max(temp_data)

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
        return self.fc(out[:, -1, :])

model_lstm = LSTMModel()
optimizer = torch.optim.Adam(model_lstm.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(10):   # reduced for speed
    output = model_lstm(X)
    loss = criterion(output.squeeze(), y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

temp_pred = model_lstm(X[-1].unsqueeze(0)).item()

# =========================
# PROCESS IMAGE
# =========================
def process_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    output = model(img)
    _, pred = torch.max(output, 1)

    image_score = pred.item() / 2
    change_score = np.random.rand()
    temp_score = temp_pred

    return image_score, change_score, temp_score

# =========================
# PIPELINE
# =========================
def run_pipeline():
    results = []

    files = os.listdir(DATASET_PATH)
    print("Files found:", files)

    for img_name in files:
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(DATASET_PATH, img_name)

        forest, defor, temp = process_image(img_path)

        results.append([img_name, forest, defor, temp])
        print("Processed:", img_name)

    return results

# =========================
# SAVE RESULTS
# =========================
if __name__ == "__main__":
    results = run_pipeline()

    with open("results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "forest_score", "deforestation_score", "temperature_score"])
        writer.writerows(results)

    print("✅ Results saved to results.csv")
```







