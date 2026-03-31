
import pandas as pd
import numpy as np
import os
from PIL import Image
import torch



import torch.nn as nn
import torchvision.models as models
import csv





DATASET_PATH = "./images"


model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 3)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)



# Load dataset (optional training)
print("Skipping ImageFolder for GitHub run")




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





