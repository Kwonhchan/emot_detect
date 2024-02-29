import numpy as np
import pandas as pd
import torch
import torchaudio
import librosa
import os
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import Wav2Vec2Model
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

SAMPLING_RATE = 16000
TARGET_LENGTH = 16000

def load_raw_audio(audio_path, target_length):
    if not os.path.exists(audio_path):
        print(f"File not found: {audio_path}")
        return None
    y, sr = librosa.load(audio_path, sr=SAMPLING_RATE)
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)), 'constant')
    elif len(y) > target_length:
        y = y[:target_length]
    return y

class AudioDataset(Dataset):
    def __init__(self, dataframe, root_dir):
        self.dataframe = dataframe
        self.root_dir = os.path.normpath(root_dir)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        relative_path = self.dataframe.iloc[idx]['path']
        audio_path = os.path.normpath(os.path.join(self.root_dir, relative_path))
        label = self.dataframe.iloc[idx]['label']
        audio = load_raw_audio(audio_path, TARGET_LENGTH)
        if audio is None:
            return torch.zeros(1, TARGET_LENGTH), torch.tensor(label)
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        return audio_tensor, torch.tensor(label, dtype=torch.long)

class EmotionRecognitionModel(nn.Module):
    def __init__(self, num_labels):
        super(EmotionRecognitionModel, self).__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h")
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.wav2vec2.config.hidden_size, num_labels)

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values=input_values).last_hidden_state
        output = outputs[:, 0, :]
        output = self.dropout(output)
        logits = self.classifier(output)
        return logits

# CUDA 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("CUDA is available. Using CUDA.")
else:
    print("CUDA is not available. Using CPU.")

# 데이터셋 준비
df = pd.read_csv(r'C:\Users\kwonh\Desktop\wvp\data\train.csv')
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

train_dataset = AudioDataset(train_df, r'C:\Users\kwonh\Desktop\wvp\data')
val_dataset = AudioDataset(val_df, r'C:\Users\kwonh\Desktop\wvp\data')

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# 모델, 손실 함수, 옵티마이저 설정
model = EmotionRecognitionModel(num_labels=6).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# TensorBoard 설정
writer = SummaryWriter('runs/emotion_recognition_experiment')

# 학습 및 검증
best_val_loss = float('inf')
epochs = 30
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs.squeeze(1))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # 검증
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs.squeeze(1))
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    accuracy = 100 * correct / total
    print(f"Epoch {epoch + 1}: Training Loss = {avg_train_loss:.4f}, Validation Loss = {avg_val_loss:.4f}, Accuracy = {accuracy:.2f}%")

    # TensorBoard에 기록
    writer.add_scalar('Loss/train', avg_train_loss, epoch)
    writer.add_scalar('Loss/val', avg_val_loss, epoch)
    writer.add_scalar('Accuracy/val', accuracy, epoch)

    # 최고 모델 저장
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'bm/best_model.pth')
        print(f"New best model saved with Validation Loss: {best_val_loss:.4f}")

writer.close()
print(f"Training completed. Best Validation Loss: {best_val_loss:.4f}")


