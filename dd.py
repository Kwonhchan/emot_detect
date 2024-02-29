import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import os
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Model
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR

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
        self.dropout1 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(self.wav2vec2.config.hidden_size, 512)
        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_labels)

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values=input_values).last_hidden_state
        output = outputs[:, 0, :]
        output = self.dropout1(F.relu(self.fc1(output)))
        output = self.dropout2(F.relu(self.fc2(output)))
        logits = self.fc3(output)
        return logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {'CUDA' if device.type == 'cuda' else 'CPU'} for computation.")

df = pd.read_csv(r'C:\Users\kwonh\Desktop\wvp\data\train.csv')
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

train_dataset = AudioDataset(train_df, r'C:\Users\kwonh\Desktop\wvp\data')
val_dataset = AudioDataset(val_df, r'C:\Users\kwonh\Desktop\wvp\data')

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

model = EmotionRecognitionModel(num_labels=6).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

writer = SummaryWriter('runs/emotion_recognition_experiment')

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc='Training', leave=True)
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs.squeeze(1))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix({'Training Loss': f'{loss.item():.4f}'})
    return total_loss / len(train_loader)

def validate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    progress_bar = tqdm(val_loader, desc='Validation', leave=True)
    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs.squeeze(1))
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            progress_bar.set_postfix({'Validation Loss': f'{loss.item():.4f}'})
    accuracy = 100 * correct / total
    return val_loss / len(val_loader), accuracy

best_val_loss = float('inf')
for epoch in range(30):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, accuracy = validate_model(model, val_loader, criterion, device)
    scheduler.step()
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/val', accuracy, epoch)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"\nEpoch {epoch+1}: New best model saved with val loss {best_val_loss:.4f}")
    else:
        print(f"\nEpoch {epoch+1}: Train loss {train_loss:.4f}, Val loss {val_loss:.4f}, Accuracy {accuracy:.2f}%")
        if epoch > 4 and val_loss >= best_val_loss:  # Simple early stopping
            print("Early stopping triggered.")
            break

writer.close()
print("Training completed.")