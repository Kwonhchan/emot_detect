import torch



def predict_emotion(audio_path, model, device):
    model.eval()  # 모델을 평가 모드로 설정
    audio = load_raw_audio(audio_path, TARGET_LENGTH)
    if audio is None:
        return "오디오 파일을 찾을 수 없습니다."
    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, L]
    
    with torch.no_grad():
        outputs = model(audio_tensor)
        _, predicted = torch.max(outputs, dim=1)
        predicted_emotion = predicted.item()
    
    return predicted_emotion

# 모델 로드
model_path = 'best_model.pth'
model = EmotionRecognitionModel(num_labels=6).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))

# 새로운 오디오 파일에 대한 감정 예측
audio_path = 'path/to/your/new/audio_file.wav'
predicted_emotion = predict_emotion(audio_path, model, device)
print(f"Predicted Emotion: {predicted_emotion}")
