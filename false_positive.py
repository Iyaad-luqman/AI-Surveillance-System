from classify_video import initialize_model, save_model
import cv2
import torch
import open_clip
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import os

class FalsePositiveDataset(Dataset):
    def __init__(self, false_positives, preprocess, correct_classification, tokenizer, device):
        self.false_positives = false_positives
        self.preprocess = preprocess
        self.correct_classification = correct_classification
        self.tokenizer = tokenizer
        self.device = device

    def __len__(self):
        return len(self.false_positives)

    def __getitem__(self, idx):
        clip = self.false_positives[idx]
        image = self.preprocess(Image.fromarray(clip).convert("RGB")).to(self.device)
        text_token = self.tokenizer([self.correct_classification]).to(self.device)
        return image, text_token
    
def train_model(model, dataset, epochs=1, lr=0.001, device='cpu'):
    model.train()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for images, text_tokens in dataloader:
            # images = images.unsqueeze(0)  # Add batch dimension
            optimizer.zero_grad()
            image_features = model.encode_image(images)
            text_features = model.encode_text(text_tokens)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarity = 100.0 * image_features @ text_features.T
            labels = torch.arange(len(text_tokens)).to(device)
            loss = criterion(similarity, labels)
            loss.backward()
            optimizer.step()


def report_false_positive(video_path, correct_classification, model_path='fine_tuned_model.pth', device='cpu'):
    # Load the video and extract the frames
    video = cv2.VideoCapture(video_path)
    success, frame = video.read()

    frames = []
    frame_count = 0
    while success:
        frames.append(frame)
        success, frame = video.read()
        frame_count += 1

    video.release()

    if frame_count == 0:
        raise ValueError("No frames were extracted from the video. Please check the video file.")

    model, preprocess, tokenizer = initialize_model(model_path=model_path, device=device)

    dataset = FalsePositiveDataset(frames, preprocess, correct_classification, tokenizer, device)
    
    if len(dataset) == 0:
        raise ValueError("The dataset is empty. Please check the video frames extraction.")

    train_model(model, dataset, device=device)

    # Save the updated model
    save_model(model, model_path)
    print(f"Model saved to {model_path}")
