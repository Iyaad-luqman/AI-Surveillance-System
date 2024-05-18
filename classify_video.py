import torch
from PIL import Image
import open_clip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import time
import cv2
import concurrent.futures
from tqdm import tqdm
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')


def classify_frame(frame, video_path):
    start_time = time.time()
    default_category = "Unknown"
    threshold = 0.25
    # Define the categories
    categories = [
        'fight on a street',
        'fire on a street',
        'car crash',
        "Accident" ,
        'violence in office',
        'fire in office',
    ]
    # Preprocess the frame
    image = preprocess(Image.fromarray(frame).convert("RGB")).unsqueeze(0)

    # Tokenize the categories
    text_tokens = tokenizer(categories)

    # Perform inference
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tokens)

        # Normalize features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Calculate the similarity between image and text features
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        # Get the highest probability category
        top_category_index = similarity.argmax()
        if similarity[0, top_category_index] < threshold:  # Replace threshold with your desired value
            top_category = default_category
        else:
            top_category = categories[top_category_index]

    end_time = time.time()

    print(f"The frame in {video_path} is classified as: {top_category}")
    print(f"Time taken: {end_time - start_time} seconds")

def classify_videos(video_path, skip_seconds=0.1):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)  # Get the frames per second of the video
    skip_frames = int(fps * skip_seconds)  # Calculate the number of frames to skip

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # Get the total number of frames in the video
    processed_frames = 0

    pbar = tqdm(total=total_frames)  # Initialize the progress bar

    success, frame = video.read()
    while success:
        if processed_frames % skip_frames == 0:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                executor.submit(classify_frame, frame, video_path)
        success, frame = video.read()
        processed_frames += 1
        pbar.update(1)  # Update the progress bar

    pbar.close() 
        
classify_videos("testing-data/accident.mp4")