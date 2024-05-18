import torch
from PIL import Image
import open_clip
import time
import cv2
from tqdm import tqdm
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')


def classify_frame(frame, frame_count, fps):
    # start_time = time.time()
    default_category = "Unknown"
    threshold = 0.47
    # Define the categories
    categories = [
        'car crash',
        'Cars passing by',
        'Unknown'
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
            # Calculate the time in the video for the current frame
    time_in_seconds = frame_count / fps
    minutes, seconds = divmod(time_in_seconds, 60)
    milliseconds = (seconds % 1) * 1000
    time_string = f"{int(minutes)}:{int(seconds)}:{int(milliseconds)}"

    return top_category, time_string

    # print(f"The frame in {video_path} is classified as: {top_category}")
    # print(f"Time taken: {end_time - start_time} seconds")

def classify_videos(video_path, skip_seconds=0.5):
    start_time = time.time()
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)  # Get the frames per second of the video
    skip_frames = int(fps * skip_seconds)  # Calculate the number of frames to skip

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # Get the total number of frames in the video
    processed_frames = 0

    pbar = tqdm(total=total_frames)  # Initialize the progress bar

    category_dict = {}  # Initialize the dictionary to store the top category for each processed frame

    success, frame = video.read()
    while success:
        if processed_frames % skip_frames == 0:
            top_category, time_string = classify_frame(frame, processed_frames, fps)
            category_dict[time_string] = top_category
        success, frame = video.read()
        processed_frames += 1
        pbar.update(1)  # Update the progress bar

    pbar.close()  # Close the progress bar when done
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

    return category_dict
        
result = classify_videos("testing-data/accident.mp4")

print(result)
# classify_videos("exdata/long_accident.mp4")
