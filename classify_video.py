import torch
from PIL import Image
import open_clip
import time
import cv2
from tqdm import tqdm
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import os
from moviepy.editor import VideoFileClip

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
    with torch.no_grad():
    # with torch.no_grad(), torch.cuda.amp.autocast():
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
        hours, remainder = divmod(time_in_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = (seconds % 1) * 1000
        time_string = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}:{int(milliseconds):03d}"

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
        
def group_categories(category_dict):
    grouped_dict = {}
    prev_category = None
    start_time = None
    categories = [
        'car crash',
        'Cars passing by',
        'Unknown'
    ]
    true_case = [ 
        True,
        False,
        False
    ]

    for time_string, category in category_dict.items():
        if category != prev_category:
            if prev_category is not None and true_case[categories.index(prev_category)]:
                grouped_dict[f"{start_time}-{prev_time}"] = prev_category
            start_time = time_string
        prev_category = category
        prev_time = time_string

    # Add the last category
    if prev_category is not None and true_case[categories.index(prev_category)]:
        grouped_dict[f"{start_time}-{prev_time}"] = prev_category
    print('Before:', grouped_dict)
    grouped_dict = adjust_timeframes(grouped_dict)
    trim_videos("testing-data/accident.mp4", grouped_dict, "output")
    return grouped_dict

def adjust_timeframes(input_dict):
    output_dict = {}
    
    for time_range, event in input_dict.items():
        start_time_str, end_time_str = time_range.split('-')
        
        # Convert time strings to seconds
        start_time = convert_to_seconds(start_time_str)
        end_time = convert_to_seconds(end_time_str)
        
        # Calculate duration
        duration = end_time - start_time
        
        # If duration is less than 3 seconds
        if duration < 3:
            # Calculate how much to subtract/add to start/end times
            diff = (4 - duration) / 2
            
            # Update start and end times
            start_time -= diff
            end_time += diff
            
            # Format adjusted times back to string
            adjusted_start_time_str = convert_to_time_string(start_time)
            adjusted_end_time_str = convert_to_time_string(end_time)
            
            # Update output dictionary
            output_dict[f"{adjusted_start_time_str}-{adjusted_end_time_str}"] = event
        else:
            # If duration is already 3 seconds or more, keep the original time range
            output_dict[time_range] = event
    
    return output_dict


def convert_to_time_string(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{milliseconds:03d}"

def trim_videos(input_video_path, grouped_dict, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    name_count = {}

    for time_range, name in grouped_dict.items():
        start_time_str, end_time_str = time_range.split('-')

        # Convert time strings to seconds
        start_time = convert_to_seconds(start_time_str)
        end_time = convert_to_seconds(end_time_str)

        # Ensure we have unique names for each trimmed video
        if name not in name_count:
            name_count[name] = 1
        else:
            name_count[name] += 1

        output_filename = f"{name.replace(' ', '_')}-{name_count[name]}.mp4"
        output_path = os.path.join(output_folder, output_filename)

        # Load video and trim
        with VideoFileClip(input_video_path) as video:
            trimmed_clip = video.subclip(start_time, end_time)
            trimmed_clip.write_videofile(output_path, codec="libx264")

def convert_to_seconds(time_str):
    hours, minutes, seconds, milliseconds = map(int, time_str.split(':'))
    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000

result = classify_videos("testing-data/accident.mp4")

result = group_categories(result)
print(result)