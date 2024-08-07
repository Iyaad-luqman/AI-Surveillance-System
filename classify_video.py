import torch
from PIL import Image
import open_clip
import time
import cv2
from tqdm import tqdm
import os
from moviepy.editor import VideoFileClip
import numpy as np
from datetime import datetime


def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)


def load_model(model, filepath, device='cpu'):
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.eval()
    return model

def initialize_model(model_name='ViT-B-32', pretrained='laion2b_s34b_b79k', model_path='fine_tuned_model.pth', device='cpu'):
    model,_, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    
    # if os.path.exists(model_path):
    #     print(f"Loading fine-tuned model from {model_path}")
    #     model = load_model(model, model_path, device)
    # else:
    #     print("No fine-tuned model found. Using pre-trained model.")
    
    return model, preprocess, tokenizer


def classify_frame(frame, frame_count, fps, categories, model, preprocess, tokenizer,  device='cpu'):
    # start_time = time.time()
    default_category = "Unknown"
    threshold = 0.57
    # Define the categories
        
    # Preprocess the frame
    image = preprocess(Image.fromarray(frame).convert("RGB")).unsqueeze(0).to(device)

    # Tokenize the categories
    text_tokens = tokenizer(categories)

    # Perform inference
    if device == 'cuda':    
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
            hours, remainder = divmod(time_in_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            milliseconds = (seconds % 1) * 1000
            time_string = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}:{int(milliseconds):03d}"
    else:
        with torch.no_grad():
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

def classify_videos(video_path, categories, true_case, dir_name, model_path='fine_tuned_model.pth', remove_duplicate_frames=False, skip_seconds=0.5, device='cpu'):
    start_time = time.time()
    
    model, preprocess, tokenizer = initialize_model(model_path=model_path, device=device)
    
    
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)  # Get the frames per second of the video
    skip_frames = int(fps * skip_seconds)  # Calculate the number of frames to skip

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # Get the total number of frames in the video
    processed_frames = 0

    pbar = tqdm(total=total_frames)  # Initialize the progress bar

    category_dict = {}  # Initialize the dictionary to store the top category for each processed frame

    success, frame = video.read()
    if not remove_duplicate_frames :
        while success:
            if processed_frames % skip_frames == 0:
                top_category, time_string = classify_frame(frame, processed_frames, fps, categories, model, preprocess, tokenizer, device=device)
                category_dict[time_string] = top_category
            success, frame = video.read()
            processed_frames += 1
            pbar.update(1) 
    else:
        prev_frame = None
        while success:
            if processed_frames % skip_frames == 0:
                if prev_frame is not None and np.sum(np.abs(frame - prev_frame)) < 0.95:
                    continue
                top_category, time_string = classify_frame(frame, processed_frames, fps, categories, model, preprocess, tokenizer, device=device)
                category_dict[time_string] = top_category
            prev_frame = frame
            success, frame = video.read()
            processed_frames += 1
            pbar.update(1)  ## Update the progress bar

    pbar.close()  # Close the progress bar when done
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    category_dict = group_categories(category_dict, categories, true_case, dir_name)
    return category_dict
        
def group_categories(category_dict, categories, true_case,dir_name):
    grouped_dict = {}
    prev_category = None
    start_time = None
    true_case = [True if x.lower() == 'true' else False if x.lower() == 'false' else x for x in true_case]
    for time_string, category in category_dict.items():
        if category != prev_category:
            if prev_category is not None and prev_category in categories and true_case[categories.index(prev_category)]:
                grouped_dict[f"{start_time}-{prev_time}"] = prev_category
            start_time = time_string
        prev_category = category
        prev_time = time_string

    # Add the last category
    if prev_category is not None and prev_category in categories and true_case[categories.index(prev_category)]:
        grouped_dict[f"{start_time}-{prev_time}"] = prev_category
    print('Before:', grouped_dict)
    grouped_dict = adjust_timeframes(grouped_dict)
    grouped_dict = merge_overlapping_timeframes(grouped_dict)
    print('After:', grouped_dict)
    trim_videos(f"static/run-test/{dir_name}/original.mp4", grouped_dict, f"static/run-test/{dir_name}" )
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
        if duration < 2:
            # Calculate how much to subtract/add to start/end times
            diff = (2 - duration) / 2
            
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

def merge_overlapping_timeframes(timeframes):
    # Convert the timeframes to a list of tuples with start and end times as datetime objects
    timeframes_list = [(datetime.strptime(start, "%H:%M:%S:%f"), datetime.strptime(end, "%H:%M:%S:%f"), category)
        for timeframe, category in timeframes.items() if timeframe.count('-') == 1 for start, end in [timeframe.split('-')]]

    # Sort the list by start time
    timeframes_list.sort()

    merged_timeframes = []
    for timeframe in timeframes_list:
        # If the list of merged timeframes is empty, or the current timeframe does not overlap with the previous one, append it to the list
        if not merged_timeframes or timeframe[0] > merged_timeframes[-1][1]:
            merged_timeframes.append(list(timeframe))
        else:
            # Otherwise, merge the current timeframe with the previous one by extending the end time
            merged_timeframes[-1][1] = max(merged_timeframes[-1][1], timeframe[1])

    # Convert the merged timeframes back to the original format
    merged_timeframes_dict = {f"{start.strftime('%H:%M:%S:%f')[:-3]}-{end.strftime('%H:%M:%S:%f')[:-3]}": category for start, end, category in merged_timeframes}

    return merged_timeframes_dict

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

# result = classify_videos("testing-data/accident.mp4")

# result = group_categories(result)
# print(result)