
# SurvAI - AI Automated CCTV Surveillance System


## Overview of the Project
The AI Automated CCTV Surveillance system is designed to process CCTV footage, detect incidents based on user-provided descriptions, and extract relevant video clips for review. This system combines natural language processing (NLP) and computer vision to analyze and extract meaningful data from video feeds. It is implemented as a Flask web application, allowing users to upload CCTV footage, enter a description of the incident, and view the results on a web interface.

## Features

**User-Friendly Web Interface:** Allows users to upload CCTV footage and enter incident descriptions in natural language.

**NLP Integration:** Uses advanced models to understand and extract keywords from user-provided descriptions.

**Video Processing:** Converts CCTV footage into frames, removes duplicate frames, and processes them to detect incidents.

**Incident Detection:** Compares text and image features to identify incidents in the footage.

**Result Extraction and Saving:** Extracts relevant video clips based on detected incidents and allows users to view and save the results.

**False Positive Reporting:** Users can report false positives to fine-tune the model for better accuracy.

## How It Works (Detailed Workflow)

### User Input

**Prompt (Natural Language):** The user provides a description of the incident they are interested in detecting within the CCTV footage. This prompt is given in natural language.

**CCTV Footage:** The user uploads CCTV footage through the web interface. This footage is then used for analysis.

### Preprocessing

**Convert Video to Frames:** The uploaded video is converted into individual frames. This process involves reading the video file and extracting frames at regular intervals. Duplicate frames are removed to reduce redundancy and optimize processing.

### Feature Extraction

**Normalize Text Features:** The user-provided prompt is processed using NLP models (Gemini or Ollama). These models extract possible keywords and normalize the text features, creating a representation of the incident description that the system can work with.

**Normalize Image Features:** Image features are extracted from the video frames using a model like OpenCLIP. This involves converting each frame into a format that can be analyzed for features relevant to the incident description.

### Incident Detection

**Calculate Similarity:** The system calculates the similarity between the normalized text features (from the incident description) and the image features (from the video frames). This comparison helps in identifying frames or sequences of frames that match the incident description with a high degree of similarity.

**Extract Incident Clips:** When a match is found, the system extracts the relevant part of the video corresponding to the detected incident. If the duration of the detected incident is less than four seconds, the system extends the clip before and after to fit a minimum duration of four seconds.

### Postprocessing

**Save and View Results:** The extracted clips are saved and labeled with the detected incident type. Users can view these results on the web interface and have the option to save them for future reference.

**False Positive Reporting:** Users can report false positives through the web interface. This feedback is used to fine-tune the model, improving its accuracy over time.

## Summary

The AI Automated CCTV Surveillance system offers a comprehensive solution for automated incident detection in CCTV footage. By leveraging advanced NLP and computer vision techniques, it provides an efficient and user-friendly platform for monitoring and analyzing video feeds. The flowchart illustrates the seamless integration of various components, from user input to incident detection and result extraction, ensuring accurate and timely surveillance outcomes.