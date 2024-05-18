import torch
from PIL import Image
import open_clip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

# Load the model and tokenizer
model,_, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# Define the categories
categories = ["animal", "dogs", "dog"]  # Replace with your categories

# Preprocess the image
image_path = "what.jpg"  # Replace with your image path
image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0)

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
    top_category = categories[similarity.argmax()]

print(f"The image is classified as: {top_category}")
    