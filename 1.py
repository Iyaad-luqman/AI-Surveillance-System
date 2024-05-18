import torch
from PIL import Image
import open_clip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')

def classify_images(image_paths):
    # Load the model and tokenizer

    default_category = "Unknown"
    threshold = 0.85
    # Define the categories
    categories = [
        'people walking on a street',
        'buildings',
        'fight on a street',
        'fire on a street',
        'street violence',
        'road',
        'car crash',
        'cars on a road',
        'car parking area',
         "snatching","kshdkankygey " , "theft", "People walking on road", "People passing by", "Accident" ,
        'cars',
        'office environment',
        'office corridor',
        'violence in office',
        'fire in office',
        'people talking',
        'people walking in office',
        'person walking in office',
        'group of people'
    ]

    for image_path in image_paths:
        # Preprocess the image
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
            top_category_index = similarity.argmax()
            if similarity[0, top_category_index] < threshold:  # Replace threshold with your desired value
                top_category = default_category
            else:
                top_category = categories[top_category_index]

        print(f"The image {image_path} is classified as: {top_category}")
        
classify_images(["testing-data/2.png"])