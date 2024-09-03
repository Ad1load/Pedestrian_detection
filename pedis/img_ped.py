import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image, ImageDraw

def detect_pedestrians(image_path):
    # Load pre-trained Faster R-CNN model
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    # Load and preprocess the image
    image = Image.open(image_path)
    transform = transforms.Compose([transforms.ToTensor()])
    input_image = transform(image)
    input_image = input_image.unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        prediction = model(input_image)

    # Draw bounding boxes on the image
    draw = ImageDraw.Draw(image)
    for box in prediction[0]['boxes']:
        draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red")

    # Display the image with bounding boxes
    image.show()

if __name__ == "__main__":
    image_path = "/Users/adityasharma/Downloads/pdd_new/pedis/test1.webp"  # Replace "image.jpg" with the path to your image
    detect_pedestrians(image_path)
