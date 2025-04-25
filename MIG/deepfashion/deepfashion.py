import os
import torch
import time
from PIL import Image
from torchvision import transforms

def load_test_data(test_file):
    with open(test_file, 'r') as file:
        image_paths = file.readlines()
    image_paths = [path.strip() for path in image_paths]
    return image_paths

def preprocess_image(image_path, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image.unsqueeze(0).to(device)

def load_classes(class_file):
    with open(class_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

def main():
    # Start timing the execution
    start_time = time.time()
    
    # Set device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    class_file = './Anno_coarse/list_category_img.txt' 
    classes = load_classes(class_file)
    
    model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
    # Move model to the selected device
    model = model.to(device)
    model.eval()  
    
    test_file = './Anno_fine/test.txt'  
    image_paths = load_test_data(test_file)
    
    processed_images = 0
    for image_path in image_paths:
        image = preprocess_image(image_path, device)
        with torch.no_grad():
            output = model(image)  
        # 获取预测的类别
        _, predicted_idx = torch.max(output, 1)
        predicted_class = classes[predicted_idx.item()]
        
        #print(f"Processed {image_path}: Predicted class: {predicted_class}")
        processed_images += 1
    
    # Calculate total execution time and display summary
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n--- Execution Summary ---")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Images processed: {processed_images}")
    print(f"Average time per image: {total_time/processed_images:.4f} seconds")
    print(f"Using device: {device}")

if __name__ == "__main__":
    main()


