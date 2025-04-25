#!/usr/bin/env python3
import os
import torch
import urllib.request
from PIL import Image
from torchvision import transforms

def download_if_missing(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename} ...")
        urllib.request.urlretrieve(url, filename)

def main():
    print("Available ResNeSt models:")
    print(torch.hub.list('zhanghang1989/ResNeSt', force_reload=True))

    model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
    model.eval()

    img_url = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
    img_file = "dog.jpg"
    download_if_missing(img_url, img_file)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])
    input_image = Image.open(img_file).convert("RGB")
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    input_batch = input_batch.to(device)

    with torch.no_grad():
        output = model(input_batch)[0] 

    probabilities = torch.nn.functional.softmax(output, dim=0)
    top5_prob, top5_catid = torch.topk(probabilities, 5)


    labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    labels_file = "imagenet_classes.txt"
    download_if_missing(labels_url, labels_file)
    with open(labels_file, "r") as f:
        categories = [s.strip() for s in f.readlines()]

    print("\nTop-5 Predictions:")
    for i, (prob, catid) in enumerate(zip(top5_prob, top5_catid), 1):
        print(f"{i}. {categories[catid]} — {prob.item():.4f}")

if __name__ == "__main__":
    main()

