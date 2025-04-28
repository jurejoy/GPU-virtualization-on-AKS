import os
import argparse
import time
import torch
import torch.nn as nn
from tqdm import tqdm
import PIL
import torchvision.transforms as transforms
from PIL import Image
import warnings

# Set CUDA environment variables to limit cores
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# By default, we'll use first GPU only
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

class ECenterCrop:
    """Crop the given PIL Image and resize it to desired size."""
    def __init__(self, imgsize):
        self.imgsize = imgsize
        self.resize_method = transforms.Resize((imgsize, imgsize), interpolation=PIL.Image.BICUBIC)

    def __call__(self, img):
        image_width, image_height = img.size
        image_short = min(image_width, image_height)

        crop_size = float(self.imgsize) / (self.imgsize + 32) * image_short

        crop_height, crop_width = crop_size, crop_size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        img = img.crop((crop_left, crop_top, crop_left + crop_width, crop_top + crop_height))
        return self.resize_method(img)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        avg = 0 if self.count == 0 else self.sum / self.count
        return avg

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def load_classes(class_file):
    with open(class_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

class DeepFashionDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, 0  # Placeholder target for inference-only
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder image
            dummy_image = torch.zeros(3, 224, 224)
            return dummy_image, 0

def load_test_data(test_file):
    with open(test_file, 'r') as file:
        image_paths = file.readlines()
    image_paths = [path.strip() for path in image_paths]
    return image_paths

def parse_args():
    parser = argparse.ArgumentParser(description='DeepFashion Batch Processing')
    parser.add_argument('--crop-size', type=int, default=64,  # Further reduced to 64
                        help='crop image size')
    parser.add_argument('--workers', type=int, default=1,      # Reduced to 1
                        help='dataloader threads')
    parser.add_argument('--no-cuda', action='store_true',
                        default=False, help='disables CUDA training')
    parser.add_argument('--model', type=str, default='mobilenet_v2',  # Switched to even lighter model
                        help='network model type (default: mobilenet_v2)')
    parser.add_argument('--test-file', type=str, default='./Anno_fine/test.txt',
                        help='path to test file list')
    parser.add_argument('--class-file', type=str, default='./Anno_coarse/list_category_img.txt',
                        help='path to class file')
    parser.add_argument('--gpu-id', type=str, default='0',
                        help='GPU ID to use (e.g. "0" or "0,1")')
    parser.add_argument('--batch-size', type=int, default=2,   # Further reduced to 2
                        help='batch size for inference')
    parser.add_argument('--threads-per-core', type=int, default=1,
                        help='number of threads per GPU core')
    parser.add_argument('--sleep-interval', type=float, default=0.05,  # Added sleep parameter
                        help='sleep interval between batches (seconds)')
    parser.add_argument('--num-loops', type=int, default=2,    # Reduced loops to 2
                        help='number of loops through the dataset')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set GPU ID to use
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    
    # Limit number of threads to reduce core utilization
    if torch.cuda.is_available():
        torch.set_num_threads(args.threads_per_core)
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    
    print(f"Using device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Model: {args.model}")
    print(f"Image size: {args.crop_size}")
    print(f"GPU ID: {args.gpu_id}")
    print(f"Threads per core: {args.threads_per_core}")
    print(f"Sleep interval: {args.sleep_interval}")
    print(f"Number of loops: {args.num_loops}")
    
    # Start timing
    start_time = time.time()
    
    # Load class list
    try:
        classes = load_classes(args.class_file)
        print(f"Loaded {len(classes)} classes")
    except Exception as e:
        print(f"Error loading class file: {e}")
        classes = ["unknown"]
    
    # Prepare model - use smaller models to reduce compute requirements
    try:
        if args.model in ['mobilenet_v2', 'mobilenet_v3_small', 'efficientnet_b0', 'shufflenet_v2_x0_5']:
            # Use more efficient models from torchvision
            import torchvision.models as models
            model_fn = getattr(models, args.model, None)
            if model_fn is not None:
                model = model_fn(pretrained=True)
            else:
                print(f"Model {args.model} not found in torchvision, falling back to mobilenet_v2")
                model = models.mobilenet_v2(pretrained=True)
        elif args.model.startswith('resnet'):
            # Use torchvision resnet models
            import torchvision.models as models
            model_fn = getattr(models, args.model, None)
            if model_fn is not None:
                model = model_fn(pretrained=True)
            else:
                print(f"Model {args.model} not found in torchvision, falling back to ResNeSt")
                model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest18', pretrained=True)
        else:
            # Fall back to original ResNeSt models but use smallest one
            model_name = args.model if args.model in ['resnest18', 'resnest50'] else 'resnest18'
            model = torch.hub.load('zhanghang1989/ResNeSt', model_name, pretrained=True)
        
        # Set to eval mode before sending to device
        model.eval()
        model = model.to(device)
        
        # Don't use DataParallel to limit core usage
        print(f"Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Use simpler transforms with smaller image size
    transform = transforms.Compose([
        ECenterCrop(args.crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    # Load test data
    try:
        image_paths = load_test_data(args.test_file)
        print(f"Loaded {len(image_paths)} test images")
    except Exception as e:
        print(f"Error loading test file: {e}")
        return
    
    # Create dataset and dataloader
    dataset = DeepFashionDataset(image_paths, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.workers, 
        pin_memory=args.cuda
    )
    
    # Process batches
    processed_images = 0
    processing_time = AverageMeter()
    
    # Use number of loops from args
    num_loops = args.num_loops
    print(f"Looping through the dataset {num_loops} times")
    
    with torch.no_grad():
        for loop in range(num_loops):
            print(f"\nLoop {loop+1}/{num_loops}")
            tbar = tqdm(dataloader, desc=f'Processing (loop {loop+1}/{num_loops})')
            for batch_idx, (data, _) in enumerate(tbar):
                batch_start = time.time()
                
                # Move data to device
                data = data.to(device)
                
                # Forward pass
                output = model(data)
                
                # Get predictions
                _, predicted_idx = torch.max(output, 1)
                
                # Update statistics
                batch_time = time.time() - batch_start
                batch_size = data.size(0)
                processing_time.update(batch_time, batch_size)
                processed_images += batch_size
                
                tbar.set_description(f'Loop {loop+1}/{num_loops} | Avg time/img: {processing_time.avg/batch_size:.4f}s')
                
                # Insert sleep after every batch to reduce GPU utilization
                time.sleep(args.sleep_interval)
    
    # Calculate total execution time and display summary
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n--- Execution Summary ---")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Images processed: {processed_images}")
    print(f"Dataset size: {len(image_paths)}")
    print(f"Number of loops: {num_loops}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.crop_size}")
    print(f"Average time per image: {total_time/processed_images:.4f} seconds")
    print(f"Average batch processing time: {processing_time.avg:.4f} seconds")
    print(f"Using device: {device}")
    print(f"Model: {args.model}")

if __name__ == "__main__":
    main()
