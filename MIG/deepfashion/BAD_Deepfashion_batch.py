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

def preprocess_images_batch(image_paths, device, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    all_batches = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        for image_path in batch_paths:
            try:
                image = Image.open(image_path).convert('RGB')
                image_tensor = transform(image)
                batch_images.append(image_tensor)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue

        if batch_images:
            # Stack tensors into a batch
            batch_tensor = torch.stack(batch_images).to(device)
            all_batches.append((batch_tensor, batch_paths))

    return all_batches

def load_classes(class_file):
    with open(class_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

def get_available_memory():
    """Get available GPU memory in MB."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        free_memory = torch.cuda.mem_get_info()[0] / (1024 * 1024)  # Convert to MB
        return free_memory
    return 0  # Return 0 for CPU

def calculate_safe_batch_size(current_batch_size, available_memory):
    """Dynamically adjust batch size based on available memory."""
    # If available memory is less than 500MB, reduce batch size significantly
    if available_memory < 500:
        return max(1, current_batch_size // 8)
    elif available_memory < 1000:
        return max(1, current_batch_size // 4)
    elif available_memory < 2000:
        return max(1, current_batch_size // 2)
    return current_batch_size

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

    # Get initial available memory and set dynamic batch size
    initial_memory = get_available_memory()
    initial_batch_size = 128  # Start with a smaller default batch size
    batch_size = calculate_safe_batch_size(initial_batch_size, initial_memory)
    print(f"Available GPU memory: {initial_memory:.2f} MB")
    print(f"Processing images in batches of {batch_size}...")

    # Process images in smaller chunks to avoid memory issues
    processed_images = 0
    total_images = len(image_paths)
    chunk_size = 5000  # Process 5000 images at a time

    for chunk_start in range(0, total_images, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_images)
        chunk_paths = image_paths[chunk_start:chunk_end]

        # Prepare data in batches for this chunk
        batches = preprocess_images_batch(chunk_paths, device, batch_size)

        for batch_idx, (batch_tensor, batch_paths) in enumerate(batches):
            # Check available memory before processing
            available_memory = get_available_memory()
            if available_memory < 200 and torch.cuda.is_available():  # Critical memory threshold
                print(f"Low GPU memory ({available_memory:.2f} MB). Clearing cache...")
                torch.cuda.empty_cache()
                time.sleep(1)  # Short pause to allow memory cleanup

                # Recalculate batch size if needed
                new_batch_size = calculate_safe_batch_size(batch_size, get_available_memory())
                if new_batch_size < batch_size:
                    print(f"Reducing batch size from {batch_size} to {new_batch_size}")
                    batch_size = new_batch_size

                    # If we had to reduce batch size, reprocess the current chunk with new batch size
                    if batch_idx > 0:
                        remaining_paths = []
                        for remaining_batch in batches[batch_idx:]:
                            remaining_paths.extend(remaining_batch[1])
                        batches = preprocess_images_batch(remaining_paths, device, batch_size)
                        batch_idx = 0  # Reset the batch index
                        continue

            try:
                # Process the batch
                with torch.no_grad():
                    outputs = model(batch_tensor)

                # Get predictions
                _, predicted_indices = torch.max(outputs, 1)

                for idx, image_path in enumerate(batch_paths):
                    predicted_class = classes[predicted_indices[idx].item()]
                    #print(f"Processed {image_path}: Predicted class: {predicted_class}")

                # Update count and show progress
                processed_images += len(batch_paths)
                if processed_images % 100 == 0:
                    elapsed = time.time() - start_time
                    print(f"Processed {processed_images}/{total_images} images " +
                          f"({(processed_images/total_images)*100:.1f}%) " +
                          f"in {elapsed:.2f}s")

                # Clear memory
                del outputs, predicted_indices, batch_tensor
                if batch_idx % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e) and batch_size > 1:
                    print(f"GPU OOM error. Reducing batch size and retrying...")
                    torch.cuda.empty_cache()
                    # Cut batch size in half
                    batch_size = max(1, batch_size // 2)
                    print(f"New batch size: {batch_size}")

                    # Reprocess the current chunk with new batch size
                    remaining_paths = []
                    for remaining_batch in batches[batch_idx:]:
                        remaining_paths.extend(remaining_batch[1])
                    batches = preprocess_images_batch(remaining_paths, device, batch_size)
                    batch_idx = 0  # Reset the batch index
                else:
                    print(f"Error processing batch: {e}")

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
