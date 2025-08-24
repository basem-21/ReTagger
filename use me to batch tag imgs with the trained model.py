import os
import glob
import onnxruntime
import pandas as pd
from PIL import Image
from torchvision import transforms
import numpy as np
from tqdm import tqdm


def get_user_input_yes_no(prompt):
    while True:
        response = input(prompt).strip().lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print("Invalid input. Please enter 'y' or 'n'.")


if __name__ == "__main__":
    print("--- Bulk Image Tagger (Auto-detect ONNX + CSV) ---")
    base_dir = input("Enter the path to the folder containing model + csv: ").strip()
    image_dir = input("Enter the path to the folder of images to tag: ").strip()

    # Confidence threshold
    while True:
        try:
            threshold_str = input("Enter the confidence threshold (e.g., 0.35, 0.5): ")
            confidence_threshold = float(threshold_str)
            if 0.0 < confidence_threshold < 1.0:
                break
            else:
                print("Please enter a value between 0.0 and 1.0.")
        except ValueError:
            print("Invalid number. Please try again.")

    '''exclude_characters = get_user_input_yes_no("Exclude character tags? [y/n]: ")
    replace_underscores = get_user_input_yes_no("Replace underscores '_' with spaces? [y/n]: ")'''
    exclude_characters = str('n')
    replace_underscores = str('n')

    # --- Auto-detect model and csv ---
    onnx_files = glob.glob(os.path.join(base_dir, "*.onnx"))
    csv_files = glob.glob(os.path.join(base_dir, "*.csv"))

    if not onnx_files:
        print("Error: No .onnx model file found in", base_dir)
        exit()
    if not csv_files:
        print("Error: No .csv tag file found in", base_dir)
        exit()

    model_path = onnx_files[0]
    tags_csv_path = csv_files[0]

    print(f"\nUsing model: {model_path}")
    print(f"Using tags: {tags_csv_path}")

    # --- Load Tags ---
    tags_df = pd.read_csv(tags_csv_path)
    all_tags = tags_df['name'].tolist()

    character_tags = set()
    if exclude_characters and 'category' in tags_df.columns:
        character_tags = set(tags_df[tags_df['category'] == 4]['name'])
        print(f"Found {len(character_tags)} character tags to filter.")

    # --- Load ONNX model ---
    print("\nLoading ONNX model...")

    available_providers = onnxruntime.get_available_providers()
    if "CUDAExecutionProvider" in available_providers:
        session = onnxruntime.InferenceSession(
            model_path,
            providers=['CUDAExecutionProvider']
        )
        device_str = "CUDA (GPU)"
    else:
        session = onnxruntime.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        device_str = "CPU"

    print(f"Model loaded successfully. Using device: {device_str}")

    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    image_size = input_shape[1]
    print(f"Expected image size: {image_size}x{image_size}")

    # --- Image Transform ---
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # --- Collect Images ---
    image_paths = glob.glob(os.path.join(image_dir, '*.png')) + \
                  glob.glob(os.path.join(image_dir, '*.jpg')) + \
                  glob.glob(os.path.join(image_dir, '*.jpeg'))

    if not image_paths:
        print(f"Error: No images found in '{image_dir}'.")
        exit()

    print(f"\nFound {len(image_paths)} images to process...")

    # --- Process Images ---
    for img_path in tqdm(image_paths, desc="Tagging images"):
        try:
            with Image.open(img_path) as img:
                image = img.convert("RGB")

            tensor = transform(image).unsqueeze(0).numpy()
            tensor = np.transpose(tensor, (0, 2, 3, 1))  # NHWC for ONNX

            # Run inference
            outputs = session.run(None, {input_name: tensor})[0]
            probs = 1 / (1 + np.exp(-outputs))  # sigmoid

            # Select tags
            final_tags = []
            for i, prob in enumerate(probs.flatten()):
                if prob > confidence_threshold:
                    tag = all_tags[i]
                    if exclude_characters and tag in character_tags:
                        continue
                    if replace_underscores:
                        tag = tag.replace('_', ' ')
                    final_tags.append(tag)

            # Save results
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            txt_path = os.path.join(image_dir, f"{base_name}.txt")
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(", ".join(final_tags))

        except Exception as e:
            print(f"\nWarning: Failed to process image {img_path}. Error: {e}")

    print("\n--- Tagging Complete! ---")
    print(f"Text files with tags saved in: {image_dir}")
