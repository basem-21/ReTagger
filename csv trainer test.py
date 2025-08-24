import os
import glob
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import onnx
import onnxruntime
import numpy as np
import copy
import json
import shutil

from onnx import helper
from onnx import numpy_helper

# --- Configuration ---
IMAGE_SIZE = 448
CONFIG_FILENAME = "csv trainer config.json"


# --- ALL CORE FUNCTIONS ---
def create_feature_extractor(original_onnx_path, output_path):
    if os.path.exists(output_path): return
    print("Creating temporary feature extractor for training...")
    model = onnx.load(original_onnx_path)
    graph = model.graph
    if not graph.node or graph.node[-1].op_type != 'Sigmoid': raise RuntimeError("Model structure error: Last node must be a Sigmoid.")
    linear_node_output = graph.node[-1].input[0]
    feature_node = next((n for n in reversed(graph.node[:-1]) if linear_node_output in n.output), None)
    if not feature_node: raise RuntimeError("Could not find the final linear (Gemm/MatMul) layer.")
    weight_name = feature_node.input[1]
    weight_initializer = next((i for i in graph.initializer if i.name == weight_name), None)
    if not weight_initializer: raise RuntimeError(f"Could not find weight initializer tensor: {weight_name}")
    input_feature_size = weight_initializer.dims[1]
    original_feature_output_name = feature_node.input[0]
    clean_feature_output_name = "features"
    producer_node = next((n for n in graph.node if original_feature_output_name in n.output), None)
    if producer_node:
        for i, name in enumerate(producer_node.output):
            if name == original_feature_output_name: producer_node.output[i] = clean_feature_output_name
    new_output_info = helper.make_tensor_value_info(name=clean_feature_output_name, elem_type=onnx.TensorProto.FLOAT, shape=['batch_size', input_feature_size])
    nodes_to_remove = [n for n in graph.node if n.name in [feature_node.name, graph.node[-1].name]]
    for node in nodes_to_remove: graph.node.remove(node)
    while(len(graph.output)): graph.output.pop()
    graph.output.append(new_output_info)
    used_initializers = {i for n in graph.node for i in n.input}
    initializers_to_keep = [i for i in graph.initializer if i.name in used_initializers]
    del graph.initializer[:]
    graph.initializer.extend(initializers_to_keep)
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print("Temporary feature extractor created successfully.")

def extract_and_save_features(extractor_path, image_paths, output_dir, transform):
    print(f"Extracting features from {len(image_paths)} images...")
    os.makedirs(output_dir, exist_ok=True)
    session = onnxruntime.InferenceSession(extractor_path, providers=['AzureExecutionProvider', 'CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    new_files_processed = 0
    for img_path in tqdm(image_paths, desc="Processing images"):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        output_feature_path = os.path.join(output_dir, f"{base_name}.npy")
        if os.path.exists(output_feature_path): continue
        new_files_processed += 1
        try:
            with Image.open(img_path) as img: image = img.convert("RGB")
            image_tensor = transform(image).unsqueeze(0).numpy()
            image_tensor = np.transpose(image_tensor, (0, 2, 3, 1))
            features = session.run(None, {input_name: image_tensor})[0]
            np.save(output_feature_path, features.squeeze())
        except Exception as e:
            print(f"Warning: Failed to process {img_path}. Error: {e}")
    if new_files_processed == 0:
        print("No new images to process. All features are cached.")

class FeatureDataset(Dataset):
    def __init__(self, df, d): self.df, self.d = df, d
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        p = self.df.iloc[i, 0]
        b = os.path.splitext(os.path.basename(p))[0]
        f = os.path.join(self.d, f"{b}.npy")
        return torch.from_numpy(np.load(f)), torch.tensor(self.df.iloc[i, 1:].values.astype('float32'))

class SimpleClassifier(nn.Module):
    def __init__(self, i, o): super().__init__(); self.head = nn.Linear(i, o)
    def forward(self, x): return self.head(x)

def process_folder_tags(p):
    if not p or not os.path.isdir(p): return [], []
    paths = glob.glob(os.path.join(p, '*.png')) + glob.glob(os.path.join(p, '*.jpg'))
    data = []
    for ip in tqdm(paths, desc="Reading tag files"):
        b = os.path.splitext(os.path.basename(ip))[0]
        tp = os.path.join(p, f"{b}.txt")
        if os.path.exists(tp):
            with open(tp, 'r', encoding='utf-8') as f: data.append({'path': ip, 'tags': {t.strip().lower() for t in f.read().split(',') if t.strip()}})
    return paths, data

def train_model(m, tr, v, d, lr, e):
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    best_loss = float('inf')
    best_w = copy.deepcopy(m.state_dict())
    for ep in range(e):
        m.train()
        t_loss = 0.0
        for feat, lab in tqdm(tr, desc=f"Epoch {ep+1}/{e} [Train]"):
            feat, lab = feat.to(d), lab.to(d)
            opt.zero_grad()
            out = m(feat)
            loss = crit(out, lab)
            loss.backward()
            opt.step()
            t_loss += loss.item()
        avg_t_loss = t_loss / len(tr)
        v_loss = float('inf')
        if v and len(v) > 0:
            m.eval()
            val_l = 0.0
            with torch.no_grad():
                for feat, lab in tqdm(v, desc=f"Epoch {ep+1}/{e} [Val]"):
                    feat, lab = feat.to(d), lab.to(d)
                    out = m(feat)
                    loss = crit(out, lab)
                    val_l += loss.item()
            v_loss = val_l / len(v)
        print(f"Epoch {ep+1}/{e} -> Train Loss: {avg_t_loss:.4f}, Val Loss: {v_loss if v_loss != float('inf') else 'N/A'}")
        if v_loss < best_loss:
            best_loss = v_loss
            best_w = copy.deepcopy(m.state_dict())
    m.load_state_dict(best_w)
    return m

def load_and_edit_config(p):
    if not os.path.exists(p):
        cfg = {"model_folder": "path/to/model", "train_folder": "path/to/train", "val_folder": "", "learning_rate": 0.0001, "epochs": 50, "batch_size": 16, "num_workers": 4, "keep_temp_folder": False}
        with open(p, 'w') as f: json.dump(cfg, f, indent=4)
        print(f"'{p}' not found. Default created. Please edit and run again."); exit()
    with open(p, 'r') as f: cfg = json.load(f)
    if 'keep_temp_folder' not in cfg: cfg['keep_temp_folder'] = False
    print("\n--- Current Config ---")
    for k, v in cfg.items(): print(f"{k}: {v}")
    if input("Change settings? (y/n): ").lower() == 'y':
        for k, v in cfg.items():
            nv = input(f"{k} [{v}]: ").strip()
            if nv:
                if isinstance(v, bool): cfg[k] = nv.lower() in ['true', 't', 'y', 'yes', '1']
                else: cfg[k] = type(v)(nv)
        with open(p, 'w') as f: json.dump(cfg, f, indent=4)
    return cfg

def perform_weight_transplant(original_model_path, trained_pytorch_head, num_new_classes):
    print("\n--- Starting Weight Transplant Surgery ---")
    model = onnx.load(original_model_path)
    graph = model.graph
    if not graph.node or graph.node[-1].op_type != 'Sigmoid': raise RuntimeError("Original model structure error.")
    linear_node_output = graph.node[-1].input[0]
    gemm_node = next((n for n in reversed(graph.node[:-1]) if linear_node_output in n.output), None)
    if not gemm_node: raise RuntimeError("Could not find Gemm node in original model.")
    old_weight_name = gemm_node.input[1]
    old_bias_name = gemm_node.input[2]
    print(f"Located target layer '{gemm_node.name}'. Old weights: '{old_weight_name}', Old bias: '{old_bias_name}'")
    trained_state_dict = trained_pytorch_head.state_dict()
    new_weights = trained_state_dict['head.weight'].cpu().numpy()
    new_biases = trained_state_dict['head.bias'].cpu().numpy()
    new_weight_tensor = numpy_helper.from_array(new_weights, name=old_weight_name)
    new_bias_tensor = numpy_helper.from_array(new_biases, name=old_bias_name)
    initializers_to_remove = {old_weight_name, old_bias_name}
    new_initializer_list = [init for init in graph.initializer if init.name not in initializers_to_remove]
    new_initializer_list.extend([new_weight_tensor, new_bias_tensor])
    del graph.initializer[:]
    graph.initializer.extend(new_initializer_list)
    print("New trained weights and biases have been transplanted into the model.")
    final_output = graph.output[0]
    final_output.type.tensor_type.shape.dim[1].dim_value = num_new_classes
    print(f"Updated model output shape for {num_new_classes} classes.")
    onnx.checker.check_model(model)
    onnx.save(model, "model.onnx")
    print("Weight transplant successful. Final model 'model.onnx' has been saved.")

# --- NEW FUNCTION FOR CUMULATIVE LEARNING ---
def extract_classifier_weights(onnx_model_path):
    """Extracts weight and bias tensors from the final linear layer of an ONNX model."""
    print("Attempting to extract existing classifier weights for continuous training...")
    try:
        model = onnx.load(onnx_model_path)
        graph = model.graph
        if not graph.node or graph.node[-1].op_type != 'Sigmoid':
            print("Warning: Model structure error, could not find Sigmoid. Cannot extract weights.")
            return None, None
        linear_node_output = graph.node[-1].input[0]
        gemm_node = next((n for n in reversed(graph.node[:-1]) if linear_node_output in n.output), None)
        if not gemm_node or gemm_node.op_type not in ['Gemm']:
            print("Warning: Could not find the final Gemm layer. Cannot extract weights.")
            return None, None
        weight_name, bias_name = gemm_node.input[1], gemm_node.input[2]
        weight_initializer = next((i for i in graph.initializer if i.name == weight_name), None)
        bias_initializer = next((i for i in graph.initializer if i.name == bias_name), None)
        if weight_initializer is None or bias_initializer is None:
            print("Warning: Could not find weight or bias initializers. Cannot extract weights.")
            return None, None
        weights = numpy_helper.to_array(weight_initializer)
        biases = numpy_helper.to_array(bias_initializer)
        print(f"Successfully extracted weights ({weights.shape}) and biases ({biases.shape}).")
        return weights, biases
    except Exception as e:
        print(f"An error occurred during weight extraction: {e}. Starting with a fresh model.")
        return None, None

# --- Main Execution (Re-engineered with Cumulative Learning) ---
if __name__ == "__main__":
    config = load_and_edit_config(CONFIG_FILENAME)
    model_folder, train_folder, val_folder, lr, epochs, batch_size, num_workers, keep_temp_folder = \
        config["model_folder"], config["train_folder"], config["val_folder"], config["learning_rate"], \
        config["epochs"], config["batch_size"], config["num_workers"], config["keep_temp_folder"]

    if not os.path.isdir(train_folder): print(f"FATAL: Training folder not found: '{train_folder}'"); exit()

    temp_dir = "finetuning_temp"
    if os.path.exists(temp_dir) and not keep_temp_folder:
        print("`keep_temp_folder` is False. Clearing old temporary directory...")
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    original_onnx_path = glob.glob(os.path.join(model_folder, '*.onnx'))[0]
    original_csv_path = glob.glob(os.path.join(model_folder, '*.csv'))[0]

    # Step 1: Create feature extractor & Pre-extract existing weights for continuous training
    temp_extractor_path = os.path.join(temp_dir, "feature_extractor.onnx")
    create_feature_extractor(original_onnx_path, temp_extractor_path)
    initial_weights, initial_biases = extract_classifier_weights(original_onnx_path)

    # Step 2: Prepare datasets and tags
    original_tags = pd.read_csv(original_csv_path)['name'].tolist()
    train_paths, train_data = process_folder_tags(train_folder)
    val_paths, val_data = process_folder_tags(val_folder)
    if not train_data: print(f"FATAL: No tagged images found in '{train_folder}'."); exit()

    all_custom_tags = set.union(*(d['tags'] for d in train_data + val_data))
    newly_added_tags = sorted(list(all_custom_tags - set(t.lower() for t in original_tags)))
    combined_tags_list = original_tags + newly_added_tags
    num_classes = len(combined_tags_list)
    print(f"\nInjecting {len(newly_added_tags)} new tags. Total classes: {num_classes}")

    def create_df(data, cols):
        rows = [[item['path']] + [1 if t.lower() in item['tags'] else 0 for t in cols] for item in data]
        return pd.DataFrame(rows, columns=['path'] + [c.lower() for c in cols])

    train_df = create_df(train_data, [t.lower() for t in combined_tags_list])
    val_df = create_df(val_data, [t.lower() for t in combined_tags_list])

    # Step 3: Pre-compute features
    features_dir = os.path.join(temp_dir, "features")
    image_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    extract_and_save_features(temp_extractor_path, train_paths + val_paths, features_dir, image_transform)

    # Step 4: Train the PyTorch head
    train_ds = FeatureDataset(train_df, features_dir)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(FeatureDataset(val_df, features_dir), batch_size=batch_size) if not val_df.empty else None

    input_feature_size = np.load(os.path.join(features_dir, os.listdir(features_dir)[0])).shape[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier_head = SimpleClassifier(input_feature_size, num_classes).to(device)

    # --- CUMULATIVE LEARNING LOGIC ---
    if initial_weights is not None and initial_biases is not None:
        print("Initializing new classifier with previously trained weights...")
        num_original_classes = initial_weights.shape[0]
        new_state_dict = classifier_head.state_dict()
        new_weights = new_state_dict['head.weight']
        new_biases = new_state_dict['head.bias']
        
        print(f"Copying weights for {num_original_classes} original classes.")
        new_weights[:num_original_classes, :] = torch.from_numpy(initial_weights)
        new_biases[:num_original_classes] = torch.from_numpy(initial_biases)

        new_state_dict['head.weight'] = new_weights
        new_state_dict['head.bias'] = new_biases
        classifier_head.load_state_dict(new_state_dict)
        print("Weight transfer complete. Ready for fine-tuning.")
    
    trained_head = train_model(classifier_head, train_loader, val_loader, device, lr, epochs)

    # Step 5: Perform the "Weight Transplant"
    perform_weight_transplant(original_onnx_path, trained_head, num_classes)

    # Step 6: Save the final, compatible tags file
    print("Creating final 4-column CSV for ComfyUI compatibility...")
    num_tags = len(combined_tags_list)
    final_df = pd.DataFrame({'tag_id': list(range(num_tags)), 'name': combined_tags_list,
                             'category': [9, 9, 9, 9] + [0] * (num_tags - 4), 'count': [0] * num_tags})
    final_df.to_csv("model.csv", index=False)
    print("Final tags saved to 'model.csv'.")

    # Step 7: Clean up temporary files (or not)
    if not keep_temp_folder:
        print(f"Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)
    else:
        print(f"Keeping temporary directory as configured: {temp_dir}")

    print("\n--- ✅ All-in-One Process Complete ---")
    print("Your final, robust model is ready: model.onnx, model.csv")