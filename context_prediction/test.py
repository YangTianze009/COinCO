import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import argparse
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json
from datetime import datetime
from diffusers import AutoencoderKL
from dataset import CustomDataset, transform
from model import SemanticNet, VisualSemanticNet, VisualNet
from transformers import BertTokenizer, BertModel

def convert_tuples_to_strings(data):
    """
    Recursively converts all tuples in a nested structure to strings
    """
    # Base case: if data is a tuple, convert it to a string
    if isinstance(data, tuple):
        return str(data[0]) if len(data) == 1 else data
    
    # If data is a list, process each element
    elif isinstance(data, list):
        return [convert_tuples_to_strings(item) for item in data]
    
    # Return data as is for other types
    else:
        return data

class CachedEmbeddingDataset(Dataset):
    def __init__(self, embedding_dir, data_df):
        self.embedding_dir = embedding_dir
        self.data_df = data_df
        
        # Verify a sample embedding file
        sample_idx = data_df['coco_index'].iloc[0]
        sample_file_path = os.path.join(embedding_dir, f"{sample_idx}.pt")
        if not os.path.exists(sample_file_path):
            raise RuntimeError(f"Embedding file not found: {sample_file_path}")
            
        sample_data = torch.load(sample_file_path)
        required_keys = ['latent1', 'latent2', 'label', 'objects_embeddings', 'replacement_embedding']
        if not all(k in sample_data for k in required_keys):
            raise RuntimeError(f"Embedding files missing required data fields: {required_keys}")
    
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        coco_idx = self.data_df['coco_index'].iloc[idx]
        file_path = os.path.join(self.embedding_dir, f"{coco_idx}.pt")
        data = torch.load(file_path)
        
        return (data['latent1'], data['latent2'], data['label'], 
                data['objects_embeddings'], data['replacement_embedding'])

def prepare_test_embeddings(args, test_df, objects_df, device, embedding_dir):
    """Prepare and cache VAE and BERT embeddings for test data if they don't exist"""
    os.makedirs(embedding_dir, exist_ok=True)
    
    # Initialize BERT model for text embeddings
    print("Loading BERT model for text embeddings...")
    tokenizer = BertTokenizer.from_pretrained('/data2/Tianze/weights/bert-base-uncased')
    bert_model = BertModel.from_pretrained('/data2/Tianze/weights/bert-base-uncased').to(device)
    bert_model.eval()
    
    # Check which embeddings need to be created
    missing_indices = []
    for coco_idx in test_df['coco_index']:
        file_path = os.path.join(embedding_dir, f"{coco_idx}.pt")
        if not os.path.exists(file_path):
            missing_indices.append(coco_idx)
    
    if not missing_indices:
        print(f"All embeddings exist for this dataset, skipping encoding...")
        return
    
    print(f"Found {len(missing_indices)} missing embeddings. Processing...")
    
    # Initialize VAE model for image embeddings, choose your own path
    model_id = "/data2/Tianze/weights/stable-diffusion-2-1"
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device)
    vae.eval()
    
    # Filter data for missing indices
    missing_df = test_df[test_df['coco_index'].isin(missing_indices)].reset_index(drop=True)
    
    # Create mapping from coco_index to row index in objects_df for quick lookup
    objects_lookup = {}
    for _, row in objects_df.iterrows():
        objects_lookup[row['coco_index']] = row
    
    # Prepare data paths
    base_path = "../task_data"
    image_paths1 = [os.path.join(base_path, "images/testing_images", f"{idx}.png") 
                   for idx in missing_df['coco_index']]
    image_paths2 = [os.path.join(base_path, "masks/bbox_masks_testing", f"{idx}.png") 
                   for idx in missing_df['coco_index']]
    
    # Create dataset for processing
    dataset = CustomDataset(
        image_paths1,
        image_paths2,
        missing_df['label'].tolist(),
        objects_df,
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    pbar = tqdm(total=len(missing_indices), desc=f"Processing missing embeddings")
    
    with torch.no_grad():
        for batch_idx, (images1, images2, labels, objects_lists, replacement_objects, coco_indices) in enumerate(dataloader):
            if len(labels) == 1:
                objects_lists = [objects_lists]
                replacement_objects = [replacement_objects]

            objects_lists = convert_tuples_to_strings(objects_lists)
            replacement_objects = convert_tuples_to_strings(replacement_objects)

            images1, images2 = images1.to(device), images2.to(device)
            
            # Get VAE latents for images
            latent1 = vae.encode(images1).latent_dist.mean
            latent2 = vae.encode(images2).latent_dist.mean

            # Process each item in the batch
            for i in range(len(labels)):
                coco_idx = coco_indices[i].item()
                
                # Get BERT embeddings for objects
                objects_list = objects_lists[i]
                objects_embeddings = []
                replacement_object = replacement_objects[i]
                
                for obj in objects_list:
                    if obj and obj.strip():  # Check if object is not empty
                        
                        # Tokenize and get embedding
                        inputs = tokenizer(obj, return_tensors="pt", padding=True, 
                                         truncation=True, max_length=128).to(device)
                        outputs = bert_model(**inputs)
                        # Use [CLS] token embedding as object embedding
                        obj_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu()
                        objects_embeddings.append(obj_embedding)
                
                # If no valid objects, create a dummy empty tensor for consistency
                if objects_embeddings:
                    # average pooling
                    objects_embeddings = torch.stack(objects_embeddings).mean(dim=0)
                else:
                    objects_embeddings = torch.zeros(768)
                # Get embedding for replacement object
                if replacement_object and replacement_object.strip():
                    inputs = tokenizer(replacement_object, return_tensors="pt", padding=True, 
                                     truncation=True, max_length=128).to(device)
                    outputs = bert_model(**inputs)
                    replacement_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu()
                else:
                    replacement_embedding = torch.zeros(768)
                
                # Save all embeddings in a single file
                save_path = os.path.join(embedding_dir, f"{coco_idx}.pt")
                torch.save({
                    'latent1': latent1[i].cpu(),
                    'latent2': latent2[i].cpu(),
                    'label': labels[i].item(),
                    'objects_embeddings': objects_embeddings,
                    'replacement_embedding': replacement_embedding,
                }, save_path)
                
                pbar.update(1)
    
    pbar.close()
    del vae, bert_model
    torch.cuda.empty_cache()

def evaluate_model(model, dataloader, device, model_type):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for latent1, latent2, labels, objects_embeddings, replacement_embedding in tqdm(dataloader, desc="Evaluating"):
            latent1, latent2, labels = latent1.to(device), latent2.to(device), labels.to(device)
            objects_embeddings, replacement_embedding = objects_embeddings.to(device), replacement_embedding.to(device)
            
            # Process image latents
            latent1_flat = latent1.view(latent1.size(0), -1)
            latent2_flat = latent2.view(latent2.size(0), -1)
            
            # Use different model interface based on model_type
            if model_type == "semantic":
                outputs = model(objects_embeddings, replacement_embedding)
            elif model_type == "combine":
                outputs = model(latent1_flat, latent2_flat, objects_embeddings, replacement_embedding)
            elif model_type == "visual":
                outputs = model(latent1_flat, latent2_flat)
                
            predictions = (torch.sigmoid(outputs) >= 0.5).float()
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    conf_matrix = confusion_matrix(all_labels, all_preds)
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds) * 100,
        'precision': precision_score(all_labels, all_preds) * 100,
        'recall': recall_score(all_labels, all_preds) * 100,
        'f1': f1_score(all_labels, all_preds) * 100,
        'confusion_matrix': conf_matrix.tolist(),
        'tn': int(conf_matrix[0, 0]),
        'fp': int(conf_matrix[0, 1]),
        'fn': int(conf_matrix[1, 0]),
        'tp': int(conf_matrix[1, 1])
    }
    
    return metrics

def test_all_models(args):
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    base_path = "../task_data/context_prediction"
    data_path = os.path.join(base_path)
    test_df = pd.read_csv(os.path.join(data_path, 'testing_data.csv'))
    
    try:
        objects_test_df = pd.read_csv(os.path.join(data_path, 'testing_object_list.csv'))
        print("\nLoaded object data:")
        print(f"Testing objects: {len(objects_test_df)} rows")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Object list files are required. Please provide the object list files.")
        exit(1)
    
    print(f"\nTest data distribution:")
    print(test_df['label'].value_counts(normalize=True))
    print(f"Total test samples: {len(test_df)}")

    embedding_dir = "../task_data/cache/cached_embeddings_testing_cp"
    prepare_test_embeddings(args, test_df, objects_test_df, device, embedding_dir)

    test_dataset = CachedEmbeddingDataset(embedding_dir, test_df)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)
    
    embedding_dim = 4 * 32 * 32
    bert_dim = 768
    
    checkpoint_dir = f"../checkpoints/context_prediction/{args.model}_{args.data_source}"
    model_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('classification_net_best_')]
    
    results_dir = "test_results"
    os.makedirs(results_dir, exist_ok=True)
    
    all_results = {
        'test_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'data_source': args.data_source,
        'model_type': args.model,
        'test_samples': len(test_df),
        'model_results': {}
    }
    
    for model_file in model_files:
        print(f"\nTesting model: {model_file}")
        model_path = os.path.join(checkpoint_dir, model_file)
        
        if args.model == "semantic":
            model = SemanticNet(bert_dim).to(device)
        elif args.model == "combine":
            model = VisualSemanticNet(embedding_dim, bert_dim).to(device)
        elif args.model == "visual":
            model = VisualNet(embedding_dim).to(device)
        
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        metrics = evaluate_model(model, test_loader, device, model_type=args.model)
        
        model_name = model_file.replace('classification_net_best_', '').replace('.pth', '')
        all_results['model_results'][model_name] = {
            'metrics': metrics,
            'training_epoch': checkpoint['epoch'],
            'validation_metrics': checkpoint['metrics']
        }
        
        # print the result
        if model_name == "accuracy":
            print(f"\nResults for model trained with best {model_name}:")
            print(f"Test Accuracy: {metrics['accuracy']:.2f}%")
            print(f"Test Precision: {metrics['precision']:.2f}%")
            print(f"Test Recall: {metrics['recall']:.2f}%")
            print(f"Test F1 Score: {metrics['f1']:.2f}%")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f'test_results_{args.model}_{args.data_source}_{timestamp}.json')
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print(f"\nAll test results have been saved to: {results_file}")
    
    best_models = {}
    metrics_of_interest = ['accuracy', 'precision', 'recall', 'f1']
    
    for metric in metrics_of_interest:
        best_model = max(all_results['model_results'].items(), 
                        key=lambda x: x[1]['metrics'][metric])
        best_models[metric] = {
            'model': best_model[0],
            'value': best_model[1]['metrics'][metric]
        }
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test all saved best models.")
    parser.add_argument('--device', type=int, default=0,
                       help="GPU device ID to use (e.g., 0, 1, 2, etc.)")
    parser.add_argument('--data_source', type=str, choices=['balanced', 'inpainting_only'],
                       default="balanced", help="Choose the data source: 'balanced' or 'inpainting_only'")
    parser.add_argument('--model', type=str, choices=['semantic', 'combine', 'visual'], default="semantic",
                      help="Model type to test: 'semantic', 'combine', or 'visual'")
    
    args = parser.parse_args()
    
    test_all_models(args)