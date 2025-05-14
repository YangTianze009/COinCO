import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from diffusers import AutoencoderKL
from dataset import CustomDataset, transform
from model import SemanticNet, VisualSemanticNet, VisualNet
import pandas as pd
import os
import argparse
from tqdm import tqdm, trange
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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

def prepare_embeddings(args, data_df, objects_df, device, embedding_dir):
    os.makedirs(embedding_dir, exist_ok=True)
    
    # Initialize BERT model for text embeddings
    print("Loading BERT model for text embeddings...")
    tokenizer = BertTokenizer.from_pretrained('/data2/Tianze/weights/bert-base-uncased')
    bert_model = BertModel.from_pretrained('/data2/Tianze/weights/bert-base-uncased').to(device)
    bert_model.eval()
    
    # Check which embeddings need to be created
    missing_indices = []
    for coco_idx in data_df['coco_index']:
        file_path = os.path.join(embedding_dir, f"{coco_idx}.pt")
        if not os.path.exists(file_path):
            missing_indices.append(coco_idx)
    
    if not missing_indices:
        print(f"All embeddings exist for this dataset, skipping encoding...")
        return
    
    print(f"Found {len(missing_indices)} missing embeddings. Processing...")
    
    # Initialize VAE model for image embeddings
    model_id = "/data2/Tianze/weights/stable-diffusion-2-1"
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device)
    vae.eval()
    
    # Filter data for missing indices
    missing_df = data_df[data_df['coco_index'].isin(missing_indices)].reset_index(drop=True)
    
    # Create mapping from coco_index to row index in objects_df for quick lookup
    objects_lookup = {}
    for _, row in objects_df.iterrows():
        objects_lookup[row['coco_index']] = row
    
    # Prepare data paths
    base_path = "../task_data"
    image_paths1 = [os.path.join(base_path, "images/training_val_images", f"{idx}.png") 
                   for idx in missing_df['coco_index']]
    image_paths2 = [os.path.join(base_path, "masks/bbox_masks_training_val", f"{idx}.png") 
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
                # print(objects_embeddings.shape)
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

def evaluate(model, dataloader, device, model_type):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for latent1, latent2, labels, objects_embeddings, replacement_embedding in dataloader:
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
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    return {
        'accuracy': accuracy * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100
    }
def train(args, embedding_dir, train_df, val_df):
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = CachedEmbeddingDataset(embedding_dir, train_df)
    val_dataset = CachedEmbeddingDataset(embedding_dir, val_df)
    
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=4)
    
    embedding_dim = 4 * 32 * 32
    bert_dim=768

    if args.model == "semantic":
        classification_net = SemanticNet(bert_dim).to(device)
    elif args.model == "combine":
        classification_net = VisualSemanticNet(embedding_dim, bert_dim).to(device)
    elif args.model == "visual":
        classification_net = VisualNet(embedding_dim).to(device)
    
    if args.data_source == 'inpainting_only':
        label_counts = train_df['label'].value_counts()
        count_1 = label_counts.get(1, 0)  # count of label 1
        count_0 = label_counts.get(0, 0)  # count of label 0
        pos_weight = torch.tensor([count_0/count_1]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()
        print("Using standard BCE loss (equal weights)")
    
    optimizer = optim.Adam(classification_net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    num_epochs = 30
    os.makedirs("logs", exist_ok=True)
    log_file = f"logs/training_log_{args.data_source}.txt"
    
    # Initialize metrics tracking
    best_metrics = {
        'f1': {'value': 0.0, 'epoch': -1},
        'accuracy': {'value': 0.0, 'epoch': -1},
        'precision': {'value': 0.0, 'epoch': -1},
        'recall': {'value': 0.0, 'epoch': -1}
    }
    
    checkpoint_dir = f"../checkpoints/context_prediction/{args.model}_{args.data_source}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    epoch_pbar = trange(num_epochs, desc="Training Progress")
    for epoch in epoch_pbar:
        classification_net.train()
        running_loss = 0.0
        
        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs-1}", 
                         leave=False, unit="batch")
        
        for batch_idx, (latent1, latent2, labels, objects_embeddings, replacement_embedding) in enumerate(batch_pbar):
            latent1, latent2, labels, objects_embeddings, replacement_embedding = latent1.to(device), latent2.to(device), labels.to(device), objects_embeddings.to(device), replacement_embedding.to(device)
            
            # Process image latents
            latent1_flat = latent1.view(latent1.size(0), -1)
            latent2_flat = latent2.view(latent2.size(0), -1)
            
            if args.model == "semantic":
                outputs = classification_net(objects_embeddings, replacement_embedding)
            elif args.model == "combine":
                outputs = classification_net(latent1_flat, latent2_flat, objects_embeddings, replacement_embedding)
            elif args.model == "visual":
                outputs = classification_net(latent1_flat, latent2_flat)
            loss = criterion(outputs, labels.float().view(-1, 1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            batch_pbar.set_postfix({
                'loss': f'{running_loss / (batch_idx + 1):.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        metrics = evaluate(classification_net, val_loader, device, model_type=args.model)
        average_loss = running_loss / len(train_loader)
        
        # Update progress bar display
        epoch_pbar.set_postfix({
            'avg_loss': f'{average_loss:.4f}',
            'val_f1': f'{metrics["f1"]:.2f}%',
            'val_acc': f'{metrics["accuracy"]:.2f}%'
        })
        
        # Log training results
        with open(log_file, "a") as f:
            f.write(f"Epoch [{epoch}/{num_epochs-1}], "
                   f"Average Loss: {average_loss:.4f}, "
                   f"Validation Metrics: {metrics}\n")
        
        # Save latest model
        latest_ckpt_path = os.path.join(checkpoint_dir, "classification_net_latest.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': classification_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'metrics': metrics,
        }, latest_ckpt_path)
        
        # Check and save best models for each metric
        for metric_name in best_metrics.keys():
            current_value = metrics[metric_name]
            if current_value > best_metrics[metric_name]['value']:
                best_metrics[metric_name]['value'] = current_value
                best_metrics[metric_name]['epoch'] = epoch
                
                best_ckpt_path = os.path.join(checkpoint_dir, f"classification_net_best_{metric_name}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': classification_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'metrics': metrics,
                }, best_ckpt_path)
                print(f"\nNew best model for {metric_name}! Value: {current_value:.2f}% at epoch {epoch}")
        
        scheduler.step()
    
    # Print final results
    print("\nTraining completed! Best results for each metric:")
    for metric_name, info in best_metrics.items():
        print(f"Best {metric_name}: {info['value']:.2f}% (Epoch {info['epoch']})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a binary classification model.")
    parser.add_argument('--device', type=int, default=0,
                       help="GPU device ID to use (e.g., 0, 1, 2, etc.)")
    parser.add_argument('--data_source', type=str, choices=['balanced', 'inpainting_only'],
                       default="balanced", help="Choose the data source: 'balanced' or 'inpainting_only'")
    parser.add_argument('--model', type=str, choices=['semantic', 'combine', 'visual'], default="semantic",
                      help="Model type to test: 'semantic' or 'combine'")
    
    args = parser.parse_args()
    
    # Set up data paths
    base_path = "../task_data/context_prediction"
    data_path = os.path.join(base_path, args.data_source)
    embedding_dir = "../task_data/cache/cached_embeddings_train_val_cp"
    
    # Read training and validation data
    train_df = pd.read_csv(os.path.join(data_path, 'training_data.csv'))
    val_df = pd.read_csv(os.path.join(data_path, 'validation_data.csv'))
    
    # Read objects data
    try:
        objects_train_df = pd.read_csv(os.path.join(data_path, 'training_object_list.csv'))
        objects_val_df = pd.read_csv(os.path.join(data_path, 'validation_object_list.csv'))
        print("\nLoaded object data:")
        print(f"Training objects: {len(objects_train_df)} rows")
        print(f"Validation objects: {len(objects_val_df)} rows")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Object list files are required. Please provide the object list files.")
        exit(1)
    
    # Print dataset statistics
    print("\nDataset statistics:")
    print("Training data distribution:")
    print(train_df['label'].value_counts(normalize=True))
    print("\nValidation data distribution:")
    print(val_df['label'].value_counts(normalize=True))
    
    # Prepare embeddings
    print("\nChecking and preparing training embeddings...")
    prepare_embeddings(args, train_df, objects_train_df, torch.device(f"cuda:{args.device}"), embedding_dir)
    
    print("\nChecking and preparing validation embeddings...")
    prepare_embeddings(args, val_df, objects_val_df, torch.device(f"cuda:{args.device}"), embedding_dir)
    
    print(f"\nDataset statistics:")
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    print(f"\nStarting training with {args.data_source} dataset on cuda:{args.device}")
    train(args, embedding_dir, train_df, val_df)