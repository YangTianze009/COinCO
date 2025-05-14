import torch
import torch.nn as nn
import torch.nn.functional as F

# Semantic classification network using BERT embeddings
class SemanticNet(nn.Module):
    def __init__(self, bert_dim=768):
        """
        Neural network for semantic classification based on BERT embeddings
        
        Args:
            bert_dim: Dimension of BERT embeddings, default is 768
        """
        super(SemanticNet, self).__init__()
        self.fc1 = nn.Linear(bert_dim * 2, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)  # Binary classification
        
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        
    def forward(self, objects_embeddings, replacement_embedding):
        """
        Forward pass of the semantic network
        
        Args:
            objects_embeddings: BERT embeddings of objects in the scene
            replacement_embedding: BERT embedding of the replacement object
            
        Returns:
            Output logits (without sigmoid)
        """
        x = torch.cat((objects_embeddings, replacement_embedding), dim=1)
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        
        x = self.fc4(x)
        
        return x

# Combined visual and semantic network
class VisualSemanticNet(nn.Module):
    def __init__(self, vae_dim, bert_dim=768):
        """
        Neural network combining visual VAE latents and semantic BERT embeddings
        
        Args:
            vae_dim: Dimension of the flattened VAE latent
            bert_dim: Dimension of BERT embeddings, default is 768
        """
        super(VisualSemanticNet, self).__init__()
        
        # Visual branch
        self.visual_fc1 = nn.Linear(vae_dim * 2, 512)
        self.visual_fc2 = nn.Linear(512, 256)
        self.visual_fc3 = nn.Linear(256, 128)
        
        # Semantic branch
        self.semantic_fc1 = nn.Linear(bert_dim * 2, 512)
        self.semantic_fc2 = nn.Linear(512, 256)
        self.semantic_fc3 = nn.Linear(256, 128)
        
        # Combined branch
        self.combined_fc = nn.Linear(128 + 128, 64)
        self.output_fc = nn.Linear(64, 1)
        
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        
    def forward(self, latent1_flat, latent2_flat, objects_embeddings, replacement_embedding):
        """
        Forward pass of the combined network
        
        Args:
            latent1_flat: Flattened VAE latent of the original image
            latent2_flat: Flattened VAE latent of the mask
            objects_embeddings: BERT embeddings of objects in the scene
            replacement_embedding: BERT embedding of the replacement object
            
        Returns:
            Output logits (without sigmoid)
        """
        # Process visual branch
        visual_x = torch.cat((latent1_flat, latent2_flat), dim=1)
        visual_x = self.relu(self.visual_fc1(visual_x))
        visual_x = self.dropout(visual_x)
        visual_x = self.relu(self.visual_fc2(visual_x))
        visual_x = self.dropout(visual_x)
        visual_x = self.relu(self.visual_fc3(visual_x))
        visual_x = self.dropout(visual_x)
        
        # Process semantic branch
        semantic_x = torch.cat((objects_embeddings, replacement_embedding), dim=1)
        semantic_x = self.relu(self.semantic_fc1(semantic_x))
        semantic_x = self.dropout(semantic_x)
        semantic_x = self.relu(self.semantic_fc2(semantic_x))
        semantic_x = self.dropout(semantic_x)
        semantic_x = self.relu(self.semantic_fc3(semantic_x))
        semantic_x = self.dropout(semantic_x)
        
        # Combine features
        combined = torch.cat((visual_x, semantic_x), dim=1)
        combined = self.relu(self.combined_fc(combined))
        combined = self.dropout(combined)
        
        # Output layer
        output = self.output_fc(combined)
        
        return output
    
class VisualNet(nn.Module):
    def __init__(self, vae_dim):
        super(VisualNet, self).__init__()
        self.fc1 = nn.Linear(vae_dim * 2, 512)     
        self.fc2 = nn.Linear(512, 256)     
        self.fc3 = nn.Linear(256, 128)     
        self.fc4 = nn.Linear(128, 1)  # Changed to 1 output for binary classification
        
        self.dropout = nn.Dropout(0.3)  # Dropout layer to prevent overfitting
        self.relu = nn.ReLU()  # ReLU activation function
        self.sigmoid = nn.Sigmoid()  # Added sigmoid for binary classification
    
    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)  # Concatenate inputs along the feature dimension
        
        x = self.relu(self.fc1(x)) 
        x = self.dropout(x)     
        
        x = self.relu(self.fc2(x))     
        x = self.dropout(x)     
        
        x = self.relu(self.fc3(x))     
        x = self.dropout(x)    
        
        x = self.fc4(x)  # Linear layer
        # x = self.sigmoid(x)  # Apply sigmoid for binary classification
        
        return x


def load_visual_model(filepath, input_dim, device):
    model = VisualNet(input_dim).to(device)
    model.load_state_dict(torch.load(filepath))
    model.eval()
    return model

# Load functions
def load_semantic_model(filepath, bert_dim=768, device='cuda'):
    """
    Load a pretrained semantic model
    
    Args:
        filepath: Path to the model state dict
        bert_dim: Dimension of BERT embeddings, default is 768
        device: Device to load the model on
        
    Returns:
        Loaded SemanticNet model
    """
    model = SemanticNet(bert_dim).to(device)
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.eval()
    return model

def load_visual_semantic_model(filepath, vae_dim, bert_dim=768, device='cuda'):
    """
    Load a pretrained visual-semantic model
    
    Args:
        filepath: Path to the model state dict
        vae_dim: Dimension of the flattened VAE latent
        bert_dim: Dimension of BERT embeddings, default is 768
        device: Device to load the model on
        
    Returns:
        Loaded VisualSemanticNet model
    """
    model = VisualSemanticNet(vae_dim, bert_dim).to(device)
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.eval()
    return model



if __name__ == "__main__":
    # Test usage
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Parameters
    batch_size = 4
    bert_dim = 768
    vae_dim = 4 * 32 * 32  # 4 channels x 32 x 32 image size
    
    # Create example inputs
    latent1 = torch.randn(batch_size, 4, 32, 32).to(device)
    latent2 = torch.randn(batch_size, 4, 32, 32).to(device)
    objects_embeddings = torch.randn(batch_size, bert_dim).to(device)
    replacement_embedding = torch.randn(batch_size, bert_dim).to(device)
    
    # Flatten latents for model input
    latent1_flat = latent1.view(batch_size, -1)
    latent2_flat = latent2.view(batch_size, -1)
    
    print(f"Latent1 shape: {latent1.shape}, flattened: {latent1_flat.shape}")
    print(f"Latent2 shape: {latent2.shape}, flattened: {latent2_flat.shape}")
    print(f"Objects embeddings shape: {objects_embeddings.shape}")
    print(f"Replacement embedding shape: {replacement_embedding.shape}")
    
    # Initialize models
    semantic_model = SemanticNet().to(device)
    visual_semantic_model = VisualSemanticNet(vae_dim).to(device)
    
    # Forward pass through semantic model
    semantic_output = semantic_model(objects_embeddings, replacement_embedding)
    print(f"\nSemantic model output shape: {semantic_output.shape}")
    
    # Forward pass through visual-semantic model
    visual_semantic_output = visual_semantic_model(
        latent1_flat, latent2_flat, objects_embeddings, replacement_embedding
    )
    print(f"Visual-semantic model output shape: {visual_semantic_output.shape}")
    
    # Apply sigmoid to convert logits to probabilities
    semantic_probs = torch.sigmoid(semantic_output)
    visual_semantic_probs = torch.sigmoid(visual_semantic_output)
    
    print(f"\nSemantic model predictions:")
    print(f"Logits: {semantic_output.detach().cpu().numpy()}")
    print(f"Probabilities: {semantic_probs.detach().cpu().numpy()}")
    
    print(f"\nVisual-semantic model predictions:")
    print(f"Logits: {visual_semantic_output.detach().cpu().numpy()}")
    print(f"Probabilities: {visual_semantic_probs.detach().cpu().numpy()}")
    
    # Simulate loading a pretrained model (for testing only)
    try:
        # This would fail unless the files exist
        # semantic_model_loaded = load_semantic_model("path/to/semantic_model.pth", device=device)
        # visual_semantic_model_loaded = load_visual_semantic_model(
        #    "path/to/visual_semantic_model.pth", vae_dim=vae_dim, device=device
        # )
        print("\nNote: Load functions are ready for use with saved model files")
    except:
        print("\nNote: Load functions are ready for use with saved model files")