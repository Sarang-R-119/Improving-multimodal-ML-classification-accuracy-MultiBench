import torch
import torch.nn as nn
import torch.optim as optim
from mamba_ssm import Mamba

# 1. Define the Mamba-based Architecture
class SimpleMambaClassifier(nn.Module):
    def __init__(self, d_model=64, num_classes=2):
        super().__init__()
        
        # The core Mamba block
        self.mamba = Mamba(
            d_model=d_model, # The dimension of your token embeddings
            d_state=16,      # State size (standard is 16)
            d_conv=4,        # Local convolution width
            expand=2,        # Expansion factor for the inner linear layers
        )
        
        # A simple linear layer to map the Mamba output to your classes
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x expected shape: (Batch, Sequence_Length, d_model)
        
        # Pass the sequence through Mamba
        mamba_out = self.mamba(x) 
        # mamba_out shape is same as input: (Batch, Sequence_Length, d_model)
        
        # Pool the sequence: we take the average across the Sequence_Length dimension
        pooled_out = mamba_out.mean(dim=1) 
        # pooled_out shape: (Batch, d_model)
        
        # Pass through the final classification head
        logits = self.classifier(pooled_out)
        # logits shape: (Batch, num_classes)
        return logits

# 2. Setup Device and Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

batch_size = 32
seq_length = 100  # e.g., 100 time-steps or tokens
d_model = 64      # Number of features per time-step
num_classes = 2

# Initialize the model and move to GPU
model = SimpleMambaClassifier(d_model=d_model, num_classes=num_classes).to(device)

# 3. Create Dummy Data
# Think of this as your aligned sequences (e.g., text and images concatenated)
# Shape: (B, L, D) -> Batch Size, Sequence Length, Feature Dimension
X_dummy = torch.randn(batch_size, seq_length, d_model).to(device)

# Binary labels: 0 or 1
y_dummy = torch.randint(0, num_classes, (batch_size,)).to(device) 

# 4. Define Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. A Quick Training Loop
print("\nStarting training loop...")
epochs = 5
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(X_dummy)
    
    # Calculate loss
    loss = criterion(outputs, y_dummy)
    
    # Backward pass and optimize
    loss.backward()
    optimizer.step()
    
    # Quick accuracy check
    predictions = torch.argmax(outputs, dim=1)
    accuracy = (predictions == y_dummy).float().mean() * 100
    
    print(f"Epoch [{epoch+1}/{epochs}] | Loss: {loss.item():.4f} | Accuracy: {accuracy:.2f}%")

print("\nMamba test completed successfully! Your environment is ready.")