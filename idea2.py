import torch
import torch.nn as nn
import torch.optim as optim
import math
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. Device Setup (with MPS support)
# -----------------------------------------------------------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

# -----------------------------------------------------------------------------
# 2. Synthetic Data Generation with Concept Drift
# -----------------------------------------------------------------------------
def generate_batch(batch_size, phase, device):
    """
    Generates 2D points. 
    Phase 1: Decision boundary is y > x.
    Phase 2: Decision boundary undergoes drift to y > -x.
    """
    x = torch.randn(batch_size, 2, device=device)
    if phase == 1:
        y = (x[:, 1] > x[:, 0]).float().unsqueeze(1)
    else:
        y = (x[:, 1] > -x[:, 0]).float().unsqueeze(1)
    return x, y

# -----------------------------------------------------------------------------
# 3. Model Definition
# -----------------------------------------------------------------------------
class ToyMLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

# -----------------------------------------------------------------------------
# 4. Training Parameters & Optimization Setup
# -----------------------------------------------------------------------------
batch_size = 64
total_steps = 1200
drift_step = 600  # Shift the data distribution halfway through

model = ToyMLP().to(device)
criterion = nn.BCEWithLogitsLoss()

# AdamW is preferred here as it decouples weight decay from the gradient updates
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

# DPR Hyperparameters
lambda_min = 1e-5  # High-flexibility state (low prior precision)
lambda_max = 5e-2  # High-compression state (high prior precision)
eta = 3.5          # Sensitivity to surprise/loss

# Logging lists for analysis
loss_history = []
wd_history = []
accuracy_history = []

# -----------------------------------------------------------------------------
# 5. Training Loop
# -----------------------------------------------------------------------------
print("\nStarting training loop...")
for step in range(total_steps):
    # Determine the current phase (introducing concept drift at step 600)
    phase = 1 if step < drift_step else 2
    
    # Generate data
    inputs, targets = generate_batch(batch_size, phase, device)
    
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # Calculate surrogate accuracy
    preds = (torch.sigmoid(outputs) > 0.5).float()
    accuracy = (preds == targets).float().mean().item()
    
    # --- Dynamic Precision Regularization (DPR) Step ---
    # Convert loss into surprise proxy
    loss_val = loss.item()
    
    # Prior precision update rule:
    # High loss -> low decay (stability sacrificed for plasticity)
    # Low loss  -> high decay (network parameters compressed)
    current_wd = lambda_min + (lambda_max - lambda_min) * math.exp(-eta * loss_val)
    
    # Apply the dynamically calculated weight decay to the optimizer
    for param_group in optimizer.param_groups:
        param_group['weight_decay'] = current_wd
    # ----------------------------------------------------
    
    # Backward pass & update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Log values
    loss_history.append(loss_val)
    wd_history.append(current_wd)
    accuracy_history.append(accuracy)
    
    if (step + 1) % 150 == 0 or step == drift_step:
        phase_str = f"Phase {phase}" if step != drift_step else "DRIFT SHIFT"
        print(f"Step {step+1:04d} | {phase_str:<11} | Loss: {loss_val:.4f} | Accuracy: {accuracy:.2f} | Dynamic WD: {current_wd:.6f}")

# -----------------------------------------------------------------------------
# 6. Plotting the Results
# -----------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot Loss and Accuracy
ax1.plot(loss_history, label="BCE Loss", color="crimson", alpha=0.8)
ax1.plot(accuracy_history, label="Accuracy", color="teal", alpha=0.6)
ax1.axvline(x=drift_step, color="black", linestyle="--", alpha=0.7, label="Concept Drift")
ax1.set_ylabel("Loss / Accuracy")
ax1.set_title("Dynamic Precision Regularization (DPR) Test")
ax1.legend(loc="lower left")
ax1.grid(True, alpha=0.3)

# Plot Weight Decay modulation
ax2.plot(wd_history, label="Dynamic Weight Decay ($\lambda$)", color="darkorange")
ax2.axvline(x=drift_step, color="black", linestyle="--", alpha=0.7)
ax2.set_xlabel("Training Steps")
ax2.set_ylabel("Weight Decay Coeff")
ax2.legend(loc="upper left")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()