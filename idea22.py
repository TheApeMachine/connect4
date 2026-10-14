import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
from torchvision import datasets, transforms
import math
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. Hardware-Accelerated Device Selection & Reproducibility
# -----------------------------------------------------------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Running Hierarchical DPR experiment on: {device}")

torch.manual_seed(42)

# Generate a fixed, reproducible pixel permutation key
perm_key = torch.randperm(28 * 28)

def apply_permutation(images, perm_key):
    B, C, H, W = images.shape
    flat = images.view(B, -1)
    permuted = flat[:, perm_key]
    return permuted.view(B, C, H, W)

# -----------------------------------------------------------------------------
# 2. Data Preparation (Train & Split-Validation Sets)
# -----------------------------------------------------------------------------
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# Train Subset (15,000 images)
mnist_train_full = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_train_subset = data_utils.Subset(mnist_train_full, range(15000))
train_loader = data_utils.DataLoader(mnist_train_subset, batch_size=64, shuffle=True)

# Held-out Test Sets (1,000 images each) for continuous evaluation
mnist_test_full = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_subset = data_utils.Subset(mnist_test_full, range(1000))
test_loader = data_utils.DataLoader(test_subset, batch_size=256, shuffle=False)

# Pre-cache unpermuted and permuted test batches for fast, high-frequency evaluation
test_batches_orig = []
test_batches_perm = []
for test_inputs, test_targets in test_loader:
    test_inputs, test_targets = test_inputs.to(device), test_targets.to(device)
    test_batches_orig.append((test_inputs, test_targets))
    test_batches_perm.append((apply_permutation(test_inputs, perm_key), test_targets))

# -----------------------------------------------------------------------------
# 3. Model Definition (Hierarchical 3-Layer MLP)
# -----------------------------------------------------------------------------
class MNISTMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10):
        super().__init__()
        # We explicitly separate the feature extractor and classifier groups
        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.classifier = nn.Linear(hidden_dim // 2, output_dim)
        
    def forward(self, x):
        return self.classifier(self.features(x))

# -----------------------------------------------------------------------------
# 4. Evaluation Routine
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate_model(model, batches):
    model.eval()
    correct = 0
    total = 0
    for inputs, targets in batches:
        outputs = model(inputs)
        preds = outputs.argmax(dim=1, keepdim=True)
        correct += preds.eq(targets.view_as(preds)).sum().item()
        total += inputs.size(0)
    model.train()
    return correct / total

# -----------------------------------------------------------------------------
# 5. Training Engine with Parameter Group Modulation
# -----------------------------------------------------------------------------
TOTAL_EPOCHS = 6
DRIFT_EPOCH = 3  # Permutation drift begins at this epoch (shared with plotting)

def train_experiment(model_type, alpha=0.2):
    model = MNISTMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Configure parameter groups with specific dynamic behaviors
    if model_type == "H-DPR":
        # Hierarchical DPR: Features are highly protected; Classifier is highly plastic
        param_groups = [
            {"params": model.features.parameters(), "lambda_min": 1e-5, "lambda_max": 0.1, "eta": 0.5, "name": "features"},
            {"params": model.classifier.parameters(), "lambda_min": 1e-5, "lambda_max": 3.5, "eta": 3.0, "name": "classifier"}
        ]
    elif model_type == "G-DPR":
        # Global DPR: All layers are treated equally (same as the previous run)
        param_groups = [
            {"params": model.parameters(), "lambda_min": 1e-5, "lambda_max": 3.5, "eta": 2.0, "name": "global"}
        ]
    else:  # Constant Low
        param_groups = [
            {"params": model.parameters(), "lambda_min": 1e-5, "lambda_max": 1e-5, "eta": 0.0, "name": "low"}
        ]
        
    optimizer = optim.AdamW(param_groups, lr=1e-3)
    
    # Trackers for plotting
    step_history = []
    orig_test_acc = []
    perm_test_acc = []
    wd_history_features = []
    wd_history_classifier = []
    
    surprise_ema = 2.3  # Initial guess
    global_step = 0
    
    for epoch in range(TOTAL_EPOCHS):
        is_drift_phase = epoch >= DRIFT_EPOCH
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            if is_drift_phase:
                inputs = apply_permutation(inputs, perm_key)
                
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss_val = loss.item()
            
            # Update surprise metric (EMA)
            surprise_ema = alpha * loss_val + (1 - alpha) * surprise_ema
            
            # Dynamic updates applied individually per parameter group
            current_wd_feat = 1e-5
            current_wd_cls = 1e-5
            for group in optimizer.param_groups:
                l_min = group["lambda_min"]
                l_max = group["lambda_max"]
                g_eta = group["eta"]
                
                # Prior precision update rule
                current_wd = l_min + (l_max - l_min) * (1.0 - math.exp(-g_eta * surprise_ema))
                group["weight_decay"] = current_wd
                
                if model_type == "H-DPR":
                    if group["name"] == "features":
                        current_wd_feat = current_wd
                    elif group["name"] == "classifier":
                        current_wd_cls = current_wd
                else:
                    current_wd_feat = current_wd
                    current_wd_cls = current_wd
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # High-frequency evaluation on held-out splits every 50 steps
            if global_step % 50 == 0:
                acc_orig = evaluate_model(model, test_batches_orig)
                acc_perm = evaluate_model(model, test_batches_perm)
                
                step_history.append(global_step)
                orig_test_acc.append(acc_orig)
                perm_test_acc.append(acc_perm)
                wd_history_features.append(current_wd_feat)
                wd_history_classifier.append(current_wd_cls)
                
            global_step += 1
            
        print(f"Model: {model_type:<13} | Epoch {epoch+1} done | Features WD: {current_wd_feat:.6f} | Classifier WD: {current_wd_cls:.6f}")
        
    return step_history, orig_test_acc, perm_test_acc, wd_history_features, wd_history_classifier

# -----------------------------------------------------------------------------
# 6. Execute Run Configurations
# -----------------------------------------------------------------------------
print("\n--- Running Model 1: Hierarchical DPR (Features Protected, Classifier Plastic) ---")
steps, h_orig, h_perm, h_wd_feat, h_wd_cls = train_experiment("H-DPR")

print("\n--- Running Model 2: Global DPR (Uniform Decay) ---")
_, g_orig, g_perm, _, _ = train_experiment("G-DPR")

print("\n--- Running Model 3: Constant Low Weight Decay ($10^{-5}$) ---")
_, low_orig, low_perm, _, _ = train_experiment("Constant_Low")

# -----------------------------------------------------------------------------
# 7. Analysis & Plotting
# -----------------------------------------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
drift_step = DRIFT_EPOCH * len(train_loader)

# Panel 1: Task 1 Retention (Original Image Test Accuracy)
ax1.plot(steps, h_orig, label="Hierarchical DPR (H-DPR)", color="darkorange", linewidth=2.5)
ax1.plot(steps, g_orig, label="Global DPR (G-DPR)", color="purple", alpha=0.7)
ax1.plot(steps, low_orig, label="Constant Low ($10^{-5}$)", color="teal", alpha=0.7)
ax1.axvline(x=drift_step, color="black", linestyle="--", alpha=0.7, label="Permutation Drift")
ax1.set_ylabel("Original Test Acc")
ax1.set_title("Hierarchical DPR: Synaptic Protection vs. Targeted Erasing")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Panel 2: Task 2 Acquisition (Permuted Image Test Accuracy)
ax2.plot(steps, h_perm, color="darkorange", linewidth=2.5, label="H-DPR")
ax2.plot(steps, g_perm, color="purple", alpha=0.7, label="G-DPR")
ax2.plot(steps, low_perm, color="teal", alpha=0.7, label="Constant Low")
ax2.axvline(x=drift_step, color="black", linestyle="--", alpha=0.7)
ax2.set_ylabel("Permuted Test Acc")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Panel 3: Hierarchical Weight Decay Trajectories
ax3.plot(steps, h_wd_feat, label=r"Features Decay ($\lambda_f$)", color="forestgreen", linewidth=1.5)
ax3.plot(steps, h_wd_cls, label=r"Classifier Decay ($\lambda_c$)", color="darkorange", linewidth=2)
ax3.axvline(x=drift_step, color="black", linestyle="--", alpha=0.7)
ax3.set_ylabel("Prior Precision ($\lambda$)")
ax3.set_xlabel("Cumulative Training Steps")
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()