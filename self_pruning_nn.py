import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

        nn.init.kaiming_uniform_(self.weight, a=0.01)

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

    def sparsity_loss(self):
        return torch.sigmoid(self.gate_scores).sum()

    def sparsity_level(self, threshold=0.05):
        gates = torch.sigmoid(self.gate_scores).detach()
        return (gates < threshold).float().mean().item()

    def gate_values(self):
        return torch.sigmoid(self.gate_scores).detach().cpu().flatten()


class SelfPruningNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = PrunableLinear(3072, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 128)
        self.fc4 = PrunableLinear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

    def sparsity_loss(self):
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                total = total + m.sparsity_loss()
        return total

    def overall_sparsity(self, threshold=0.05):
        pruned, total = 0, 0
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                gates = torch.sigmoid(m.gate_scores).detach()
                pruned += (gates < threshold).sum().item()
                total += gates.numel()
        return pruned / total if total > 0 else 0.0

    def all_gate_values(self):
        parts = []
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                parts.append(m.gate_values())
        return torch.cat(parts)


def get_dataloaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    train_set = torchvision.datasets.CIFAR10(root="./data", train=True,
                                              download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root="./data", train=False,
                                             download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=256,
                             shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


def train_one_epoch(model, loader, optimizer, lam, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        logits = model(images)
        loss = F.cross_entropy(logits, labels) + lam * model.sparsity_loss()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        correct += (model(images).argmax(1) == labels).sum().item()
        total += images.size(0)
    return correct / total


def run_experiment(lam, epochs, train_loader, test_loader, device, seed=42):
    torch.manual_seed(seed)
    model = SelfPruningNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\n--- lambda = {lam} ---")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, lam, device)
        scheduler.step()
        if epoch % 5 == 0 or epoch == epochs:
            sparsity = model.overall_sparsity()
            print(f"  epoch {epoch}/{epochs}  loss={train_loss:.4f}  "
                  f"train_acc={train_acc*100:.1f}%  sparsity={sparsity*100:.1f}%")

    test_acc = evaluate(model, test_loader, device)
    sparsity = model.overall_sparsity()
    print(f"  => test_acc={test_acc*100:.2f}%  sparsity={sparsity*100:.2f}%")
    return test_acc, sparsity, model


def plot_gates(model, lam):
    gates = model.all_gate_values().numpy()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(gates, bins=100, color="#4C72B0", edgecolor="white", linewidth=0.3)
    ax.axvline(x=0.05, color="red", linestyle="--", linewidth=1.2, label="threshold=0.05")
    ax.set_title(f"Gate Value Distribution  (λ={lam})")
    ax.set_xlabel("Gate Value")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    plt.xlim(-0.005, 0.15)
    plt.savefig("gate_dist.png", dpi=150)
    plt.close()
    print("saved gate_dist.png")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    epochs = 30
    lambdas = [1e-5, 1e-4, 1e-3]
    train_loader, test_loader = get_dataloaders()

    results = []
    best_model, best_lam, best_acc = None, None, -1.0

    for lam in lambdas:
        acc, sparsity, model = run_experiment(lam, epochs, train_loader, test_loader, device)
        results.append((lam, acc * 100, sparsity * 100))
        if acc > best_acc:
            best_acc, best_model, best_lam = acc, model, lam

    print("\n" + "=" * 50)
    print(f"{'Lambda':<12} {'Test Acc':>12} {'Sparsity':>12}")
    print("-" * 50)
    for lam, acc, sparsity in results:
        print(f"{lam:<12} {acc:>11.2f}% {sparsity:>11.2f}%")
    print("=" * 50)

    plot_gates(best_model, best_lam)


if __name__ == "__main__":
    main()