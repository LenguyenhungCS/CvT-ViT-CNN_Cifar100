import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import argparse
import os
import time
import sys
import math
import numpy as np
from cvt_model import get_cvt_model

class SuppressOutput:
    def __enter__(self):
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

class MultiViewCIFAR100(Dataset):
    def __init__(self, root, train=True, download=False, transform_original=None, transform_augment=None, num_augment_versions=3):
        with SuppressOutput():
            self.base_dataset = torchvision.datasets.CIFAR100(
                root=root, train=train, download=download, transform=None)
        self.transform_original = transform_original
        self.transform_augment = transform_augment
        self.num_total_versions = 1 + num_augment_versions

    def __len__(self):
        return len(self.base_dataset) * self.num_total_versions

    def __getitem__(self, idx):
        base_idx = idx // self.num_total_versions
        version_type = idx % self.num_total_versions

        img, target = self.base_dataset[base_idx]

        if version_type == 0:
            if self.transform_original:
                img = self.transform_original(img)
        else:
            if self.transform_augment:
                img = self.transform_augment(img)
        
        return img, target

# --- Argument Parser ---
parser = argparse.ArgumentParser(description='PyTorch CIFAR-100 CvT Training')
parser.add_argument('--lr', default=2e-3, type=float, help='learning rate')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--epochs', default=200, type=int, help='number of epochs to train')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--seed', default=42, type=int, help='random seed')
parser.add_argument('--data_path', default='./data', type=str, help='path to dataset')
parser.add_argument('--num_workers', default=4, type=int, help='number of data loading workers')
parser.add_argument('--checkpoint_path', default='./checkpoint', type=str, help='path to save checkpoints')
parser.add_argument('--run_name', default='cvt_cifar100', type=str, help='name for this run (used for checkpoint saving)')
parser.add_argument('--warmup_epochs', default=10, type=int, help='number of warmup epochs')
parser.add_argument('--mixup_alpha', default=0.2, type=float, help='mixup alpha')
args = parser.parse_args()

# --- Setup ---
if not os.path.exists(args.checkpoint_path):
    os.makedirs(args.checkpoint_path)

# Set device (GPU if available, else CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# For reproducibility
torch.manual_seed(args.seed)
if device == 'cuda':
    torch.cuda.manual_seed(args.seed)
    cudnn.deterministic = False
    cudnn.benchmark = True

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# --- Data ---
print('==> Preparing data..')
# Normalization values for CIFAR-100
normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                               std=[0.2675, 0.2565, 0.2761])

transform_original_view = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

transform_augmented_view = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomPerspective(0.2),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
    transforms.RandomApply([transforms.RandomAutocontrast()], p=0.3),
    transforms.ToTensor(),
    normalize,
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

# Load CIFAR-100 dataset
try:
    trainset = MultiViewCIFAR100(
        root=args.data_path, 
        train=True, 
        download=True,
        transform_original=transform_original_view,
        transform_augment=transform_augmented_view,
        num_augment_versions=3
    )
    
    with SuppressOutput():
        testset = torchvision.datasets.CIFAR100(
            root=args.data_path, train=False, download=True, transform=transform_test)

except Exception as e:
    print(f"Error downloading or loading CIFAR-100 dataset: {e}")
    print("Please check your internet connection or the specified data path.")
    sys.exit(1)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

# --- Model ---
print('==> Building model...')
model = get_cvt_model(num_classes=100)
model = model.to(device)

# Support multi-GPU
if device == 'cuda' and torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

# --- Resume from Checkpoint ---
if args.resume:
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(args.checkpoint_path), f'Error: no checkpoint directory found at {args.checkpoint_path}'
    checkpoint_file = os.path.join(args.checkpoint_path, f'{args.run_name}_ckpt.pth')
    if os.path.isfile(checkpoint_file):
        checkpoint = torch.load(checkpoint_file, map_location=device)
        state_dict = checkpoint['model']
        if device == 'cuda' and torch.cuda.device_count() > 1 and not isinstance(model, nn.DataParallel):
            state_dict = {'module.' + k: v for k, v in state_dict.items()}
        elif device == 'cpu' and isinstance(model, nn.DataParallel):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        elif device == 'cuda' and torch.cuda.device_count() == 1 and isinstance(model, nn.DataParallel):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1
        print(f"Loaded checkpoint '{checkpoint_file}' (epoch {checkpoint['epoch']}, acc {best_acc:.2f}%)")
    else:
        print(f'Error: no checkpoint found at {checkpoint_file}')
        start_epoch = 0
        best_acc = 0

# --- Loss and Optimizer ---
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)

# Mixup augmentation
def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# Calculate total steps
num_training_steps = len(trainloader) * args.epochs
# num_warmup_steps = len(trainloader) * args.warmup_epochs # No longer directly needed for OneCycleLR config here
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=args.lr,
    epochs=args.epochs,
    steps_per_epoch=len(trainloader),
    pct_start=args.warmup_epochs / args.epochs if args.epochs > 0 else 0.1, # Use warmup_epochs to set pct_start
    div_factor=25,
    final_div_factor=1e4
)

# --- Training Function ---
def train(epoch):
    print(f'\nEpoch: {epoch + 1}/{args.epochs}')
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    start_time = time.time()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Apply mixup
        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, args.mixup_alpha)

        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Mixup loss
        loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if (batch_idx + 1) % 50 == 0 or batch_idx == len(trainloader) - 1:
            progress = 100. * (batch_idx + 1) / len(trainloader)
            current_lr = optimizer.param_groups[0]['lr']
            print(f'  Train Batch: [{batch_idx + 1}/{len(trainloader)}] ({progress:.0f}%) | Loss: {train_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.3f}% ({correct}/{total}) | LR: {current_lr:.6f}')

    end_time = time.time()
    print(f'Epoch {epoch+1} Training Time: {end_time - start_time:.2f} seconds')

# --- Testing Function ---
def test(epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if (batch_idx + 1) % 100 == 0 or batch_idx == len(testloader) - 1:
                progress = 100. * (batch_idx + 1) / len(testloader)
                print(f'  Test Batch: [{batch_idx + 1}/{len(testloader)}] ({progress:.0f}%) | Loss: {test_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.3f}% ({correct}/{total})')

    end_time = time.time()
    print(f'Epoch {epoch+1} Testing Time: {end_time - start_time:.2f} seconds')

    # Save checkpoint
    acc = 100. * correct / total
    print(f'==> Test Results: Accuracy: {acc:.3f}% ({correct}/{total})')
    if acc > best_acc:
        print(f'Saving.. New best accuracy: {acc:.3f}% (previously {best_acc:.3f}%)')
        state = {
            'model': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'args': args
        }
        checkpoint_file = os.path.join(args.checkpoint_path, f'{args.run_name}_ckpt.pth')
        torch.save(state, checkpoint_file)
        best_acc = acc
    else:
        print(f'Accuracy ({acc:.3f}%) did not improve from best ({best_acc:.3f}%)')

    return test_loss, acc

# --- Main Loop ---
if __name__ == '__main__':
    print("==> Starting Training Loop ==")
    patience = 20
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(start_epoch, args.epochs):
        train(epoch)
        val_loss, val_acc = test(epoch)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"==> Training Finished. Best Test Accuracy: {best_acc:.3f}% ==")