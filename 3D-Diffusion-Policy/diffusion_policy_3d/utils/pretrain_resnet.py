import zarr
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18, resnet34
from tqdm import tqdm
from torch.utils.data import random_split
import wandb

# 1. 读取 Zarr 数据
class ZarrDataset(Dataset):
    def __init__(self, zarr_path, transform=None):
        self.zarr_root = zarr.open(zarr_path, mode='r')
        self.images = self.zarr_root['data/img2']  # 图像数据: (N, C, H, W)
        self.labels = self.zarr_root['data/label']  # 标签: (N,)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]  # 读取单张图片
        label = int(self.labels[idx])  # 读取对应的标签

        image = torch.tensor(image, dtype=torch.float32)  # 转换为 PyTorch Tensor
        if self.transform:
            image = self.transform(image)

        return image, label

# 2. 数据预处理（确保适配 ResNet）
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet 需要 224x224 输入
    transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化
])

# 3. 创建 DataLoader
# save_data_path = '/home/robotics/crq/3D-Diffusion-Policy/3D-Diffusion-Policy/data/data_for_pretrain.zarr'

zarr_path = "/home/robotics/crq/3D-Diffusion-Policy/3D-Diffusion-Policy/data/data_for_pretrain.zarr"  # 替换成你的 Zarr 数据路径
dataset = ZarrDataset(zarr_path, transform)

# 按 80% 训练，20% 验证拆分
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

print(f"训练样本: {train_size}, 验证样本: {val_size}")

# 4. 测试数据加载是否正确
sample_img, sample_label = next(iter(train_loader))
print(f"Sample image shape: {sample_img.shape}, Label: {sample_label}")

# 1. 定义 ResNet 分类模型
class ResNetClassifier(nn.Module):
    def __init__(self, model_type='resnet18', num_classes=3, pretrained=True):
        super().__init__()
        if model_type == 'resnet18':
            self.resnet = resnet18(pretrained=pretrained)
            self.resnet.conv1 = nn.Conv2d(1, self.resnet.conv1.out_channels, 
                                          kernel_size=self.resnet.conv1.kernel_size, 
                                          stride=self.resnet.conv1.stride, 
                                          padding=self.resnet.conv1.padding, 
                                          bias=self.resnet.conv1.bias)
        elif model_type == 'resnet34':
            self.resnet = resnet34(pretrained=pretrained)
            self.resnet.conv1 = nn.Conv2d(4, self.resnet.conv1.out_channels, 
                                          kernel_size=self.resnet.conv1.kernel_size, 
                                          stride=self.resnet.conv1.stride, 
                                          padding=self.resnet.conv1.padding, 
                                          bias=self.resnet.conv1.bias)
        else:
            raise ValueError("Unsupported model type")

        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)  # 修改最后的全连接层

    def forward(self, x):
        return self.resnet(x)

# 2. 训练函数
def train_classifier(model, train_loader, val_loader, epochs=10, lr=1e-4, device='cuda'):
    wandb.init(project="resnet_pretrain", config={"epochs": epochs, "learning_rate": lr})

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        # 训练阶段
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        print(f"Epoch {epoch+1}, Train Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}")
        wandb.log({"train_loss": total_loss, "train_acc": train_acc})

        # 评估阶段
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        wandb.log({"val_loss": val_loss, "val_acc": val_acc})

        model_save_path = "resnet2_model.pth"
        torch.save(model.state_dict(), model_save_path)
        print("Model saved successfully!")
    wandb.finish()


num_epoches = 3

# 3. 训练 ResNet
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ultrasound_model = ResNetClassifier(model_type='resnet18', num_classes=3).to(device)
# train_classifier(ultrasound_model, train_loader, val_loader, epochs=num_epoches, device=device)

# 训练 ResNet-34 处理腕部深度图像
wrist_model = ResNetClassifier(model_type='resnet34', num_classes=3)
train_classifier(wrist_model, train_loader, val_loader, epochs=num_epoches, device=device)
