# =========================================================
# 🧠 訓練主程式流程簡介（虛擬碼）/ Main Training Script Overview (Pseudocode)
#
# Step 0: 設定超參數（可手動調整）/ Set hyperparameters (adjust manually)
# BATCH_SIZE = 32
# LEARNING_RATE = 1e-4
# NUM_EPOCHS = 10
# IMG_SIZE = 224
# TRAIN_SPLIT = 0.8
#
# Step 1: 匯入必要模組 / Import necessary modules
# import torch
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from rfmid_dataset import RFMiDDataset
# from torchvision.models import resnet18  # 可替換為 / can be replaced with: vgg16, resnet50
#
# Step 2: 定義圖像轉換（Resize, Normalize 等）/ Define image transforms (Resize, Normalize, etc.)
# transform = transforms.Compose([
#     # Resize to (224, 224)
#     # Convert to Tensor
#     # Normalize
# ])
#
# Step 3: 建立 RFMiD Dataset / Create RFMiD Dataset
# dataset = RFMiDDataset(csv_file='path/to/labels.csv',
#                        img_dir='path/to/images',
#                        transform=transform)
#
# Step 4: 切分訓練集與驗證集 / Split into training and validation sets
# train_dataset, val_dataset = split_dataset(dataset)
#
# Step 5: 建立 DataLoader / Create DataLoader
# train_loader = DataLoader(train_dataset, batch_size=..., shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=..., shuffle=False)
#
# Step 6: 初始化模型 / Initialize model
# model = resnet18(pretrained=True)
# # 修改輸出層以對應 RFMiD 類別數 / Modify output layer to match the number of RFMiD classes
#
# Step 7: 定義損失函數與 optimizer / Define loss function and optimizer
# criterion = CrossEntropyLoss()  # 或 / or FocalLoss()
# optimizer = Adam(model.parameters(), lr=...)
#
# Step 8: 開始訓練迴圈 / Start training loop
# for epoch in range(num_epochs):
#     model.train()
#     for images, labels in train_loader:
#         # 前向傳播 / Forward pass
#         # 計算 loss / Compute loss
#         # 反向傳播與更新參數 / Backpropagation and parameter update
#
#     model.eval()
#     with torch.no_grad():
#         for images, labels in val_loader:
#             # 驗證階段：計算 loss 與準確率 / Validation: compute loss and accuracy
#
#     # 記錄到 wandb（可選）/ Log to wandb (optional)
#
# Step 9: 儲存模型或畫出 loss/acc 曲線（可選）/ Save model or plot loss/acc curves (optional)
# =========================================================
