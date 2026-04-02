import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score

# ==============================================================================
# 🧠 1. DeepLOB 模型定义（与实盘代码完全一致）
# ==============================================================================
class DeepLOB(nn.Module):
    def __init__(self):
        super().__init__()
        lrelu_neg_slope = 0.01

        self.conv1 = nn.Conv2d(1, 16, kernel_size=(1, 2), stride=(1, 2))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(1, 2), stride=(1, 2))
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(1, 10))

        # 分支 1: 1x1 卷积 -> 3x1 卷积
        self.incep1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(1, 1)),
            nn.LeakyReLU(negative_slope=lrelu_neg_slope),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(3, 1), padding=(1, 0)),
            nn.LeakyReLU(negative_slope=lrelu_neg_slope),
            nn.BatchNorm2d(32)
        )
        
        # 分支 2: 1x1 卷积 -> 5x1 卷积
        self.incep2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(1, 1)),
            nn.LeakyReLU(negative_slope=lrelu_neg_slope),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(5, 1), padding=(2, 0)),
            nn.LeakyReLU(negative_slope=lrelu_neg_slope),
            nn.BatchNorm2d(32)
        )
        
        # 分支 3: 3x1 MaxPool -> 1x1 卷积
        self.incep3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(64, 32, kernel_size=(1, 1)),
            nn.LeakyReLU(negative_slope=lrelu_neg_slope),
            nn.BatchNorm2d(32)
        )

        self.lstm = nn.LSTM(96, 64, batch_first=True, num_layers=1)
        self.fc = nn.Linear(64, 3)
        self.relu = nn.LeakyReLU(negative_slope=lrelu_neg_slope)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        x1 = self.incep1(x)
        x2 = self.incep2(x)
        x3 = self.incep3(x)

        x = torch.cat([x1, x2, x3], dim=1) 
        x = x.squeeze(-1).transpose(1, 2)

        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

# ==============================================================================
# 📊 2. 5档盘口 Dataset 构建 (100步滑动窗口 + 40维特征拼接)
# ==============================================================================
class LOBDataset(Dataset):
    def __init__(self, csv_path, seq_length=100, prediction_horizon=50):
        """
        :param csv_path: 你的历史 LOB 数据路径
        :param seq_length: 时间序列长度，DeepLOB 默认为 100
        :param prediction_horizon: 预测未来多少个 tick 后的涨跌 (比如 10 或 50 个 tick)
        """
        print(f"正在加载数据: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # 1. 剔除第一行可能存在的全是 0.0 的脏数据
        df = df[(df['bid_0_price'] > 0) & (df['ask_0_price'] > 0)]
        
        # 2. 提取 5 档盘口（共 20 维特征）
        feature_cols = [
            'bid_0_price', 'bid_0_size', 'ask_0_price', 'ask_0_size',
            'bid_1_price', 'bid_1_size', 'ask_1_price', 'ask_1_size',
            'bid_2_price', 'bid_2_size', 'ask_2_price', 'ask_2_size',
            'bid_3_price', 'bid_3_size', 'ask_3_price', 'ask_3_size',
            'bid_4_price', 'bid_4_size', 'ask_4_price', 'ask_4_size'
        ]
        
        # 提取 20 维数据
        raw_data = df[feature_cols].values.astype(np.float32)
        
        # ⚡ 核心对齐：像实盘一样，将 20 维数据复制 2 遍，拼成 40 维！
        self.data = np.concatenate([raw_data, raw_data], axis=1)
        
        # 3. 标签生成 (使用中间价 mid_price 来计算未来涨跌)
        mid_prices = df['mid'].values
        self.labels = []
        
        # 阈值：未来价格变动百分之多少算涨跌？根据你的股票活跃度调整
        # 模拟盘可以设得极小，比如 0.0001
        threshold = 0.0001 
        
        for i in range(len(mid_prices) - prediction_horizon):
            current_mid = mid_prices[i]
            future_mid = mid_prices[i + prediction_horizon]
            
            return_pct = (future_mid - current_mid) / current_mid
            
            if return_pct > threshold:
                self.labels.append(2)  # 上涨
            elif return_pct < -threshold:
                self.labels.append(0)  # 下跌
            else:
                self.labels.append(1)  # 走平
                
        self.labels = np.array(self.labels, dtype=np.int64)
        
        # 滑动窗口能生成的有效样本数
        self.num_samples = len(self.labels) - seq_length
        self.seq_length = seq_length
        
        print(f"数据加载完成。总样本数: {self.num_samples}")
        print(f"类别分布: 下跌={np.sum(self.labels==0)}, 走平={np.sum(self.labels==1)}, 上涨={np.sum(self.labels==2)}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 提取长为 100 的时间序列
        seq = self.data[idx : idx + self.seq_length]
        
        # ⚡ 终极 Z-score 防 nan 改造
        mean = seq.mean(axis=0)
        std = seq.std(axis=0)
        
        # 只要标准差小于 1e-4 或者本身是无效数，直接不进行除法，避免放飞
        std[std < 1e-4] = 1.0
        
        norm_seq = (seq - mean) / std
        
        # ⚡ 再次强行安全检查：万一还有 nan，直接全部用 0 填充
        norm_seq = np.nan_to_num(norm_seq, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 转换成 PyTorch Tensor
        x = torch.tensor(norm_seq, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(self.labels[idx + self.seq_length], dtype=torch.long)
        
        return x, y
# ==============================================================================
# 🚂 3. 训练与验证流程
# ==============================================================================
def train_model(csv_file):
    # 超参数
    BATCH_SIZE = 64
    EPOCHS = 10
    LEARNING_RATE = 0.0003
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {DEVICE}")

    # 1. 准备数据
    dataset = LOBDataset(csv_file, seq_length=100, prediction_horizon=50)
    
    # 简易切分 80% 训练，20% 验证 (实际量化中建议按时间切分，这里简化)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 2. 实例化模型
    model = DeepLOB().to(DEVICE)
    
    # 使用类别权重平衡损失（防止走平的样本太多吃掉涨跌样本）
    # 如果你的数据里 1 极其多，可以用下面的方式平衡：
    # criterion = nn.CrossEntropyLoss(weight=torch.tensor([2.0, 1.0, 2.0]).to(DEVICE))
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_acc = 0.0

    print("开始训练...")
    for epoch in range(EPOCHS):
        # --- 训练阶段 ---
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        train_acc = 100. * correct / total
        
        # --- 验证阶段 ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
        val_acc = 100. * val_correct / val_total
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"| Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% "
              f"| Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "deeplob_weight.pth")
            print(f"🌟 最高验证集准确率提升至 {val_acc:.2f}%, 权重已保存为 deeplob_weight.pth")
            
    # 打印最终详细分类报告
    print("\n📊 最佳模型的验证集分类报告:")
    print(classification_report(all_targets, all_preds, target_names=['下跌 (0)', '走平 (1)', '上涨 (2)']))

if __name__ == "__main__":
    # 请将此处替换为你收集的 LOB CSV 文件路径
    csv_file_path = "order_book.csv" 
    
    # 自动生成一份假数据仅供代码跑通测试（你真正跑的时候把这几行删掉）
    if not os.path.exists(csv_file_path):
        print("未找到真实的 CSV，正在生成一份随机测试数据用于跑通代码...")
        dummy_data = []
        for i in range(2000):
            dummy_data.append([
                177.10 + i*0.01, 2.0, 177.15 + i*0.01, 4.0,
                177.05, 5.0, 177.20, 3.0,
                177.00, 10.0, 177.25, 6.0,
                176.95, 12.0, 177.30, 8.0,
                176.90, 15.0, 177.35, 10.0,
                177.125 + i*0.01 # 模拟 mid 价格
            ])
        cols = [
            'bid_0_price', 'bid_0_size', 'ask_0_price', 'ask_0_size',
            'bid_1_price', 'bid_1_size', 'ask_1_price', 'ask_1_size',
            'bid_2_price', 'bid_2_size', 'ask_2_price', 'ask_2_size',
            'bid_3_price', 'bid_3_size', 'ask_3_price', 'ask_3_size',
            'bid_4_price', 'bid_4_size', 'ask_4_price', 'ask_4_size',
            'mid'
        ]
        pd.DataFrame(dummy_data, columns=cols).to_csv(csv_file_path, index=False)
        
    train_model(csv_file_path)