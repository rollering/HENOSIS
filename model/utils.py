import torch
from tqdm import tqdm
import numpy as np

class Accumulator:  #@save
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def evaluate_accuracy(net, data_iter):  #@save
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def accuracy(y_hat, y):  #@save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    pbar=tqdm(train_iter,desc='Processing')
    for X, y in pbar:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        pbar.set_description(f'loss: {l.item():.4f}')
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        print(train_metrics, test_acc)
        train_loss, train_acc = train_metrics



def split_data(data,seed=42):
    rng = np.random.default_rng(seed)
    data=np.array(data)
    shuffled = rng.permutation(data) 
    split_idx = int(len(data) * 4 / 5)
    return shuffled[:split_idx], shuffled[split_idx:]

def get_batch_grad_mean(net):
    total_grad = []
    for param in net.parameters():
        if param.grad is not None:
            total_grad.append(param.grad.view(-1))  # 展平成一维
    if total_grad:
        total_grad = torch.cat(total_grad)  # 合并所有梯度
        return total_grad.mean().item()  # 计算均值
    return 0  # 避免 NoneType 错误


class EarlyStopping:
    """当验证集指标停止提升时，提前停止训练，并保存最佳模型。"""
    def __init__(self, patience=7, verbose=False, delta=0, path='best_model.pt', mode='min'):
        """
        Args:
            patience (int): 没有提升的轮数上限，超过该轮数则停止训练
            verbose (bool): 是否打印每次保存的信息
            delta (float): 算作显著提升的最小变化量
            path (str): 最佳模型权重的保存路径
            mode (str): 'min' (如 val_loss) 或 'max' (如 val_acc)
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.mode = mode
        
        # 根据 mode 初始化最优值
        self.val_score_min = np.inf if mode == 'min' else -np.inf

    def __call__(self, current_score, model, extra_results=None):
        # 统一转化为“分数越高越好”的逻辑进行比较
        score = -current_score if self.mode == 'min' else current_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(current_score, model, extra_results)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(current_score, model, extra_results)
            self.counter = 0

    def save_checkpoint(self, current_score, model, extra_results):
        """保存当前最佳模型和相关结果"""
        if self.verbose:
            print(f'Validation metric improved ({self.val_score_min:.6f} --> {current_score:.6f}). Saving model...')
        
        # 构建保存的字典，包含模型状态和额外的结果（如当前的epoch、loss、metrics等）
        save_dict = {
            'model_state_dict': model.state_dict(),
            'best_score': current_score
        }
        if extra_results is not None:
            save_dict.update(extra_results)
            
        torch.save(model.state_dict(), self.path)
        self.val_score_min = current_score