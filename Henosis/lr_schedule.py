import math
import torch
from torch.optim.lr_scheduler import _LRScheduler
class DynamicLRScheduler(_LRScheduler):
    def __init__(self, optimizer, threshold=0.005, factor=0.5, patience=3,min_lr=1e-6, max_lr=1e-3):
        self.threshold = threshold  # loss变化幅度阈值
        self.factor = factor        # 调整系数（增大或减小）
        self.patience = patience    # 连续触发次数
        self.last_loss = float('inf')
        self.stable_count = 0
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.train_loss = None
        super().__init__(optimizer)
    def get_lr(self):
        if self.train_loss is None:  # 添加空值检查
            return [group['lr'] for group in self.optimizer.param_groups]
        current_loss=self.train_loss
        loss_change=abs(current_loss-self.last_loss)
        new_lr = [group['lr'] for group in self.optimizer.param_groups]  
        if loss_change < self.threshold:
            self.stable_count += 1
            if self.stable_count >= self.patience:
                new_lr = [base_lr * self.factor for base_lr in self.base_lrs]
                self.stable_count = 0 
        else:
            self.stable_count = 0
            if loss_change > 2*self.threshold and current_loss<self.train_loss:  # 变化过大时反向调整
                new_lr = [base_lr*0.75 / self.factor for base_lr in self.base_lrs]
        self.last_loss = current_loss
        return new_lr
    
class CustomSchedule(object):
    def __init__(self,d_model,warmup_steps=4000,optimizer=None):
        super(CustomSchedule,self).__init__()
        self.d_model=torch.tensor(d_model,dtype=torch.float32)
        self.warmup_steps = warmup_steps
        self.steps=1.
        self.optimizer = optimizer
    def step(self):
        arg1=self.steps**-0.5
        arg2=self.steps*(self.warmup_steps**-1.5)
        self.steps+=1.
        lr=(self.d_model**-0.5)*min(arg1,arg2)
        for p in self.optimizer.param_groups:
            p['lr'] = lr
        return lr

class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_steps : int = 0,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr