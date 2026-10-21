import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, kl_divergence
#from geomloss import SamplesLoss

class MMD(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5,eps=1e-4):
        super(MMD, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type
        self.eps = eps 

    def gaussian_kernel(self, source, target, kernel_mul=2.0, kernel_num=2, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
        total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance).detach() / (n_samples**2 - n_samples)

        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth = torch.clamp(bandwidth, min=self.eps)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        mean_x = f_of_X.mean(0)
        mean_y = f_of_Y.mean(0)
        return torch.sum((mean_x - mean_y) ** 2)

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.gaussian_kernel(
                source, target, kernel_mul=self.kernel_mul,
                kernel_num=self.kernel_num, fix_sigma=self.fix_sigma
            )
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = XX + YY - XY - YX
            return loss

class MMMD(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5,eps=1e-4,min_n_samples=2,each=True):
        super(MMMD, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type
        self.eps = eps
        self.min_n_samples=min_n_samples
        self.mmd=MMD(kernel_type='rbf', kernel_mul=2.0, kernel_num=5,eps=1e-4)
        self.each=each

    def each_mmd(self,z,cat):
        if len(torch.unique(cat))==1:
            return torch.tensor(0.0, device=z.device)
        zs = []
        loss = torch.tensor(0.0, device=z.device)
        zs = [z[cat == i] for i in torch.unique(cat)]
        count = 0
        for i in range(len(torch.unique(cat))):
            for j in range(i+1,len(torch.unique(cat))):
                if zs[i].shape[0] > self.min_n_samples and zs[j].shape[0] > self.min_n_samples:
                    loss +=self.mmd(zs[i], zs[j])
                    count += 1
        return loss / count
    
    def zero_mmd(self,z,cat):
        if len(torch.unique(cat))==1:
            return 0#torch.tensor(0.0, device=z.device)
        else:
            zs = []
            loss = 0#torch.tensor(0.0, device=z.device)
            zs = [z[cat == i] for i in torch.unique(cat)]
            for i in range(1,len(torch.unique(cat))):
                if zs[i].shape[0] > self.min_n_samples and zs[0].shape[0] > self.min_n_samples:
                    loss +=self.mmd(zs[i], zs[0])
            return loss

    def forward(self,z,cat):
        if self.each:
            loss=self.each_mmd(z,cat)
        else:
            loss=self.zero_mmd(z,cat)
        return loss

'''class MSampleLoss(nn.Module):
    def __init__(self,p=2, blur=0.05, scaling=0.9,min_n_samples=2,each=True):
        super(MSampleLoss, self).__init__()
        self.loss=SamplesLoss(p=p, blur=blur, scaling=scaling)
        self.min_n_samples=min_n_samples
        self.each=each
    def each_sample_loss(self,z,cat):
        zs = []
        loss = torch.tensor(0.0, device=z.device)
        zs = [z[cat == i] for i in torch.unique(cat)]
        for i in range(len(torch.unique(cat))):
            for j in range(i+1,len(torch.unique(cat))):
                if zs[i].shape[0] > self.min_n_samples and zs[j].shape[0] > self.min_n_samples:
                    loss +=self.loss(zs[i], zs[j])
        return loss
    
    def zero_sample_loss(self,z,cat):
        if len(torch.unique(cat))==1:
            return torch.tensor(0.0, device=z.device)
        else:
            zs = []
            loss = torch.tensor(0.0, device=z.device)
            zs = [z[cat == i] for i in torch.unique(cat)]
            for i in range(1,len(torch.unique(cat))):
                if zs[i].shape[0] > self.min_n_samples and zs[0].shape[0] > self.min_n_samples:
                    loss +=self.loss(zs[i], zs[0])
            return loss
        
    def forward(self,z,cat):
        if self.each:
            loss=self.each_sample_loss(z,cat)
        else:
            loss=self.zero_sample_loss(z,cat)
        return loss'''
    


def kl_mvn(mu0, Sigma0, mu1, Sigma1, eps=1e-6):
    d = mu0.shape[0]

    # 转成 float32 保证可逆分解
    Sigma0 = Sigma0.to(torch.float32) + eps * torch.eye(d, device=Sigma0.device)
    Sigma1 = Sigma1.to(torch.float32) + eps * torch.eye(d, device=Sigma1.device)
    mu0 = mu0.to(torch.float32)
    mu1 = mu1.to(torch.float32)

    inv_Sigma1 = torch.linalg.inv(Sigma1)
    term_trace = torch.trace(inv_Sigma1 @ Sigma0)
    term_mahalanobis = (mu1 - mu0).T @ inv_Sigma1 @ (mu1 - mu0)
    term_logdet = torch.logdet(Sigma1) - torch.logdet(Sigma0)

    kl = 0.5 * (term_trace + term_mahalanobis - d + term_logdet)
    return kl

def kl_divergence_mv(emb0, emb1, eps=1e-6):
    mu0 = emb0.mean(dim=0)
    sigma0 = torch.cov(emb0.T) + eps * torch.eye(emb0.shape[1], device=emb0.device)

    mu1 = emb1.mean(dim=0)
    sigma1 = torch.cov(emb1.T) + eps * torch.eye(emb1.shape[1], device=emb1.device)

    dist0 = MultivariateNormal(mu0, covariance_matrix=sigma0)
    dist1 = MultivariateNormal(mu1, covariance_matrix=sigma1)

    return kl_divergence(dist0, dist1)

def clip_distance_matrix(pts_src: torch.Tensor, pts_dst: torch.Tensor, p: int = 2):
    """
    Returns the matrix of ||x_i-y_j||_p, while restricting x_i,y_j in unit balls by multiplying a constant.

    Parameters
    ----------
    pts_src
        [R, D] matrix
    pts_dst
        [C, D] matrix
    p
        p-norm
    
    Return
    ------
    [R, C] matrix
         clipped distance matrix
    """
    clip_value=torch.tensor(1000.0)
    max_norm=torch.max(torch.abs(pts_src).max()+  torch.abs(pts_dst).max() , 2*clip_value )/2
    pts_src_clip=clip_value*pts_src/max_norm
    pts_dst_clip=clip_value*pts_dst/max_norm
    x_col = pts_src_clip.unsqueeze(1)
    y_row = pts_dst_clip.unsqueeze(0)
    distance = torch.sum((torch.abs(x_col - y_row)) ** p, 2)
    return distance

def unbalanced_ot_z( x1, x2,reg=0.1, reg_m=1.0,max_iteration={'outer':10,'inner':5}):
    '''
    Calculate a unbalanced optimal transport matrix between mini subsets.
    Parameters
    ----------
    reg:
        Entropy regularization parameter in OT
    reg_m:
        Unbalanced OT parameter. Larger values means more balanced OT
    device
        training device

    Returns
    -------
    matrix
        mini-subset unbalanced optimal transport matrix
    '''
    device=x1.device
    outer_iter=max_iteration['outer']
    inner_iter=max_iteration['inner']
    ns = x1.size(0)
    nt = x2.size(0)
    cost_pp = clip_distance_matrix(x1, x2, p=2).to(device)
    
    p_s = torch.ones(ns, 1) / ns   
    p_t = torch.ones(nt, 1) / nt
    

    p_s = p_s.to(device)
    p_t = p_t.to(device)

    tran = torch.ones(ns, nt) / (ns * nt)
    tran = tran.to(device)

    dual = (torch.ones(ns, 1) / ns).to(device)
    f = reg_m / (reg_m + reg)

    for m in range(outer_iter):
        cost = cost_pp
        kernel = torch.exp(-cost / (reg*torch.max(torch.abs(cost)))) * tran
        b = p_t / (torch.t(kernel) @ dual)
        for i in range(inner_iter):
            dual =( p_s / (kernel @ b) )**f
            b = ( p_t / (torch.t(kernel) @ dual) )**f
        tran = (dual @ torch.t(b)) * kernel
        
    out=tran.detach()
    if torch.isnan(out).sum() > 0:        
        out=None
    return  out

if __name__ == '__main__':
    emb0=torch.randn(10,10)
    emb1=torch.randn(10,10)
    kl=kl_divergence_mv(emb0,emb1)
    print(kl.shape)