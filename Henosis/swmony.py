from functools import partial
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import logging
import time
from scipy.special import softmax
#from model.harmo import optimized_moe_correct_ridge
#import os
#import hashlib

# create logger
logger = logging.getLogger('harmonypy')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# from IPython.core.debugger import set_trace

def run_harmony(
    data_mat: np.ndarray,
    meta_data: pd.DataFrame,
    vars_use,
    theta = None,
    lamb = None,
    sigma = 0.1, 
    nclust = None,
    tau = 0,
    block_size = 0.05, 
    max_iter_harmony = 10,
    max_iter_kmeans = 20,
    epsilon_cluster = 1e-5,
    epsilon_harmony = 1e-4, 
    plot_convergence = False,
    verbose = True,
    reference_values = None,
    cluster_prior = None,
    random_state = 0,
    cluster_fn = 'kmeans',
    clust_mode='softmax',
    comatrix_path=None,
    pcs=None,
    ref_mean=None,
    test_mode=False
):
    """Run Harmony.
    """

    # theta = None
    # lamb = None
    # sigma = 0.1
    # nclust = None
    # tau = 0
    # block_size = 0.05
    # epsilon_cluster = 1e-5
    # epsilon_harmony = 1e-4
    # plot_convergence = False
    # verbose = True
    # reference_values = None
    # cluster_prior = None
    # random_state = 0
    # cluster_fn = 'kmeans'. Also accepts a callable object with data, num_clusters parameters

    N = meta_data.shape[0]
    if data_mat.shape[1] != N:
        data_mat = data_mat.T

    assert data_mat.shape[1] == N, \
       "data_mat and meta_data do not have the same number of cells" 

    if nclust is None:
        nclust = np.min([np.round(N / 30.0), 100]).astype(int)

    if type(sigma) is float :#and nclust > 1:
        sigma = np.repeat(sigma, nclust)

    if isinstance(vars_use, str):
        vars_use = [vars_use]

    unique_batches = sorted(meta_data[vars_use[0]].unique())
    meta_data[vars_use[0]] = pd.Categorical(
        meta_data[vars_use[0]], 
        categories=unique_batches, 
        ordered=True
    )
    phi = pd.get_dummies(meta_data[vars_use[0]]).to_numpy().T
    phi_n = meta_data[vars_use].describe().loc['unique'].to_numpy().astype(int)
    sp_mapping = {batch_name: i for i, batch_name in enumerate(unique_batches)}
    print('sp_mapping',sp_mapping)
    '''sp = list(set(meta_data[vars_use].iloc[:, 0].tolist()))
    df = pd.get_dummies(sp)
    sp_mapping={v: i for i, v in enumerate(df.columns)}'''

    if theta is None:
        theta = np.repeat([1] * len(phi_n), phi_n)
        #theta = np.repeat([2] * len(phi_n), phi_n)
    elif isinstance(theta, float) or isinstance(theta, int):
        theta = np.repeat([theta] * len(phi_n), phi_n)
    elif len(theta) == len(phi_n):
        theta = np.repeat([theta], phi_n)

    assert len(theta) == np.sum(phi_n), \
        "each batch variable must have a theta"

    if lamb is None:
        lamb = np.repeat([1] * len(phi_n), phi_n)
        #lamb = np.repeat([1e-4] * len(phi_n), phi_n)
    elif isinstance(lamb, float) or isinstance(lamb, int):
        lamb = np.repeat([lamb] * len(phi_n), phi_n)
    elif len(lamb) == len(phi_n):
        lamb = np.repeat([lamb], phi_n)

    assert len(lamb) == np.sum(phi_n), \
        "each batch variable must have a lambda"

    # Number of items in each category.
    N_b = phi.sum(axis = 1)
    # Proportion of items in each category.
    Pr_b = N_b / N

    if tau > 0:
        theta = theta * (1 - np.exp(-(N_b / (nclust * tau)) ** 2))

    lamb_mat = np.diag(np.insert(lamb, 0, 0))

    phi_moe = np.vstack((np.repeat(1, N), phi))

    np.random.seed(random_state)

    
    ho = Harmony(
        data_mat, phi, phi_moe, Pr_b, sigma, theta, max_iter_harmony, max_iter_kmeans,
        epsilon_cluster, epsilon_harmony, nclust, block_size, lamb_mat, verbose,
        random_state,cluster_fn,clust_mode,test_mode
    )
    #ref_mean = np.mean(data_mat.T, axis=0)
    if comatrix_path is not None:
        np.savez_compressed(comatrix_path,W=ho.W,Y=ho.Y,sp_mapping=sp_mapping,pcs=pcs,ref_mean=ref_mean)
    #print(f"Z_corr hash: {hashlib.sha256(ho.Z_corr.tobytes()).hexdigest()}")
    if test_mode:
        return ho.Z_corr.T,ho.test_corr.T
    else:
        return ho.Z_corr.T
    



class Harmony(object):
    def __init__(
            self, Z, Phi, Phi_moe, Pr_b, sigma,
            theta, max_iter_harmony, max_iter_kmeans, 
            epsilon_kmeans, epsilon_harmony, K, block_size,
            lamb, verbose, random_state=None, cluster_fn='kmeans',clust_mode='softmax',test_mode=False
    ):
        self.Z_corr = np.array(Z)
        self.Z_orig = np.array(Z)

        self.Z_cos = self.Z_orig / self.Z_orig.max(axis=0)
        self.Z_cos = self.Z_cos / np.linalg.norm(self.Z_cos, ord=2, axis=0)

        self.Phi             = Phi
        self.Phi_moe         = Phi_moe
        self.N               = self.Z_corr.shape[1]
        self.Pr_b            = Pr_b
        self.B               = self.Phi.shape[0] # number of batch variables
        self.d               = self.Z_corr.shape[0]
        self.window_size     = 3
        self.epsilon_kmeans  = epsilon_kmeans
        self.epsilon_harmony = epsilon_harmony

        self.lamb            = lamb
        self.sigma           = sigma
        self.sigma_prior     = sigma
        self.block_size      = block_size
        self.K               = K                # number of clusters
        self.max_iter_harmony = max_iter_harmony
        self.max_iter_kmeans = max_iter_kmeans
        self.verbose         = verbose
        self.theta           = theta
        self.clust_mode      = clust_mode

        self.objective_harmony        = []
        self.objective_kmeans         = []
        self.objective_kmeans_dist    = []
        self.objective_kmeans_entropy = []
        self.objective_kmeans_cross   = []
        self.kmeans_rounds  = []

        self.allocate_buffers()
        if cluster_fn == 'kmeans':
            cluster_fn = partial(Harmony._cluster_kmeans, random_state=random_state)
        self.init_cluster(cluster_fn)
        self.harmonize(self.max_iter_harmony, self.verbose)
        self.test_mode=test_mode
        if test_mode:
            self.test_corr=self.transform(self.Z_orig,self.Phi_moe)
    def result(self):
        if self.test_mode:
            return self.Z_corr,self.test_corr
        else:
            return self.Z_corr

    def allocate_buffers(self):
        self._scale_dist = np.zeros((self.K, self.N))
        self.dist_mat    = np.zeros((self.K, self.N))
        self.O           = np.zeros((self.K, self.B))
        self.E           = np.zeros((self.K, self.B))
        self.W           = np.zeros((self.B + 1, self.d))
        self.Phi_Rk      = np.zeros((self.B + 1, self.N))

    @staticmethod
    def _cluster_kmeans(data, K, random_state):
        # Start with cluster centroids
        logger.info("Computing initial centroids with sklearn.KMeans...")
        model = KMeans(n_clusters=K, init='k-means++',
                       n_init=10, max_iter=25, random_state=random_state)
        model.fit(data)
        km_centroids, km_labels = model.cluster_centers_, model.labels_
        logger.info("sklearn.KMeans initialization complete.")
        transfer_matrix=np.eye(K)[km_labels]
        beta=np.sum(transfer_matrix,axis=1)
        alpha=np.sum(transfer_matrix,axis=0)/data.shape[0]
        return km_centroids,beta,alpha

    def init_cluster(self, cluster_fn):
        self.Y,self.beta,self.alpha= cluster_fn(self.Z_cos.T, self.K)
        self.Y=self.Y.T
        # (1) Normalize
        self.Y = self.Y / np.linalg.norm(self.Y, ord=2, axis=0)
        # (2) Assign cluster probabilities
        self.dist_mat = 2 * (1 - np.dot(self.Y.T, self.Z_cos))
        self.R = -self.dist_mat
        if self.clust_mode=='softmax':
            self.R=softmax(self.R/self.sigma[:,None],axis=0)#
        else:
            self.R=sinkhorn_knopp(self.R.T,self.alpha,self.beta,0.1,1.0).T#
        '''self.R = self.R / self.sigma[:,None]
        self.R -= np.max(self.R, axis = 0)
        self.R = np.exp(self.R)
        self.R = self.R / np.sum(self.R, axis = 0)'''
        # (3) Batch diversity statistics
        self.E = np.outer(np.sum(self.R, axis=1), self.Pr_b)
        self.O = np.inner(self.R , self.Phi)
        self.compute_objective()
        # Save results
        self.objective_harmony.append(self.objective_kmeans[-1])

    def compute_objective(self):
        kmeans_error = np.sum(np.multiply(self.R, self.dist_mat))
        # Entropy
        _entropy = np.sum(safe_entropy(self.R) * self.sigma[:,np.newaxis])
        # Cross Entropy
        x = (self.R * self.sigma[:,np.newaxis])
        y = np.tile(self.theta[:,np.newaxis], self.K).T
        z = np.log((self.O + 1) / (self.E + 1))
        w = np.dot(y * z, self.Phi)
        _cross_entropy = np.sum(x * w)
        # Save results
        self.objective_kmeans.append(kmeans_error + _entropy + _cross_entropy)
        self.objective_kmeans_dist.append(kmeans_error)
        self.objective_kmeans_entropy.append(_entropy)
        self.objective_kmeans_cross.append(_cross_entropy)

    def harmonize(self, iter_harmony=10, verbose=True):
        converged = False
        for i in range(1, iter_harmony + 1):
            if verbose:
                logger.info("Iteration {} of {}".format(i, iter_harmony))
            # STEP 1: Clustering
            t0 = time.perf_counter()
            self.cluster()
            t_cluster = time.perf_counter() - t0
            # STEP 2: Regress out covariates
            # self.moe_correct_ridge()
            t0 = time.perf_counter()
            '''self.Z_cos, self.Z_corr, self.W, self.Phi_Rk = moe_correct_ridge(#moe_correct_ridge(
                self.Z_orig, self.Z_cos, self.Z_corr, self.R, self.W, self.K,
                self.Phi_Rk, self.Phi_moe, self.lamb
            )'''
            self.Z_cos,self.Z_corr, self.W,=optimized_moe_correct_ridge(self.Z_orig, self.Phi_moe, self.R, self.lamb)
            #import model.moe as moe
            #self.Z_cos,self.Z_corr, self.W,=moe.moe_train(self.Z_orig, self.Phi_moe, self.R, self.lamb)
            t_ridge = time.perf_counter() - t0
            if verbose:
                logger.info(
                    f"Timing | cluster: {t_cluster:.3f}s | "
                    f"moe_ridge: {t_ridge:.3f}s"
                )
            # STEP 3: Check for convergence
            converged = self.check_convergence(1)
            '''if converged:
                if verbose:
                    logger.info(
                        "Converged after {} iteration{}"
                        .format(i, 's' if i > 1 else '')
                    )
                break'''
        if verbose and not converged:
            logger.info("Stopped before convergence")
        return 0

    def cluster(self):
        # Z_cos has changed
        # R is assumed to not have changed
        # Update Y to match new integrated data
        self.dist_mat = 2 * (1 - np.dot(self.Y.T, self.Z_cos))
        for i in range(self.max_iter_kmeans):
            # print("kmeans {}".format(i))
            # STEP 1: Update Y
            self.Y = np.dot(self.Z_cos, self.R.T)/np.sum(self.R, axis=1)
            self.Y = self.Y / np.linalg.norm(self.Y, ord=2, axis=0)
            # STEP 2: Update dist_mat
            self.dist_mat = 2 * (1 - np.dot(self.Y.T, self.Z_cos))
            # STEP 3: Update R
            self.update_R()
            # STEP 4: Check for convergence
            self.compute_objective()
            if i > self.window_size:
                converged = self.check_convergence(0)
                if converged:
                    break
        self.kmeans_rounds.append(i)
        self.objective_harmony.append(self.objective_kmeans[-1])
        return 0

    def update_R(self):
        self._scale_dist = -self.dist_mat
        #self._scale_dist = self._scale_dist / self.sigma[:,None]
        '''self.O=np.dot(self.R, self.Phi.T)
        self.E = np.outer(np.sum(self.R, axis=1), self.Pr_b)
        self.logEO=self.theta*np.log((self.O + 1) / (self.E + 1))
        self.logEO=np.dot(self.logEO,self.Phi)
        self._scale_dist+=self.logEO'''
        if self.clust_mode=='softmax':
            self.R=softmax(self._scale_dist/self.sigma[:,None],axis=0)
        else:
            self.R=sinkhorn_knopp(self._scale_dist.T,self.alpha,self.beta,0.1,1.0).T#softmax(self._scale_dist/self.sigma[:,None],axis=0)#
        self.alpha=np.sum(self.R,axis=1)
        return 0

    def check_convergence(self, i_type):
        obj_old = 0.0
        obj_new = 0.0
        # Clustering, compute new window mean
        if i_type == 0:
            okl = len(self.objective_kmeans)
            for i in range(self.window_size):
                obj_old += self.objective_kmeans[okl - 2 - i]
                obj_new += self.objective_kmeans[okl - 1 - i]
            if abs(obj_old - obj_new) / abs(obj_old) < self.epsilon_kmeans:
                return True
            return False
        # Harmony
        if i_type == 1:
            obj_old = self.objective_harmony[-2]
            obj_new = self.objective_harmony[-1]
            if (obj_old - obj_new) / abs(obj_old) < self.epsilon_harmony:
                return True
            return False
        return True

    def transform(self, Z_new, Phi_new_moe):
        """
        映射新样本到已有的 Harmony 空间。
        Z_new: (D, N_new) 新样本的 PCA 降维结果
        Phi_new_moe: (B, N_new) 新样本的批次设计矩阵（类别需与 Reference 严格对齐）
        """
        # 1. 对新样本进行余弦归一化
        Z_new_cos = Z_new / np.linalg.norm(Z_new, ord=2, axis=0, keepdims=True)
        
        # 2. 计算新样本到“冻结”的聚类中心 Y 的距离
        dist_mat_new = 2 * (1 - np.dot(self.Y.T, Z_new_cos))
        
        # 3. 计算软聚类分配概率 R_new (这里以 softmax 为例，为了极速计算)
        # 注意：预测阶段通常直接用 softmax 即可，不需要做复杂的 Sinkhorn 约束
        R_new = softmax(-dist_mat_new / self.sigma[:, None], axis=0)
        
        # 4. 使用“冻结”的修正矩阵 W 进行反向修正
        # 将 W (K, B, D) 压平以复用 einsum 逻辑
        B = self.Phi_moe.shape[0]
        D = self.d
        K = self.K
        
        # 将自有的 W 重塑以便与 R_new 相乘
        W_flat = self.W.transpose(1, 2, 0).reshape(B*D, K)
        correction_intermediate = W_flat @ R_new  # (B*D, N_new)
        correction_intermediate = correction_intermediate.reshape(B, D, -1)
        
        # 应用批次设计矩阵，计算最终的新样本修正坐标
        Z_new_corr = Z_new - np.einsum('bn,bdn->dn', Phi_new_moe, correction_intermediate)
        return Z_new_corr

def sinkhorn_knopp(scores,alpha,beta,epsilon,reg, n_iterations=3):
    #beta是行约束,alpha是列约束
    kernel = np.exp(scores/epsilon) 
    v=np.ones(scores.shape[1])
    f=reg / (reg + epsilon)
    for i in range(n_iterations):
        u = beta / np.dot(kernel, v)
        v = (alpha / np.dot(kernel.T, u))**f
    P = u[:, np.newaxis] * kernel * v[np.newaxis, :]
    return P


def safe_entropy(x: np.array):
    y = np.multiply(x, np.log(x))
    y[~np.isfinite(y)] = 0.0
    return y


def moe_correct_ridge(Z_orig, Z_cos, Z_corr, R, W, K, Phi_Rk, Phi_moe, lamb):
    Z_corr = Z_orig.copy()
    for i in range(K):
        Phi_Rk = np.multiply(Phi_moe, R[i,:])
        x = np.dot(Phi_Rk, Phi_moe.T) + lamb
        W = np.dot(np.dot(np.linalg.inv(x), Phi_Rk), Z_orig.T)
        W[0,:] = 0 # do not remove the intercept
        Z_corr -= np.dot(W.T, Phi_Rk)
    Z_cos = Z_corr / np.linalg.norm(Z_corr, ord=2, axis=0)
    return Z_cos, Z_corr, W, Phi_Rk

def optimized_moe_correct_ridge(Z_orig, Phi_moe, R, lamb):
    """
    针对 N 很大、B 和 D 较小场景优化的 Ridge 修正
    Phi_moe: (B, N)
    Z_orig: (D, N)
    R: (K, N)
    lamb: scalar or (B, B)
    """
    B, N = Phi_moe.shape
    D = Z_orig.shape[0]
    K = R.shape[0]

    # --- Step 1: 批量计算所有 K 个 W 矩阵 ---
    # 计算 Phi_moe 的自外积和与 Z_orig 的外积，用于构建 Ridge 方程
    # 形状说明: (B, B, N) -> (B*B, N) 
    # 这里巧妙利用广播避开 K，先处理 N
    Phi_outer = (Phi_moe[:, None, :] * Phi_moe[None, :, :]).reshape(B*B, N)
    Phi_Z_outer = (Phi_moe[:, None, :] * Z_orig[None, :, :]).reshape(B*D, N)

    # 通过 R 将 N 维度缩并掉，得到所有 K 对应的组件
    # x_all 形状: (K, B, B), M_all 形状: (K, B, D)
    x_all = (R @ Phi_outer.T).reshape(K, B, B)

    x_all+= lamb
    M_all = (R @ Phi_Z_outer.T).reshape(K, B, D)

    # 解方程 W = x_inv @ M
    W = np.linalg.solve(x_all, M_all) 
    W[:, 0, :] = 0  # 对应原代码 W[0,:] = 0

    # --- Step 2: 合并修正项并应用 ---
    # 原式: Z_corr = Z_orig - sum_k(W_k.T @ (Phi_moe * R_k))
    # 变换: Z_corr = Z_orig - sum_b(Phi_moe_b * sum_k(R_kn * W_k_bd))
    
    # 将 W 与 R 结合: (B*D, K) @ (K, N) -> (B*D, N)
    W_flat = W.transpose(1, 2, 0).reshape(B*D, K)
    correction_intermediate = W_flat @ R # 形状 (B*D, N)
    correction_intermediate = correction_intermediate.reshape(B, D, N)

    # 最后一步：沿着 B 维度与 Phi_moe 相乘并求和
    # (B, N) 与 (B, D, N) 在 B 上缩并
    # 使用 einsum 以保持内存高效
    Z_corr = Z_orig - np.einsum('bn,bdn->dn', Phi_moe, correction_intermediate)

    # 计算余弦归一化
    Z_cos = Z_corr / np.linalg.norm(Z_corr, ord=2, axis=0, keepdims=True)

    return Z_cos, Z_corr, W



def load_comatrix(comatrix_path,X,sp):
    Z_orig=X.copy()
    cache = np.load(comatrix_path, allow_pickle=True)
    #Z=Z-np.mean(Z,axis=0)
    ref_mean = cache["ref_mean"]
    Z=Z_orig.T-ref_mean
    pcs=cache["pcs"]
    Z=Z@pcs
    Z_orig=Z.T
    W = cache["W"]
    K,B,D=W.shape
    #R = cache["R"]

    Y=cache["Y"]
    Z_cos = Z_orig / Z_orig.max(axis=0)
    Z_cos = Z_cos / np.linalg.norm(Z_cos, ord=2, axis=0)
    dist_mat = 2 * (1 - np.dot(Y.T, Z_cos))
    sigma=0.1
    sigma = np.repeat(sigma, K)
    R = softmax(-dist_mat / sigma[:, None], axis=0)

    _,N=R.shape
    sp_mapping=cache["sp_mapping"].item()
    L = len(sp_mapping)
    def encode(x):
        vec = np.zeros(L)
        if x in sp_mapping:
            vec[sp_mapping[x]] = 1
        return vec

    phi = np.array([encode(s) for s in sp]).T
    #phi = pd.get_dummies(sp).to_numpy().T
    Phi_moe = np.vstack((np.repeat(1, N), phi))
    B, _ = Phi_moe.shape
    D = Z_orig.shape[0]
    print(f"K={K}, B={B}, N={N}, D={D}")
    print("W shape:",W.shape)
    W_flat = W.transpose(1, 2, 0).reshape(B*D, K)
    correction_intermediate = W_flat @ R # 形状 (B*D, N)
    correction_intermediate = correction_intermediate.reshape(B, D, N)
    Z_corr = Z_orig - np.einsum('bn,bdn->dn', Phi_moe, correction_intermediate)
    return Z_corr.T
