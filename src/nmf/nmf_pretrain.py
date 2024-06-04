import numpy
import torch
from sklearn import preprocessing
import tqdm

EPS=1e-6
class NMF(torch.nn.Module):
    """
    Non-negative Matrix Factorization with GPU accelration
    Approach proposed in [1] to decompose a positive matrix X into two non-negative matrices W and H.
    H and W are updated with a 2-step algorithm following multiplicative updates.
    The `infer` method uses the current W matrix (should be pretrained first) and infer over a given input X by updating H through a few iterations.

    Params:
        n_components: number of components on which to decompose the input data (int)
        n_feats: first dimension of the data named as features (int)
        n_frames: second dimension of the input data named as frames (int)
        mu_fact: sparsity factor (float > 0)

    Returns:
        W and H: non-negative matrices after ONE update

    Théo Mariotte - 06/2023
    """
    def __init__(self,
                 n_components,
                 n_feats,
                 n_frames,):
        super(NMF,self).__init__()
        self.n_components=n_components
        #self.mat11 = torch.ones((n_feats,n_feats))
        #self.gamma_pow0 = torch.ones((n_feats,n_frames))
        
        self.register_buffer("H",torch.rand((n_components,n_frames)))
        self.register_buffer("W",torch.rand((n_feats,n_components)))

    def forward(self,X):
        with torch.no_grad():
            W_t_W = self.W.T.matmul(self.W)
            H_update = self.W.T.matmul(X) / (W_t_W.matmul(self.H))
            self.H = self.H * H_update
            H_H_t = self.H.matmul(self.H.T)
            W_update = X.matmul(self.H.T) / (self.W.matmul(H_H_t))
            self.W = self.W * W_update

        return self.W, self.H
    
    def infer(self,X,n_iter=25):
        """
        Inference
        Only the H matrix is optimized, the W dictionary is fixed.
        """
        H_infer = torch.rand((self.n_components,X.shape[-1])).to(X.device)
        for i in range(n_iter):
            W_t_W = self.W.T.matmul(self.W)
            H_update = self.W.T.matmul(X) / (W_t_W.matmul(H_infer))
            H_infer = H_infer * H_update
           
        return H_infer
    
    def _norm(self,W):
        Wn = torch.nn.functional.normalize(W,dim=1,eps=EPS)
        return Wn

    def get_dictionary(self):
        return self.W
    
    
class SNMF(torch.nn.Module):
    """
    Sparse Non-negative Matrix Factorization with GPU accelration
    Approach proposed in [1] to decompose a positive matrix X into two non-negative matrices W and H.
    H and W are updated with a 2-step algorithm following multiplicative updates.
    The `infer` method uses the current W matrix (should be pretrained first) and infer over a given input X by updating H through a few iterations.

    Params:
        n_components: number of components on which to decompose the input data (int)
        n_feats: first dimension of the data named as features (int)
        n_frames: second dimension of the input data named as frames (int)
        mu_fact: sparsity factor (float > 0)

    Returns:
        W and H: non-negative matrices after ONE update

    Théo Mariotte - 06/2023
    """
    def __init__(self,
                 n_components,
                 n_feats,
                 n_frames,
                 mu_fact=1.0,
                 beta=2,):
        super(SNMF,self).__init__()
        self.n_components=n_components
        self.mu = mu_fact
        self.beta=beta
        #self.mat11 = torch.ones((n_feats,n_feats))
        #self.gamma_pow0 = torch.ones((n_feats,n_frames))
        
        self.register_buffer("gamma_pow0",torch.ones((n_feats,n_frames)))
        self.register_buffer("mat11",torch.ones((n_feats,n_feats)))
        self.register_buffer("H",torch.rand((n_components,n_frames)))
        self.register_buffer("W",torch.rand((n_feats,n_components)))

    def forward(self,X):
        with torch.no_grad():
            self.W = self._norm(self.W)
            G = self.W.matmul(self.H)
            G_pow_beta_1=G**(self.beta-1)
            G_pow_beta_2=G**(self.beta-2)            
            # Update H
            num_h = (self.W.T).matmul(X*G_pow_beta_2)
            den_h = (self.W.T).matmul(G_pow_beta_1) + self.mu
            H_update = num_h/den_h
            self.H = self.H * H_update
            G = self.W.matmul(self.H)
            
            # Update W
            num_val1 = (G_pow_beta_2*X).matmul(self.H.T)
            tmp_ = self.W * (G_pow_beta_1.matmul(self.H.T))
            num_val2 = self.W * ((self.mat11.T).matmul(tmp_))
            den_val1 = G_pow_beta_1.matmul(self.H.T)
            tmp_ = num_val1*self.W
            den_val2 = self.W*(self.mat11.matmul(tmp_))

            W_update = (num_val1 + num_val2)/(den_val1 + den_val2 + EPS)
            
            self.W = self.W * W_update
            
            self.W = self._norm(self.W)

        return self.W, self.H
    
    def infer(self,X,n_iter=25):
        """
        Inference
        Only the H matrix is optimized, the W dictionary is fixed.
        """
        H_infer = torch.rand((self.n_components,X.shape[-1])).to(X.device)
        gamma_pow0 = torch.ones((X.shape[0],X.shape[-1])).to(X.device)
        for i in range(n_iter):
            G = self.W.matmul(H_infer)
            num_h = (self.W.T).matmul(X*gamma_pow0)
            den_h = (self.W.T).matmul(G) + self.mu
            H_update = num_h/den_h
            H_infer = H_infer * H_update
           
        return H_infer
    
    def _norm(self,W):
        Wn = torch.nn.functional.normalize(W,dim=1,eps=EPS)
        return Wn

    def get_dictionary(self):
        return self.W


def snmf_pretrain(X,
                  n_components,
                  device,
                  n_iter=100,
                  mu_fact=1.0,
                  beta=2,
                  save_measure=False,
                  save_dictionary=None,
                  is_sparse=True):
    """
    Solve Sparse Non-negative Matrix Factorization [1] from a given dataset. 
    We highly recommend to run this un a GPU!

    Params:
        X: dataset to fit the W and H matrices (torch.Tensor)
        n_components: number of compnonents to represent the data e.g. W \in R^{F x n_components} (int)
        device: device on which to do the training. (torch.cuda.device)
        n_iter: number of iteration to fit (int)
        mu_fact: sparsity fator (float)
        save_dictionary: path where to save dictionary matrix $W$ (str). Set this parameter to `None` to avoid the saving.

    References:
        [1] J. Le Roux et al. "Sparse NMF – half-baked or well done?" (2015)

    Théo Mariotte - 06/2023
    """
    if is_sparse:
        nmf_solver = SNMF(n_components=n_components,n_frames=X.shape[-1],n_feats=X.shape[0],mu_fact=mu_fact,beta=2)
    else:
        nmf_solver = NMF(n_components=n_components,n_frames=X.shape[-1],n_feats=X.shape[0],)
    meas_set=[]
    nmf_solver.to(device)
    for i in tqdm.tqdm(range(n_iter)):
        X = X.squeeze().to(device)
        W, H = nmf_solver(X)
        if save_measure:
            meas_set.append(torch.mean((X-W.matmul(H))**2).detach().cpu())
    
    if save_dictionary is not None:
        torch.save(W.detach().cpu(),save_dictionary)
    
    if save_measure: 
        return W, H, meas_set, nmf_solver
    else:
        return W,H, nmf_solver



