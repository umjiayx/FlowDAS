import torch

class Interpolant:

    def __init__(self, config):
        self.config = config

    def wide(self, t):
        return t[:, None, None, None]

    def alpha(self, t):
        return self.wide(1-t) 

    def alpha_dot(self, t):
        return self.wide(-1.0 * torch.ones_like(t))

    def beta(self, t):
        is_squared = self.config.beta_fn == 't^2'
        return self.wide(t.pow(2) if is_squared else t)

    def beta_dot(self, t):
        is_squared = self.config.beta_fn == 't^2'
        return self.wide(2.0 * t if is_squared else torch.ones_like(t))

    # we sometimes multiply sigma + sigma_dot by avg pixel norm, 
    # but when standardized (centered cifar), 
    # or when norm 1 (we rescale nse), not needed
    def sigma(self, t):
        return self.config.sigma_coef * self.wide(1-t) 

    def sigma_dot(self, t):
        return self.config.sigma_coef * self.wide(-torch.ones_like(t)) 
    
    def gamma(self, t):
        return self.wide(t.sqrt()) * self.sigma(t)

    def compute_zt(self, D):
        return D['at'] * D['z0'] + D['bt'] * D['z1'] + D['gamma_t'] * D['noise']

    def compute_zt_new(self, D): ## here use new because in the Dz0 has lookback frames rather than just z0 
        # print('z0', D['z0'][:,-1].shape)
        # print('z1', D['z1'].shape)
        # print(D['at'].squeeze(1).shape)
        # print((D['at'].squeeze(1) * D['z0'][:,-1]).shape)
        # print((D['bt'].squeeze(1) * D['z1']).shape)
        z0 = D['z0'][:,-1].unsqueeze(1)
        # print('zt_z0',z0.shape)
        if len(D['z1'].shape)==3:
            z1 = D['z1'].unsqueeze(1)
        else:
            z1 = D['z1']
        # print('zt_z1',z1.shape)
        # print('last term', (D['gamma_t'] * D['noise']).shape)
        return D['at'] * z0 + D['bt'] * z1 + D['gamma_t'] * D['noise']

    def compute_target(self, D):
        return D['adot'] * D['z0'] + D['bdot'] * D['z1'] +  (D['sdot'] * D['root_t']) * D['noise']

    def compute_target_new(self, D): ## here use new because in the Dz0 has lookback frames rather than just z0 
        z0 = D['z0'][:,-1].unsqueeze(1)
        # print('target_z0',z0.shape)
        z1 = D['z1']
        # print('target_z1',z1.shape)
        return D['adot'] * z0 + D['bdot'] * z1 +  (D['sdot'] * D['root_t']) * D['noise']
    
    def interpolant_coefs(self, D):
        return self(D)

    def __call__(self, D):
        D['at'] = self.alpha(D['t'])
        D['bt'] = self.beta(D['t'])
        D['adot'] = self.alpha_dot(D['t'])
        D['bdot'] = self.beta_dot(D['t'])
        D['root_t'] = self.wide(D['t'].sqrt())
        D['gamma_t'] = self.gamma(D['t'])
        D['st'] = self.sigma(D['t'])
        D['sdot'] = self.sigma_dot(D['t'])
        return D

