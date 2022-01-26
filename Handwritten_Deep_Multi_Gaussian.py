"""
Created on Wednesday 2020/08/12

@author Luzhe Sun

Python3.7
pytorch
"""
import torch
import torch.nn as nn


class GP:
    # Initialize the class
    def __init__(self, X, y, feature_dim):
        self.D = X.shape[1]
        self.X = X
        self.y = y

        self.hyp = self.init_params()
        self.jitter = 1e-8
        self.likelihood(self.hyp)

        '''Here is what I add'''
        self.feature_dim = feature_dim
        hidden_dim = 10 * feature_dim
        self.Linear1 = nn.Linear(self.D, hidden_dim)
        self.Linear2 = nn.Linear(hidden_dim, feature_dim)
        self.ReLU = nn.ReLU()

        print("Total number of parameters:%d" % (self.hyp.shape[0]))

    '''
    Here is what I add to implement nonlinear relationship extract
    '''

    # Deep Net to extract nonlinear relationship
    def feature_get(self, X):
        hidden_layer = self.Linear1(X)
        ReLU_layer = self.ReLU(hidden_layer)
        output_layer = self.Linear2(ReLU_layer)
        return output_layer

    # Initialize hyper
    def init_params(self):
        hyp = torch.log(torch.ones(self.feature_dim + 1))
        self.idx_theta = torch.arange(hyp.shape[0])
        logsigma_n = torch.tensor([-4.0])
        hyp = torch.cat([hyp, logsigma_n])
        return hyp

    # A simple vectorized rbf kernel
    def kernel(self, x, xp, hyp):
        output_scale = torch.exp(hyp[0])
        lengthscales = torch.exp(hyp[1:])
        diffs = torch.expand_dims(x / lengthscales, 1) - \
                torch.expand_dims(xp / lengthscales, 0)
        return output_scale * torch.exp(-0.5 * torch.sum(diffs ** 2, axis=2))

    # Computes the negative log-marginal likelihood
    def likelihood(self, hyp):
        X = self.feature_get(self.X)
        y = self.y

        N = y.shape[0]

        logsigma_n = hyp[-1]
        sigma_n = torch.exp(logsigma_n)

        theta = hyp[self.idx_thata]

        K = self.kernel(X, X, theta) + torch.eye(N) * sigma_n
        '''
        Cholesky 分解是把一个对称正定的矩阵表示成一个下三角矩阵L
        和其转置的乘积的分解。它要求矩阵的所有特征值必须大于零，
        故分解的下三角的对角元也是大于零的。Cholesky分解法又称平方根法，
        是当A为实对称正定矩阵时，LU三角分解法的变形。'''
        L = torch.cholesky(K + torch.eye(N) * self.jitter)
        self.L = L

        alpha = torch.solve(torch.transpose(L), torch.solve(L, y))
        NLML = 0.5 * torch.matmul(torch.transpose(y), alpha) + \
               torch.sum(torch.log(torch.diag(L))) + 0.5 * torch.log(2. * torch.pi) * N
        return NLML[0, 0]

    #      # Minimizes the negative log-marginal likelihood
    #     def train(self):
    #         result = minimize(value_and_grad(self.likelihood), self.hyp, jac=True,
    #                           method='L-BFGS-B', callback=self.callback)
    #         self.hyp = result.x

    def predict(self, X_star_raw):
        X = self.feature_get(self.X)
        y = self.y
        X_star = self.feature_get(X_star_raw)

        L = self.L

        theta = self.hyp[self.idx_theta]

        psi = self.kernel(X_star, X, theta)

        alpha = torch.solve(torch.transpose(L), torch.solve(L, y))
        pred_u_star = torch.matmul(psi, alpha)

        beta = torch.solve(torch.transpose(L), torch.solve(L, psi.T))
        var_u_star = self.kernel(X_star, X_star, theta) - torch.matmul(psi, beta)

        return pred_u_star, var_u_star


# A minimal GP multi-fidelity class (two levels of fidelity)
class Multifidelity_GP:
    # Initialize the class
    def __init__(self, X_L, y_L, X_H, y_H, feature_dim):
        self.D = X_H.shape[1]
        self.X_L = X_L
        self.y_L = y_L
        self.X_H = X_H
        self.y_H = y_H

        '''Here is what I add'''
        self.feature_dim = feature_dim
        hidden_dim = 10 * feature_dim
        self.Linear1 = nn.Linear(self.D, hidden_dim)
        self.Linear2 = nn.Linear(hidden_dim, feature_dim)
        self.ReLU = nn.ReLU()

        self.hyp = self.init_params()
        print("Total number of parameters: %d" % (self.hyp.shape[0]))

        self.jitter = 1e-8

    '''
    Here is what I add to implement nonlinear relationship extract
    '''

    # Deep Net to extract nonlinear relationship
    def feature_get(self, X):
        hidden_layer = self.Linear1(X)
        ReLU_layer = self.ReLU(hidden_layer)
        output_layer = self.Linear2(ReLU_layer)
        return output_layer

    # Initialize hyper-parameters
    def init_params(self):
        hyp = torch.log(torch.ones(self.feature_dim + 1))
        self.idx_theta_L = torch.arange(hyp.shape[0])

        hyp = torch.cat([hyp, torch.log(torch.ones(self.feature_dim + 1))])
        self.idx_theta_H = torch.arange(self.idx_theta_L[-1] + 1, hyp.shape[0])

        rho = torch.tensor([1.0])
        logsigma_n = torch.tensor([-4.0, -4.0])
        hyp = torch.cat([hyp, rho, logsigma_n])
        return hyp

    # A simple vectorized rbf kernel
    def kernel(self, x, xp, hyp):
        output_scale = torch.exp(hyp[0])
        lengthscales = torch.exp(hyp[1:])
        diffs = torch.expand_dims(x / lengthscales, 1) - \
                torch.expand_dims(xp / lengthscales, 0)
        return output_scale * torch.exp(-0.5 * torch.sum(diffs ** 2, axis=2))

    # Computes the negative log-marginal likelihood
    def likelihood(self, hyp):
        X_L = self.feature_get(self.X_L)
        y_L = self.y_L
        X_H = self.feature_get(self.X_H)
        y_H = self.y_H

        y = torch.vstack((y_L, y_H))

        NL = y_L.shape[0]
        NH = y_H.shape[0]
        N = y.shape[0]

        rho = hyp[-3]
        logsigma_n_L = hyp[-2]
        logsigma_n_H = hyp[-1]
        sigma_n_L = torch.exp(logsigma_n_L)
        sigma_n_H = torch.exp(logsigma_n_H)

        theta_L = hyp[self.idx_theta_L]
        theta_H = hyp[self.idx_theta_H]

        K_LL = self.kernel(X_L, X_L, theta_L) + torch.eye(NL) * sigma_n_L
        K_LH = rho * self.kernel(X_L, X_H, theta_L)
        K_HH = rho ** 2 * self.kernel(X_H, X_H, theta_L) + \
               self.kernel(X_H, X_H, theta_H) + torch.eye(NH) * sigma_n_H
        K = torch.vstack((torch.hstack((K_LL, K_LH)),
                          torch.hstack((K_LH.T, K_HH))))
        L = torch.cholesky(K + torch.eye(N) * self.jitter)
        self.L = L

        alpha = torch.solve(torch.transpose(L), torch.solve(L, y))
        NLML = 0.5 * torch.matmul(torch.transpose(y), alpha) + \
               torch.sum(torch.log(torch.diag(L))) + 0.5 * torch.log(2. * torch.pi) * N
        return NLML[0, 0]

    #     # Minimizes the negative log-marginal likelihood
    #     def train(self):
    #         result = minimize(value_and_grad(self.likelihood), self.hyp, jac=True,
    #                           method='L-BFGS-B', callback=self.callback)
    #         self.hyp = result.x

    # Return posterior mean and variance at a set of test points
    def predict(self, X_star_raw):
        X_L = self.feature_get(self.X_L)
        y_L = self.y_L
        X_H = self.feature_get(self.X_H)
        y_H = self.y_H
        L = self.L
        X_star=self.feature_get(self.X_star_raw)

        y = torch.vstack((y_L, y_H))

        rho = self.hyp[-3]
        theta_L = self.hyp[self.idx_theta_L]
        theta_H = self.hyp[self.idx_theta_H]

        psi1 = rho * self.kernel(X_star, X_L, theta_L)
        psi2 = rho ** 2 * self.kernel(X_star, X_H, theta_L) + \
               self.kernel(X_star, X_H, theta_H)
        psi = torch.hstack((psi1, psi2))

        alpha = torch.solve(torch.transpose(L), torch.solve(L, y))
        pred_u_star = torch.matmul(psi, alpha)

        beta = torch.solve(torch.transpose(L), torch.solve(L, psi.T))
        var_u_star = rho ** 2 * self.kernel(X_star, X_star, theta_L) + \
                     self.kernel(X_star, X_star, theta_H) - torch.matmul(psi, beta)

        return pred_u_star, var_u_star

    #  Prints the negative log-marginal likelihood at each training step
    def callback(self, params):
        print("Log likelihood {}".format(self.likelihood(params)))