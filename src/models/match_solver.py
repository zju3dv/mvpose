"""
This file is pytorch implementation of :
    Wang, Qianqian, Xiaowei Zhou, and Kostas Daniilidis. "Multi-Image Semantic Matching by Mining Consistent Features." arXiv preprint arXiv:1711.07641 (2017).
"""
import torch


def myproj2dpam(Y, tol=1e-4):
    X0 = Y
    X = Y
    I2 = 0

    for iter_ in range ( 10 ):

        X1 = projR ( X0 + I2 )
        I1 = X1 - (X0 + I2)
        X2 = projC ( X0 + I1 )
        I2 = X2 - (X0 + I1)

        chg = torch.sum ( torch.abs ( X2[:] - X[:] ) ) / X.numel ()
        X = X2
        if chg < tol:
            return X
    return X


def projR(X):
    for i in range ( X.shape[0] ):
        X[i, :] = proj2pav ( X[i, :] )
        # X[i, :] = proj2pavC ( X[i, :] )
    return X


def projC(X):
    for j in range ( X.shape[1] ):
        # X[:, j] = proj2pavC ( X[:, j] )
        # Change to tradition implementation
        X[:, j] = proj2pav ( X[:, j] )
    return X


def proj2pav(y):
    y[y < 0] = 0
    x = torch.zeros_like ( y )
    if torch.sum ( y ) < 1:
        x += y
    else:
        u, _ = torch.sort ( y, descending=True )
        sv = torch.cumsum ( u, 0 )
        to_find = u > (sv - 1) / (torch.arange ( 1, len ( u ) + 1, device=u.device, dtype=u.dtype ))
        rho = torch.nonzero ( to_find.reshape ( -1 ) )[-1]
        theta = torch.max ( torch.tensor ( 0, device=sv.device, dtype=sv.dtype ), (sv[rho] - 1) / (rho.float () + 1) )
        x += torch.max ( y - theta, torch.tensor ( 0, device=sv.device, dtype=y.dtype ) )
    return x


def proj2pavC(y):
    # % project an n-dim vector y to the simplex Dn
    # % Dn = { x : x n-dim, 1 >= x >= 0, sum(x) = 1}
    #
    # % (c) Xiaojing Ye
    # % xyex19@gmail.com
    # %
    # % Algorithm is explained as in the linked document
    # % http://arxiv.org/abs/1101.6081
    # % or
    # % http://ufdc.ufl.edu/IR00000353/
    # %
    # % Jan. 14, 2011.

    m = len ( y )
    bget = False

    s, _ = torch.sort ( y, descending=True )
    tmpsum = 0

    for ii in range ( m - 1 ):
        tmpsum = tmpsum + s[ii]
        # tmax = (tmpsum - 1) / ii
        tmax = (tmpsum - 1) / (ii + 1)  # change since index starts from 0
        if tmax >= s[ii + 1]:
            bget = True
            break

    if not bget:
        tmax = (tmpsum + s[m - 1] - 1) / m

    x = torch.max ( y - tmax, torch.zeros_like ( y ) )
    return x



