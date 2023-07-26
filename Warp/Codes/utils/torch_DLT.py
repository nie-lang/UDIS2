import torch
import numpy as np
import cv2

# src_p: shape=(bs, 4, 2)
# det_p: shape=(bs, 4, 2)
#
#                                     | h1 |
#                                     | h2 |                   
#                                     | h3 |
# | x1 y1 1  0  0  0  -x1x2  -y1x2 |  | h4 |  =  | x2 |
# | 0  0  0  x1 y1 1  -x1y2  -y1y2 |  | h5 |     | y2 |
#                                     | h6 |
#                                     | h7 |
#                                     | h8 |

def tensor_DLT(src_p, dst_p):
   
    bs, _, _ = src_p.shape

    ones = torch.ones(bs, 4, 1)
    if torch.cuda.is_available():
        ones = ones.cuda()
    xy1 = torch.cat((src_p, ones), 2)
    zeros = torch.zeros_like(xy1)
    if torch.cuda.is_available():
        zeros = zeros.cuda()

    xyu, xyd = torch.cat((xy1, zeros), 2), torch.cat((zeros, xy1), 2)
    M1 = torch.cat((xyu, xyd), 2).reshape(bs, -1, 6)
    M2 = torch.matmul(
        dst_p.reshape(-1, 2, 1), 
        src_p.reshape(-1, 1, 2),
    ).reshape(bs, -1, 2)
    
    # Ah = b
    A = torch.cat((M1, -M2), 2)
    b = dst_p.reshape(bs, -1, 1)
    
    #h = A^{-1}b
    Ainv = torch.inverse(A)
    h8 = torch.matmul(Ainv, b).reshape(bs, 8)
 
    H = torch.cat((h8, ones[:,0,:]), 1).reshape(bs, 3, 3)
    return H