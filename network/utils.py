import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    K =1

def init_weights(m):
    if type(m) == nn.Conv3d or type(m) == nn.ConvTranspose3d:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.normal_(m.weight, std=0.001)
        #nn.init.normal_(m.bias, std=0.001)
        truncated_normal_(m.bias, mean=0, std=0.001)
        k=1

def init_weights_orthogonal_normal(m):
    if type(m) == nn.Conv3d or type(m) == nn.ConvTranspose3d:
        nn.init.orthogonal_(m.weight)
        truncated_normal_(m.bias, mean=0, std=0.001)
        #nn.init.normal_(m.bias, std=0.001)

def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg

def save_mask_prediction_example(mask, pred, iter):
	plt.imshow(pred[0,:,:],cmap='Greys')
	plt.savefig('images/'+str(iter)+"_prediction.png")
	plt.imshow(mask[0,:,:],cmap='Greys')
	plt.savefig('images/'+str(iter)+"_mask.png")

def get_soft_label(input_tensor, num_class):
    """
        convert a label tensor to soft label
        input_tensor: tensor with shape [N, C, H, W]
        output_tensor: shape [N, H, W, num_class]
    """
    tensor_list = []
    if input_tensor.ndim == 5:
        input_tensor = input_tensor.permute(0, 2, 3, 4, 1)
    else:
        input_tensor = input_tensor.permute(0, 2, 3, 1)
    # input_tensor = input_tensor.permute(0, 2, 3, 1)
    for i in range(num_class):
        temp_prob = torch.eq(input_tensor, i * torch.ones_like(input_tensor))
        tensor_list.append(temp_prob)
    output_tensor = torch.cat(tensor_list, dim=-1)
    output_tensor = output_tensor.float()
    return output_tensor