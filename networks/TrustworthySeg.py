import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from utils.criterions import KL,ce_loss,mse_loss,dce_evidence_loss,dce_evidence_u_loss
from networks.nnUnet import BasicUNet

# from models.lib.TransU_zoo import Transformer_U
# from models.lib.vit_seg_modeling import ViT
# from sklearn.preprocessing import MinMaxScaler

class TMSU(nn.Module):

    def __init__(self, args):
        """
        :param classes: Number of classification categories
        :param modes: Number of modes
        :param classifier_dims: Dimension of the classifier
        :param annealing_epoch: KL divergence annealing epoch during training
        """
        super(TMSU, self).__init__()
        # ---- Net Backbone ----
        num_classes = args.out_channels
        # modes = args.modes
        total_epochs = args.max_epochs #
        lambda_epochs = (args.max_epochs)/2 #
        self.backbone = BasicUNet(
            spatial_dims=args.spatial_dims,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            features=args.features
            )
        
        self.backbone.cuda()
        self.classes = num_classes
        self.disentangle = False
        self.eps = 1e-10
        self.lambda_epochs = lambda_epochs
        self.total_epochs = total_epochs 

    def forward(self, X, y, global_step, mode):
        # X data
        # y target
        # global_step : epochs

        # step zero: backbone

        backbone_output = self.backbone(X)

        # step one
        evidence = self.infer(backbone_output) # batch_size * class * image_size
        backbone_pred = F.softmax(backbone_output,1)  # batch_size * class * image_size

        # step two
        alpha = evidence + 1
        if mode == 'train':
            loss = dce_evidence_u_loss(y.to(torch.int64), alpha, self.classes, global_step, self.lambda_epochs,self.total_epochs,self.eps,self.disentangle,evidence,backbone_pred)
            loss = torch.mean(loss)
            return evidence, loss
        else:
            return evidence

    def infer(self, input):
        """
        :param input: modal data
        :return: evidence of modal data
        """
        evidence = F.softplus(input)
        return evidence
