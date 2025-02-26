import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init
from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d
from mmcv.runner import load_checkpoint
from mmcv.cnn import ConvModule
from models.common import PixelShufflePack, flow_warp, make_layer, ResidualBlockNoBN
from models.dino import DINOFeatureExtractor
from utils_basicvsr.logger import get_root_logger
import random

class RBSR(nn.Module):
    """RBSR network structure.
    """
    def __init__(self,
                 mid_channels=64,
                 num_blocks=7,
                 max_residue_magnitude=10,
                 is_low_res_input=True,
                 spynet_pretrained=None,
                 cpu_cache_length=100,
                 align_type=None):

        super().__init__()

        self.align_type = align_type if align_type is not None else 'flow_guided_dcn_alignment'

        self.mid_channels = mid_channels

        if self.align_type == 'flow_guided_dcn_alignment':
            # optical flow
            self.spynet = SPyNet(pretrained='./pretrained_networks/spynet_20210409-c6c1bd09.pth')
            self.dcn_alignment = DeformableAlignment(mid_channels,mid_channels,3,padding=1,deform_groups=8,
                    max_residue_magnitude=max_residue_magnitude)
        elif self.align_type == 'flow_alignment':
            self.spynet = SPyNet(pretrained='./pretrained_networks/spynet_20210409-c6c1bd09.pth')
        elif self.align_type == 'dcn_alignment':
            self.dcn_alignment_without_flow = DeformableAlignmentWithoutFlow(mid_channels, mid_channels, 3, padding=1,
                    max_residue_magnitude=max_residue_magnitude)
        else:
            raise Exception

        # feature extraction module
        self.feat_extract = ResidualBlocksWithInputConv(4, mid_channels, 5)

        # propagation branches
        self.backbone = nn.ModuleDict()
        self.backbone['backward_1'] = ResidualBlocksWithInputConv(3*mid_channels, mid_channels, 40)
        # upsampling module
        self.reconstruction = ResidualBlocksWithInputConv(
            2* mid_channels, mid_channels, 5)
        self.upsample1 = PixelShufflePack(
            mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(
            mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample3 = PixelShufflePack(
            mid_channels, mid_channels, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.conv_last = nn.Conv2d(mid_channels, 3, 3, 1, 1)
        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.skipup1 = PixelShufflePack(4, mid_channels, 2, upsample_kernel=3)
        self.skipup2 = PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3)
        self.skipup3 = PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3)

    def compute_flow(self, lqs):
        lqs = torch.stack((lqs[:, :, 0], lqs[:, :, 1:3].mean(dim=2), lqs[:, :, 3]), dim=2)
        n, t, c, h, w = lqs.size()
        oth = lqs[:, 1:, :, :, :].reshape(-1, c, h, w)
        ref = lqs[:,:1, :, :, :].repeat(1,t-1,1,1,1).reshape(-1, c, h, w)
        flows_backward = self.spynet(ref, oth).view(n, t - 1, 2, h, w)
        return flows_backward
    
    def burst_propagate(self, feats, module_name, feat_base):   
        feat_prop = torch.zeros_like(feats['spatial'][0])
        for i in range(0, len(feats['spatial'])):
            feat_current = feats['spatial'][i]
            feat = [feat_base] + [feat_current] + [feat_prop]
            feat = torch.cat(feat, dim=1)
            feat_prop = feat_prop + self.backbone[module_name](feat)
            feats[module_name].append(feat_prop)
        return feats
    def upsample(self, lqs, feats, base_feat):
        outputs = []
        skip1 = self.skipup1(lqs[:, 0, :, :, :])
        skip2 = self.skipup2(skip1)
        skip3 = self.skipup3(skip2)
        i = -1
        hr = [feats[k][i] for k in feats if k != 'spatial']
        hr.insert(0, base_feat)    
        hr = torch.cat(hr, dim=1)
        hr = self.reconstruction(hr) 
        hr = self.lrelu(self.upsample1(hr)) + skip1
        hr = self.lrelu(self.upsample2(hr)) + skip2
        hr = self.lrelu(self.upsample3(hr)) + skip3
        hr = self.lrelu(self.conv_hr(hr))
        hr = self.conv_last(hr)
        outputs.append(hr)               

        i = random.randint(1, 12)
        hr = [feats[k][i] for k in feats if k != 'spatial']
        hr.insert(0, base_feat)     
        hr = torch.cat(hr, dim=1)
        hr = self.reconstruction(hr) 
        hr = self.lrelu(self.upsample1(hr)) + skip1
        hr = self.lrelu(self.upsample2(hr)) + skip2
        hr = self.lrelu(self.upsample3(hr)) + skip3
        hr = self.lrelu(self.conv_hr(hr))
        hr = self.conv_last(hr)
        outputs.append(hr)
        return torch.stack(outputs, dim=1)
    def forward(self, lqs):

        if self.align_type == 'flow_alignment':
            n, t, c, h, w = lqs.size() #(n, t, c, h,w)    
            feats = {}
            feats_ = self.feat_extract(lqs.view(-1, c, h, w))   # (*, C, H, W)
            feats_ = feats_.view(n, t, -1, h, w)
            
            oth_feat = feats_[:, 1:, :, :, :].contiguous().view(-1, *feats_.shape[-3:])

            flows_backward = self.compute_flow(lqs) # (n, t-1, 2, h, w)
            flows_backward = flows_backward.view(-1, 2, *feats_.shape[-2:]) # (n(t-1), 2, h, w)

            oth_feat_warped = flow_warp(oth_feat, flows_backward.permute(0, 2, 3, 1))   # (n(t-1), c, h, w)
            
            oth_feat = oth_feat_warped.view(n, t-1, -1, h, w)
            ref_feat = feats_[:, :1, :, :, :]
            feats_ = torch.cat((ref_feat, oth_feat), dim=1)

            feats['spatial'] = [feats_[:, i, :, :, :] for i in range(0, t, 1)]
            base_feat = feats_[:, 0, :, :, :]

        elif self.align_type == 'dcn_alignment':
            n, t, c, h, w = lqs.size() #(n, t, c, h,w)
            feats = {}
            feats_ = self.feat_extract(lqs.view(-1, c, h, w))   # (*, C, H, W)
            h, w = feats_.shape[2:]
            feats_ = feats_.view(n, t, -1, h, w)
            ref_feat = feats_[:, :1, :, :, :].repeat(1, t-1, 1, 1, 1).view(-1, *feats_.shape[-3:])
            oth_feat = feats_[:, 1:, :, :, :].contiguous().view(-1, *feats_.shape[-3:])

            oth_feat = self.dcn_alignment_without_flow(oth_feat, ref_feat)
            oth_feat = oth_feat.view(n, t-1, -1, h, w)
            ref_feat = ref_feat.view(n, t-1, -1, h, w)[:, :1, :, :, :]
            feats_ = torch.cat((ref_feat, oth_feat), dim=1)  
            
            feats['spatial'] = [feats_[:, i, :, :, :] for i in range(0, t, 1)]
            base_feat = feats_[:, 0, :, :, :]
            

        elif self.align_type == 'flow_guided_dcn_alignment':
            n, t, c, h, w = lqs.size() #(n, t, c, h,w)
            feats = {}
            feats_ = self.feat_extract(lqs.view(-1, c, h, w))   # (*, C, H, W)
            h, w = feats_.shape[2:]
            feats_ = feats_.view(n, t, -1, h, w)
            ref_feat = feats_[:, :1, :, :, :].repeat(1, t-1, 1, 1, 1).view(-1, *feats_.shape[-3:])
            oth_feat = feats_[:, 1:, :, :, :].contiguous().view(-1, *feats_.shape[-3:])

            flows_backward = self.compute_flow(lqs)     # (n, t-1, 2, h, w)
            flows_backward = flows_backward.view(-1, 2, *feats_.shape[-2:])     # (n(t-1), 2, h, w)

            oth_feat_warped = flow_warp(oth_feat, flows_backward.permute(0, 2, 3, 1))
            oth_feat = self.dcn_alignment(oth_feat, ref_feat, oth_feat_warped, flows_backward)
            oth_feat = oth_feat.view(n, t-1, -1, h, w)
            ref_feat = ref_feat.view(n, t-1, -1, h, w)[:, :1, :, :, :]
            feats_ = torch.cat((ref_feat, oth_feat), dim=1)  

            feats['spatial'] = [feats_[:, i, :, :, :] for i in range(0, t, 1)]
            base_feat = feats_[:, 0, :, :, :]

        # feature propagation
        module = 'backward_1'
        feats[module] = []
        feats = self.burst_propagate(feats, module, base_feat)
        return self.upsample(lqs, feats, base_feat), {}

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Default: None.
            strict (bool, optional): Whether strictly load the pretrained
                model. Default: True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')


class RBSR_DINO(nn.Module):
    """RBSR network structure.
    """
    def __init__(self,
                 mid_channels=64,
                 dino_channels=256,
                 ):

        super().__init__()

        self.mid_channels = mid_channels
        self.dino_channels = dino_channels

        # optical flow
        self.spynet = SPyNet(pretrained='./pretrained_networks/spynet_20210409-c6c1bd09.pth')

        # feature extraction module
        self.feat_extract = ResidualBlocksWithInputConv(4, mid_channels, 5)

        # propagation branches
        self.backbone = nn.ModuleDict()
        self.backbone['backward_1'] = ResidualBlocksWithInputConv(4 * mid_channels, mid_channels, 40)

        # DINO feature extractor
        self.dino_feat_extract = DINOFeatureExtractor()
        for param in self.dino_feat_extract.parameters():
            param.requires_grad = False
        self.conv_shallow = nn.Conv2d(dino_channels, mid_channels, 1, 1)
        self.conv_mid = nn.Conv2d(dino_channels * 2, mid_channels, 1, 1)
        self.conv_deep = nn.Conv2d(dino_channels * 4, mid_channels, 1, 1)
        self.conv_dino = nn.Conv2d(mid_channels * 3, mid_channels, 1, 1)

        # upsampling module
        self.reconstruction = ResidualBlocksWithInputConv(
            2* mid_channels, mid_channels, 5)
        self.upsample1 = PixelShufflePack(
            mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(
            mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample3 = PixelShufflePack(
            mid_channels, mid_channels, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.conv_last = nn.Conv2d(mid_channels, 3, 3, 1, 1)
        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.skipup1 = PixelShufflePack(4, mid_channels, 2, upsample_kernel=3)
        self.skipup2 = PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3)
        self.skipup3 = PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3)

    def compute_flow(self, lqs):
        lqs = torch.stack((lqs[:, :, 0], lqs[:, :, 1:3].mean(dim=2), lqs[:, :, 3]), dim=2)
        n, t, c, h, w = lqs.size()
        oth = lqs[:, 1:, :, :, :].reshape(-1, c, h, w)
        ref = lqs[:,:1, :, :, :].repeat(1,t-1,1,1,1).reshape(-1, c, h, w)
        flows_backward = self.spynet(ref, oth).view(n, t - 1, 2, h, w)
        return flows_backward
    
    def upsample(self, lq, feat, base_feat):
        skip1 = self.skipup1(lq)
        skip2 = self.skipup2(skip1)
        skip3 = self.skipup3(skip2)
        hr = torch.cat([base_feat, feat], dim=1)
        hr = self.reconstruction(hr) 
        hr = self.lrelu(self.upsample1(hr)) + skip1
        hr = self.lrelu(self.upsample2(hr)) + skip2
        hr = self.lrelu(self.upsample3(hr)) + skip3
        hr = self.lrelu(self.conv_hr(hr))
        hr = self.conv_last(hr)
        return hr

    def get_dino_feat(self, img):
        dino_feats = self.dino_feat_extract(img)
        feat_shallow = self.conv_shallow(F.interpolate(dino_feats[0], scale_factor=0.5, mode='nearest'))
        feat_mid = self.conv_mid(dino_feats[1])
        feat_deep = self.conv_deep(F.interpolate(dino_feats[2], scale_factor=2, mode='nearest'))
        feat_fused = torch.cat([feat_shallow, feat_mid, feat_deep], dim=1)
        feat_fused = self.lrelu(self.conv_dino(feat_fused))
        return feat_fused

    def forward(self, lqs, random_idx=None):
        n, t, c, h, w = lqs.size() #(n, t, c, h,w)
        feats = {}
        feats_ = self.feat_extract(lqs.view(-1, c, h, w))   # (*, C, H, W)
        h, w = feats_.shape[2:]
        feats_ = feats_.view(n, t, -1, h, w)
        ref_feat = feats_[:, :1, :, :, :].repeat(1, t-1, 1, 1, 1).view(-1, *feats_.shape[-3:])
        oth_feat = feats_[:, 1:, :, :, :].contiguous().view(-1, *feats_.shape[-3:])

        flows_backward = self.compute_flow(lqs)     # (n, t-1, 2, h, w)
        flows_backward = flows_backward.view(-1, 2, *feats_.shape[-2:])     # (n(t-1), 2, h, w)
        oth_feat_warped = flow_warp(oth_feat, flows_backward.permute(0, 2, 3, 1))

        oth_feat = oth_feat_warped.view(n, t-1, -1, h, w)
        ref_feat = ref_feat.view(n, t-1, -1, h, w)[:, :1, :, :, :]
        feats_ = torch.cat((ref_feat, oth_feat), dim=1)  

        feats['spatial'] = [feats_[:, i, :, :, :] for i in range(0, t, 1)]  # t * (n, c, h, w)
        base_feat = feats_[:, 0, :, :, :]   # (n, c, h, w)

        # feature propagation
        if random_idx is not None:
            res_list = []
            feat_prop = torch.zeros_like(feats['spatial'][0])
            feat_dino_prev = torch.zeros_like(feat_prop)
            for i in range(0, len(feats['spatial'])):
                feat_current = feats['spatial'][i]
                feat = [base_feat] + [feat_current] + [feat_prop] + [feat_dino_prev]
                feat = torch.cat(feat, dim=1)
                feat_prop = feat_prop + self.backbone['backward_1'](feat)
                res = self.upsample(lqs[:, 0, :, :, :], feat_prop, base_feat)
                if random_idx == i:
                    res_list.append(res)
                if i != len(feats['spatial']) - 1:
                    feat_dino_prev = self.get_dino_feat(res)
            res_list.insert(0, res)
            return torch.stack(res_list, dim=1)
        
        else:
            res_list = []
            feat_prop = torch.zeros_like(feats['spatial'][0])
            feat_dino_prev = torch.zeros_like(feat_prop)
            for i in range(0, len(feats['spatial'])):
                feat_current = feats['spatial'][i]
                feat = [base_feat] + [feat_current] + [feat_prop] + [feat_dino_prev]
                feat = torch.cat(feat, dim=1)
                feat_prop = feat_prop + self.backbone['backward_1'](feat)
                res = self.upsample(lqs[:, 0, :, :, :], feat_prop, base_feat)
                if i != len(feats['spatial']) - 1:
                    feat_dino_prev = self.get_dino_feat(res)
            res_list.append(res)
            return torch.stack(res_list, dim=1)


class ResidualBlocksWithInputConv(nn.Module):
    """Residual blocks with a convolution in front.

    Args:
        in_channels (int): Number of input channels of the first conv.
        out_channels (int): Number of channels of the residual blocks.
            Default: 64.
        num_blocks (int): Number of residual blocks. Default: 30.
    """

    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()

        main = []

        # a convolution used to match the channels of the residual blocks
        main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # residual blocks
        main.append(
            make_layer(
                ResidualBlockNoBN, num_blocks, mid_channels=out_channels))

        self.main = nn.Sequential(*main)

    def forward(self, feat):
        """Forward function for ResidualBlocksWithInputConv.

        Args:
            feat (Tensor): Input feature with shape (n, in_channels, h, w)

        Returns:
            Tensor: Output feature with shape (n, out_channels, h, w)
        """
        return self.main(feat)


class DeformableAlignment(ModulatedDeformConv2d):
    """flow guided deformable alignment module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.
    """
    # 128 64 3 1 1 1 1 [0.,0.,..,0.] 10 
    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)
        super(DeformableAlignment, self).__init__(*args, **kwargs)
        self.conv_offset = nn.Sequential(
            nn.Conv2d(2 * self.out_channels + 2, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        )
        self.init_offset()
    def init_offset(self):
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, cur_feat, ref_feat, warped_feat, flow):
        extra_feat = torch.cat([warped_feat, ref_feat, flow], dim=1)
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset = offset + flow.flip(1).repeat(1,offset.size(1) // 2, 1, 1)
        mask = torch.sigmoid(mask)
        return modulated_deform_conv2d(cur_feat, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)


class DeformableAlignmentWithoutFlow(ModulatedDeformConv2d):
    """deformable alignment module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.
    """
    # 128 64 3 1 1 1 1 [0.,0.,..,0.] 10 
    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)
        super(DeformableAlignmentWithoutFlow, self).__init__(*args, **kwargs)
        self.conv_offset = nn.Sequential(
            nn.Conv2d(2 * self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        )
        self.init_offset()
    def init_offset(self):
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, cur_feat, ref_feat):
        extra_feat = torch.cat([cur_feat, ref_feat], dim=1)
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        mask = torch.sigmoid(mask)
        return modulated_deform_conv2d(cur_feat, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)


class SPyNet(nn.Module):
    """SPyNet network structure.

    The difference to the SPyNet in [tof.py] is that
        1. more SPyNetBasicModule is used in this version, and
        2. no batch normalization is used in this version.

    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017

    Args:
        pretrained (str): path for pre-trained SPyNet. Default: None.
    """

    def __init__(self, pretrained):
        super().__init__()

        self.basic_module = nn.ModuleList(
            [SPyNetBasicModule() for _ in range(6)])

        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=True, logger=logger)
        elif pretrained is not None:
            raise TypeError('[pretrained] should be str or None, '
                            f'but got {type(pretrained)}.')

        self.register_buffer(
            'mean',
            torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            'std',
            torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def compute_flow(self, ref, supp):
        """Compute flow from ref to supp.

        Note that in this function, the images are already resized to a
        multiple of 32.

        Args:
            ref (Tensor): Reference image with shape of (n, 3, h, w).
            supp (Tensor): Supporting image with shape of (n, 3, h, w).

        Returns:
            Tensor: Estimated optical flow: (n, 2, h, w).
        """
        n, _, h, w = ref.size()

        # normalize the input images
        ref = [(ref - self.mean) / self.std]
        supp = [(supp - self.mean) / self.std]

        # generate downsampled frames
        for level in range(5):
            ref.append(
                F.avg_pool2d(
                    input=ref[-1],
                    kernel_size=2,
                    stride=2,
                    count_include_pad=False))
            supp.append(
                F.avg_pool2d(
                    input=supp[-1],
                    kernel_size=2,
                    stride=2,
                    count_include_pad=False))
        ref = ref[::-1]
        supp = supp[::-1]

        # flow computation
        flow = ref[0].new_zeros(n, 2, h // 32, w // 32)
        for level in range(len(ref)):
            if level == 0:
                flow_up = flow
            else:
                flow_up = F.interpolate(
                    input=flow,
                    scale_factor=2,
                    mode='bilinear',
                    align_corners=True) * 2.0

            # add the residue to the upsampled flow
            flow = flow_up + self.basic_module[level](
                torch.cat([
                    ref[level],
                    flow_warp(
                        supp[level],
                        flow_up.permute(0, 2, 3, 1),
                        padding_mode='border'), flow_up
                ], 1))

        return flow

    def forward(self, ref, supp):
        """Forward function of SPyNet.

        This function computes the optical flow from ref to supp.

        Args:
            ref (Tensor): Reference image with shape of (n, 3, h, w).
            supp (Tensor): Supporting image with shape of (n, 3, h, w).

        Returns:
            Tensor: Estimated optical flow: (n, 2, h, w).
        """

        # upsize to a multiple of 32
        h, w = ref.shape[2:4]
        w_up = w if (w % 32) == 0 else 32 * (w // 32 + 1)
        h_up = h if (h % 32) == 0 else 32 * (h // 32 + 1)
        ref = F.interpolate(
            input=ref, size=(h_up, w_up), mode='bilinear', align_corners=False)
        supp = F.interpolate(
            input=supp,
            size=(h_up, w_up),
            mode='bilinear',
            align_corners=False)

        # compute flow, and resize back to the original resolution
        flow = F.interpolate(
            input=self.compute_flow(ref, supp),
            size=(h, w),
            mode='bilinear',
            align_corners=False)

        # adjust the flow values
        flow[:, 0, :, :] *= float(w) / float(w_up)
        flow[:, 1, :, :] *= float(h) / float(h_up)

        return flow

class SPyNetBasicModule(nn.Module):
    """Basic Module for SPyNet.

    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017
    """

    def __init__(self):
        super().__init__()

        self.basic_module = nn.Sequential(
            ConvModule(
                in_channels=8,
                out_channels=32,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=32,
                out_channels=64,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=64,
                out_channels=32,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=32,
                out_channels=16,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=16,
                out_channels=2,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=None))

    def forward(self, tensor_input):
        """
        Args:
            tensor_input (Tensor): Input tensor with shape (b, 8, h, w).
                8 channels contain:
                [reference image (3), neighbor image (3), initial flow (2)].

        Returns:
            Tensor: Refined flow with shape (b, 2, h, w)
        """
        return self.basic_module(tensor_input)
