import torch
import torch.nn as nn
from torch.nn import functional as F

import cv2

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


# DINO
class DINO(nn.Module):
    def __init__(self):
        super(DINO, self).__init__()
        self.backbone = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
        
        # # original (ImageNet RGB)
        # self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        # self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # modified (ImageNet BGR)
        self.register_buffer('mean', torch.FloatTensor([0.406, 0.456, 0.485]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.225, 0.224, 0.229]).view(1, 3, 1, 1))
    
    def forward(self, img):
        with torch.no_grad():
            img = (img - self.mean) / self.std
            return self.backbone(img)


class DINOFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.outputs = {}
        self.model = DINO()

    def hook_fn(self, module, input, output):
        self.outputs[module] = output
    
    def get_module(self):
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model
    
    def forward(self, img):
        with torch.no_grad():
            backbone = self.get_module().backbone
            hook_handle_shallow = backbone.layer1.register_forward_hook(self.hook_fn)
            hook_handle_mid = backbone.layer2.register_forward_hook(self.hook_fn)
            hook_handle_deep = backbone.layer3.register_forward_hook(self.hook_fn)
            _ = self.model(img)
            output_list = list(self.outputs.values())

            # # PCAs
            # shallow_feat = DINO2PCA(F.normalize(output_list[0], dim=-3, p=2))
            # mid1_feat = DINO2PCA(F.normalize(output_list[1], dim=-3, p=2))
            # mid2_feat = DINO2PCA(F.normalize(output_list[2], dim=-3, p=2))
            # deep_feat = DINO2PCA(F.normalize(output_list[3], dim=-3, p=2))
            # cv2.imwrite('feat_shallow.png', shallow_feat)
            # cv2.imwrite('feat_mid1.png', mid1_feat)
            # cv2.imwrite('feat_mid2.png', mid2_feat)
            # cv2.imwrite('feat_deep.png', deep_feat)
            # print()

            hook_handle_shallow.remove()
            hook_handle_mid.remove()
            hook_handle_deep.remove()
            return output_list


# def DINO2PCA(features):
#     B, C, H, W = features.shape
#     features = features.detach().cpu().numpy().reshape(B, C, H*W)

#     for i in range(B):
#         pca = PCA(n_components=3)
#         pca_result = pca.fit_transform(features[i].T)
#         pca_result = pca_result.T.reshape(3, H, W)

#         pca_image = np.transpose(pca_result, (1, 2, 0))
#         pca_image = (pca_image - pca_image.min()) / (pca_image.max() - pca_image.min())
#         pca_image = np.clip(pca_image * 255, 0, 255).astype(np.uint8)

#         pca_image_tensor = torch.tensor(pca_image).permute(2, 0, 1).unsqueeze(
#             0).float()
#         pca_image_upsampled = F.interpolate(pca_image_tensor, size=(384, 384), mode='nearest')
#         pca_image_upsampled = pca_image_upsampled.squeeze(0).permute(1, 2,
#                                                                      0).byte().numpy()
        
#     return pca_image_upsampled




# if __name__ == '__main__':

#     # burst
#     img = cv2.imread('im_raw_00.png', cv2.IMREAD_UNCHANGED)
#     img_t = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1).float() / (2**14)
#     img_t = img_t.unsqueeze(0).to('cuda:0') # (1, 4, 48, 48)

#     x = torch.cat([img_t[:, 0:1], torch.mean(img_t[:, 1:3], dim=1, keepdim=True), img_t[:, 3:4]], dim=1)
#     img_t = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=False)


    
#     # x = torch.empty((1, 1, 96, 96), dtype=torch.float32)
#     # x[:, :, 0::2, 0::2] = img_t[:, 0, :, :]
#     # x[:, :, 0::2, 1::2] = img_t[:, 1, :, :]
#     # x[:, :, 1::2, 0::2] = img_t[:, 2, :, :]
#     # x[:, :, 1::2, 1::2] = img_t[:, 3, :, :]
#     # x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
#     # img_t = torch.cat([x]*3, dim=1).to('cuda:0')


#     print()

#     # # GT srgb
#     # img = cv2.imread('im_rgb.png')
#     # img_t = torch.from_numpy(img) / 255.
#     # img_t = img_t.permute(2, 0, 1).unsqueeze(0).to('cuda:0')

#     model = DINOFeature().to('cuda:0')

#     feat = model(img_t)

#     print()


