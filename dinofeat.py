import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob, os
from sklearn.decomposition import PCA
from PIL import Image

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms


# DINOv2
class DINOv2(nn.Module):
    def __init__(self):
        super(DINOv2, self).__init__()
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, img, name):
        with torch.no_grad():
            _, _, h, w = img.size()
            # img = F.interpolate(img, [(h+13) // 14 * 14, (w+13) // 14 * 14])
            img = F.interpolate(img, [392, 392])
            _, _, h, w = img.size()
            for i in range(img.shape[0]):
                img_i = img[i]
                output_path = './image/' + name
                plt.imsave(output_path, img_i.permute(1, 2, 0).detach().cpu().numpy())
            img = (img - self.mean) / self.std
            
            # # DINO final feature
            # x = self.backbone.forward_features(img)["x_norm_patchtokens"]
            # x = x.permute(0, 2, 1)
            # x = x.reshape(x.shape[0], x.shape[1], h // 14, w // 14)
            # x = F.normalize(x, dim=-3, p=2)
            # DINO2PCA(x, name)

            # DINO shallow / middle / deep feature
            layers = [1, 6, 11]
            features = []
            y = img
            y = self.backbone.prepare_tokens_with_masks(y, None)
            for idx, block in enumerate(self.backbone.blocks):
                y = block(y)
                if idx in layers:
                    features.append(y)
            for i, feature in enumerate(features):
                feature = feature[:, self.backbone.num_register_tokens + 1:]
                print(f'{i}: {feature.shape}')
                x = feature.permute(0, 2, 1)
                x = x.reshape(x.shape[0], x.shape[1], h // 14, w // 14)
                x = F.normalize(x, dim=-3, p=2)
                name_ = name.replace('.png', '_' + str(layers[i]) + '.png')
                DINO2PCA(x, name_)

            f = F.interpolate(x, [32, 32])
            return f


def DINO2PCA(features, name):
    B, C, H, W = features.shape
    features = features.detach().cpu().numpy().reshape(B, C, H * W)

    for i in range(B):
        pca = PCA(n_components=3)  # Reduce to 3 components for visualization
        pca_result = pca.fit_transform(features[i].T)  # Apply PCA to each batch sample
        pca_result = pca_result.T.reshape(3, H, W)  # Reshape back to image format

        # Convert PCA result to a visualizable image
        pca_image = np.transpose(pca_result, (1, 2, 0))  # Change to [16, 16, 3] for visualization
        pca_image = (pca_image - pca_image.min()) / (pca_image.max() - pca_image.min())  # Normalize to [0, 1]
        pca_image = np.clip(pca_image * 255, 0, 255).astype(np.uint8)  # Convert to [0, 255]

        # Upsample the image to 256x256
        pca_image_tensor = torch.tensor(pca_image).permute(2, 0, 1).unsqueeze(
            0).float()  # Convert to tensor [1, 3, 16, 16]
        pca_image_upsampled = F.interpolate(pca_image_tensor, size=(H*14, W*14), mode='bilinear',
                                            align_corners=False)
        pca_image_upsampled = pca_image_upsampled.squeeze(0).permute(1, 2,
                                                                     0).byte().numpy()  # Convert back to [256, 256, 3]

        # Save the PCA result as an image file
        output_path = './feature/' + name  # Path to save the image for each batch
        plt.imsave(output_path, pca_image_upsampled)
        print(f"PCA image saved to {output_path}")


if __name__ == '__main__':

    model = DINOv2().cuda()

    # gt srgb
    gt_srgb_dir = '/data/gjh8760/datasets/SyntheticBurstVal_srgb/gt'
    subfolders = sorted(os.listdir(gt_srgb_dir))
    for subfolder in subfolders:
        im_path = os.path.join(gt_srgb_dir, subfolder, 'im_rgb.png')
        im_gt = np.array(Image.open(im_path).convert('RGB'))
        im_gt_t = torch.from_numpy(im_gt.astype(np.float32) / 255.).permute(2, 0, 1).unsqueeze(0).float().cuda()
        
        name = 'gt_srgb/{}/im_rgb.png'.format(subfolder)
        if not os.path.exists(os.path.join('image', os.path.dirname(name))):
            os.makedirs(os.path.join('image', os.path.dirname(name)))
        if not os.path.exists(os.path.join('feature', os.path.dirname(name))):
            os.makedirs(os.path.join('feature', os.path.dirname(name)))
        res = model(im_gt_t, name)

    # # gt linear
    # gt_linear_dir = '/data/gjh8760/datasets/SyntheticBurstVal/gt'
    # subfolders = sorted(os.listdir(gt_linear_dir))
    # for subfolder in subfolders:
    #     im_path = os.path.join(gt_linear_dir, subfolder, 'im_rgb.png')
    #     im_gt = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
    #     im_gt_t = torch.from_numpy(im_gt.astype(np.float32) / 2 ** 14).permute(2, 0, 1).unsqueeze(0).float().cuda()
        
    #     name = 'gt_linear/{}/im_rgb.png'.format(subfolder)
    #     if not os.path.exists(os.path.join('image', os.path.dirname(name))):
    #         os.makedirs(os.path.join('image', os.path.dirname(name)))
    #     if not os.path.exists(os.path.join('feature', os.path.dirname(name))):
    #         os.makedirs(os.path.join('feature', os.path.dirname(name)))
    #     res = model(im_gt_t, name)

    # burst srgb
    burst_srgb_dir = '/data/gjh8760/datasets/SyntheticBurstVal_srgb/bursts'
    subfolders = sorted(os.listdir(burst_srgb_dir))
    for subfolder in subfolders:
        im_paths = sorted(glob.glob(os.path.join(burst_srgb_dir, subfolder, '*.png')))
        for im_path in im_paths:
            im_burst = np.array(Image.open(im_path).convert('RGB'))
            im_burst_t = torch.from_numpy(im_burst.astype(np.float32) / 255.).permute(2, 0, 1).unsqueeze(0).float().cuda()

            name = 'burst_srgb/{}/{}'.format(subfolder, os.path.basename(im_path))
            if not os.path.exists(os.path.join('image', os.path.dirname(name))):
                os.makedirs(os.path.join('image', os.path.dirname(name)))
            if not os.path.exists(os.path.join('feature', os.path.dirname(name))):
                os.makedirs(os.path.join('feature', os.path.dirname(name)))
            res = model(im_burst_t, name)
    





# layers = [2, 7, 11]
# features = []
# y = img
# y = self.backbone.prepare_tokens_with_masks(y, None)
# for idx, block in enumerate(self.backbone.blocks):
#     y = block(y)
#     if idx in layers:
#         features.append(y)
# for i, feature in enumerate(features):
#     feature = feature[:, self.backbone.num_register_tokens + 1:]
#     print(f"{i}: {feature.shape}")