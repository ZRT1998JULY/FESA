import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.deep_lab import DeepLabv3_plus
from collections import OrderedDict
from networks.tools import save_image
from networks import tools

# The Code of baseline network is referenced from https://github.com/icoz69/CaNet
# The code of training & testing is referenced from https://github.com/xiaomengyc/SG-One

class network(nn.Module):
    """
    Fewshot Segmentation model

    Args:
        in_channels:
            number of input channels
        pretrained_path:
            path of the model for initialization
        cfg:
            model configurations
    """
    def __init__(self, num_heads=15): # + background  part:29  ori:15
        super().__init__()
        self.num_heads = num_heads

        # Encoder os=8
        self.encoder = nn.Sequential(OrderedDict([('backbone', DeepLabv3_plus(nInputChannels=3, n_classes=num_heads, os=16, pretrained=True, _print=True))]))
        #self.encoder = nn.Sequential(OrderedDict([('backbone', Encoder(pretrained_path='./pretrained_model/vgg16-397923af.pth'))]))

        self.GMP = nn.AdaptiveMaxPool2d((1,1))
        self.GAP = nn.AdaptiveAvgPool2d((1,1))

        self.save_pred_list = []
        self.save_bipred_list = []
        self.save_mask_list = []
        self.save_part_mask_list = []

        self.loss_weights = torch.tensor([1.72, 46.1, 47.2, 32.4, 40.9, 49.2, 49.8, 48.7, 47.5, 50.4, 11.8, 49.8,
                                         16.9, 45.1, 32.9, 38.6, 45.3, 50.0, 22.2, 50.1, 39.9, 36.3, 44.0, 29.9,
                                         32.4, 47.8, 45.8, 41.6, 37.9]).type(torch.cuda.FloatTensor)

        # self.loss_weights = torch.tensor([1.80, 33.0, 37.4, 38.4, 18.9, 17.7, 34.5, 9.9, 37.2, 34.0, 23.5, 31.2,
        #                                  34.2, 30.0, 30.2]).type(torch.cuda.FloatTensor)

    def forward(self, img):
        # Extractor
        fts,_ = self.encoder(img) # torch.Size([8, 256, 28, 28])
        fts = F.interpolate(fts, size=img.size()[2:], mode='bilinear', align_corners=True)
        return fts

    def forward_gcn(self, img, mask): # (32,3,224,224)
        b,c,w,h = img.size()

        _, fts = self.encoder(img)  # torch.Size([32, 44, 224, 224]),torch.Size([32, 256, 56, 56])

        mask_c = torch.zeros_like(mask)
        mask_c[mask!=0] = 1.0
        mask_c = F.interpolate(mask_c, size=fts.shape[-2:], mode='bilinear', align_corners=True)

        fts = fts * mask_c #* mask # torch.Size([32, 256, 56, 56])
        fts = self.GAP(fts).view(fts.size(0), -1)

        sum = 0
        count = 0
        for i, itemi in enumerate(fts):
            for j, itemj in enumerate(fts[i+1:]):
                cos = F.cosine_similarity(itemi, itemj, dim=0)
                sum += cos.item()
                count += 1
        return sum/count


    def get_loss(self, cos_map, part_mask):
        b,c,w,h = cos_map.size()
        #loss_fuc = torch.nn.MSELoss(size_average = False)

        loss_fuc = torch.nn.CrossEntropyLoss(weight=self.loss_weights)
        #cos_map = cos_map * part_idx[:, :, None, None]
        mse_loss = loss_fuc(cos_map, part_mask[:,0,:,:].type(torch.cuda.LongTensor))# torch.Size([8, 43, 224, 224])
        return mse_loss


    def save_pred(self,cos_map, mask, part_mask, ii, mode, img=None, cat_idx=None): # torch.Size([8, 43, 224, 224])  torch.Size([8, 224, 224])
        b,c,w,h = cos_map.size()

        mask = mask.detach().cpu().numpy()[:, 0, :, :]

        cos_map = cos_map.detach().cpu().numpy()
        cos_map = np.argmax(cos_map, axis=1)
        bi_cos_map = np.zeros_like(cos_map)
        bi_cos_map[cos_map!=0]=1
        # category = np.unique(mask)[1]
        # bi_cos_map[cos_map != 0] = category

        if mode == 'train' or mode == 'test_voc':
            mask[mask != 0] = 1
            part_mask = part_mask.detach().cpu().numpy()
            save_map = np.concatenate((cos_map, part_mask[:,0,:,:], bi_cos_map, mask), axis=2)
            for j, item in enumerate(save_map):
                tools.labelTopng(item[:,:224], 'for_paper/%d_%d_attpred.png'%(ii,j))
                tools.labelTopng(item[:, 224:448], 'for_paper/%d_%d_attgt.png' % (ii, j))
                tools.labelTopng(item[:, 448:672], 'for_paper/%d_%d_bipred.png' % (ii, j), mode='binary')
                tools.labelTopng(item[:, 672:], 'for_paper/%d_%d_bigt.png' % (ii, j), mode='binary')
                img_s = img[j].cpu().numpy()
                img_s = img_s.swapaxes(0,1).swapaxes(1,2)
                img_s = tools.norm_image(img_s)
                save_image(img_s, 'for_paper/%d_%d_img.jpg'%(ii,j))

            for i in range(b):
                self.save_pred_list.append(np.uint8(cos_map[i]))
                self.save_bipred_list.append(np.uint8(bi_cos_map[i]))
                self.save_mask_list.append(np.uint8(mask[i]))
                self.save_part_mask_list.append(np.uint8(part_mask[i]))

        elif mode == 'test_coco':
            mask[mask != cat_idx] = 0
            mask[mask == cat_idx] = 1
            save_map = np.concatenate((bi_cos_map, mask), axis=2)
            # for j, item in enumerate(save_map):
            #     idx = cat_idx[j]
            #     tools.labelTopng(item[:,:224], 'for_paper_semantic/%d_%d_bipred.png'%(ii,j))
            #     tools.labelTopng(item[:, 224:448], 'for_paper_semantic/%d_%d_bigt.png' % (ii, j))
            #     img_s = img[j].cpu().numpy()
            #     img_s = img_s.swapaxes(0,1).swapaxes(1,2)
            #     img_s = tools.norm_image(img_s)
            #     save_image(img_s, 'for_paper_semantic/%d_%d_img.jpg'%(ii,j))
            for i in range(b):
                self.save_bipred_list.append(np.uint8(bi_cos_map[i]))
                self.save_mask_list.append(np.uint8(mask[i]))

    def save_semantic(self,cos_map, mask,ii): # torch.Size([8, 43, 224, 224])  torch.Size([8, 224, 224])
        b,w,h = cos_map.shape
        mask = mask[:,0]
        save_map = np.concatenate((cos_map, mask), axis=2)
        for i in range(b):
            #print(np.unique(cos_map[i]), np.unique(mask[i]))
            tools.labelTopng(save_map[i], 'check_semantic/%d_%d.png'%(ii,i))
            self.save_pred_list.append(np.uint8(cos_map[i]))
            self.save_mask_list.append(np.uint8(mask[i]))


    def get_pred(self, mode='train'):
        if mode == 'train' or mode == 'test_voc':
            _,_,miou,_ = tools.compute_miou(self.save_bipred_list, self.save_mask_list, n_class=2)
            _, _, part_miou, _ = tools.compute_miou(self.save_pred_list, self.save_part_mask_list, n_class=self.num_heads)
            self.save_pred_list = []
            self.save_bipred_list = []
            self.save_mask_list = []
            self.save_part_mask_list = []
            return miou, part_miou

        elif mode == 'test_coco':
            _,_,miou,_ = tools.compute_miou(self.save_bipred_list, self.save_mask_list, n_class=2)
            self.save_pred_list = []
            self.save_bipred_list = []
            self.save_mask_list = []
            self.save_part_mask_list = []
            return miou, 0.0

    def get_pred_semantic(self):
        _,_,miou,_ = tools.compute_miou(self.save_pred_list, self.save_mask_list, n_class=21)
        self.save_pred_list = []
        self.save_bipred_list = []
        self.save_mask_list = []
        self.save_part_mask_list = []
        return miou

