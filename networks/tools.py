import random

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.binomial import Binomial
import numpy as np
from skimage import io
import PIL.Image as Image

def getPartPalette(mode):
    '''
    http://blog.csdn.net/yhl_leo/article/details/52185581
    '''
    if mode == 'color':
        pal = np.array([[0, 0, 0],
                        [128, 0, 0],
                        [0, 128, 0],
                        [128, 128, 0],
                        [0, 0, 128],
                        [128, 0, 128],
                        [0, 128, 128],
                        [128, 128, 128],
                        [64, 0, 0],
                        [192, 0, 0],
                        [64, 128, 0],
                        [192, 128, 0],
                        [64, 0, 128],
                        [192, 0, 128],
                        [64, 128, 128],
                        [192, 128, 128],
                        [0, 64, 0],
                        [128, 64, 0],
                        [0, 192, 0],
                        [128, 192, 0],
                        [0, 64, 128],
                       [128, 32, 32],
                       [32, 128, 32],
                       [128, 128, 32],
                       [32, 32, 128],
                       [128, 32, 128],
                       [32, 128, 128],
                       [128, 128, 128],
                       [64, 32, 32],
                       [192, 32, 32],
                       [64, 128, 32],
                       [192, 128, 32],
                       [64, 32, 128],
                       [192, 32, 128],
                       [64, 128, 128],
                       [192, 128, 128],
                       [32, 64, 32],
                       [128, 64, 32],
                       [32, 192, 32],
                       [128, 192, 32],
                       [32, 64, 128],
                       [128, 16, 16],
                       [16, 128, 16],
                       [128, 128, 16],
                       [16, 16, 128],
                       [128, 16, 128],
                       [16, 128, 128],
                       [128, 128, 128],
                       [64, 16, 16],
                       [192, 16, 16]], dtype='uint8').flatten()
    elif mode =='binary':
        pal = np.array([[0, 0, 0],
                        [255, 255, 255]], dtype='uint8').flatten()
    return pal

def colorize_mask(mask, mode):
    """
    :param mask: ?????????????????????????????????????????????
    :return:
    """
    new_mask = Image.fromarray(mask.astype(np.uint8), 'P')  # ???????????????????????????

    pal = getPartPalette(mode)
    new_mask.putpalette(pal)
    # print(new_mask.show())
    return new_mask

def labelTopng(label, img_name, mode='color'):
    '''
    convert tensor cpu label to png and save
    '''
    #label = label.numpy()             # 320 320
    label_pil = colorize_mask(label, mode)
    label_pil.save(img_name)

def gen_cam(image, mask, index=None):
    """
    ??????CAM???
    :param image: [H,W,C],????????????
    :param mask: [H,W],??????0~1
    :return: tuple(cam,heatmap)
    """
    # mask??????heatmap
    # save_file=open("test.txt",'w')
    # save_file.write(str(np.uint8(255 * mask)))
    # save_file.close()
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]  # gbr to rgb

    # ??????heatmap???????????????
    cam = heatmap*5 + np.float32(image)
    cam = norm_image(cam)
    if not index is None:
        index = str(index)
        cv2.putText(cam, index, (0, 50), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
    return cam

def norm_image(image):
    """
    ???????????????
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)

def save_image(image_cam, save_path):
    io.imsave(save_path, image_cam)

def get_query_output(proto, fts, scalar=20): # torch.Size([1, 512]) torch.Size([1, 512, 417, 417])

    cos_map = torch.cosine_similarity(proto[:,:, None, None], fts, dim=1) * scalar

    return cos_map

def get_cam(proto, fts, mask, seq, img):
    # torch.Size([1, 50, 256]) torch.Size([1, 256, 128, 128]) torch.Size([50]) torch.Size([1, 3, 512, 512])
    n_shots, c, dim = proto.size()

    img = img[0].detach()
    img = img.permute((1, 2, 0))
    img = img * (mask.unsqueeze(-1))
    fts = F.interpolate(fts, size=(512,512), mode='bilinear')

    cam_list = []
    for shot in range(n_shots):
        proto_mask = proto[shot] * (seq.unsqueeze(-1)) # (50,256)
        for channel in range(c):
            cos = F.cosine_similarity(proto_mask[channel].unsqueeze(0).unsqueeze(-1).unsqueeze(-1), fts, dim=1)
            cos = F.relu(cos)

            cos = cos.detach().cpu().numpy()
            cos = cos - np.min(cos, (1, 2))
            cos = cos / np.max(cos + 1e-7, (1, 2))

            cam = gen_cam(img.cpu().numpy(), np.max(cos, 0), index=channel)
            cam_list.append(cam)

    cam_list = np.stack(cam_list, axis=0) # (50,512,512,3) * (50,1,1,1)
    #cam_list = cam_list * seq.clone().detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).cpu().numpy()
    cam_1 = cam_list[:10].reshape((512 * 10, 512, 3))
    cam_2 = cam_list[10:20].reshape((512 * 10, 512, 3))
    cam_3 = cam_list[20:30].reshape((512 * 10, 512, 3))
    #cam_4 = cam_list[30:40].reshape((512 * 10, 512, 3))
    #cam_5 = cam_list[40:].reshape((512 * 10, 512, 3))
    cam_summary = np.concatenate((cam_1, cam_2, cam_3), axis=1)

    return cam_summary


def get_seg(proto, fts, seq, img):
    n_shots, c, dim = proto.size()

    img = img[0].detach()
    img = img.permute((1, 2, 0))

    fts = F.interpolate(fts, size=(512, 512), mode='bilinear')

    cos_list = []
    for shot in range(n_shots):
        proto_mask = proto[shot] * (seq.unsqueeze(-1))  # (50,256)
        for channel in range(c):
            cos = F.cosine_similarity(proto_mask[channel].unsqueeze(0).unsqueeze(-1).unsqueeze(-1), fts, dim=1)
            cos = F.relu(cos)
            cos_list.append(cos.detach().cpu().numpy())

    seg = np.concatenate(cos_list, axis=0) # (50,512,512)
    seg = np.sum(seg, axis=0, keepdims=True) # (1,512,512)

    seg = seg - np.min(seg, (1, 2))
    seg = seg / np.max(seg + 1e-7, (1, 2))

    zeros = np.zeros_like(seg[0])
    background = np.where(seg[0] < 0.15, seg, zeros)
    foreground = np.where(seg[0] >= 0.15, seg, zeros)
    seg_return = np.concatenate((background, foreground), axis=0)

    # save seg img
    seg = np.argmax(seg_return, axis=0)
    seg = seg[:, :, np.newaxis]
    seg_img = norm_image(seg * img.cpu().numpy())

    return seg_img, seg_return

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

def compute_miou(label_trues, label_preds, n_class=21):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)   # n_class, n_class
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + 1e-5)
    print(iu)
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc

if __name__ == '__main__':
    a = torch.tensor([-1,5])
    print(torch.relu(a))