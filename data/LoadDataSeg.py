from __future__ import print_function
from __future__ import absolute_import

from data.transforms import transforms
from torch.utils.data import DataLoader
from data.all_voc_train import all_voc_train
from PIL import Image

def all_data_loader(args):

    batch = args.batch_size
    mean_vals = [0.485, 0.456, 0.406]
    std_vals = [0.229, 0.224, 0.225]
    size = 321

    if args.mode == 'train':
        tsfm = transforms.Compose([transforms.ToPILImage(),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.RandomResizedCrop(size=size, scale=(0.5, 1.0)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean_vals, std_vals)
                                         ])
    elif args.mode == 'test_voc' or args.mode == 'test_coco':
        tsfm = transforms.Compose([transforms.ToPILImage(),
                                   transforms.Resize(size=(size, size)),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean_vals, std_vals)
                                   ])

    else:
        print('running mode error!')
    # if args.dataset == 'coco':
    #     img_train = coco_train(args, transform=tsfm_train)
    if args.dataset == 'voc':
        img_train = all_voc_train(args, size, transform=tsfm, mode=args.mode)

    if args.mode == 'train':
        loader = DataLoader(img_train, batch_size=batch, shuffle=True, num_workers=1, drop_last=True)
    elif args.mode == 'test_voc' or args.mode == 'test_coco':
        loader = DataLoader(img_train, batch_size=batch, shuffle=False, num_workers=1, drop_last=True)

    return loader
