# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import os
import math
import random
from glob import glob
import os.path as osp
import json
from PIL import Image

from utils import frame_utils
from utils.augmentor import FlowAugmentor, SparseFlowAugmentor


AWI_ROOT = {'dubbo': '/home/tpatten/Data/AWI/AWI_Dataset',
            'denilquin': '/home/tpatten/Data/AWI/AWI_Dataset_2',
            'awi_uv': '/home/tpatten/Data/AWI/TechLab_UV_Annotation'}

AWI_IMAGE_RES = (600, 2464)

AWI_UV_EPS = 50


class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, halve_image=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []
        self.halve_image = halve_image

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]

            if self.halve_image:
                half_width = int(img1.shape[1] / 2)
                if self.extra_info[index]['camera'] == 'GX301187':
                    img1 = img1[:, :half_width, :]
                    img2 = img2[:, :half_width, :]
                elif self.extra_info[index]['camera'] == 'GX300643':
                    img1 = img1[:, half_width:, :]
                    img2 = img2[:, half_width:, :]
                elif self.extra_info[index]['camera'] == 'uv':
                    # Use the mask to crop the image
                    msk = np.asarray(Image.open(self.extra_info[index]['mask'])).astype(np.uint8)
                    valid_cols = np.where(msk > 0)[-1]
                    max_valid_col = valid_cols.max()
                    max_valid_col = min(img1.shape[1] - 1, max_valid_col + AWI_UV_EPS)
                    img1 = img1[:, max_valid_col - half_width:max_valid_col, :]
                    img2 = img2[:, max_valid_col - half_width:max_valid_col, :]
                else:
                    raise NotImplementedError

            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None
        if self.sparse:
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        else:
            flow = frame_utils.read_gen(self.flow_list[index])
            if type(flow) is tuple:
                flow, valid = flow

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        if self.halve_image:
            half_width = int(flow.shape[1] / 2)
            if self.extra_info[index]['camera'] == 'GX301187':
                flow = flow[:, :half_width, :]
                valid = valid[:, :half_width]
                img1 = img1[:, :half_width, :]
                img2 = img2[:, :half_width, :]
            else:
                flow = flow[:, half_width:, :]
                valid = valid[:, half_width:]
                img1 = img1[:, half_width:, :]
                img2 = img2[:, half_width:, :]

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        return img1, img2, flow, valid.float()

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self
        
    def __len__(self):
        return len(self.image_list)
        

class MpiSintel(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/Sintel', dstype='clean'):
        super(MpiSintel, self).__init__(aug_params)
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)

        if split == 'test':
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            for i in range(len(image_list)-1):
                self.image_list += [ [image_list[i], image_list[i+1]] ]
                self.extra_info += [ (scene, i) ] # scene and frame_id

            if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))


class FlyingChairs(FlowDataset):
    def __init__(self, aug_params=None, split='train', root='datasets/FlyingChairs_release/data'):
        super(FlyingChairs, self).__init__(aug_params)

        images = sorted(glob(osp.join(root, '*.ppm')))
        flows = sorted(glob(osp.join(root, '*.flo')))
        assert (len(images)//2 == len(flows))

        split_list = np.loadtxt('chairs_split.txt', dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split=='training' and xid==1) or (split=='validation' and xid==2):
                self.flow_list += [ flows[i] ]
                self.image_list += [ [images[2*i], images[2*i+1]] ]


class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/FlyingThings3D', dstype='frames_cleanpass'):
        super(FlyingThings3D, self).__init__(aug_params)

        for cam in ['left']:
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(osp.join(root, dstype, 'TRAIN/*/*')))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TRAIN/*/*')))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(osp.join(idir, '*.png')) )
                    flows = sorted(glob(osp.join(fdir, '*.pfm')) )
                    for i in range(len(flows)-1):
                        if direction == 'into_future':
                            self.image_list += [ [images[i], images[i+1]] ]
                            self.flow_list += [ flows[i] ]
                        elif direction == 'into_past':
                            self.image_list += [ [images[i+1], images[i]] ]
                            self.flow_list += [ flows[i+1] ]
      

class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/KITTI'):
        super(KITTI, self).__init__(aug_params, sparse=True)
        if split == 'testing':
            self.is_test = True

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [ [frame_id] ]
            self.image_list += [ [img1, img2] ]

        if split == 'training':
            self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))


class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/HD1k'):
        super(HD1K, self).__init__(aug_params, sparse=True)

        seq_ix = 0
        while 1:
            flows = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
            images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))

            if len(flows) == 0:
                break

            for i in range(len(flows)-1):
                self.flow_list += [flows[i]]
                self.image_list += [ [images[i], images[i+1]] ]

            seq_ix += 1


class AWI_Dubbo(FlowDataset):
    def __init__(self, aug_params=None, halve_image=False, split='training', root=''):
        super(AWI_Dubbo, self).__init__(aug_params, sparse=False, halve_image=halve_image)

        # Constants
        cameras = ['GX300643', 'GX301187']
        flow_dir = 'flow_annotation'
        annotation_dir = 'json'
        image_pair_dir = 'before_skirt'

        if split == 'test':
            self.is_test = True

        # Setting to generate the flow for all available folders
        flow_dir = os.path.join(root, flow_dir)
        fleece_dirs = sorted([f for f in os.listdir(flow_dir) if f.startswith('fleece')])

        if split == 'test':
            self.is_test = True

        for subdir in fleece_dirs:
            for c in cameras:
                path_to_annotations = os.path.join(root, subdir, c, annotation_dir)
                path_to_flows = os.path.join(flow_dir, subdir, c)
                if os.path.exists(path_to_flows):
                    # Get all files in the directory
                    for i, f in enumerate(os.listdir(path_to_flows)):
                        f_name, f_ext = os.path.splitext(f)
                        flow_file = os.path.join(path_to_flows, f)
                        anno_file = os.path.join(path_to_annotations, f.replace(f_ext, '.json'))
                        if os.path.isfile(anno_file):
                            with open(anno_file) as json_file:
                                anno_data = json.load(json_file)
                            image_1 = os.path.join(path_to_annotations, image_pair_dir, f.replace(f_ext, '.png'))
                            image_2 = os.path.join(path_to_annotations, anno_data['correspondence'])

                            self.image_list += [[image_2, image_1]]  # Reverse because want flow from after to before
                            self.flow_list += [flow_file]
                            # scene, camera and frame_id with no extension
                            self.extra_info += [{'scene': subdir, 'camera': c, 'frame': f_name}]


# class AWI_Deniliquin(FlowDataset):


class AWI_UV(FlowDataset):
    def __init__(self, aug_params=None, halve_image=False, split='test', root=''):
        super(AWI_UV, self).__init__(aug_params, sparse=False, halve_image=halve_image)

        # Constants
        image_pairs = [('00_00_rgb.png', '00_02_rgb.png'),
                       ('01_00_rgb.png', '01_02_rgb.png'),
                       ('02_00_rgb.png', '02_02_rgb.png')]
        flow_dir = 'flows'
        mask_dir = 'masks'

        if split == 'test':
            self.is_test = True
        
        # Setting to generate the flow for all available folders
        fleece_dirs = sorted([f for f in os.listdir(root) if f.startswith('fleece')])

        for subdir in fleece_dirs:
            for image_p in image_pairs:
                image_1 = os.path.join(root, subdir, image_p[0])
                image_2 = os.path.join(root, subdir, image_p[1])

                flow_file = os.path.join(root, flow_dir, subdir, image_p[0].replace('_rgb.png', '.mat'))

                mask_1 = os.path.join(root, mask_dir, subdir, image_p[0])
                if not os.path.exists(mask_1):
                    mask_1 = mask_1.replace('rgb', 'mask')

                self.image_list += [[image_2, image_1]]  # Reverse because want flow from after to before skirted
                self.flow_list += [flow_file]
                # scene and frame_id with no extension
                self.extra_info += [{'scene': subdir, 'camera': 'uv', 'frame': os.path.splitext(image_p[0])[0],
                                     'mask': mask_1}]


def fetch_dataloader(args, TRAIN_DS='C+T+K+S+H'):
    """ Create the data loader for the corresponding trainign set """

    if args.stage == 'chairs':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
        train_dataset = FlyingChairs(aug_params, split='training')
    
    elif args.stage == 'things':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        clean_dataset = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        final_dataset = FlyingThings3D(aug_params, dstype='frames_finalpass')
        train_dataset = clean_dataset + final_dataset

    elif args.stage == 'sintel':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')        

        if TRAIN_DS == 'C+T+K+S+H':
            kitti = KITTI({'crop_size': args.image_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True})
            hd1k = HD1K({'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True})
            train_dataset = 100*sintel_clean + 100*sintel_final + 200*kitti + 5*hd1k + things

        elif TRAIN_DS == 'C+T+K/S':
            train_dataset = 100*sintel_clean + 100*sintel_final + things

    elif args.stage == 'kitti':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = KITTI(aug_params, split='training')

    elif args.stage == 'dubbo':
        aug_params = None  # {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        if args.image_size[1] != AWI_IMAGE_RES[1]:
            train_dataset = AWI_Dubbo(aug_params, split='training', root=AWI_ROOT['dubbo'], halve_image=True)
        else:
            train_dataset = AWI_Dubbo(aug_params, split='training', root=AWI_ROOT['dubbo'], halve_image=False)

    elif args.stage == 'deniliquin':
        raise NotImplementedError

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
        pin_memory=False, shuffle=True, num_workers=4, drop_last=True)

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader

