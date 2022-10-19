import sys
sys.path.append('core')

from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import imageio

import datasets
from utils import flow_viz
from utils import frame_utils

from raft import RAFT
from utils.utils import InputPadder, forward_interpolate


@torch.no_grad()
def create_sintel_submission(model, iters=32, warm_start=False, output_path='sintel_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    for dstype in ['clean', 'final']:
        test_dataset = datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)
        
        flow_prev, sequence_prev = None, None
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None
            
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

            flow_low, flow_pr = model(image1, image2, iters=iters, flow_init=flow_prev, test_mode=True)
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()
            
            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame+1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)
            sequence_prev = sequence


@torch.no_grad()
def create_kitti_submission(model, iters=24, output_path='kitti_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id, ) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)


@torch.no_grad()
def validate_chairs(model, iters=24):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []

    val_dataset = datasets.FlyingChairs(split='validation')
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt)**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe = np.mean(np.concatenate(epe_list))
    print("Validation Chairs EPE: %f" % epe)
    return {'chairs': epe}


@torch.no_grad()
def validate_sintel(model, iters=32):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype)
        epe_list = []

        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)

    return results


@torch.no_grad()
def validate_kitti(model, iters=24):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.KITTI(split='training')

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).cpu()

        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1}


@torch.no_grad()
def validate_awi_d(model, dataset_name, iters=6, halve_image=False, save=False):
    """ Perform validation using the AWI Dubbo or Deniliquin datasets """
    assert dataset_name in {'dubbo', 'deniliquin'}, 'Dataset must be named dubbo or deniliquin'

    model.eval()
    if dataset_name == 'dubbo':
        val_dataset = datasets.AWI_Dubbo(split='test', root=datasets.AWI_ROOT['dubbo'], halve_image=halve_image)
        out_dir = 'awi_dubbo'
    elif dataset_name == 'deniliquin':
        val_dataset = datasets.AWI_Deniliquon(split='test', root=datasets.AWI_ROOT['deniliquin'], halve_image=halve_image)
        out_dir = 'awi_deniliquin'

    print('Evaluating on {} image pairs'.format(len(val_dataset)))

    for val_id in tqdm(range(len(val_dataset))):
        image1, image2, frame_info = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        flow_pr = model.module(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).cpu()

        # Visualizations
        if save:
            output_flow = flow.permute(1, 2, 0).numpy()
            flow_img = flow_viz.flow_to_image(output_flow)
            flow_img = Image.fromarray(flow_img)
            fleece_id = frame_info['scene']
            camera = frame_info['camera']
            ts = frame_info['frame']
            if not os.path.exists(f'vis/{out_dir}/{fleece_id}/{camera}/vis/'):
                os.makedirs(f'vis/{out_dir}/{fleece_id}/{camera}/vis/')
            if not os.path.exists(f'vis/{out_dir}/{fleece_id}/{camera}/flow/'):
                os.makedirs(f'vis/{out_dir}/{fleece_id}/{camera}/flow/')
            imageio.imwrite(f'vis/{out_dir}/{fleece_id}/{camera}/vis/{ts}.png', flow_img)
            frame_utils.writeFlow(f'vis/{out_dir}/{fleece_id}/{camera}/flow/{ts}.flo', output_flow)



@torch.no_grad()
def validate_awi_uv(model, iters=24, halve_image=False, save=False):
    """ Perform validation using the AWI UV dataset """
    model.eval()
    val_dataset = datasets.AWI_UV(split='validation', root=datasets.AWI_ROOT['awi_uv'], halve_image=halve_image)
    out_dir = 'awi_uv'

    print('Evaluating on {} image pairs'.format(len(val_dataset)))

    results = {}
    epe_list = []
    for val_id in tqdm(range(len(val_dataset))):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).cpu()

        epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
        epe = epe.view(-1)
        val = valid_gt.reshape(-1) >= 0.5
        if val.sum() > 0:
            epe_list.append(epe[val].mean().item())

        # Visualizations
        if save:
            output_flow = flow.permute(1, 2, 0).numpy()
            flow_img = flow_viz.flow_to_image(output_flow)
            flow_img = Image.fromarray(flow_img)
            fleece_id, ts = val_dataset.extra_info[val_id]['scene'], val_dataset.extra_info[val_id]['frame']
            if not os.path.exists(f'vis/{out_dir}/{fleece_id}/vis/'):
                os.makedirs(f'vis/{out_dir}/{fleece_id}/vis/')
            if not os.path.exists(f'vis/{out_dir}/{fleece_id}/flow/'):
                os.makedirs(f'vis/{out_dir}/{fleece_id}/flow/')
            imageio.imwrite(f'vis/{out_dir}/{fleece_id}/vis/{ts}.png', flow_img)
            frame_utils.writeFlow(f'vis/{out_dir}/{fleece_id}/flow/{ts}.flo', output_flow)

    epe_all = np.array(epe_list)

    epe = np.mean(epe_all)
    px1 = np.mean(epe_all < 1)
    px3 = np.mean(epe_all < 3)
    px5 = np.mean(epe_all < 5)

    print("Validation EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (epe, px1, px3, px5))
    results['awi_epe'] = np.mean(epe)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--save', action='store_true', help='save predictions to file')
    parser.add_argument('--halve', action='store_true', help='halve the image')
    args = parser.parse_args()

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model.cuda()
    model.eval()

    # create_sintel_submission(model.module, warm_start=True)
    # create_kitti_submission(model.module)

    with torch.no_grad():
        if args.dataset == 'chairs':
            validate_chairs(model.module)

        elif args.dataset == 'sintel':
            validate_sintel(model.module)

        elif args.dataset == 'kitti':
            validate_kitti(model.module)

        elif args.dataset == 'dubbo' or args.dataset == 'deniliquin':
            validate_awi_d(model, args.dataset, iters=24, halve_image=args.halve, save=args.save)

        elif args.dataset == 'awi_uv':
            validate_awi_uv(model, iters=24, halve_image=args.halve, save=args.save)


