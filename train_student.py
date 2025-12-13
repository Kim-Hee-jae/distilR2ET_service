# -*- coding: cp949 -*- 
import os, sys
import time
import datetime
import random
from xml.etree.ElementInclude import include
import yaml
import argparse
import shutil
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import torch.optim as optim
from os import listdir, makedirs
from os.path import exists, join
import json

from src.ops import get_wjs
from datasets.train_feeder_student import Feeder
from src.model_student import DistilledRetNet


def get_parser():
    # parameter priority: command line > config file > default
    parser = argparse.ArgumentParser(description='R2ET for motion retargeting')
    parser.add_argument(
        '--config',
        default='./config/train_cfg.yaml',
        help='path to the configuration file',
    )
    parser.add_argument('--phase', default='train', help='train or test')
    parser.add_argument(
        '--work_dir',
        default='./work_dir/r2et_shape_aware',
        help='the work folder for storing results',
    )
    parser.add_argument(
        '--mesh_path', default='./datasets/mixamo_train_mesh', help='the mesh file path'
    )
    parser.add_argument(
        '--model_save_name', default='distilled_r2et', help='model saved name'
    )
    parser.add_argument(
        '--train_feeder_args',
        default=dict(),
        help='the arguments of data loader for training',
    )
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing',
    )
    parser.add_argument(
        '--base_lr', type=float, default=0.0001, help='initial learning rate'
    )
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument(
        '--weight_dqs', type=float, default=1.0, help='weight factor for delta qs loss'
    )
    parser.add_argument(
        '--weight_dqg', type=float, default=1.0, help='weight factor for delta qg loss'
    )
    parser.add_argument(
        '--weight_quat', type=float, default=1.0, help='weight factor for quat loss'
    )
    parser.add_argument(
        '--weight_fc', type=float, default=1.0, help='weight factor for foot contact loss'
    )
    parser.add_argument(
        '--max_length', type=int, default=60, help='max sequence length: T'
    )
    parser.add_argument(
        '--num_joint', type=int, default=22, help='number of the joints'
    )
    parser.add_argument(
        '--kp', type=float, default=0.8, help='keep prob in dropout layers'
    )
    parser.add_argument(
        '--ret_model_args',
        type=dict,
        default=dict(),
        help='the arguments of retargetor',
    )
    parser.add_argument(
        '--weight_decay', type=float, default=0.0005, help='weight decay for optimizer'
    )
    parser.add_argument(
        '--step',
        type=int,
        default=[],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate',
    )
    parser.add_argument(
        '--epoch', type=int, default=100, help='stop training in which epoch'
    )
    parser.add_argument(
        '--ret_weights',
        default='./work_dir/pmnet.pt',
        help='the path of the weights file for ret net',
    )
    parser.add_argument(
        '--ignore_weights', default=[], help='the ret weights that should be ignored'
    )

    parser.add_argument(
        '--job_id',
        default='',
        help='service/jobs/<job_id> root for per-job training',
    )

    return parser


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def print_log_txt(s, work_dir, print_time=True):
    if print_time:
        localtime = time.asctime(time.localtime(time.time()))
        s = f'[ {localtime} ] {s}'
    print(s)
    with open(os.path.join(work_dir, 'log.txt'), 'a') as f:
        print(s, file=f)


def load_model(ret_model, arg):
    output_device = arg.device[0]

    if arg.ret_weights:
        print_log_txt(f'Loading ret weights from {arg.ret_weights}', arg.work_dir)
        ret_weights = torch.load(arg.ret_weights)

        ret_weights = OrderedDict(
            [
                [k.split('module.')[-1], v.cuda(output_device)]
                for k, v in ret_weights.items()
            ]
        )

        pop_lst = []
        for k in ret_weights.keys():
            for w in arg.ignore_weights:
                if w in k:
                    pop_lst.append(k)
        state = ret_model.state_dict()
        for k in ret_weights.keys():
            if k not in state or ret_weights[k].shape != state[k].shape:
                pop_lst.append(k)
        for k in pop_lst:
            ret_weights.pop(k)
            print_log_txt(f'Sucessfully Remove Weights: {k}', arg.work_dir)

        try:
            ret_model.load_state_dict(ret_weights)
        except:
            state = ret_model.state_dict()
            diff = list(set(state.keys()).difference(set(ret_weights.keys())))
            print_log_txt('Can not find these weights:', arg.work_dir, False)
            for d in diff:
                print_log_txt('  ' + d, arg.work_dir, False)
            state.update(ret_weights)
            ret_model.load_state_dict(state)


def quat_angle_error_deg(q_pred, q_gt, eps=1e-7):
    """
    q_pred, q_gt: (..., 4) normalized quaternions
    return: (...,) rotation angle error in degrees
    """
    # inner product
    dot = torch.sum(q_pred * q_gt, dim=-1).abs()

    # numerical safety
    dot = torch.clamp(dot, -1.0 + eps, 1.0 - eps)

    # angle in radians
    angle = 2.0 * torch.acos(dot)

    # radians -> degrees
    angle_deg = angle * (180.0 / torch.pi)

    return angle_deg

def ensure_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def train(
    retarget_net,
    data_loader,
    optimizer_ret,
    scheduler,
    global_mean,
    global_std,
    local_mean,
    local_std,
    quat_mean,
    quat_std,
    parents,
    mesh_file_dic,  # None to
    all_names,
    body_vertices_dic,
    head_vertices_dic,
    leftarm_vertices_dic,
    rightarm_vertices_dic,
    leftleg_vertices_dic,
    rightleg_vertices_dic,
    hands_vertices_dic, # None
    epoch,
    max_epoch,
    logger,
    arg,
    include_fc,
):
    pbar = tqdm(total=len(data_loader), ncols=140)
    epoch_loss_ret = AverageMeter()
    epoch_loss_quat = AverageMeter()
    epoch_loss_foot = AverageMeter()
    epoch_time = AverageMeter()

    total_steps = len(data_loader)
    epoch_start_time = time.time()

    local_mean = torch.from_numpy(local_mean).cuda(arg.device[0])
    local_std = torch.from_numpy(local_std).cuda(arg.device[0])
    global_mean = torch.from_numpy(global_mean).cuda(arg.device[0])
    global_std = torch.from_numpy(global_std).cuda(arg.device[0])

    for batch_idx, (
        indexesA,
        indexesB,
        seqA,
        skelA,
        seqB,
        skelB,
        aeReg,
        mask,
        heightA,
        heightB,
        shapeA,
        shapeB,
        quatA_cp,
        quatB_gt,
        delta_qs_gt,
        delta_qg_gt,
        foot_contact_mask,
        offsetA,
        offsetB
    ) in enumerate(data_loader):
        seqA = seqA.float().cuda(arg.device[0])
        skelA = skelA.float().cuda(arg.device[0])
        seqB = seqB.float().cuda(arg.device[0])
        skelB = skelB.float().cuda(arg.device[0])
        aeReg = aeReg.float().cuda(arg.device[0])
        mask = mask.float().cuda(arg.device[0])
        heightA = heightA.float().cuda(arg.device[0])
        heightB = heightB.float().cuda(arg.device[0])
        quatA_cp = quatA_cp.float().cuda(arg.device[0])
        shapeA = shapeA.float().cuda(arg.device[0])
        shapeB = shapeB.float().cuda(arg.device[0])
        quatB_gt = quatB_gt.float().cuda(arg.device[0])
        delta_qs_gt = delta_qs_gt.float().cuda(arg.device[0])
        delta_qg_gt = delta_qg_gt.float().cuda(arg.device[0])
        foot_contact_mask = foot_contact_mask.float().cuda(arg.device[0])
        offsetA = offsetA.float().cuda(arg.device[0])
        offsetB = offsetB.float().cuda(arg.device[0])

        pbar.set_description("Train Epoch %i  Step %i" % (epoch + 1, batch_idx))
        start_time = time.time()

        # ------------------------------------ train generator ------------------------------------#
        retarget_net.train()
        optimizer_ret.zero_grad()

        (
            localA_gt,
            localB_rt,
            localB_gt,
            globalA_gt,
            globalB_rt,
            quatB_rt,
            delta_qs,
            delta_qg,
            quatB_base,
            localB_base,
            weights_sp,
        ) = retarget_net(
            seqA,
            seqB,
            skelA,
            skelB,
            shapeA,
            shapeB,
            quatA_cp,
            heightA,
            heightB,
            local_mean,
            local_std,
            quat_mean,
            quat_std,
            parents,
        )

        delta_qs_loss = DistilledRetNet.get_quat_loss_(mask, delta_qs_gt, delta_qs).mean()
        delta_qg_loss = DistilledRetNet.get_quat_loss_(mask, delta_qg_gt, delta_qg).mean()
        quat_loss = DistilledRetNet.get_quat_loss_(mask, quatB_gt, quatB_rt).mean()

        joint_lst = [8, 9, 12, 13]

        base_loss = (
            arg.weight_dqs * delta_qs_loss + arg.weight_dqg * delta_qg_loss + arg.weight_quat * quat_loss
        )
        if include_fc:
            localB_rt_denorm = localB_rt * local_std[None, :] + local_mean[None, :]
            foot_contact_loss = DistilledRetNet.get_foot_contact_loss_(joint_lst, foot_contact_mask, mask, heightB, localB_rt_denorm, globalB_rt, offsetB).mean()
        else:
            foot_contact_loss = torch.zeros_like(base_loss)



        ret_loss = base_loss + arg.weight_fc * foot_contact_loss

        for para in retarget_net.parameters():
            para.requires_grad = True
        ret_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(retarget_net.parameters(), max_norm=25)

        optimizer_ret.step()

        end_time = time.time()
        epoch_time.update(end_time - start_time)
        epoch_loss_ret.update(float(ret_loss.item()))
        epoch_loss_quat.update(float(quat_loss.item()))
        epoch_loss_foot.update(float(foot_contact_loss.item()))

        pbar.set_postfix(
            loss_r=float(ret_loss.item()),
            loss_q=float(quat_loss.item()),
            loss_f=float(foot_contact_loss.item()),
            time=end_time - start_time,
        )
        pbar.update(1)

        done_steps = batch_idx + 1 
        elapsed_epoch = time.time() - epoch_start_time 
        avg_step = elapsed_epoch / max(done_steps, 1) 
        remaining_in_epoch = max(total_steps - done_steps, 0)
        remaining_epochs = max_epoch - (epoch + 1)
        remaining_steps = remaining_in_epoch + total_steps * max(remaining_epochs, 0)
        eta = avg_step * remaining_steps
        tqdm.write( f"STUDENT_PROGRESS {epoch + 1} {max_epoch} {eta:.1f}", 
                   file=sys.stderr,
        )


    scheduler.step()
    pbar.close()

    logger.add_scalar('train_loss_ret', epoch_loss_ret.avg, epoch)
    logger.add_scalar('train_loss_quat', epoch_loss_quat.avg, epoch)
    logger.add_scalar('train_loss_foot', epoch_loss_foot.avg, epoch)

    return epoch_loss_ret, epoch_loss_quat, epoch_loss_foot, epoch_time

def evaluate_generalization(
    retarget_net,
    data_feeder,
    arg,
):
    """
    학습이 끝난 student 모델과 teacher(R2ET)의 quatB 차이를
    phase='test_uu' split에서 평가하는 함수.

    - Feeder의 phase를 'test_uu'로 바꿔서 DataLoader 생성
    - quatB_rt (student) vs quatB_gt (teacher) 의 quat loss 평균을 계산
    """

    eval_feeder_args = dict(arg.train_feeder_args)
    eval_feeder_args['phase'] = 'tmp_test_uu'
    eval_feeder = Feeder(**eval_feeder_args)
    eval_loader = torch.utils.data.DataLoader(
        dataset=eval_feeder,
        batch_size=arg.batch_size,
        num_workers=4,
        shuffle=False,
        persistent_workers=False,
    )

    device = arg.device[0]

    local_mean  = torch.from_numpy(data_feeder.local_mean).cuda(device)
    local_std   = torch.from_numpy(data_feeder.local_std).cuda(device)
    global_mean = torch.from_numpy(data_feeder.global_mean).cuda(device)
    global_std  = torch.from_numpy(data_feeder.global_std).cuda(device)
    quat_mean   = data_feeder.quat_mean
    quat_std    = data_feeder.quat_std
    parents     = data_feeder.parents   # numpy 그대로 (train과 동일)

    quat_meter = AverageMeter()

    angle_list = []
    retarget_net.eval()
    with torch.no_grad():
        pbar = tqdm(total=len(eval_loader), ncols=140)
        pbar.set_description("Eval quatB gap")

        for batch_idx, (
            indexesA,
            indexesB,
            seqA,
            skelA,
            seqB,
            skelB,
            aeReg,
            mask,
            heightA,
            heightB,
            shapeA,
            shapeB,
            quatA_cp,
            quatB_gt,
            delta_qs_gt,
            delta_qg_gt,
            foot_contact_mask,
            offsetA,
            offsetB
        ) in enumerate(eval_loader):

            seqA = seqA.float().cuda(device)
            skelA = skelA.float().cuda(device)
            seqB = seqB.float().cuda(device)
            skelB = skelB.float().cuda(device)
            aeReg = aeReg.float().cuda(device)
            mask = mask.float().cuda(device)
            heightA = heightA.float().cuda(device)
            heightB = heightB.float().cuda(device)
            quatA_cp = quatA_cp.float().cuda(device)
            shapeA = shapeA.float().cuda(device)
            shapeB = shapeB.float().cuda(device)
            quatB_gt = quatB_gt.float().cuda(device)
            delta_qs_gt = delta_qs_gt.float().cuda(device)
            delta_qg_gt = delta_qg_gt.float().cuda(device)
            foot_contact_mask = foot_contact_mask.float().cuda(device)
            offsetA = offsetA.float().cuda(device)
            offsetB = offsetB.float().cuda(device)

            (
                localB_rt, 
                globalB_rt, 
                quatB_rt, 
                delta_qs, 
                delta_qg
            ) = retarget_net(
                seqA,
                seqB,
                skelA,
                skelB,
                shapeA,
                shapeB,
                quatA_cp,
                heightA,
                heightB,
                local_mean,
                local_std,
                quat_mean,
                quat_std,
                parents,
            )

            # teacher(R2ET)의 quatB_gt와 student의 quatB_rt의 차이
            # quatB_rt, quatB_gt : (B, T, J, 4)
            angle_err = quat_angle_error_deg(quatB_rt, quatB_gt)  # (B,T,J)

            # mask: (B,T) -> joint 차원 넓히기
            mask3d = mask.unsqueeze(-1).expand_as(angle_err)

            angle_err = angle_err * mask3d

            # 유효한 값만 모으기
            valid_angles = angle_err[mask3d > 0]

            # batch별 누적
            angle_list.append(valid_angles.detach().cpu())

            pbar.update(1)

        pbar.close()
        all_angles = torch.cat(angle_list, dim=0)

        mean_deg = all_angles.mean().item()
        std_deg  = all_angles.std().item()

    return mean_deg, std_deg




def main(arg):
    if not exists(arg.work_dir):
        makedirs(arg.work_dir)

    data_feeder = Feeder(**arg.train_feeder_args)
    retarget_net = DistilledRetNet(**arg.ret_model_args).cuda(arg.device[0])

    load_model(retarget_net, arg)

    retarget_net = nn.DataParallel(retarget_net, device_ids=arg.device)

    data_loader = torch.utils.data.DataLoader(
        dataset=data_feeder, batch_size=arg.batch_size, num_workers=8, shuffle=True, persistent_workers=True
    )

    optimizer_ret = optim.Adam(
        retarget_net.parameters(), lr=arg.base_lr, weight_decay=arg.weight_decay, betas=(0.5, 0.999)
    )
    scheduler_ret = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_ret, milestones=arg.step, gamma=0.1, last_epoch=-1
    )

    train_writer = SummaryWriter(
        os.path.join(arg.work_dir, arg.model_save_name, 'train_log'), 'train'
    )

    # ----------------------------------------- load mesh data: X ----------------------------------------------
    body_vertices_dic = {}
    head_vertices_dic = {}
    leftarm_vertices_dic = {}
    rightarm_vertices_dic = {}
    leftleg_vertices_dic = {}
    rightleg_vertices_dic = {}
    hands_vertices_dic = {}

    # save cfg file
    arg_dict = vars(arg)
    with open(join(arg.work_dir, 'config.yaml'), 'w') as f:
        yaml.dump(arg_dict, f)

    
    max_epoch = arg.epoch + int(np.ceil(arg.epoch / 2))
    tqdm.write(
            f"STUDENT_TOTAL_STEPS {max_epoch}",
            file=sys.stderr,
        )

    # foot contact X
    for i in range(arg.epoch):
        epoch_loss_r, epoch_loss_q, epoch_loss_foot, epoch_time = train(
            retarget_net,
            data_loader,
            optimizer_ret,
            scheduler_ret,
            data_feeder.global_mean,
            data_feeder.global_std,
            data_feeder.local_mean,
            data_feeder.local_std,
            data_feeder.quat_mean,
            data_feeder.quat_std,
            data_feeder.parents,
            None,
            data_feeder.all_names,
            body_vertices_dic,
            head_vertices_dic,
            leftarm_vertices_dic,
            rightarm_vertices_dic,
            leftleg_vertices_dic,
            rightleg_vertices_dic,
            hands_vertices_dic,
            i,
            max_epoch,
            train_writer,
            arg,
            False,  # Stage 1: foot-contact loss off
        )
        lr = optimizer_ret.param_groups[0]['lr']
        log_txt = (
            'epoch:'
            + str(i + 1)
            + '  ret loss:'
            + str(epoch_loss_r.avg)
            + '  quat loss:'
            + str(epoch_loss_q.avg)
            + '  foot loss:'
            + str(epoch_loss_foot.avg)
            + '  epoch time:'
            + str(epoch_time.avg)
            + '  lr:'
            + str(lr)
        )
        print_log_txt(log_txt, arg.work_dir)
        # quat loss가 충분히 작아지면 조기 종료
        if epoch_loss_q.avg < 0.0002:
            break

    # foot contact O
    i += 1
    for j in range(i, i + int(np.ceil(arg.epoch / 2))):
        epoch_loss_r, epoch_loss_q, epoch_loss_foot, epoch_time = train(
            retarget_net,
            data_loader,
            optimizer_ret,
            scheduler_ret,
            data_feeder.global_mean,
            data_feeder.global_std,
            data_feeder.local_mean,
            data_feeder.local_std,
            data_feeder.quat_mean,
            data_feeder.quat_std,
            data_feeder.parents,
            None,
            data_feeder.all_names,
            body_vertices_dic,
            head_vertices_dic,
            leftarm_vertices_dic,
            rightarm_vertices_dic,
            leftleg_vertices_dic,
            rightleg_vertices_dic,
            hands_vertices_dic,
            j,    
            max_epoch,      
            train_writer,
            arg,
            True,           
        )
        lr = optimizer_ret.param_groups[0]['lr']
        log_txt = (
            'epoch:'
            + str(j + 1)
            + '  ret loss:'
            + str(epoch_loss_r.avg)
            + '  quat loss:'
            + str(epoch_loss_q.avg)
            + '  foot loss:'
            + str(epoch_loss_foot.avg)
            + '  epoch time:'
            + str(epoch_time.avg)
            + '  lr:'
            + str(lr)
        )
        print_log_txt(log_txt, arg.work_dir)

    state_dict_ret = retarget_net.state_dict()
    weights_gen = OrderedDict([[k, v.cpu()] for k, v in state_dict_ret.items()])
    torch.save(
        weights_gen,
        os.path.join(
            arg.work_dir, arg.model_save_name + '.pt'
        ),
    )
    log_txt = arg.model_save_name + '.pt has been saved!'
    print_log_txt(log_txt, arg.work_dir)

    # 학습 후 일반화 성능 평가
    try:
        mean_deg, std_deg = evaluate_generalization(retarget_net, data_feeder, arg)
        log_txt = (
            "[Generalization] phase=test_uu | "
            f"Rotation Error (Teacher vs Student): "
            f"{mean_deg:.2f} ± {std_deg:.2f} degrees"
        )
        print_log_txt(log_txt, arg.work_dir)
    except Exception as e:
        mean_deg = None
        print_log_txt(f"[Generalization] 평가 중 오류 발생: {repr(e)}", arg.work_dir)

    # 기록
    if getattr(arg, 'job_id', ''):
        job_root = os.path.join('.', 'service', 'jobs', arg.job_id)
        output_dir = os.path.join(job_root, 'output')
        os.makedirs(output_dir, exist_ok=True)

        txt_path = os.path.join(output_dir, 'generalization_gap.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("대형 모델(teacher: R2ET) 대비 일반화 성능 차이\n")
            f.write("평가 unseen 캐릭터 & unseen 애니메이션\n")
            if mean_deg is not None:
                f.write(f"평균 quatB 차이 (teacher: R2ET): {mean_deg:.2f} ± {std_deg:.2f} degrees\n")
            else:
                f.write("평가에 실패했습니다. 로그를 확인하세요.\n")



if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    init_seed(3047)
    parser = get_parser()

    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG:', k)
                assert k in key
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()

    if getattr(arg, 'job_id', ''):
        job_root = os.path.join('.', 'service', 'jobs', arg.job_id)

        arg.work_dir = os.path.join(job_root, 'student')
        os.makedirs(arg.work_dir, exist_ok=True)

        tfa = arg.train_feeder_args

        data_cfg_path = tfa.get('data_config_path')
        with open(data_cfg_path, 'r', encoding='utf-8') as f:
            data_cfg = json.load(f)
        target_char = data_cfg['target_char']

        shape_root = tfa['shape_path']
        os.makedirs(shape_root, exist_ok=True)
        src_shape = os.path.join(job_root, 'input.npz')
        dst_shape = os.path.join(shape_root, target_char + '.npz')
        shutil.copyfile(src_shape, dst_shape)

        tfa['gt_path'] = os.path.join(job_root, 'r2et')

        tfa['stats_path'] = os.path.join(job_root, 'r2et', 'stats')

        arg.train_feeder_args = tfa

    main(arg)


