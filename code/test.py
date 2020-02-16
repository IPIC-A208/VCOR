import argparse
from torch.utils.data import DataLoader
from utils import YUVread
from model import Model
from Transformer import *
from data import MDataYUV, Data_codec_isolate
import time
import yaml
import numpy as np
import os

version = float(torch.__version__[0:3])
if version >= 1.1:
    from torch.utils.tensorboard import SummaryWriter
else:
    from tensorboardX import SummaryWriter




def get_model(args):
    if args.model == 'BaseModel':
        args.model = args.model + 'B_' + str(args.block)
        print(args.model)
        # print('???***************************************')
        return BaseModel(block=args.block, fea_channel=args.channel)
    if args.model == 'BaseModelV2':
        args.model = args.model + '_C' + str(args.channel)
        print(args.model)
        # print('???***************************************')
        return BaseModelV2(fea_channel=args.channel)
    if args.model == 'BaseModelV3':
        args.model = args.model + '_C' + str(args.channel)
        print(args.model)
        # print('???***************************************')
        return BaseModelV3(fea_channel=args.channel)
    if args.model == 'BaseV3NonLocal':
        args.model = args.model + '_C' + str(args.channel)
        print(args.model)
        return BaseV3NonLocal(fea_channel=args.channel)
    if args.model == 'BaseV3NonLocalV2':
        args.model = args.model + '_C' + str(args.channel)
        print(args.model)
        return BaseV3NonLocalV2(fea_channel=args.channel)
    if args.model == 'BaseV4NonLocal':
        args.model = args.model + '_C' + str(args.channel)
        print(args.model)
        return BaseV4NonLocal(fea_channel=args.channel)
    if args.model == 'BaseModelV4':
        args.model = args.model + '_C' + str(args.channel)
        print(args.model)
        # print('???***************************************')
        return BaseModelV4(fea_channel=args.channel)
    if args.model == 'BaseV3Attention':
        args.model = args.model + '_C' + str(args.channel)
        print(args.model)
        # print('???***************************************')
        return BaseV3Attention(fea_channel=args.channel)        
    if args.model == 'BaseV4Attention':
        args.model = args.model + '_C' + str(args.channel) + 'B' + str(args.block)
        print(args.model)
        return BaseV4Attention(fea_channel=args.channel, deep=args.block)
    if args.model == 'BaseV4CAT':
        args.model = args.model + '_C' + str(args.channel) + 'B' + str(args.block)
        print(args.model)
        return BaseV4CAT(fea_channel=args.channel, deep=args.block)
    if args.model == 'BaseV5CAT':
        args.model = args.model + '_C' + str(args.channel) + 'B' + str(args.block)
        print(args.model)
        return BaseV5CAT(fea_channel=args.channel, deep=args.block)
    if args.model == 'BaseV6CAT':
        args.model = args.model + '_C' + str(args.channel) + 'B' + str(args.block)
        print(args.model)
        return BaseV6CAT(fea_channel=args.channel, deep=args.block)
    if args.model == 'BaseUNet':
        args.model = args.model + '_C' + str(args.channel) + 'B' + str(args.block)
        print(args.model)
        return BaseUNet(fea_channel=args.channel, deep=args.block)
    if args.model == 'BaseV7CAT':
        args.model = args.model + '_C' + str(args.channel) + 'B' + str(args.block)
        print(args.model)
        return BaseV7CAT(fea_channel=args.channel, deep=args.block)
    if args.model == 'BaseV5CATV1':
        args.model = args.model + '_C' + str(args.channel) + 'B' + str(args.block)
        print(args.model)
        return BaseV5CATV1(fea_channel=args.channel, deep=args.block)
    if args.model == 'bprn_Attention':
        args.model = args.model + '_C' + str(args.channel) + 'B' + str(args.block)
        print(args.model)
        return bprn_Attention(fea_channel=args.channel, block=args.block)
    if args.model == 'BaseV5Multi':
        args.model = args.model + '_C' + str(args.channel) + 'B' + str(args.block)
        print(args.model)
        return BaseV5Multi(fea_channel=args.channel, deep=args.block)
    if args.model == 'BaseV5MultiA':
        args.model = args.model + '_C' + str(args.channel) + 'B' + str(args.block)
        print(args.model)
        return BaseV5MultiA(fea_channel=args.channel, deep=args.block)
    if args.model == 'BaseV5MultiABi':
        args.model = args.model + '_C' + str(args.channel) + 'B' + str(args.block)
        print(args.model)
        return BaseV5MultiABi(fea_channel=args.channel, deep=args.block)
    if args.model == 'BaseV5MultiAPool':
        args.model = args.model + '_C' + str(args.channel) + 'B' + str(args.block)
        print(args.model)
        return BaseV5MultiAPool(fea_channel=args.channel, deep=args.block)
    if args.model == 'BaseV5MultiAPoolV2':
        args.model = args.model + '_C' + str(args.channel) + 'B' + str(args.block)
        print(args.model)
        return BaseV5MultiAPoolV2(fea_channel=args.channel, deep=args.block)
    if args.model == 'BaseV5MultiAPoolV3':
        args.model = args.model + '_C' + str(args.channel) + 'B' + str(args.block)
        print(args.model)
        return BaseV5MultiAPoolV3(fea_channel=args.channel, deep=args.block)
    if args.model == 'BaseV5MultiAPoolV4':
        args.model = args.model + '_C' + str(args.channel) + 'B' + str(args.block)
        print(args.model)
        return BaseV5MultiAPoolV4(fea_channel=args.channel, deep=args.block)
    if args.model == 'BaseV5MultiAPoolV5':
        args.model = args.model + '_C' + str(args.channel) + 'B' + str(args.block)
        print(args.model)
        return BaseV5MultiAPoolV5(fea_channel=args.channel, deep=args.block)




def get_data(args):
    if args.data == 'RA':
        rec_path='../data/videos/RA/'+args.sequence+'_QP'+args.qp+'_RA_rec_HM.yuv'
        label_path='../data/videos/RA/'+args.sequence+'.yuv'
        rec_file = open(rec_path, 'rb')
        label_file = open(label_path, 'rb')
        if args.frame < 4:
            rec_y, _, _ = YUVread(rec_file, [args.heigh, args.width], frame_num=args.frame+3, start_frame=0)
            label_y, _, _ = YUVread(label_file, [args.heigh, args.width], frame_num=args.frame, start_frame=0)
        else:
            rec_y, _, _ = YUVread(rec_file, [args.heigh, args.width], frame_num=args.frame, start_frame=0)
            label_y, _, _ = YUVread(label_file, [args.heigh, args.width], frame_num=args.frame, start_frame=0)
        return MDataYUV(rec_y=rec_y, label_y=label_y, nFrames=args.neighbor, width=args.width, heigh=args.heigh, totalFrames=args.frame, width_cut=args.width_cut, heigh_cut=args.heigh_cut)
    if args.data == 'pre':
        args.model = 'pre_' + args.model
        with open("/home/disk1/lilei/H265-overfit/data/derf/part1_GT_info.yaml", "r") as yaml_part1:
            yaml_obj1 = yaml.load(yaml_part1.read())
        with open("/home/disk1/lilei/H265-overfit/data/derf/part2_GT_info.yaml", "r") as yaml_part2:
            yaml_obj2 = yaml.load(yaml_part2.read())
        with open("/home/disk1/lilei/H265-overfit/data/derf/part3_GT_info.yaml", "r") as yaml_part3:
            yaml_obj3 = yaml.load(yaml_part3.read())
        return Data_codec_isolate(args.qp, yaml_obj1, yaml_obj2, yaml_obj3)




### python test.py --qp 37 --frame 200 --model BaseV5MultiAPoolV3 --width_cut 60 --heigh_cut 60 --width 1920 --heigh 1080 --data RA --sequence BasketballDrive_1920x1080_50 --save BasketballDrive_1920x1080_50_QP37-2_RA_rec_HM --resume BaseV5MultiAPoolV3_C16B2_BasketballDrive_1920x1080_50_allFrame_200_QP37_gain_0.5175_.pth

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1.0e-02, help='learning rate')
    parser.add_argument('--resume', type=str, default='none', help='resume model')
    parser.add_argument('--qp', type=str, default='22', help='training qp')
    parser.add_argument('--decay', type=int, default=3000, help='learning rate decay')
    parser.add_argument('--epoch', type=int, default=1000, help='all epoch want to run')
    parser.add_argument('--frame', type=int, default=50, help='frames need to test')
    parser.add_argument('--epoch_start', type=int, default=0, help='start epoch')
    parser.add_argument('--batch_size', type=int, default=64, help='batch')
    parser.add_argument('--model', type=str, default='BaseModel', help='model want to use')
    parser.add_argument('--width_cut', type=int, default=960, help='width cut')
    parser.add_argument('--heigh_cut', type=int, default=540, help='heigh cut')
    parser.add_argument('--width', type=int, default=1920, help='width')
    parser.add_argument('--heigh', type=int, default=1080, help='heigh')
    parser.add_argument('--data', type=str, default='RA', help='HM, H266, X265, RA, RAdelta0')
    parser.add_argument('--sequence', type=str, default='BasketballDrive_1920_1080', help='sequence name')
    parser.add_argument('--neighbor', type=int, default=6, help='neighbor frame number')
    parser.add_argument('--block', type=int, default=2, help='blocks of Model')
    parser.add_argument('--channel', type=int, default=16, help='channels of Model')
    parser.add_argument('--optim', type=str, default='Adam', help='optimizer')
    parser.add_argument('--save', type=str, default='False', help='saved name, default False')
    # parser.add_argument('--block', type=int, default=16, help='channels of Model')


    args = parser.parse_args()
    # frame_num = args.frame
    # width, heigh = args.width, args.heigh
    # sequence = args.sequence
    blocks_per_frame = args.width * args.heigh // args.width_cut // args.heigh_cut

    model_ = get_model(args)
    

    model = Model(lr=args.lr, model=model_, optim=args.optim)
    model.print_network()
    model.epoch = args.epoch_start + 1

    if args.resume != 'none':
        model.resume_network(args.resume)
        print('checkpoint {} has resumed'.format(args.resume))
    
    whole_out = torch.zeros([1, 1, args.heigh , args.width], dtype=torch.float32)
    whole_label = torch.zeros([1, 1, args.heigh , args.width], dtype=torch.float32)
    whole_input = torch.zeros([1, 1, args.heigh , args.width], dtype=torch.float32)
    uv = np.zeros(args.heigh*args.width>>1, dtype=np.uint8)
    # v = np.zeros(args.heigh*args.width>>2, dtype=np.uint8)
    DataYUV = get_data(args)

    if args.save != 'False':
        path = os.path.join('..', 'data','videos','RA', args.save+'.yuv')
        print('file will be saved in : ', path)
        f_save = open(path, 'wb+')

    psnr_gain_avg = 0
    psnr_bef_avg = 0
    psnr_aft_avg = 0
    for i in range(args.frame):
        for j in range(blocks_per_frame):
            idx = i*blocks_per_frame + j
            input_, neighbor, label_ = DataYUV[idx]
            N, C, H, W = neighbor.shape
            input_ = input_.view(1, C, H, W)
            neighbor = neighbor.view(1, N, C, H, W)
            label_ = label_.view(1, C, H, W)
            model.feed_data(input_, neighbor, label_)
            out = model.test()
            # model.log(idx, 0, len(DataYUV), args)
            h, w = idx%blocks_per_frame//(args.width//args.width_cut), idx%blocks_per_frame%(args.width//args.width_cut)
            whole_out[:, :, h*args.heigh_cut:(h+1)*args.heigh_cut, w*args.width_cut:(w+1)*args.width_cut] = out.detach().to(torch.device('cpu'))
            whole_label[:, :, h*args.heigh_cut:(h+1)*args.heigh_cut, w*args.width_cut:(w+1)*args.width_cut] = label_.detach().to(torch.device('cpu'))
            whole_input[:, :, h*args.heigh_cut:(h+1)*args.heigh_cut, w*args.width_cut:(w+1)*args.width_cut] = input_.detach().to(torch.device('cpu'))

        psnr_bef = -20 * ((whole_label - whole_input).pow(2).mean().pow(0.5)).log10()
        psnr_aft = -20 * ((whole_label - whole_out).pow(2).mean().pow(0.5)).log10()
        print('[frame {}] > psnr_bef: {:.6f} | psnr_aft: {:.6f} | psnr_gain: {:.6f}'.format(i, psnr_bef.item(), psnr_aft.item(), psnr_aft.item()-psnr_bef.item()))
        psnr_gain_avg += psnr_aft.item() - psnr_bef.item()
        psnr_aft_avg += psnr_aft.item()
        psnr_bef_avg += psnr_bef.item()
        if args.save != 'False':
            f_save.write(np.uint8(whole_out.detach()*255).tobytes())
            f_save.write(uv.tobytes())
    psnr_gain_avg = psnr_gain_avg / args.frame
    psnr_bef_avg = psnr_bef_avg / args.frame
    psnr_aft_avg = psnr_aft_avg / args.frame
    print('complate! before_avg: {:.6} | after_avg: {:.6f} | average_gain: {:.6f}'.format(psnr_bef_avg, psnr_aft_avg, psnr_gain_avg))








if __name__ == '__main__':
    main()


