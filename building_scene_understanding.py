import random

import numpy as np
import torch
from pypcd import pypcd
from gridencoder import GridEncoder
import bagpy
from bagpy import bagreader
import json
import math
from random import *
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import time
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

SUM_FREQ = 20
class Logger:
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None
        self.last_time = time.time()
    def _print_training_status(self):
        now_time = time.time()
        metrics_data = [self.running_loss[k]/SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        time_str = ("time = %.2f, "%(now_time-self.last_time))
        data = open("records.txt", 'a')
        # print the training status
        print(training_str + metrics_str +time_str)
        data.write(training_str + metrics_str+"\n")
        data.close()
        self.last_time = now_time
        if self.writer is None:
            self.writer = SummaryWriter()

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()

#[0, -46.08, -3, 92.16, 46.08, 1]
def point_cloud_2_birdseye(points,
                           res=0.05,
                           side_range=(-46.08, 46.08),  # left-most to right-most
                           fwd_range=(0., 92.16),  # back-most to forward-most
                           height_range=(-3., 1.),  # bottom-most to upper-most
                           ):
    """ Creates an 2D birds eye view representation of the point cloud data.

    Args:
        points:     (numpy array)
                    N rows of points data
                    Each point should be specified by at least 3 elements x,y,z
        res:        (float)
                    Desired resolution in metres to use. Each output pixel will
                    represent an square region res x res in size.
        side_range: (tuple of two floats)
                    (-left, right) in metres
                    left and right limits of rectangle to look at.
        fwd_range:  (tuple of two floats)
                    (-behind, front) in metres
                    back and front limits of rectangle to look at.
        height_range: (tuple of two floats)
                    (min, max) heights (in metres) relative to the origin.
                    All height values will be clipped to this min and max value,
                    such that anything below min will be truncated to min, and
                    the same for values above max.
    Returns:
        2D numpy array representing an image of the birds eye view.
    """
    # EXTRACT THE POINTS FOR EACH AXIS
    x_points = points[:, 0]
    y_points = points[:, 1]
    z_points = points[:, 2]

    # FILTER - To return only indices of points within desired cube
    # Three filters for: Front-to-back, side-to-side, and height ranges
    # Note left side is positive y axis in LIDAR coordinates
    f_filt = np.logical_and((x_points > fwd_range[0]), (x_points < fwd_range[1]))
    s_filt = np.logical_and((y_points > -side_range[1]), (y_points < -side_range[0]))
    filter = np.logical_and(f_filt, s_filt)
    indices = np.argwhere(filter).flatten()

    # KEEPERS
    x_points = x_points[indices]
    y_points = y_points[indices]
    z_points = z_points[indices]

    # CONVERT TO PIXEL POSITION VALUES - Based on resolution
    x_img = (-y_points / res).astype(np.int32)  # x axis is -y in LIDAR
    y_img = (-x_points / res).astype(np.int32)  # y axis is -x in LIDAR

    # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
    # floor & ceil used to prevent anything being rounded to below 0 after shift
    x_img -= int(np.floor(side_range[0] / res))
    y_img += int(np.ceil(fwd_range[1] / res))

    # CLIP HEIGHT VALUES - to between min and max heights
    pixel_values = np.clip(a=z_points,
                           a_min=height_range[0],
                           a_max=height_range[1])

    # RESCALE THE HEIGHT VALUES - to be between the range 0-255
    '''
    pixel_values = scale_to_255(pixel_values,
                                min=height_range[0],
                                max=height_range[1])'''

    # INITIALIZE EMPTY ARRAY - of the dimensions we want
    x_max = 2 + int((side_range[1] - side_range[0]) / res)
    y_max = 2 + int((fwd_range[1] - fwd_range[0]) / res)
    im = np.zeros([y_max, x_max], dtype=np.uint8)

    # FILL PIXEL VALUES IN IMAGE ARRAY
    im[y_img, x_img] = pixel_values

    return im

def read_pcd(pcd_path):
    pcd = pypcd.PointCloud.from_path(pcd_path)
    pcd_np_points = np.zeros((pcd.points, 4), dtype=np.float32)
    pcd_np_points[:, 0] = np.transpose(pcd.pc_data['x'])
    pcd_np_points[:, 1] = np.transpose(pcd.pc_data['y'])
    pcd_np_points[:, 2] = np.transpose(pcd.pc_data['z'])
    pcd_np_points[:, 3] = np.transpose(pcd.pc_data['intensity'])
    del_index = np.where(np.isnan(pcd_np_points))[0]
    pcd_np_points = np.delete(pcd_np_points, del_index, axis=0)

    del_index_ = (pcd_np_points[:, 0] >= -46.08) * (pcd_np_points[:, 0] <= 46.08) * (pcd_np_points[:, 1] >= -46.08) * (
                pcd_np_points[:, 1] <= 46.08) * (pcd_np_points[:, 2] >= -5) * (pcd_np_points[:, 2] <= -0.5)
    '''
    del_index_ = (pcd_np_points[:, 0] >= 0) * (pcd_np_points[:, 0] <= 92.16) * (pcd_np_points[:, 1] >= -46.08) * (
            pcd_np_points[:, 1] <= 46.08) * (pcd_np_points[:, 2] >= -3) * (pcd_np_points[:, 2] <= 1)'''

    pcd_np_points = pcd_np_points[del_index_ == True]
    pcd_ts_points = torch.tensor(pcd_np_points)
    #pcd_ts_points = pcd_ts_points[:10000]
    return pcd_ts_points

def get_pcd(pcd_path, split):

    pcd = []
    idx = 0
    while idx < len(split):
        start = int(split[idx])
        end = int(split[idx+1])
        idx+=2
        for i in range(start, end):
            pcd.append(read_pcd(pcd_path + '{:0>6d}'.format(i) + '.pcd'))
    print("read finish")

    lenth = len(pcd)
    interval = lenth / 30
    pcd_return = []
    for i in range(0, lenth, math.ceil(interval)):
        pcd_return.append(pcd[i].cuda())

    return pcd_return

def get_fsk_pcd():
    fsk_pth = '/home/tools/fsk/'
    pcd = []
    for i in range(0, 971, 32):
        pcd.append(read_pcd(fsk_pth + str(i) + '.pcd').cuda())
    print("read finish")
    '''
    lenth = len(pcd)
    interval = lenth / 30
    pcd_return = []
    for i in range(0, lenth, math.ceil(interval)):
        pcd_return.append(pcd[i].cuda())'''

    return pcd
'''
def sample_lines(line, N):
    d_x = torch.linspace(0, line[0], N).cuda()
    t = torch.rand(N).cuda() * line[0] / N
    d_x[:99] = d_x[:99] + t[:99]
    #d_x = torch.tensor([uniform(0, line[0]) for i in range(N)]).cuda()
    d_y = d_x / line[0] * line[1]
    d_z = d_x / line[0] * line[2]
    sample_points = torch.cat((d_x.unsqueeze(1), d_y.unsqueeze(1), d_z.unsqueeze(1)), dim=1)
    
    for i in range(N):
        sample_points.append(torch.tensor([d_x[i], d_y[i], d_z[i]]).unsqueeze(0).cuda())
    sample_points.append(torch.tensor([line[0], line[1], line[2]]).unsqueeze(0).cuda())
    sample_points = torch.cat([p for p in sample_points])
    return sample_points'''

def sample_pcd(pcd, N):
    #ind = sample(range(0,len(pcd) - 1), 10)#随机选线
    ind = torch.randint(0, len(pcd), (10000,)).cuda()#在所有雷达帧里面做随机2W条
    lines = pcd[ind]#记录线
    #sample_pcd = torch.cat([sample_lines(line, N).unsqueeze(0) for line in lines])


    distance = torch.norm(lines, 2, dim=1)
    #dis_x = lines[:, 0] / distance * distance * 0.02  #左右5cm
    dis_x = lines[:, 0] * 0.02

    broad = torch.cat((torch.arange(0, 500), torch.arange(0, 500))).repeat(10000, 1).cuda()
    broad_1 = ((lines[:, 0] - dis_x) / 500).repeat(500, 1).transpose(0, 1)
    broad_2 = (dis_x / 250).repeat(500, 1).transpose(0, 1)
    broad_3 = torch.cat((broad_1, broad_2), dim = 1)
    broad = broad * broad_3
    y = (lines[:, 0] - dis_x).repeat(500, 1).transpose(0, 1)
    broad[:, 500:] = broad[:, 500:] + y

    #broad = torch.cat((torch.linspace(0, (x - dis_x) * 0.95, 20), torch.linspace(80, 119.5, 80))).repeat(20000, 1).cuda()
    #broad = torch.cat((torch.linspace(0, 76, 20), torch.linspace(80, 119.5, 80))).repeat(20000, 1).cuda()

    t = torch.rand(10000, N).cuda()
    t = t * broad_3
    t[:, 749] = dis_x / 250

    x = lines[:, 0].repeat(N, 1).transpose(0, 1)
    y = lines[:, 1].repeat(N, 1).transpose(0, 1)
    z = lines[:, 2].repeat(N, 1).transpose(0, 1)

    sample_x = broad + t
    sample_y = sample_x / x * y
    sample_z = sample_x / x * z
    #sample_in = torch.zeros(90000,200).cuda()
    #sample_in[:, 189] = lines[:, 3]

    dis_ = (lines[:, 0] / distance).repeat(500, 1).transpose(0, 1)
    gt_o_1 = torch.zeros(500).repeat(10000, 1).cuda()
    gt_o_2 = normal((sample_x[:, 500:] - lines[:, 0].repeat(500, 1).transpose(0, 1)) / dis_)
    gt_o = torch.cat((gt_o_1, gt_o_2), dim=1)

    '''
    gt_o = torch.zeros(1000).repeat(10000, 1).cuda()
    gt_o[:, 999] = 1'''
    #gt_o = torch.cat((gt_o.unsqueeze(2), sample_in.unsqueeze(2)), dim = 2)

    #sample_x = ((sample_x + 48) / 96 - 0.5) * 2
    sample_x = (sample_x / 96 - 0.5) * 2
    sample_y = ((sample_y + 48) / 96 - 0.5) * 2
    sample_z = ((sample_z + 3.5) / 4 - 0.5) * 2
    #sample_z = ((sample_z + 5.5) / 6 - 0.5) * 2

    '''
    # 创建一个画布figure，然后在这个画布上加各种元素。
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(sample_x[7, :].detach().cpu().numpy(), sample_y[7, :].detach().cpu().numpy(), sample_z[7, :].detach().cpu().numpy())
    plt.show()
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for i in range(1000):
        ax.scatter(sample_x[9+i, :][gt_o[9+i, :]>0].detach().cpu().numpy(), sample_y[9+i, :][gt_o[9+i, :]>0].detach().cpu().numpy(), sample_z[9+i, :][gt_o[9+i, :]>0].detach().cpu().numpy())
    
    plt.show()'''
    sample_pcd = torch.cat((sample_x.unsqueeze(2), sample_y.unsqueeze(2), sample_z.unsqueeze(2)), dim = 2)

    '''
    for line in lines:
        sample_pcd.append(sample_lines(line, N))#线采点
    sample_pcd = torch.cat(([s for s in sample_pcd]))'''
    '''
    line = torch.cat([ind.unsqueeze(1) / 100, ind.unsqueeze(1) % 200], dim=1)
    line[:, 0] = (line[:, 0] / (pcd.shape[0] / 100) -0.5) * 2
    line[:, 1] = (line[:, 1] / 200 - 0.5) * 2'''

    width =math.ceil(pcd.shape[0] / 100)
    line = torch.cat([(ind.unsqueeze(1) / width).ceil(), ind.unsqueeze(1) % width], dim=1)
    line[:, 0] = (line[:, 0] / 100 - 0.5) * 2
    line[:, 1] = (line[:, 1] / width - 0.5) * 2
    '''
    lines[:, 0] = (lines[:, 0] / 96 -0.5) * 2
    lines[:, 1] = ((lines[:, 1] + 48) / 96 -0.5) * 2
    lines[:, 2] = ((lines[:, 2] + 3) / 4.5 -0.5) * 2'''
    # i,j   这个是第几幅    j这个是这一幅中的那一根？  20   3-4w    (N/20-0.5)*2   (M/5w-0.5)*2
    return line, sample_pcd, gt_o

def normal(x):
    theta = 1 / math.sqrt(2 * math.pi)
    y = torch.exp(-x**2 / (2 * theta)**2)

    return y


class NGP(nn.Module):
    def __init__(self, input_dim=3, output_dim = 1):
        super(NGP, self).__init__()
        self.Confencoder = GridEncoder(input_dim=input_dim,  # 输入的维度
                                       num_levels=12,  # 这个应该是那个L的参数
                                       level_dim=2,  # 这个是F
                                       base_resolution=16,  # Nmin
                                       desired_resolution=8192,  # Nmax
                                       log2_hashmap_size=21,  # 这个是图里面的T
                                       gridtype='hash',
                                       align_corners=False).cuda()

        self.last_dim = self.Confencoder.output_dim
        self.conf_layer = nn.Sequential(nn.Linear(self.last_dim, 64),
                                        nn.ReLU(),
                                        nn.Linear(64, output_dim))
    def forward(self,pcd):
        featureconf = self.Confencoder(pcd)
        confsample = self.conf_layer(featureconf)

        return confsample


class INGP(nn.Module):
    def __init__(self):
        super(INGP, self).__init__()
        self.Confidence = NGP(input_dim=2, output_dim=1)
        self.Opacity = NGP(input_dim=3, output_dim=1)
    def forward(self, line, line_points, gt_o): #单独计算一个采样
        #loss_all = 0
        #gt_o = torch.cat((torch.zeros(99),torch.ones(1))).cuda()


         #一帧中每一条线采样99+终点 100个点
        #lines 20000,3  line_points 20000 100 3

        opacity = self.Opacity(line_points).squeeze(2)
        confidence = self.Confidence(line).squeeze(1)

        loss1 = (opacity - gt_o).abs()
        loss1[:, :500] *= (4/3)
        loss1[:, 500:] *= (2/3)

        loss = (-confidence).exp() * loss1.mean(1) + confidence
        #loss = (-confidence).exp() * (opacity - gt_o).abs().sum(1) + confidence
        #loss = (opacity - gt_o).abs().mean(1)
        #loss_all += loss.mean()
        '''
        #计算loss
        loss_line = 0
        for i in range(20000): #计算每条线的loss
            loss_i = math.exp(-confidence[i][0]) * abs(opacity.squeeze(1)[i*100:i*100+100] - gt_o).sum() + confidence[i][0]
            #神奇的loss,但有nan 4.23 问题在哪？
            loss_line += loss_i / 100
        lossz_all += loss_line / 20000'''

        #print(loss_all)
        return loss.mean()

    def test(self, pcd):
        #lines, line_points, gt_o = sample_pcd(pcd, 100)
        pcd_all = torch.clone(pcd)

        pcd[:, 0] = (pcd[:, 0] / 96 - 0.5) * 2
        pcd[:, 1] = ((pcd[:, 1] + 48) / 96 - 0.5) * 2
        pcd[:, 2] = ((pcd[:, 2] + 3) / 4.5 - 0.5) * 2

        '''
        pcd[:, 0] = ((pcd[:, 0] + 48) / 96 - 0.5) * 2
        pcd[:, 1] = ((pcd[:, 1] + 48) / 96 - 0.5) * 2
        pcd[:, 2] = ((pcd[:, 2] + 5.5) / 6 - 0.5) * 2'''

        ind = torch.arange(pcd.size(0))

        width = math.ceil(pcd.shape[0] / 100)
        line = torch.cat([(ind.unsqueeze(1) / width).ceil(), ind.unsqueeze(1) % width], dim=1)
        line[:, 0] = (line[:, 0] / 100 - 0.5) * 2
        line[:, 1] = (line[:, 1] / width - 0.5) * 2

        opacity = self.Opacity(pcd).squeeze(1)
        confidence = self.Confidence(line.cuda()).squeeze(1)
        '''
        for i in range(0,100):
            plt.plot(opacity1.detach().cpu().numpy()[1500+i,:])
        plt.show()'''
        pcd_in = torch.cat([pcd_all, confidence.unsqueeze(1)], dim=1)


        print(1)

    def separate(self,pcd):

        pcd_all = torch.clone(pcd)
        '''
        pcd[:, 0] = (pcd[:, 0] / 96 - 0.5) * 2
        pcd[:, 1] = ((pcd[:, 1] + 48) / 96 - 0.5) * 2
        pcd[:, 2] = ((pcd[:, 2] + 3) / 4.5 - 0.5) * 2

        '''
        pcd[:, 0] = ((pcd[:, 0] + 48) / 96 - 0.5) * 2
        pcd[:, 1] = ((pcd[:, 1] + 48) / 96 - 0.5) * 2
        pcd[:, 2] = ((pcd[:, 2] + 5.5) / 6 - 0.5) * 2

        opacity = self.Opacity(pcd[:, :3]).squeeze(1)

        backgroud = pcd_all[opacity>0.5]
        dynamic = pcd_all[opacity<=0.5]
        #return的是点云

        return backgroud, dynamic




pcd_path = '/new_data/DAIR/cooperative-vehicle-infrastructure/infrastructure-side/velodyne/'
'''split_list = [
    ['015596', '015775', '015786', '015975', '015986', '016165', '016176', '016365', '016376', '016495',
     '016502', '016575', '016586', '016795', '016806', '017056', '017646', '017855', '017866', '018036',
     '018047', '018166', '018173', '018236', '018246', '018415', '018426', '018625', '018636', '018805',
     '018816', '018935', '018938', '018955', '019546', '019666', '019676', '019805', '019816', '019955']]
sence = get_pcd(pcd_path, split_list[0])
sence = torch.cat([s[0:-1] for s in sence])'''

'''
sence = get_fsk_pcd()
sence = torch.cat([s[0:-1] for s in sence])'''

def train():

    ingp = INGP().cuda()
    ingp.train()

    optimizer = optim.AdamW(ingp.parameters(), lr=0.0005, weight_decay=0.0001, eps=1e-8)#0.000125

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, 0.000125, 40000,
                                              pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    scaler = GradScaler(enabled=False)
    logger = Logger(ingp, scheduler)

    #for i in range(0, len(split_list)):
    #sence = get_pcd(pcd_path, split_list[0])  #第一个场景试试

    for i in range(40000):
        line, line_points, gt_o = sample_pcd(sence, 1000)
        loss_all = ingp(line, line_points, gt_o)

        scaler.scale(loss_all).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(ingp.parameters(), 1.0)
        scaler.step(optimizer)
        scheduler.step()
        scaler.update()

        metrics = {
            'loss_all': loss_all.item()
        }

        logger.push(metrics)

        if i % 40000 == 39999:
            torch.save(ingp.state_dict(), "/new_data/ingp/dair_sence7" + ".pth")

def eval():

    ingp = INGP().cuda()
    ingp.eval()

    pretrained_dict = torch.load('/new_data/ingp/ingp_handlez_119999.pth')
    ingp.load_state_dict(pretrained_dict, strict=False)

    '''
    for pcd in sence:
        plt.imshow(point_cloud_2_birdseye(np.array(pcd.cpu())))
        plt.show()
        plt.close()'''

    ingp.test(sence)

def sepreate():
    split = [
        ['000009', '000092', '000103', '000206', '000217', '000218', '000238', '000249', '000260', '000409',
         '000419', '000461', '000480', '000512', '000885', '000906', '000927', '000969', '000980', '001073',
         '001084', '001146', '001157', '001315', '001317', '001492', '001511', '001574'],
        ['002133', '002203', '002206', '002354', '002355', '002567', '002568', '002598', '002606', '002739',
         '002740', '002921', '002922', '002983', '002989', '003040', '003041', '003171'],
        ['003805', '003810', '003823', '004004', '004015', '004216', '004227', '004429', '004440', '004551',
         '004556', '004672'],
        ['004993', '005016', '005018', '005220', '005225', '005402', '005454', '005473', '005474', '005544',
         '005616', '005757', '005758', '005979', '005983', '006081', '006085', '006142', '006688', '006901',
         '006902', '007377', '007410', '007621'],
        ['007703', '007732', '007750', '007823', '008348', '008529', '008540', '008620', '008636', '008730',
         '008741', '008892', '008903', '009094', '009105', '009145', '009165', '009296', '009307', '009518',
         '009529', '009700', '010258', '010398', '010409', '010459', '010477', '010570', '010583', '010702',
         '010713', '010863', '010874', '011014', '011025', '011215', '011243', '011407', '012014', '012175',
         '012186', '012336'],
        ['012351', '012498', '012501', '012563', '012568', '012654', '012658', '012745', '012762', '012958',
         '013594', '013813', '013818', '013832', '013834', '014037', '014101', '014250', '014251', '014404',
         '014409', '014516', '014613', '014810', '014816', '015033'],
        ['015596', '015775', '015786', '015975', '015986', '016165', '016176', '016365', '016376', '016495',
         '016502', '016575', '016586', '016795', '016806', '017056', '017646', '017855', '017866', '018036',
         '018047', '018166', '018173', '018236', '018246', '018415', '018426', '018625', '018636', '018805',
         '018816', '018935', '018938', '018955', '019546', '019666', '019676', '019805', '019816', '019955']]

    ingp = INGP().cuda()
    ingp.eval()

    pretrained_dict = torch.load('/new_data/ingp/dair_sence7.pth')
    ingp.load_state_dict(pretrained_dict, strict=False)

    idx = 0
    s = 6
    while idx < len(split[s]):
        start = int(split[s][idx])
        end = int(split[s][idx+1])
        idx+=2
        for i in range(start, end+1):
            pcd = read_pcd(pcd_path + '{:0>6d}'.format(i) + '.pcd').cuda()
            #t1 = time.time()
            _, fore = ingp.separate(pcd)
            #t2 = time.time()
            #print(t2-t1)
            fore = fore[fore[:, 2] > -1.65]
            del_index = (fore[:, 0] > 50) * (fore[:, 2]>-0.5)
            del_index = ~del_index
            fore = fore[del_index]

            fore_pcd = tensor_to_pcd(fore)
            pypcd.save_point_cloud(fore_pcd, '/new_data/ingp/dair_dynamic_pcd/' + '{:0>6d}'.format(i) + '_dynamic.pcd')
            fore_np = np.array(fore.cpu())
            fore_np.tofile(pcd_path + '{:0>6d}'.format(i) + '_.bin')

def sepreate_fsk():

    ingp = INGP().cuda()
    ingp.eval()

    pretrained_dict = torch.load('/new_data/ingp/ingp_fsk_119999.pth')
    ingp.load_state_dict(pretrained_dict, strict=False)

    fsk_pth = '/home/fsk/'

    for i in range(0, 971):
        pcd = (read_pcd(fsk_pth + str(i) + '.pcd').cuda())

        #t1 = time.time()
        _, fore = ingp.separate(pcd)
        #t2 = time.time()
        #print(t2-t1)
        #fore = fore[fore[:, 2] > -1.8]
        #del_index = (fore[:, 2]>-0.5)
        #del_index = ~del_index
        #fore = fore[del_index]

        fore_pcd = tensor_to_pcd(fore)
        pypcd.save_point_cloud(fore_pcd, '/new_data/ingp/fsk_dynamic_pcd/' + str(i) + '_dynamic.pcd')
        #fore_np = np.array(fore.cpu())
        #fore_np.tofile(pcd_path + '{:0>6d}'.format(i) + '_.bin')


def tensor_to_pcd(tensor_points):
    size_float = 4
    list_pcd = []
    for ii in range(tensor_points.shape[0]):
        if tensor_points.shape[1] == 4:
            x, y, z, intensity = tensor_points[ii, 0], tensor_points[ii, 1], tensor_points[ii, 2], tensor_points[
                ii, 3]
        else:
            x, y, z = tensor_points[ii, 0], tensor_points[ii, 1], tensor_points[ii, 2]
            intensity = 1.0
        list_pcd.append((x, y, z, intensity))

    dt = np.dtype([('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('intensity', 'f4')])
    np_pcd = np.array(list_pcd, dtype=dt)

    new_metadata = {}
    new_metadata['version'] = '0.7'
    new_metadata['fields'] = ['x', 'y', 'z', 'intensity']
    new_metadata['size'] = [4, 4, 4, 4]
    new_metadata['type'] = ['F', 'F', 'F', 'F']
    new_metadata['count'] = [1, 1, 1, 1]
    new_metadata['width'] = len(np_pcd)
    new_metadata['height'] = 1
    new_metadata['viewpoint'] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    new_metadata['points'] = len(np_pcd)
    new_metadata['data'] = 'binary'
    pc_save = pypcd.PointCloud(new_metadata, np_pcd)

    return pc_save

def show_pcd():
    #import open3d as o3d

    try:
        import open3d as o3d
        from visual_utils import open3d_vis_utils as V
        OPEN3D_FLAG = True
    except:
        import mayavi.mlab as mlab
        from visual_utils import visualize_utils as V
        OPEN3D_FLAG = False

    fsk_pth = '/home/fsk/'
    dynamic_pth = '/new_data/ingp/fsk_dynamic_pcd/'

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)

    # draw origin
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    vis.add_geometry(axis_pcd)

    for i in range(0, 971):
        #pcd = (read_pcd(dynamic_pth + str(i) + '_dynamic.pcd').cuda())

        pcd = (read_pcd('/home/fsk95.pcd').cuda())

        #pcd = o3d.io.read_point_cloud(fsk_pth + str(i) + '.pcd')

        V.draw_scenes(vis = vis, points = pcd)

        if not OPEN3D_FLAG:
            mlab.show(stop=True)
        #point = vis.get_picked_points()
        #vis.destroy_window()
        #print(point[0].index, np.asarray(point[0].coord))


if __name__ == '__main__':
    #print("120000")
    #eval()
    #train()
    #sepreate()
    #sepreate_fsk()
    show_pcd()
