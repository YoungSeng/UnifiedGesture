import os
import pdb

import numpy as np

import pdb
import yaml
from pprint import pprint
from easydict import EasyDict
import torch
from configs.parse_args import parse_args
import glob
from models.vqvae import VQVAE_ulr, VQVAE_ulr_2
import torch.nn as nn

args = parse_args()
mydevice = torch.device('cuda:' + args.gpu)

# import sys
# [sys.path.append(i) for i in ['.', '..', '../process']]
# from process.beat_data_to_lmdb import process_bvh
# from process.process_bvh import make_bvh_GENEA2020_BT
# from process.bvh_to_position import bvh_to_npy
# from process.visualize_bvh import visualize

with open(args.config) as f:
    config = yaml.safe_load(f)

for k, v in vars(args).items():
    config[k] = v
pprint(config)

config = EasyDict(config)


def main(upper, lower, root, model, save_path, prefix, model_name='vqvae_ulr_2'):

    with torch.no_grad():
        result_upper = []
        result_lower = []
        result_root = []

        x1 = torch.tensor(upper).unsqueeze(0).to(mydevice)
        x2 = torch.tensor(lower).unsqueeze(0).to(mydevice)
        x3 = torch.tensor(root).unsqueeze(0).to(mydevice)

        if model_name == 'vqvae_ulr':
            zs1, zs2, zs3 = model.module.get_code(x1.float(), x2.float(), x3.float())
        elif model_name == 'vqvae_ulr_2':
            x2 = torch.cat([x2, x3], dim=2)
            zs1, zs2 = model.module.get_code(x1.float(), x2.float())

        if model_name == 'vqvae_ulr':
            y1, y2, y3 = model.module.code_2_pose(zs1, zs2, zs3)
        elif model_name == 'vqvae_ulr_2':
            y1, y2 = model.module.code_2_pose(zs1, zs2)

        result_upper.append(y1.squeeze(0).data.cpu().numpy())
        result_lower.append(y2.squeeze(0).data.cpu().numpy())
        if model_name == 'vqvae_ulr':
            result_root.append(y3.squeeze(0).data.cpu().numpy())

    out_zs1 = np.vstack(zs1[0].squeeze(0).data.cpu().numpy())
    out_zs2 = np.vstack(zs2[0].squeeze(0).data.cpu().numpy())
    if model_name == 'vqvae_ulr':
        out_zs3 = np.vstack(zs3[0].squeeze(0).data.cpu().numpy())
        out_root = np.vstack(result_root)
    out_upper = np.vstack(result_upper)
    out_lower = np.vstack(result_lower)

    if model_name == 'vqvae_ulr':
        print(out_upper.shape, out_lower.shape, out_root.shape)
    elif model_name == 'vqvae_ulr_2':
        print(out_upper.shape, out_lower.shape)

    np.save(os.path.join(save_path, 'code_upper_' + prefix + '.npy'), out_zs1)
    np.save(os.path.join(save_path, 'code_lower_' + prefix + '.npy'), out_zs2)
    np.save(os.path.join(save_path, 'generate_upper_' + prefix + '.npy'), np.expand_dims(out_upper.transpose(), axis=0))
    np.save(os.path.join(save_path, 'generate_lower_' + prefix + '.npy'), np.expand_dims(out_lower.transpose(), axis=0))
    if model_name == 'vqvae_ulr':
        np.save(os.path.join(save_path, 'code_root_' + prefix + '.npy'), out_zs3)
        np.save(os.path.join(save_path, 'generate_root_' + prefix + '.npy'), np.expand_dims(out_root.transpose(), axis=0))
        return np.expand_dims(out_upper.transpose(), axis=0), np.expand_dims(out_lower.transpose(), axis=0), \
               np.expand_dims(out_root.transpose(), axis=0)

    elif model_name == 'vqvae_ulr_2':
        return np.expand_dims(out_upper.transpose(), axis=0), np.expand_dims(out_lower.transpose(), axis=0)


def main_2(upper, model, save_path, prefix, w_grad, code=None):
    if not w_grad:
        with torch.no_grad():
            result_upper = []
            x1 = torch.tensor(upper).unsqueeze(0).to(mydevice)
            if type(code) is np.ndarray:
                print('use code')
                zs1 = [torch.tensor(code).to(mydevice)]
                out_distance = None
            else:
                zs1, distance = model.module.get_code(x1.float())
                out_distance = np.vstack(distance[0].squeeze(0).data.cpu().numpy())

            y1 = model.module.code_2_pose(zs1)
            result_upper.append(y1.squeeze(0).data.cpu().numpy())
    else:
        result_upper = []

        x1 = upper.unsqueeze(0).to(mydevice)
        zs1, distance = model.module.get_code(x1.float())
        out_distance = torch.cat(distance, dim=0)
        y1 = model.module.code_2_pose(zs1)
        result_upper.append(y1.squeeze(0).data.cpu().numpy())

    out_zs1 = np.vstack(zs1[0].squeeze(0).data.cpu().numpy())
    out_upper = np.vstack(result_upper)
    # print(out_upper.shape)
    # np.save(os.path.join(save_path, 'code_upper_' + prefix + '.npy'), out_zs1)
    # np.save(os.path.join(save_path, 'generate_upper_' + prefix + '.npy'), np.expand_dims(out_upper.transpose(), axis=0))
    return np.expand_dims(out_upper.transpose(), axis=0), out_zs1, out_distance



def cal_distance(args, model_path, save_path, prefix, normalize=True):

    with torch.no_grad():
        # model = VQVAE(args.VQVAE, 15 * 3)  # n_joints * n_chanels
        model = VQVAE(args.VQVAE, 15 * 9)  # n_joints * n_chanels
        model = nn.DataParallel(model, device_ids=[eval(i) for i in config.no_cuda])
        model = model.to(mydevice)
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_dict'])
        model = model.eval()

        result = []
        code = []

        # for i in range(0, num_subdivision):
        for i in range(0, 512):
            # prepare pose input
            zs = [torch.tensor([i] * 30).unsqueeze(0).to(mydevice)]
            pose_sample = model.module.decode(zs).squeeze(0).data.cpu().numpy()
            code.append(zs[0].squeeze(0).data.cpu().numpy())
            result.append(pose_sample)
    # code: (512, 30)
    # poses: (512, 240, 135)
    np.savez_compressed('/mnt/nfs7/y50021900/My/codebook/BEAT_output_60fps_rotation/code.npz', code=np.array(code), poses=np.array(result), signature=np.mean(np.array(result), axis=1))


def visualize_code(model, save_path, prefix, code_upper, code_lower=None, code_root=None, model_name='vqvae_ulr_2'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with torch.no_grad():
        result_upper = []
        result_lower = []
        if model_name == 'vqvae_ulr':
            result_root = []
            code_root = torch.tensor(code_root).unsqueeze(0).to(mydevice)

        code_upper = torch.tensor(code_upper).unsqueeze(0).to(mydevice)
        if code_lower:
            code_lower = torch.tensor(code_lower).unsqueeze(0).to(mydevice)

        if model_name == 'vqvae_ulr':
            y1, y2, y3 = model.module.code_2_pose([code_upper], [code_lower], [code_root])
            result_root.append(y3.squeeze(0).data.cpu().numpy())
            out_root = np.vstack(result_root)

        elif model_name == 'vqvae_ulr_2':
            if code_lower:
                y1, y2 = model.module.code_2_pose([code_upper], [code_lower])
            else:
                y1 = model.module.code_pose([code_upper])

        if code_lower:
            result_lower.append(y2.squeeze(0).data.cpu().numpy())
        result_upper.append(y1.squeeze(0).data.cpu().numpy())


    out_upper = np.vstack(result_upper)
    if code_lower:
        out_lower = np.vstack(result_lower)

    if model_name == 'vqvae_ulr':
        print(out_upper.shape, out_lower.shape, out_root.shape)
        np.save(os.path.join(save_path, 'generate_root_' + prefix), np.expand_dims(out_root.transpose(), axis=0))
    elif model_name == 'vqvae_ulr_2':
        if code_lower:
            print(out_upper.shape, out_lower.shape)
            np.save(os.path.join(save_path, 'generate_lower_' + prefix), np.expand_dims(out_lower.transpose(), axis=0))
        else:
            print(out_upper.shape)
        np.save(os.path.join(save_path, 'generate_upper_' + prefix), np.expand_dims(out_upper.transpose(), axis=0))

    if model_name == 'vqvae_ulr':
        return np.expand_dims(out_upper.transpose(), axis=0), np.expand_dims(out_lower.transpose(), axis=0), \
           np.expand_dims(out_root.transpose(), axis=0)
    elif model_name == 'vqvae_ulr_2':
        if code_lower:
            return np.expand_dims(out_upper.transpose(), axis=0), np.expand_dims(out_lower.transpose(), axis=0)
        else:
            return np.expand_dims(out_upper.transpose(), axis=0)


def visualize_PCA_codebook(signature_path, pic_save_path):
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    signature = np.load(signature_path)['signature']
    codebook_size = signature.shape[0]
    c2s = []
    print(codebook_size)
    for i in range(codebook_size):
        c2s.append(signature[i])

    pca = PCA()
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', pca)
    ])
    Xt = pipe.fit_transform(c2s)
    plt.figure()
    plt.scatter(Xt[:, 0], Xt[:, 1], label='code')
    plt.legend()
    plt.title("PCA of Codebook")
    plt.savefig(pic_save_path + 'PCA_w_scaler.jpg')


def visualize_code_freq(code, output_path):
    from collections import Counter
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 2))
    print(code.shape)
    code = code.flatten()

    result = Counter(code)
    result_sorted = sorted(result.items(), key=lambda item: item[1], reverse=True)

    x = []
    y = []
    for d in result_sorted[:15]:
        x.append(str(d[0]))
        y.append(d[1])

    p1 = plt.bar(x[0:len(x)], y[0:len(x)])
    plt.bar_label(p1, label_type='edge')
    plt.tight_layout()
    plt.savefig(output_path + 'visualize_code_freq_top15.jpg')


def clip_code_unit(video_path, save_path):
    import subprocess
    import os

    delta_X = 4  # 每10s切割

    mark = 0

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # 获取视频的时长
    def get_length(filename):
        result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                                 "format=duration", "-of",
                                 "default=noprint_wrappers=1:nokey=1", filename],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        return float(result.stdout)

    min = int(get_length(video_path)) // 60  # file_name视频的分钟数
    second = int(get_length(video_path)) % 60  # file_name视频的秒数
    totol_sec = int(get_length(video_path))

    print(min, second, totol_sec)

    for i in range(0, totol_sec, delta_X):

        min_start = str(i // 60)
        start = str(i % 60)
        min_end = str((i + delta_X) // 60)
        end = str((i + delta_X) % 60)

        # crop video
        # 保证两位数
        if len(str(min_start)) == 1:
            min_start = '0' + str(min_start)
        if len(str(min_end)) == 1:
            min_end = '0' + str(min_end)
        if len(str(start)) == 1:
            start = '0' + str(start)
        if len(str(end)) == 1:
            end = '0' + str(end)

        # 设置保存视频的名字

        name = str(mark)
        command = 'ffmpeg -i {} -ss 00:{}:{} -to 00:{}:{} -strict -2 {}'.format(
            video_path,
            min_start, start, min_end, end,
            os.path.join(save_path, name) + '.mp4')
        mark += 1
        os.system(command)


def pick_code_freq(train_code, code_int, topk=10, txt_dataset_path=None):
    from collections import Counter

    print(train_code.shape)

    line_code_count = {}
    for line in range(len(train_code)):
        result = Counter(train_code[line])
        line_code_count[line] = result[code_int]

    dataset = np.load(txt_dataset_path, allow_pickle=True)['aux']

    print([[dataset[i[0]],i[1]] for i in sorted(line_code_count.items(), key=lambda item: item[1], reverse=True)[:topk]])


def pick_code_txt(train_code, code_int=None, txt_dataset_path=None, stride=240, num_frames_code=30, fps=60, codebook_size=512, topk=3):
    from collections import Counter

    dataset = np.load(txt_dataset_path, allow_pickle=True)
    aux = dataset['aux']
    txt = dataset['txt']

    print(train_code.shape)

    code_txt = []

    # reshape txt
    step_sz = int(stride / num_frames_code)
    stride_time = stride//fps
    for line in txt:
        tmp_code_txt = [[] for _ in range(num_frames_code)]
        while line != []:
            tmp = line.pop(0)
            tmp_code_txt[int((tmp[0] % stride_time+ (tmp[1] % stride_time if tmp[1] % stride_time != 0 else stride_time)) * 60 / 2 / step_sz)].append(tmp)      # Prevent n*stride_time from being treated as 0
        code_txt.append(tmp_code_txt)

    # init code txt
    c2txt = {}
    txt2c = {}
    for i in range(codebook_size):
        c2txt[i] = []

    for i in range(train_code.shape[0]):       # for every stride
        for j in range(num_frames_code):        # for every code
            for tmp_code_txt in code_txt[i][(j-3 if j-3 > 0 else 0):(j+4 if j+4 < num_frames_code else num_frames_code)]:
                for tmp in tmp_code_txt:
                    c2txt[train_code[i][j]].append(tmp[2])
                    if tmp[2] not in txt2c:
                        txt2c[tmp[2]] = [train_code[i][j]]
                    else:
                        txt2c[tmp[2]].append(train_code[i][j])

    for i in range(codebook_size):
        count_txt = Counter(c2txt[i])
        count_txt = sorted(count_txt.items(), key=lambda item: item[1], reverse=True)[:topk]
        c2txt[i] = count_txt
        if count_txt == []:
            del c2txt[i]

    c2txt = sorted(c2txt.items(), key=lambda item: item[1][0][1], reverse=True)[:topk]

    # print(c2txt)

    for i in txt2c.keys():
        count_code = Counter(txt2c[i])
        count_code = sorted(count_code.items(), key=lambda item: item[1], reverse=True)[:topk]
        txt2c[i] = count_code

    pdb.set_trace()
    txt2c = sorted(txt2c.items(), key=lambda item: item[1][0][1], reverse=True)[:topk]


def fold2code(path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    upper_files = sorted(glob.glob(path + "/*_upper.npy"))
    for item in upper_files:
        path = os.path.split(item)[0]
        name = os.path.split(item)[1]
        source_upper = np.load(os.path.join(path, name))[0].transpose()
        source_lower = np.load(os.path.join(path, name.replace('upper', 'lower')))[0].transpose()
        source_root = np.load(os.path.join(path, name.replace('upper', 'root')))[0].transpose()
        prefix = name.replace('_upper.npy', '')
        main(source_upper, source_lower, source_root, model, save_path, prefix, model_name='vqvae_ulr_2')



if __name__ == '__main__':
    '''
    cd codebook/
    python VisualizeCodebook.py --config=./configs/codebook.yml --train --no_cuda 2 --gpu 2
    '''

    # code_source = np.array([34, 34, 34, 34, 34, 34] * 5)      # len=30
    # visualizeCodeAndWrite(code_source=code_source, prefix='code_' + str(code_source[:6])[1:-1].replace(' ', '_'), generateGT=False)

    # code_path = './Speech2GestureMatching/output/knn_pred_wavvq.npz'
    # visualizeCodeAndWrite(code_path=code_path, prefix='knn_pred_wavvq', generateGT=False)

    # code_source = np.array([34, 34, 34, 34, 34, 34] * 5)      # len=30
    # visualizeCodeAndWrite(code_source=code_source, prefix='code_' + str(code_source[:6])[1:-1].replace(' ', '_'), generateGT=False)

    # # cal_distance(config, model_path, save_path, prefix, normalize=True)
    #
    # # signature_path = './BEAT_output_60fps_rotation/code.npz'
    # # pic_save_path = './BEAT_output_60fps_rotation/'
    # # visualize_PCA_codebook(signature_path, pic_save_path)
    #
    # train_code_path = '../data/BEAT0909/speaker_1_state_0/speaker_1_state_0_train_240_code.npz'
    # train_code = np.load(train_code_path)['code']
    # pic_save_path = './BEAT_output_60fps_rotation/'
    # # visualize_code_freq(train_code, pic_save_path)
    # txt_dataset_path = '../data/BEAT0909/speaker_1_state_0/speaker_1_state_0_train_240_txt.npz'
    # # pick_code_freq(train_code, code_int=318, topk=10, txt_dataset_path=txt_dataset_path)
    # pick_code_txt(train_code, code_int=None, txt_dataset_path=txt_dataset_path)
    #
    # # video_path = './BEAT_output_60fps_rotation/0001-122880.mkv'  # 待切割视频存储目录
    # # clip_unit_video_path = './BEAT_output_60fps_rotation/clip_unit/'
    # # clip_code_unit(video_path, clip_unit_video_path)






    model = VQVAE_ulr(config.VQVAE, 7 * 16)  # n_joints * n_chanels
    model_path = "./result/train_codebook_upper_lower_root_downsample2/codebook_checkpoint_best.bin"
    # model = VQVAE_ulr_2(config.VQVAE, config.VQVAE_2)  # n_joints * n_chanels
    # model_path = "./result/train_codebook_bvh2upper_lower_root_again/codebook_checkpoint_300.bin"
    model = nn.DataParallel(model, device_ids=[eval(i) for i in config.no_cuda])
    model = model.to(mydevice)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_dict'])
    model = model.eval()

    # test orignal latent
    source_upper = np.load('../retargeting/datasets/bvh2upper_lower_root/Trinity/Recording_006_upper.npy')[0].transpose()
    # print(source_upper.shape)       # (3164, 64)
    save_path = "./result/inference/Trinity"
    prefix = 'train_codebook_bvh2upper_lower_root_again'
    main_2(source_upper, model, save_path, prefix)


    # generate code from latent
    # ZEGGS_path = '../retargeting/datasets/bvh2upper_lower_root/ZEGGS/'
    # Trinity_path = '../retargeting/datasets/bvh2upper_lower_root/Trinity/'
    # ZEGGS_save_path = '../dataset/ZEGGS/VQVAE_result_2/ZEGGS'
    # Trinity_save_path = '../dataset/Trinity/VQVAE_result_2/Trinity/'
    # fold2code(Trinity_path, Trinity_save_path)
    # fold2code(ZEGGS_path, ZEGGS_save_path)

    # test generate code
    # test_file = '../result/inference/Recording_006.npy'
    # path = os.path.split(test_file)[0]
    # name = os.path.split(test_file)[1]
    # code_upper = np.load(os.path.join(path, 'code_upper_' + name))
    # code_lower = np.load(os.path.join(path, 'code_lower_' + name))
    # code_root = np.load(os.path.join(path, 'code_root_' + name))
    # save_path = "../result/inference/Trinity/"

    # visualize_code(model, save_path, name, code_upper, code_lower=None, code_root=None, model_name='vqvae_ulr_2')
