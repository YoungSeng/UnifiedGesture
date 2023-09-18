# to 30fps

import os
import pdb
import glob
import numpy as np
from bvh_parser import BVH_file
from bvh_writer import BVH_writer
import argparse
import sys

[sys.path.append(i) for i in ['.', '..']]
from models.IK import get_foot_vel_position, PFC_fix


def space2t(test_sentence):
    res_str = ''
    for i in range(0, len(test_sentence), 2):
        if test_sentence[i:i + 2] == '  ':
            res_str += '\t'
        else:
            res_str += test_sentence[i:]
            break
    return res_str


def downsample_process_root(trinity_path, dst_path):
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    for item in os.listdir(trinity_path):
        if not item.endswith('.bvh'):
            continue
        bvh_file = os.path.join(trinity_path, item)
        motion = []
        with open(os.path.join(dst_path, item), 'w') as output:
            with open(bvh_file, 'r') as f:
                for i, line in enumerate(f.readlines()):
                    if i == 3:
                        line = line.strip().split(' ')
                        x = float(line[-3])
                        y = float(line[-2])
                        z = float(line[-1])
                        # print('root', x, y, z)
                        line[-3] = '0.0'
                        line[-1] = '0.0'
                        line = '  ' + ' '.join(line) + '\n'
                        line = space2t(line)
                    if i < 343:
                        output.write(space2t(line))
                        continue
                    if i == 343:
                        try:
                            frames = int(line.strip().split('\t')[-1])       # \t for testing and ' ' for training
                        except:
                            frames = int(line.strip().split(' ')[-1])
                        print(frames)
                        output.write('Frames: ' + str((frames + 1)//2) + '\n')
                        continue
                    if i == 344:
                        try:
                            fps = float(line.strip().split('\t')[-1])        #
                        except:
                            fps = float(line.strip().split(' ')[-1])
                        print(fps)
                        output.write('Frame Time: ' + str(1/30.0) + '\n')
                        continue
                    else:
                        motion.append(line)
            if len(motion) != frames:
                print(len(motion), '/', frames)
                motion = motion[:frames]
            motion = motion[::2]
            assert len(motion) == (frames + 1)//2
            # fps = 1/30.0
            for i in motion:
                i = i.strip().split(' ')
                # print(i)
                i[2] = str(0- float(i[2]))
                i[0] = str(0 - float(i[0]))
                i[4] = str(0 - float(i[4]))
                i[3] = str(0 - float(i[3]))
                i[5] = str(180 + float(i[5]))
                i = ' '.join(i) + '\n'
                output.write(i)
                # break
        # break


def process_root():
    Talking_With_Hands_path = '/ceph/hdd/yangsc21/Python/My_3/trn/bvh/'
    dst_path = '/ceph/hdd/yangsc21/Python/My_3/deep-motion-editing/retargeting/datasets/Mixamo_9/Talking_With_Hands/'

    for item in os.listdir(Talking_With_Hands_path):
        bvh_file = os.path.join(Talking_With_Hands_path, item)
        motion = []
        with open(os.path.join(dst_path, item), 'w') as output:
            with open(bvh_file, 'r') as f:
                for i, line in enumerate(f.readlines()):
                    if i == 1:
                        output.write('ROOT b_root\n')
                        continue
                    if i == 3:
                        line = line.strip().split(' ')
                        z1 = float(line[-3])
                        x1 = float(line[-2])
                        y1 = float(line[-1])
                        continue
                    if i > 2 and i < 7:
                        continue
                    if i == 7:
                        line = line.strip().split(' ')
                        z2 = float(line[-3])
                        x2 = float(line[-2])
                        y2 = float(line[-1])
                        output.write('\tOFFSET {} {} {}\n'.format(z1 + z2, x1 + x2, y1 + y2))
                        continue
                    if i < 7 or 524 <= i < 527:
                        output.write(line)
                        continue
                    if i < 527:
                        output.write(space2t(line[2:]))
                        continue
                    else:
                        line = [float(j) for j in line.strip().split(' ')]
                        # print(len(line))        # 83 * 6 = 498
                        line[6] = line[6] + line[0]
                        line[7] = line[7] + line[1]
                        # line[7] = line[1]
                        line[8] = line[8] + line[2]
                        line[9] = line[9] + line[3]
                        line[10] = line[10] + line[4]
                        line[11] = line[11] + line[5]
                        line = line[6:]
                        line = [str(j) for j in line]
                        line = ' '.join(line)
                        output.write(line + '\n')
                # break


def process_T_pose(ZEGGS_path, dst_path, divide=2):

    # ZEGGS_path = "/ceph/hdd/yangsc21/Python/My_3/IJCAI-20fps/"
    # dst_path = "/ceph/hdd/yangsc21/Python/My_3/IJCAI-20fps-fix/"

    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    if ZEGGS_path.split('/')[-2] == 'ZEGGS_GT':
        GT = True
    else:
        GT = False
    for item in os.listdir(ZEGGS_path):
        if item[-4:] == '.wav':
            continue
        bvh_file = os.path.join(ZEGGS_path, item)
        motion = []
        with open(os.path.join(dst_path, item), 'w') as output:
            with open(bvh_file, 'r') as f:
                for i, line in enumerate(f.readlines()):
                    if i == 3:
                        line = line.strip().split(' ')
                        x = float(line[-3])
                        y = float(line[-2])
                        z = float(line[-1])
                        # print('root', x, y, z)
                        line[-3] = '0.0'
                        line[-1] = '0.0'
                        line = '  ' + ' '.join(line) + '\n'
                        line = space2t(line)
                    if not GT:
                        if ((51 <= i <= 75 or 87 <= i <= 99 or 111 <= i <= 123 or 135 <= i <= 147 or 159 <= i <= 171) and i % 4 ==3):     # RightArm OFFSET
                            line = line.strip('\n').split(' ')
                            x = float(line[-3])
                            y = float(line[-2])
                            z = float(line[-1])
                            line[-3] = str(0.0 - y)
                            line[-2] = str(x)
                            line[-1] = str(z)
                            line = ' '.join(line) + '\n'
                            line = space2t(line)
                        if ((209 <= i <= 233 or 245 <= i <= 257 or 269 <= i <= 281 or 293 <= i <= 305 or 317 <= i <= 329) and i % 4 ==1):     # RightArm OFFSET
                            line = line.strip('\n').split(' ')
                            x = float(line[-3])
                            y = float(line[-2])
                            z = float(line[-1])
                            line[-3] = str(y)
                            line[-2] = str(0.0 - x)
                            line[-1] = str(z)
                            line = ' '.join(line) + '\n'
                            line = space2t(line)
                        if i < 461:
                            output.write(space2t(line))
                            continue
                        if i == 461:
                            frames = int(line.strip().split(' ')[-1])  # \t for testing and ' ' for training
                            print(frames)
                            output.write('Frames: ' + str((frames + 1) // divide) + '\n')
                            continue
                        if i == 462:
                            fps = float(line.strip().split(' ')[-1])  #
                            print(fps)
                            if divide == 2:
                                output.write('Frame Time: ' + str(1 / 30.0) + '\n')
                            elif divide == 1:
                                output.write('Frame Time: ' + str(1 / 60.0) + '\n')
                            continue
                        else:
                            motion.append(line)
                    else:
                        if i < 566:
                            output.write(space2t(line))
                            continue
                        if i == 566:
                            frames = int(line.strip().split(' ')[-1])
                            print(frames)
                            output.write('Frames: ' + str((frames + 1) // 2) + '\n')
                            continue
                        if i == 567:
                            fps = float(line.strip().split(' ')[-1])  #
                            print(fps)
                            output.write('Frame Time: ' + str(1 / 30.0) + '\n')
                            continue
                        else:
                            motion.append(line)
            if len(motion) != frames:
                print(len(motion), '/', frames)
                motion = motion[:frames]
            motion = motion[::divide]
            if divide == 2:
                assert len(motion) == (frames + 1) // divide     # fps = 1/30.0
            elif divide == 1:
                assert len(motion) == (frames) // divide
            for i in motion:
                i = i.strip().split(' ')

                z = float(i[30])
                y = float(i[31])
                x = float(i[32])
                i[30] = str(z - 90.0)
                i[31] = str(x)
                i[32] = str(0.0 - y)

                # print(z, y, x, i[30], i[31], i[32])      # 92.418716 -0.394881 9.707915

                for j in range(11 * 3, 35 * 3, 3):
                    z = float(i[j])
                    y = float(i[j+1])
                    x = float(i[j+2])
                    i[j+1] = str(x)
                    i[j+2] = str(0.0 - y)
                    # print(i[j], i[j+1], i[j+2])

                z = float(i[36 * 3])
                y = float(i[36 * 3 + 1])
                x = float(i[36 * 3 + 2])
                i[36 * 3] = str(z + 90.0)
                i[36 * 3 + 1] = str(0.0 - x)
                i[36 * 3 + 2] = str(y)

                # print(z, y, x, i[36 * 3], i[36 * 3 + 1], i[36 * 3 + 2])     # -90.370065 0.051974 5.208683

                for j in range(37 * 3, 61 * 3, 3):
                    y = float(i[j+1])
                    x = float(i[j+2])
                    i[j+1] = str(0.0 - x)
                    i[j+2] = str(y)

                # GT
                # print(len(i), i[30:30+6],       # GT 576 (96*6) ['-2.624908', '11.175079', '-0.000479', '2.486827', '9.713857', '0.398656']
                #       i[30+6:30+12],        #  ['-10.435041', '-0.000056', '-0.000023', '68.264312', '4.249098', '33.534958']
                #       i[30+12:30+18],       # ['-29.627481', '-0.000870', '-0.000919', '-5.192122', '21.799514', '-13.415904']
                #       i[38 * 6:39 * 6],     # ['2.625969', '11.175263', '-0.000499', '-0.372213', '-5.206465', '0.049018']
                #       i[39 * 6:40 * 6],     # ['10.813481', '-0.000380', '0.000377', '-69.326304', '7.144804', '19.466830']
                #       i[40 * 6:41 * 6]      # ['28.725275', '-0.001194', '-0.000643', '6.746904', '-40.325830', '-11.368573']
                #       )

                i = ' '.join(i) + '\n'
                output.write(i)
                # break
        # break


def get_height(file):
    file = BVH_file(file)
    return file.get_height()


def process_foot_contact(source_path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    root_motions = sorted(glob.glob(source_path + "/*.bvh"))
    for item in root_motions:
        name = os.path.split(item)[1]
        print(name)
        result = get_foot_vel_position(item, get_height(item))  # (2065, 4)
        np.save(os.path.join(save_path, name), result)


if __name__ == '__main__':
    '''
    cd datasets/
    python process_bvh.py
    '''

    parser = argparse.ArgumentParser(description='process')
    parser.add_argument('--step', type=str, default="IK")
    parser.add_argument('--source_path', type=str, default='./Mixamo_new_2/ZEGGS/')
    parser.add_argument('--save_path', type=str, default='./Mixamo_new_2/ZEGGS_aux/')
    parser.add_argument('--ref_bvh', type=str, default="./Mixamo_new_2/ZEGGS/067_Speech_2_x_1_0.bvh")
    args = parser.parse_args()

    if args.step == "IK":

        # bvh_path = "/ceph/hdd/yangsc21/Python/My_3/deep-motion-editing/result/inference/Trinity/Recording_006_minibatch_240_[0, 0, 1, 0, 0, 0, 0]_123456.bvh"
        # # bvh_path = "/ceph/hdd/yangsc21/Python/My_3/deep-motion-editing/retargeting/datasets/Mixamo_new_2/Trinity/Recording_001.bvh"
        ref_bvh = args.ref_bvh
        # bvh_path = "/ceph/hdd/yangsc21/Python/My_3/deep-motion-editing/diffusion_latent/result_256_seed_4_aux_bvh/TestSeq001_final.bvh"
        # PFC_fix(bvh_path, get_height(ref_bvh))

        # source_path = "/ceph/hdd/yangsc21/Python/My_3/deep-motion-editing/diffusion_latent/result_model3_128_bvh/"
        # target_path = ''

        source_path = args.source_path
        for item in os.listdir(source_path):
            print(item)
            if not item.endswith('_fix.bvh'):
                bvh_path = os.path.join(source_path, item)
                PFC_fix(bvh_path, get_height(ref_bvh))

    if args.step == "Trinity":
        '''
        python process_bvh.py --step Trinity --source_path "../../dataset/Trinity/test_motion/" --save_path "../../dataset/Trinity/test_motion_downsample/"
        '''
        downsample_process_root(args.source_path, args.save_path)      # Trinity
        # process_root()      # Talking_With_Hands
    if args.step == 'ZEGGS':
        process_T_pose(args.source_path, args.save_path, divide=2)        # ZEGGS

    if args.step == 'foot_contact':
        process_foot_contact(source_path=args.source_path, save_path=args.save_path)





