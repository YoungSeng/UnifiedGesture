import os
import pdb

import torch
from models import create_model
from datasets import create_dataset
import option_parser
from os.path import join as pjoin
import numpy as np
from datasets.bvh_parser import BVH_file

def eval_prepare(args):
    character = []
    file_id = []
    character_names = []
    character_names.append(args.input_bvh.split('/')[-2])
    character_names.append(args.target_bvh.split('/')[-2])
    # if args.test_type == 'intra':
    #     if character_names[0].endswith('_m'):
    #         character = [['BigVegas', 'BigVegas'], character_names]
    #         file_id = [[0, 0], [args.input_bvh, args.input_bvh]]
    #         src_id = 1
    #     else:
    #         character = [character_names, ['Goblin_m', 'Goblin_m']]
    #         file_id = [[args.input_bvh, args.input_bvh], [0, 0]]
    #         src_id = 0
    if args.test_type == 'cross':
        if character_names[0]=='ZEGGS':
            character = [[character_names[1]], [character_names[0]]]
            # file_id = [[0], [args.input_bvh]]
            file_id = [[args.target_bvh], [args.input_bvh]]      # ,[0]        modify 2
            src_id = 1
        else:
            character = [[character_names[0]], [character_names[1]]]
            file_id = [[args.input_bvh], [args.target_bvh]]      # ,[0]        modify 2
            # file_id = [[args.input_bvh], [0]]
            src_id = 0
    else:
        raise Exception('Unknown test type')
    return character, file_id, src_id


def recover_space(file):
    l = file.split('/')
    l[-1] = l[-1].replace('_', ' ')
    return '/'.join(l)


def demo_model(input_bvh, target_bvh, test_type, output_filename):
    parser = option_parser.get_parser()
    args = parser.parse_args()
    args.input_bvh = input_bvh
    args.target_bvh = target_bvh
    args.test_type = test_type
    args.output_filename = output_filename
    character_names, file_id, src_id = eval_prepare(args)
    print(character_names, input_bvh)
    test_device = args.cuda_device
    eval_seq = args.eval_seq

    para_path = os.path.join(args.save_dir, 'para.txt')
    with open(para_path, 'r') as para_file:
        argv_ = para_file.readline().split()[1:]
        args = option_parser.get_parser().parse_args(argv_)

    args.cuda_device = test_device if torch.cuda.is_available() else 'cpu'
    args.is_train = False  #
    args.rotation = 'quaternion'
    args.eval_seq = eval_seq

    dataset = create_dataset(args, character_names)
    model = create_model(args, character_names, dataset)
    model.load(epoch=16000)     # my_model_new_3 + Mixamo_new_2
    return model


def main(input_bvh, target_bvh, test_type, output_filename, load_model=None):
    parser = option_parser.get_parser()
    args = parser.parse_args()

    args.input_bvh = input_bvh
    args.target_bvh = target_bvh
    args.test_type = test_type
    args.output_filename = output_filename

    character_names, file_id, src_id = eval_prepare(args)
    # character_names = [['Trinity'], ['Talking_With_Hands']]
    # character_names = [['Trinity'], ['ZEGGS']]
    '''
    # intra
    character_names     [['Aj', 'BigVegas'], ['Goblin_m', 'Goblin_m']]      # Aj (..., 69) BigVegas (..., 69)
    file_id             [['./datasets/Mixamo/Aj/Dancing Running Man.bvh', './datasets/Mixamo/Aj/Dancing Running Man.bvh'], [0, 0]]
    src_id              0
    # cross
    character_names     [['BigVegas'], ['Mousey_m']]
    file_id             [['./datasets/Mixamo/BigVegas/Dual Weapon Combo.bvh'], [0]]
    src_id              0
    '''

    # input_character_name = args.input_bvh.split('/')[-2]
    output_character_name = args.target_bvh.split('/')[-2]
    output_filename = args.output_filename
    output_filename = pjoin(output_filename, args.input_bvh.split('/')[-1].replace('.bvh', '.npy'))

    test_device = args.cuda_device
    eval_seq = args.eval_seq
    args.cuda_device = test_device if torch.cuda.is_available() else 'cpu'
    args.is_train = False  #
    args.rotation = 'quaternion'
    args.eval_seq = eval_seq

    dataset = create_dataset(args, character_names)

    if load_model is None:
        model = demo_model(input_bvh, target_bvh, test_type, output_filename)
    else:
        model = load_model

    # for body in ['upper', 'lower', 'root']:
    if True:
        # if body == 'upper':
        #     upper_motions = []
        # elif body == 'lower':
        #     lower_motions = []
        # elif body == 'root':
        #     root_motions = []

        input_motion = []
        for i, character_group in enumerate(character_names):
            print('\r' + str(i), end='')
            input_group = []

            # if body == 'upper':
            #     upper_motions_group = []
            # elif body == 'lower':
            #     lower_motions_group = []
            # elif body == 'root':
            #     root_motions_group = []

            for j in range(len(character_group)):
                print(file_id[i][j])
                new_motion = dataset.get_item(i, j, file_id[i][j])
                new_motion_ = new_motion.clone()
                new_motion_.unsqueeze_(0)        # [103, 14284] -> [1, 103, 14284]
                # if body == 'upper':
                #     upper_motion = new_motion[:, 10 * 4:-3].clone()
                #     upper_motions_group.append(upper_motion)
                # elif body == 'lower':
                #     lower_motion = new_motion[:, :10 * 4].clone()
                #     lower_motions_group.append(lower_motion)
                # elif body == 'root':
                #     root_motion = new_motion[:, -3:].clone()
                #     root_motions_group.append(root_motion)
                new_motion_ = (new_motion_ - dataset.mean[i][j]) / dataset.var[i][j]

                # if body == 'upper':       # 20230418
                #     new_motion_[:, :10 * 4] = 0
                #     new_motion_[:, -3:] = 0
                # elif body == 'lower':
                #     new_motion_[:, 10 * 4:] = 0
                # elif body == 'root':
                #     new_motion_[:, :-3] = 0     # 20230414

                input_group.append(new_motion_)

            input_group = torch.cat(input_group, dim=0)
            input_motion.append([input_group, list(range(len(character_group)))])

        model.set_input(input_motion)
        # input_motion [[[1, 91, 216], [0]], [[1, 111, len'], [0]]]

        # inference latent
        latent_ = model.get_latent()     # [[1, 112, 3165], [1, 112, 517]]

        for body in ['upper', 'lower', 'root']:
            latent = latent_.copy()
            if body == 'upper':
                latent[0] = latent[0][:, 2 * 16:-1 * 16]      # 20230421
                latent[1] = latent[1][:, 2 * 16:-1 * 16]

                # latent = input_motion[0]
                # latent[0][:, :4 * 10] = 0
                # latent[0][:, -1 * 3:] = 0
                # latent[0] = torch.cat((latent[0], torch.zeros_like(latent[0][:, [0], :])), dim=1)

            elif body == 'lower':
                # latent[0][:, 2 * 16:-1 * 16] = 0
                # latent[0] = torch.cat((latent[0][:, :2 * 16], latent[0][:, -1 * 16:]), dim=1)       # bvh2upper_lower

                latent[0] = latent[0][:, :2 * 16]
                latent[1] = latent[1][:, :2 * 16]

                # latent = input_motion[0]
                # latent[0][:, 4 * 10:] = 0
                # latent[0] = torch.cat((latent[0], torch.zeros_like(latent[0][:, [0], :])), dim=1)
            elif body == 'root':
                latent[0] = latent[0][:, -1 * 16:]
                latent[1] = latent[1][:, -1 * 16:]

                # latent = input_motion[0]
                # latent[0][:, :-1 * 3] = 0
                # latent[0] = torch.cat((latent[0], torch.zeros_like(latent[0][:, [0], :])), dim=1)

            output_filename_ = output_filename[:-4]
            print(output_filename_, latent[0].shape)
            if src_id == 0:
                np.save("./{}_{}.npy".format(output_filename_, body), latent[0].detach().cpu().numpy())
                # np.save("./{}_global.npy".format(output_filename), [input_motion_root[0].detach().cpu().numpy(), BVH_file(args.input_bvh).get_height()])
            elif src_id == 1:
                np.save("./{}_{}.npy".format(output_filename_, body), latent[1].detach().cpu().numpy())
                # np.save("./{}_global.npy".format(output_filename), [input_motion_root[1].detach().cpu().numpy(), BVH_file(args.input_bvh).get_height()])


def load_model():
    from datasets import create_dataset, get_character_names
    from torch.utils.data.dataloader import DataLoader

    args = option_parser.get_args()
    # characters = get_character_names(args)      # [['ch01'], ['ch02']]

    # characters = [['Trinity'], ['Talking_With_Hands']]

    # characters = [['Talking_With_Hands'], ['Trinity']]      # model_4, model_5

    characters = [['Trinity'], ['ZEGGS']]

    para_path = os.path.join(args.save_dir, 'para.txt')
    with open(para_path, 'r') as para_file:
        argv_ = para_file.readline().split()[1:]
        args = option_parser.get_parser().parse_args(argv_)

    # data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    args.is_train = False       #
    args.rotation = 'quaternion'
    dataset = create_dataset(args, characters)
    model = create_model(args, characters, dataset)

    model.load(epoch=12400)      # 20000, modify for test
    return args, characters, model


def my_main(args, character_names, model):
    # input_bvh = './datasets/Mixamo/Talking_With_Hands/trn_2022_v1_064.bvh'
    # ref_bvh = './datasets/Mixamo/Trinity/Recording_006.bvh'
    from datasets.combined_motion import MixedData
    from torch.utils.data.dataloader import DataLoader
    dataset = MixedData(args, character_names)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    print(args.cuda_device)
    for step, motions in enumerate(data_loader):
        model.set_input(motions)
        model.test()
        break
    print('Finish!')


if __name__ == '__main__':
    '''
    cd /nfs7/y50021900/My_3/deep-motion-editing/retargeting/
    cd /ceph/hdd/yangsc21/Python/My_3/deep-motion-editing/retargeting/
    python eval_single_pair.py
    '''

    # src_name = 'Talking_With_Hands'
    # dest_name = 'Trinity'
    # bvh_name = 'trn_2022_v1_064.bvh'
    # test_type = 'cross'
    # output_path = './examples_my/cross_structure'
    # ref_bvh_name = 'Recording_006.bvh'
    #
    # input_file = './datasets/Mixamo/{}/{}'.format(src_name, bvh_name)
    # ref_file = './datasets/Mixamo/{}/{}'.format(dest_name, ref_bvh_name)
    #
    # cmd = 'python eval_single_pair.py --input_bvh={} --target_bvh={} --output_filename={} --test_type={}'.format(
    #     input_file, ref_file, pjoin(output_path, 'result.bvh'), test_type
    # )
    # os.system(cmd)

    # args, characters, model = load_model()
    # my_main(args, characters, model)

    # main(input_bvh, target_bvh, test_type, output_filename)
    pass

