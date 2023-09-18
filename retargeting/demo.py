import os
import pdb
import argparse
from datasets.bvh_parser import BVH_file
from datasets.bvh_writer import BVH_writer
from models.IK import fix_foot_contact
from os.path import join as pjoin
import numpy as np
from eval_single_pair import main as eval_single_pair
from easydict import EasyDict
import yaml

# downsampling and remove redundant joints
def copy_ref_file(src, dst):
    file = BVH_file(src)
    writer = BVH_writer(file.edges, file.names)
    # writer.write_raw(file.to_tensor(quater=True)[..., ::2], 'quaternion', dst)
    writer.write_raw(file.to_tensor(quater=True)[..., :], 'quaternion', dst)

def get_height(file):
    file = BVH_file(file)
    return file.get_height()


def example(dataset_name, src_name, dest_name, bvh_name, test_type, output_path, ref_bvh_name=None, loaded_model=None):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    input_file = './datasets/' + dataset_name + '/{}/{}'.format(src_name, bvh_name)
    ref_file = './datasets/' + dataset_name + '/{}/{}'.format(dest_name, ref_bvh_name)
    # copy_ref_file(input_file, pjoin(output_path, 'input.bvh'))
    # copy_ref_file(ref_file, pjoin(output_path, 'gt.bvh'))
    # height = get_height(input_file)
    # bvh_name = bvh_name.replace(' ', '_')

    eval_single_pair(input_file, ref_file, test_type, output_path, loaded_model)

    # cmd = 'python eval_single_pair.py --input_bvh={} --target_bvh={} --output_filename={} --test_type={}'.format(
    #     input_file, ref_file, output_path, test_type
    # )
    # os.system(cmd)

    # fix_foot_contact(pjoin(output_path, 'result.bvh'),
    #                  pjoin(output_path, 'input.bvh'),
    #                  pjoin(output_path, 'result_fix_foot_contact.bvh'),
    #                  height)


def bvh_input(quaternion, edge):
    import sys
    sys.path.append("./utils")
    from Quaternions import Quaternions
    import torch

    # path = "./datasets/Mixamo_debug/Talking_With_Hands/val_2022_v1_038.bvh"
    # output_path = "./datasets/Mixamo_debug/Talking_With_Hands/val_2022_v1_038_recon.bvh"
    # new_path = "./datasets/Mixamo_debug/Talking_With_Hands/"

    path = "./datasets/Mixamo_new/ZEGGS/067_Speech_2_x_1_0.bvh"
    output_path = "./datasets/Mixamo_debug/067_Speech_2_x_1_0_recon.bvh"
    new_path = "./datasets/Mixamo_debug/"

    # path = "./datasets/Mixamo_debug/Trinity/TestSeq010_z.bvh"
    # output_path = "./datasets/Mixamo_debug/Trinity/TestSeq010_z_recon.bvh"
    # new_path = "./datasets/Mixamo_debug/Trinity/"

    # path = "./datasets/Mixamo_ch12/ch01/Hip_Hop_Dancing.bvh"
    # output_path = "./datasets/Mixamo_ch12/ch01/Hip_Hop_Dancing_recon.bvh"
    # new_path = "./datasets/Mixamo_ch12/ch01/"

    file = BVH_file(path)
    # file.set_new_root(1)        # 20230330
    new_motion = file.to_tensor(quater=False, edge=edge).permute((1, 0)).numpy()       # quater=False, edge=False
    print(new_motion.shape)        # (1800, 87)
    # motion = file.to_numpy(quater=False, edge=False)

    file.write(output_path)
    window_size = 64
    new = new_motion[0:window_size, :]
    # quaternion = False

    if quaternion and edge:
        new = new.reshape(new.shape[0], -1, 3)  # -> (64, 29, 3)
        rotations = new[:, :-1, :]  # -> (64, 28, 3)
        rotations = Quaternions.from_euler(np.radians(rotations)).qs  # -> (64, 28, 4)
        rotations = rotations.reshape(rotations.shape[0], -1)  # -> (64, 112)
        new = np.concatenate((rotations, new[:, -1, :].reshape(new.shape[0], -1)), axis=1)  # -> (64, 115)
    new = new[np.newaxis, ...]  # -> (1, 64, 115)
    new_window = torch.tensor(new, dtype=torch.float32)
    print(new_window.shape)      # [1, 64, 115]

    data = []
    data.append(new_window)
    data = torch.cat(data)
    data = data.permute(0, 2, 1)
    print(data.shape)            # [1, 115, 64]

    mean = torch.mean(data, (0, 2), keepdim=True)
    var = torch.var(data, (0, 2), keepdim=True)
    var = var ** (1 / 2)
    idx = var < 1e-5
    var[idx] = 1
    data_ = (data - mean) / var
    print(data_.shape)            # [1, 115, 64]

    gt = data_[0]
    ans = gt * var + mean
    print(ans.shape)             # [1, 115, 64]

    writer = BVH_writer(file.edges, file.names)

    # debug bvh
    # print('topology', file.topology, 'offset', file.offset, 'names', file.names)
    # writer.parent = file.topology
    # writer.offset = file.offset
    # writer.names = file.names

    # edge -> False
    if edge == False:
        from datasets.bvh_writer import write_bvh
        print(ans[0].shape)     # -> [81, 64]
        motion = ans[0].permute(1, 0).detach().cpu().numpy()        # -> [64, 81]
        rotations = motion[..., :-3].reshape(motion.shape[0], -1, 3)
        positions = motion[..., -3:]
        write_bvh(file.topology, file.offset, rotations, positions, file.names, 1.0/30, 'xyz',
                  os.path.join(new_path, '0_gt_euler_angle_fix.bvh'))
    else:
        if quaternion:
            # print('parent', writer.parent, 'offset', writer.offset, 'names', writer.names)
            writer.write_raw(ans[0], 'quaternion', os.path.join(new_path, '0_gt_quaternion.bvh'))
        else:
            writer.write_raw(ans[0], 'xyz', os.path.join(new_path, '0_gt_euler_angle.bvh'))
    

def bvh2latent(Trinity_bvh_path = "./datasets/Trinity_ZEGGS/Trinity/", ZEGGS_bvh_path = "./datasets/Trinity_ZEGGS/ZEGGS/",
               Trinity_output = './datasets/Trinity_ZEGGS/bvh2upper_lower_root/Trinity', ZEGGS_output = './datasets/Trinity_ZEGGS/bvh2upper_lower_root/ZEGGS'):


    for i, item in enumerate(os.listdir(Trinity_bvh_path)):
        print(i, item)
        if i == 0:
            from eval_single_pair import demo_model
            dataset_name = 'Trinity_ZEGGS'
            src_name = 'Trinity'
            dest_name = 'ZEGGS'
            bvh_name = item
            ref_bvh_name = '067_Speech_2_x_1_0.bvh'
            input_file = './datasets/' + dataset_name + '/{}/{}'.format(src_name, bvh_name)
            ref_file = './datasets/' + dataset_name + '/{}/{}'.format(dest_name, ref_bvh_name)
            model = demo_model(input_file, ref_file, 'cross', Trinity_output)
        example('Trinity_ZEGGS', 'Trinity', 'ZEGGS', item, 'cross', Trinity_output, ref_bvh_name='067_Speech_2_x_1_0.bvh', loaded_model=model)


    for i, item in enumerate(os.listdir(ZEGGS_bvh_path)):
        print(i, item)
        if i == 0:
            from eval_single_pair import demo_model
            dataset_name = 'Trinity_ZEGGS'
            src_name = 'ZEGGS'
            dest_name = 'Trinity'
            bvh_name = item
            ref_bvh_name = 'Recording_006.bvh'
            input_file = './datasets/' + dataset_name + '/{}/{}'.format(src_name, bvh_name)
            ref_file = './datasets/' + dataset_name + '/{}/{}'.format(dest_name, ref_bvh_name)
            model = demo_model(input_file, ref_file, 'cross', ZEGGS_output)
        example('Trinity_ZEGGS', 'ZEGGS', 'Trinity', item, 'cross', ZEGGS_output, ref_bvh_name='Recording_006.bvh', loaded_model=model)


'''
def latent2bvh(mode, item, ref_path, output_path):
    import torch

    if mode == 'ZEGGS':
        from eval_single_pair import demo_model
        dataset_name = 'Mixamo_new_2'
        src_name = 'Trinity'
        dest_name = 'ZEGGS'
        bvh_name = item
        ref_bvh_name = '067_Speech_2_x_1_0.bvh'
        input_file = './datasets/' + dataset_name + '/{}/{}'.format(src_name, bvh_name)
        ref_file = './datasets/' + dataset_name + '/{}/{}'.format(dest_name, ref_bvh_name)
        model = demo_model(input_file, ref_file, 'cross', './datasets/bvh2latent/Trinity')
        latent_0 = np.load(item)
        latent_1 = np.load(ref_path)
        latent = [torch.from_numpy(latent_0).to('cuda:5'), torch.from_numpy(latent_1).to('cuda:5')]     # 20230413
        model.latent2res(latent)
        model.compute_test_result()
        os.system('cp "{}/{}/0_{}.bvh" "./{}"'.format(model.bvh_path, 'ZEGGS', '0', output_path))

    if mode == 'Trinity':
        from eval_single_pair import demo_model
        dataset_name = 'Mixamo_new_2'
        src_name = 'ZEGGS'
        dest_name = 'Trinity'
        bvh_name = item
        ref_bvh_name = 'Recording_006.bvh'
        input_file = './datasets/' + dataset_name + '/{}/{}'.format(src_name, bvh_name)
        ref_file = './datasets/' + dataset_name + '/{}/{}'.format(dest_name, ref_bvh_name)
        model = demo_model(input_file, ref_file, 'cross', './datasets/bvh2latent/ZEGGS')
        latent_0 = np.load(item)
        latent_1 = np.load(ref_path)
        latent = [torch.from_numpy(latent_1).to('cuda:5'), torch.from_numpy(latent_0).to('cuda:5')]     # 20230413
        model.latent2res(latent)
        model.compute_test_result()
        os.system('cp "{}/{}/0_{}.bvh" "./{}"'.format(model.bvh_path, 'Trinity', '1', output_path))
'''

def latent2bvh(mode, upper, lower, root, ref_path, output_path, hide=True, model_name='vqvae_ulr', device='cuda:5'):
    import torch
    if hide:
        from eval_single_pair import demo_model
        if mode == 'ZEGGS':
            dataset_name = 'Mixamo_new_2'
            src_name = 'Trinity'
            dest_name = 'ZEGGS'
            ref_bvh_name = '067_Speech_2_x_1_0.bvh'
        elif mode == 'Trinity':
            dataset_name = 'Mixamo_new_2'
            src_name = 'ZEGGS'
            dest_name = 'Trinity'
            ref_bvh_name = 'Recording_006.bvh'
        ref_file = './datasets/' + dataset_name + '/{}/{}'.format(dest_name, ref_bvh_name)
        latent_1 = torch.from_numpy(np.load(ref_path)).to(device)
        input_file = './datasets/' + dataset_name + '/{}/{}'.format(src_name, upper)
        if mode == 'ZEGGS':
            model = demo_model(input_file, ref_file, 'cross', './datasets/bvh2latent/Trinity')
        elif mode == 'Trinity':
            model = demo_model(input_file, ref_file, 'cross', './datasets/bvh2latent/ZEGGS')

        latent_upper = np.load(upper)
        name_ = os.path.split(upper)[1][:-4]
        if model_name == 'vqvae_ulr':
            latent_root = np.load(root)
            latent_lower = np.load(lower)
            latent_0 = np.concatenate((latent_lower, latent_upper, latent_root), axis=1)
        elif model_name == 'vqvae_ulr_2':

            # root = np.zeros_like(latent_lower[:, -16 * 1:])     # [1, 16, length]
            # root[..., 0] = np.load('./datasets/bvh2upper_lower_root/Trinity/Recording_006_root.npy')[..., 0]
            # for iii in range(1, root.shape[-1]):
            #     root[..., iii] = root[..., iii - 1] + latent_lower[:, -16 * 1:, iii - 1]

            latent_0 = np.concatenate((latent_lower[:, :16*2], latent_upper, latent_lower[:, -16*1:]), axis=1)
        elif model_name == 'diffusion':
            # (1, 238, 112)
            latent_0 = latent_upper
        elif model_name == 'GT':
            latent_0 = latent_upper[:, :16 * 7].transpose(1, 0)
            latent_0 = np.expand_dims(latent_0, axis=0).astype(np.float32)
        print(latent_0.shape)
        latent_0 = torch.from_numpy(latent_0).to(device)

        latent = [latent_0, latent_1]
        model.latent2res(latent)
        if mode == 'ZEGGS':
            result = model.fake_res_denorm[1][0, ...]
        elif mode == 'Trinity':
            result = model.fake_res_denorm[0][0, ...]
        model.generate_bvh(result=result, mode=mode)
        os.system('cp "{}/{}/0_{}.bvh" "./{}/{}.bvh"'.format(model.bvh_path, mode, '0', output_path, name_))
    else:
        if mode == 'ZEGGS':
            from eval_single_pair import demo_model
            dataset_name = 'Mixamo_new_2'
            src_name = 'Trinity'
            dest_name = 'ZEGGS'
            ref_bvh_name = '067_Speech_2_x_1_0.bvh'
            ref_file = './datasets/' + dataset_name + '/{}/{}'.format(dest_name, ref_bvh_name)
            latent_1 = torch.from_numpy(np.load(ref_path)).to(device)

            if upper is not None:
                input_file = './datasets/' + dataset_name + '/{}/{}'.format(src_name, upper)
                model = demo_model(input_file, ref_file, 'cross', './datasets/bvh2latent/Trinity')
                if type(upper) is not np.ndarray:
                    latent_0 = np.load(upper)
                else:
                    latent_0 = upper
                latent = [torch.from_numpy(latent_0).to(device), latent_1]     # 20230413
                model.latent2res(latent)
                upper_result = model.fake_res_denorm[1][0, ...]       # [103, 14284]
                # if lower is None:
                #     upper_result[:4 * 10] = 0
                # if root is None:
                #     upper_result[-3:] = 0
            if lower is not None:
                input_file = './datasets/' + dataset_name + '/{}/{}'.format(src_name, lower)
                model = demo_model(input_file, ref_file, 'cross', './datasets/bvh2latent/Trinity')
                if type(lower) is not np.ndarray:
                    lower = torch.from_numpy(np.load(lower)).to(device)
                else:
                    lower = lower.to(device)

                latent = [lower, latent_1]     # 20230413
                model.latent2res(latent)
                lower_result = model.fake_res_denorm[1][0, ...]
                upper_result[:4 * 10] = lower_result[:4 * 10].clone()
                # if root is None:
                #     lower_result[-3:] = 0
                # if upper is None:
                #     lower_result[40:-3] = 0
            if root is not None:
                input_file = './datasets/' + dataset_name + '/{}/{}'.format(src_name, root)
                model = demo_model(input_file, ref_file, 'cross', './datasets/bvh2latent/Trinity')
                if type(root) is not np.ndarray:
                    root = torch.from_numpy(np.load(root)).to(device)
                else:
                    root = root.to(device)
                latent = [root, latent_1]  # 20230413
                model.latent2res(latent)
                root_result = model.fake_res_denorm[1][0, ...]
                upper_result[-3:] = root_result[-3:].clone()

            model.generate_bvh(result=upper_result, mode=mode)
            os.system('cp "{}/{}/0_{}.bvh" "./{}"'.format(model.bvh_path, 'ZEGGS', '0', output_path))

        if mode == 'Trinity':
            from eval_single_pair import demo_model
            dataset_name = 'Mixamo_new_2'
            src_name = 'ZEGGS'
            dest_name = 'Trinity'
            bvh_name = item
            ref_bvh_name = 'Recording_006.bvh'
            input_file = './datasets/' + dataset_name + '/{}/{}'.format(src_name, bvh_name)
            ref_file = './datasets/' + dataset_name + '/{}/{}'.format(dest_name, ref_bvh_name)
            model = demo_model(input_file, ref_file, 'cross', './datasets/bvh2latent/ZEGGS')
            latent_0 = np.load(item)
            latent_1 = np.load(ref_path)
            latent = [torch.from_numpy(latent_1).to(device), torch.from_numpy(latent_0).to(device)]     # 20230413
            model.latent2res(latent)
            # model.compute_test_result()
            global_position = torch.from_numpy(np.load(global_position, allow_pickle=True)[0]).to(device)
            model.compute_test_result_global(global_position, mode)
            os.system('cp "{}/{}/0_{}.bvh" "./{}"'.format(model.bvh_path, 'Trinity', '1', output_path))


def orignal2bvh(upper=None, lower=None, root=None, ref_path=None, output_path=None):

    dataset_name = 'Mixamo_new_2'
    src_name = 'Trinity'
    dest_name = 'ZEGGS'
    bvh_name = upper
    ref_bvh_name = '067_Speech_2_x_1_0.bvh'
    input_file = './datasets/' + dataset_name + '/{}/{}'.format(src_name, bvh_name)
    ref_file = './datasets/' + dataset_name + '/{}/{}'.format(dest_name, ref_bvh_name)

    import torch
    import option_parser
    from datasets import create_dataset

    parser = option_parser.get_parser()
    args = parser.parse_args()
    args.input_bvh = input_file
    args.target_bvh = ref_file
    args.output_filename = output_path

    character_names = [['Trinity'], ['ZEGGS']]

    eval_seq = args.eval_seq
    test_device = args.cuda_device
    args.cuda_device = test_device if torch.cuda.is_available() else 'cpu'

    para_path = os.path.join(args.save_dir, 'para.txt')
    with open(para_path, 'r') as para_file:
        argv_ = para_file.readline().split()[1:]
        args = option_parser.get_parser().parse_args(argv_)

    args.is_train = False  #
    args.rotation = 'quaternion'
    args.eval_seq = eval_seq

    dataset = create_dataset(args, character_names)

    if upper is not None:
        upper_result = torch.from_numpy(np.load(upper)[:, :-1]).to(args.cuda_device)

    fake_res_denorm = dataset.denorm(0, [0], upper_result)
    path = './datasets/Mixamo_new_2/Trinity/Recording_006.bvh'
    file = BVH_file(path)
    writer = BVH_writer(file.edges, file.names)
    writer.write_raw(fake_res_denorm[0], 'quaternion', os.path.join(output_path, '0_gt_quaternion.bvh'))
    # pdb.set_trace()


def generate_result(source_path, generate_path):
    for item in os.listdir(source_path):
        print(item)
        latent2bvh('ZEGGS',
                   os.path.join(source_path, item),
                   None,
                   None,
                   ref_path='./datasets/bvh2latent/ZEGGS/065_Speech_0_x_1_0.npy',
                   output_path=generate_path,
                   model_name='diffusion')

        # latent2bvh('Trinity',
        #            os.path.join(source_path, item),
        #            None,
        #            None,
        #            # ref_path='./datasets/bvh2latent/Trinity/Recording_006.npy',
        #            ref_path='./datasets/bvh2latent/ZEGGS/065_Speech_0_x_1_0.npy',
        #            output_path=generate_path,
        #            model_name='diffusion')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Retargeting')
    parser.add_argument('--model_name', type=str, default='diffusion')
    parser.add_argument('--cuda_device', type=str, default='cuda:0')
    parser.add_argument('--target', type=str, default='ZEGGS')
    parser.add_argument('--input_file', type=str, default="../diffusion_latent/result_model3_128/Trinity/005_Neutral_4_x_1_0_minibatch_30_[0, 0, 0, 0, 0, 3, 0]_123456_recon.npy")
    parser.add_argument('--ref_path', type=str, default='./datasets/bvh2latent/ZEGGS/065_Speech_0_x_1_0.npy')
    parser.add_argument('--output_path', type=str, default='../result/inference/Trinity/')
    parser.add_argument('--mode', type=str, default='quick_start')
    parser.add_argument('--save_dir', type=str, default='./my_model_new_3', help='directory for all savings')
    config = parser.parse_args()

    # example('Mixamo_new_2', 'Trinity', 'ZEGGS', 'Recording_006.bvh', 'cross', './examples_my/cross_structure', ref_bvh_name='067_Speech_2_x_1_0.bvh')
    # example('Mixamo_new_2', 'ZEGGS', 'Trinity', '067_Speech_2_x_1_0.bvh', 'cross', './examples_my/cross_structure', ref_bvh_name='Recording_006.bvh')


    # x = np.load("../codebook/result/inference/Trinity/generate_train_codebook.npy")
    # x = x[0].transpose(1, 0).reshape(-1, 28, 4)
    # x[:, -1, :] = 0
    # x = np.expand_dims(x.reshape(-1, 112).transpose(1, 0), axis=0)
    # np.save("../codebook/result/inference/Trinity/generate_train_codebook_setroot.npy", x)

    # latent2bvh('ZEGGS', "./examples_my/cross_structure/Trinity/Recording_006_2.npy",
    #            ref_path='./datasets/bvh2latent/ZEGGS/065_Speech_0_x_1_0.npy', output_path="./examples_my/cross_structure/")
    # latent2bvh('Trinity', './datasets/bvh2latent/ZEGGS/065_Speech_0_x_1_0.npy',
    #            ref_path='./datasets/bvh2latent/Trinity/Recording_004.npy', output_path='./datasets/bvh2latent/ZEGGS/')

    # locallatent_global2bvh('ZEGGS', './datasets/bvh2glocal_local_latent/Trinity/Recording_004_local.npy',
    #                        './datasets/bvh2glocal_local_latent/Trinity/Recording_004_global.npy',
    #            ref_path='./datasets/bvh2latent/ZEGGS/065_Speech_0_x_1_0.npy', output_path="./examples_my/cross_structure/")


    # generate_result("../diffusion_latent/result_model3_128/Trinity/",
    #                 "../diffusion_latent/result_model3_128_bvh/")

    # generate_result("./datasets/valid_process_2/Trinity/",
    #                 "./datasets/valid_process_2/bvh/")

    if config.mode == 'quick_start':

        # Check if the directory exists
        if not os.path.exists(config.output_path):
            os.makedirs(config.output_path)
            print(f"Directory {config.output_path} created!")
        else:
            print(f"Directory {config.output_path} already exists!")

        latent2bvh(config.target,
                   config.input_file,
                   None,
                   None,
                   # "/ceph/hdd/yangsc21/Python/My_3/deep-motion-editing/retargeting/datasets/valid_processed/Trinity/TestSeq003_lower.npy",
                   #  "/ceph/hdd/yangsc21/Python/My_3/deep-motion-editing/retargeting/datasets/valid_processed/Trinity/TestSeq003_root.npy",
                    ref_path=config.ref_path,
                    output_path=config.output_path,
                    model_name=config.model_name,
                    device=config.cuda_device)


    elif config.mode == 'bvh2latent':
        bvh2latent()


    # latent2bvh('Trinity',
    #            '../dataset/ZEGGS/VQVAE_result/ZEGGS/generate_upper_001_Neutral_0_mirror_x_1_0.npy',
    #            '../dataset/ZEGGS/VQVAE_result/ZEGGS/generate_lower_001_Neutral_0_mirror_x_1_0.npy',
    #            '../dataset/ZEGGS/VQVAE_result/ZEGGS/generate_root_001_Neutral_0_mirror_x_1_0.npy',
    #            ref_path='./datasets/bvh2latent/Trinity/Recording_006.npy',
    #            output_path="./examples_my/cross_structure/")

    # orignal2bvh('../codebook/result/inference/Trinity/generate_upper_train_codebook_upper_original_downsample_2_joint_channel_1.npy', None, None,
    #             ref_path='./datasets/bvh2latent/ZEGGS/065_Speech_0_x_1_0.npy', output_path='../codebook/result/inference/Trinity')        # './examples_my/cross_structure/'

    print('Finished!')

    # bvh_input(quaternion=True, edge=True)
    # bvh_input(quaternion=True, edge=False)
    # # bvh_input(quaternion=False, edge=False)

    