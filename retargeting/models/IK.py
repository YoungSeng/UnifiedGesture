import pdb
import sys
import torch
from models.Kinematics import InverseKinematics
from datasets.bvh_parser import BVH_file
from tqdm import tqdm

sys.path.append('../utils')
import numpy as np
import BVH as BVH
import Animation as Animation
from Quaternions_old import Quaternions
import os
from scipy.signal import savgol_filter

L = 5  # #frame to look forward/backward


def alpha(t):
    return 2.0 * t * t * t - 3.0 * t * t + 1


def lerp(a, l, r):
    return (1 - a) * l + a * r


def get_character_height(file_name):
    file = BVH_file(file_name)
    return file.get_height()


def get_foot_contact(file_name, ref_height):
    anim, names, _ = BVH.load(file_name)

    ee_ids = get_ee_id_by_names(names)

    glb = Animation.positions_global(anim)  # [T, J, 3]

    ee_pos = glb[:, ee_ids, :]
    ee_velo = ee_pos[1:, ...] - ee_pos[:-1, ...]
    ee_velo = torch.tensor(ee_velo) / ref_height
    ee_velo_norm = torch.norm(ee_velo, dim=-1)
    contact = ee_velo_norm < 0.003
    contact = contact.int()
    padding = torch.zeros_like(contact)
    contact = torch.cat([padding[:1, :], contact], dim=0)
    return contact.numpy()


def get_foot_vel_position(file_name, ref_height):
    anim, names, _ = BVH.load(file_name)
    ee_ids = get_ee_id_by_names(names)
    glb = Animation.positions_global(anim)  # [T, J, 3]
    T = glb.shape[0]
    ee_pos = glb[:, ee_ids, :]
    ee_velo = ee_pos[1:, ...] - ee_pos[:-1, ...]
    ee_velo = np.pad(ee_velo, ((1, 0), (0, 0), (0, 0)), 'constant')
    root_position = glb[:, names.index('Hips')].copy()
    result = np.concatenate([root_position, ee_pos.reshape(T, -1), ee_velo.reshape(T, -1)], axis=-1)
    return result / ref_height


def get_ee_id_by_names(joint_names):
    # ees = ['RightToeBase', 'LeftToeBase', 'LeftFoot', 'RightFoot']
    ees = ['RightToeBase', 'LeftToeBase', 'LeftToeBaseEnd', 'RightToeBaseEnd']
    ee_id = []
    for i, ee in enumerate(ees):
        ee_id.append(joint_names.index(ee))
    return ee_id


def PFC_fix(input_file, ref_height):
    path_1 = os.path.split(input_file)[0]
    path_2 = os.path.split(input_file)[1][:-4]
    output_file = os.path.join(path_1, path_2 + '_fix.bvh')
    print('output file:', output_file)
    anim, name, ftime = BVH.load(input_file)
    fid = get_ee_id_by_names(name)
    contact_original = get_foot_contact(input_file, ref_height)
    # rowIndex = np.where((contact_original == (0, 0, 0, 0)).all(axis=1))
    glb = Animation.positions_global(anim)

    ee_pos = glb[:, fid, :]
    ee_velo = (ee_pos[1:, ...] - ee_pos[:-1, ...]) / ref_height
    ee_velo_norm = torch.norm(torch.tensor(ee_velo), dim=-1)
    padding = torch.zeros_like(ee_velo_norm)
    ee_velo_norm = torch.cat([padding[:1, :], ee_velo_norm], dim=0)
    T = glb.shape[0]
    root_position = glb[:, name.index('Hips')].copy()

    smoothing = True
    # smoothing
    if smoothing:
        n_poses = root_position.shape[0]
        out_poses = np.zeros((n_poses, root_position.shape[1]))
        for i in range(out_poses.shape[1]):
            # if (13 + (njoints - 14) * 9) <= i < (13 + njoints * 9): out_poses[:, i] = savgol_filter(poses[:, i], 41, 2)  # NOTE: smoothing on rotation matrices is not optimal
            # else:
            out_poses[:, i] = savgol_filter(root_position[:, i], 41, 2)  # NOTE: smoothing on rotation matrices is not optimal
        root_position = out_poses
    glb[:, name.index('Hips')] = root_position

    root_vec = (root_position[1:, ...] - root_position[:-1, ...]) / ref_height
    root_acc = root_vec[1:, ...] - root_vec[:-1, ...]
    root_vec = np.pad(root_vec, ((1, 0), (0, 0)), 'constant')
    root_acc = np.pad(root_acc, ((2, 0), (0, 0)), 'constant')
    z_acc = root_acc[:, 1]
    z_acc[z_acc < 0] = 0.0
    root_acc[:, 1] = z_acc          # (len, 3)
    root_vec = np.linalg.norm(root_vec, axis=1)
    root_acc = np.linalg.norm(root_acc, axis=1)
    # root_contact = np.where(root_acc>=0.0003)
    # print('root contact:', len(root_contact[0]), 'frames:', T)
    # for index in root_contact[0]:
    #     ee_velo_norm[index] = ee_velo_norm[index] * 0.75

    for i in range(1, T):
        ee_velo_norm[i] = ee_velo_norm[i] * 1 / np.exp(1000 * root_acc[i])       #  + 50 * root_vec[i]

    contact = ee_velo_norm < 0.003
    contact_fix = contact.int().numpy()     # np.where((contact_fix == (0, 0, 0, 0)).all(axis=1))
    # contact_fix[np.where((contact_fix == (0, 0, 0, 0)).all(axis=1))] = (0, 1, 1, 0)
    for i, fidx in enumerate(fid):  # fidx: index of the foot joint
        fixed = contact_fix[:, i]  # [T]
        s = 0
        while s < T:
            while s < T and fixed[s] == 0:
                s += 1
            if s >= T:
                break
            t = s
            avg = glb[t, fidx].copy()
            while t + 1 < T and fixed[t + 1] == 1:
                t += 1
                avg += glb[t, fidx].copy()
            avg /= (t - s + 1)

            for j in range(s, t + 1):
                glb[j, fidx] = avg.copy()
            s = t + 1

        for s in range(T):
            if fixed[s] == 1:
                continue
            l, r = None, None
            consl, consr = False, False
            for k in range(L):
                if s - k - 1 < 0:
                    break
                if fixed[s - k - 1]:
                    l = s - k - 1
                    consl = True
                    break
            for k in range(L):
                if s + k + 1 >= T:
                    break
                if fixed[s + k + 1]:
                    r = s + k + 1
                    consr = True
                    break
            if not consl and not consr:
                continue
            if consl and consr:
                litp = lerp(alpha(1.0 * (s - l + 1) / (L + 1)),
                            glb[s, fidx], glb[l, fidx])
                ritp = lerp(alpha(1.0 * (r - s + 1) / (L + 1)),
                            glb[s, fidx], glb[r, fidx])
                itp = lerp(alpha(1.0 * (s - l + 1) / (r - l + 1)),
                           ritp, litp)
                glb[s, fidx] = itp.copy()
                continue
            if consl:
                litp = lerp(alpha(1.0 * (s - l + 1) / (L + 1)),
                            glb[s, fidx], glb[l, fidx])
                glb[s, fidx] = litp.copy()
                continue
            if consr:
                ritp = lerp(alpha(1.0 * (r - s + 1) / (L + 1)),
                            glb[s, fidx], glb[r, fidx])
                glb[s, fidx] = ritp.copy()

    # glb is ready

    anim = anim.copy()

    rot = torch.tensor(anim.rotations.qs, dtype=torch.float)
    pos = torch.tensor(anim.positions[:, 0, :], dtype=torch.float)
    offset = torch.tensor(anim.offsets, dtype=torch.float)

    glb = torch.tensor(glb, dtype=torch.float)

    ik_solver = InverseKinematics(rot, pos, offset, anim.parents, glb)

    print('Fixing foot contact using IK...')
    for i in tqdm(range(100)):
        ik_solver.step()

    rotations = ik_solver.rotations.detach()
    norm = torch.norm(rotations, dim=-1, keepdim=True)
    rotations /= norm

    anim.rotations = Quaternions(rotations.numpy())
    anim.positions[:, 0, :] = ik_solver.position.detach().numpy()

    BVH.save(output_file, anim, name, ftime)



def fix_foot_contact(input_file, foot_file, output_file, ref_height):
    anim, name, ftime = BVH.load(input_file)

    fid = get_ee_id_by_names(name)
    contact = get_foot_contact(foot_file, ref_height)

    glb = Animation.positions_global(anim)  # [T, J, 3]

    T = glb.shape[0]

    for i, fidx in enumerate(fid):  # fidx: index of the foot joint
        fixed = contact[:, i]  # [T]
        s = 0
        while s < T:
            while s < T and fixed[s] == 0:
                s += 1
            if s >= T:
                break
            t = s
            avg = glb[t, fidx].copy()
            while t + 1 < T and fixed[t + 1] == 1:
                t += 1
                avg += glb[t, fidx].copy()
            avg /= (t - s + 1)

            for j in range(s, t + 1):
                glb[j, fidx] = avg.copy()
            s = t + 1

        for s in range(T):
            if fixed[s] == 1:
                continue
            l, r = None, None
            consl, consr = False, False
            for k in range(L):
                if s - k - 1 < 0:
                    break
                if fixed[s - k - 1]:
                    l = s - k - 1
                    consl = True
                    break
            for k in range(L):
                if s + k + 1 >= T:
                    break
                if fixed[s + k + 1]:
                    r = s + k + 1
                    consr = True
                    break
            if not consl and not consr:
                continue
            if consl and consr:
                litp = lerp(alpha(1.0 * (s - l + 1) / (L + 1)),
                            glb[s, fidx], glb[l, fidx])
                ritp = lerp(alpha(1.0 * (r - s + 1) / (L + 1)),
                            glb[s, fidx], glb[r, fidx])
                itp = lerp(alpha(1.0 * (s - l + 1) / (r - l + 1)),
                           ritp, litp)
                glb[s, fidx] = itp.copy()
                continue
            if consl:
                litp = lerp(alpha(1.0 * (s - l + 1) / (L + 1)),
                            glb[s, fidx], glb[l, fidx])
                glb[s, fidx] = litp.copy()
                continue
            if consr:
                ritp = lerp(alpha(1.0 * (r - s + 1) / (L + 1)),
                            glb[s, fidx], glb[r, fidx])
                glb[s, fidx] = ritp.copy()

    # glb is ready

    anim = anim.copy()

    rot = torch.tensor(anim.rotations.qs, dtype=torch.float)
    pos = torch.tensor(anim.positions[:, 0, :], dtype=torch.float)
    offset = torch.tensor(anim.offsets, dtype=torch.float)

    glb = torch.tensor(glb, dtype=torch.float)

    ik_solver = InverseKinematics(rot, pos, offset, anim.parents, glb)

    print('Fixing foot contact using IK...')
    for i in tqdm(range(200)):
        ik_solver.step()

    rotations = ik_solver.rotations.detach()
    norm = torch.norm(rotations, dim=-1, keepdim=True)
    rotations /= norm

    anim.rotations = Quaternions(rotations.numpy())
    anim.positions[:, 0, :] = ik_solver.position.detach().numpy()

    BVH.save(output_file, anim, name, ftime)
