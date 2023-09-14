import copy
import functools
import os
import pdb

import torch.nn.functional as F

import blobfile as bf
import torch
from torch.optim import AdamW

from diffusion import logger
from diffusion.fp16_util import MixedPrecisionTrainer
from diffusion.resample import LossAwareSampler, UniformSampler
from tqdm import tqdm
from diffusion.resample import create_named_schedule_sampler
# from eval import eval_humanml, eval_humanact12_uestc

from datetime import datetime

import sys
[sys.path.append(i) for i in ['./mydiffusion', '../../My', '../../My/process', '../process', '../../ubisoft-laforge-ZeroEGGS-main', '../../ubisoft-laforge-ZeroEGGS-main/ZEGGS']]
from generate.generate import WavEncoder
# from process.process_bvh import make_bvh_GENEA2020_BT
# from process_zeggs_bvh import pose2bvh

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(self, args, model, diffusion, device, data=None):
        self.args = args
        self.data = data
        self.model = model
        self.diffusion = diffusion
        self.cond_mode = model.cond_mode
        self.batch_size = args.batch_size
        self.microbatch = args.batch_size  # deprecating this option
        self.lr = args.lr
        self.log_interval = args.log_interval
        # self.save_interval = args.save_interval
        # self.resume_checkpoint = args.resume_checkpoint
        self.use_fp16 = False  # deprecating this option
        self.fp16_scale_growth = 1e-3  # deprecating this option
        self.weight_decay = args.weight_decay
        self.lr_anneal_steps = args.lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size # * dist.get_world_size()
        # self.num_steps = args.num_steps
        self.num_epochs = 40000
        self.n_seed = args.n_seed

        self.sync_cuda = torch.cuda.is_available()

        # self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=self.fp16_scale_growth,
        )

        self.save_dir = args.save_dir
        self.overwrite = args.overwrite

        self.device = device
        if args.audio_feat == "wav encoder":
            self.WavEncoder = WavEncoder().to(self.device)
            self.opt = AdamW([
                {'params': self.mp_trainer.master_params, 'lr':self.lr, 'weight_decay':self.weight_decay},
                {'params': self.WavEncoder.parameters(), 'lr':self.lr}
            ])
        elif args.audio_feat == "mfcc" or args.audio_feat == 'wavlm':
            self.opt = AdamW([
                {'params': self.mp_trainer.master_params, 'lr':self.lr, 'weight_decay':self.weight_decay}
            ])

        # if self.resume_step:
        #     self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.

        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)
        self.eval_wrapper, self.eval_data, self.eval_gt_data = None, None, None

        self.use_ddp = False
        self.ddp_model = self.model
        self.mask_train = (torch.zeros([self.batch_size, 1, 1, args.n_poses]) < 1).to(self.device)
        self.mask_test = (torch.zeros([1, 1, 1, args.n_poses]) < 1).to(self.device)
        self.mask_local_train = torch.ones(self.batch_size, args.n_poses).bool().to(self.device)
        self.mask_local_test = torch.ones(1, args.n_poses).bool().to(self.device)

    def run_loop(self):

        for epoch in range(self.num_epochs):
            print(f'Starting epoch {epoch}')
            # for _ in tqdm(range(10)):     # 4 steps, batch size, chmod 777
            for batch in tqdm(self.data):
                if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                    break

                cond_ = {'y':{}}

                pose_seq,  _, style, wavlm = batch  # (batch, 240, 135), (batch, 30), (batch, 64000)
                # pose_vel = F.pad((pose_seq[:, 1:] - pose_seq[:, :-1]), (0, 0, 1, 0))        # 20230424
                # pose_seq = torch.cat([pose_seq, pose_vel], dim=-1)

                # vel = pose_seq[:, 1:] - pose_seq[:, :-1]
                # acc = vel[:, 1:] - vel[:, :-1]
                # vel = F.pad(vel, (0, 0, 1, 0))
                # acc = F.pad(acc, (0, 0, 2, 0))  # 20230424
                # pose_seq = torch.cat([pose_seq, vel, acc], dim=-1)

                motion = pose_seq.permute(0, 2, 1).unsqueeze(2).to(self.device)
                cond_['y']['seed'] = motion[..., 0:self.n_seed]
                cond_['y']['style'] = style.to(self.device)
                cond_['y']['mask_local'] = self.mask_local_train

                if self.args.audio_feat == 'wav encoder':
                    # cond_['y']['audio'] = torch.rand(240, 2, 32).to(self.device)
                    cond_['y']['audio'] = self.WavEncoder(audio.to(self.device)).permute(1, 0, 2)       # (batch, 240, 32)
                elif self.args.audio_feat == 'mfcc':
                    # cond_['y']['audio'] = torch.rand(80, 2, 13).to(self.device)
                    cond_['y']['audio'] = mfcc.to(torch.float32).to(self.device).permute(1, 0, 2)       # [self.n_seed:, ...]      # (batch, 80, 13)
                elif self.args.audio_feat == 'wavlm':
                    cond_['y']['audio'] = wavlm.to(torch.float32).to(self.device)

                cond_['y']['mask'] = self.mask_train        # [..., self.n_seed:]

                self.run_step(motion, cond_)
                if self.step % self.log_interval == 0:
                    for k,v in logger.get_current().name2val.items():
                        if k == 'loss':
                            print('step[{}]: loss[{:0.5f}]'.format(self.step+self.resume_step, v))

                if self.step % 50000 == 0:
                    self.save()

                    # Run for a finite amount of time in integration tests.
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        return
                self.step += 1
            if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                break


    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)      # torch.Size([64, 251, 1, 196]) cond['y'].keys() dict_keys(['mask', 'lengths', 'text', 'tokens'])
        self.mp_trainer.optimize(self.opt)
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            # Eliminates the microbatch feature
            assert i == 0
            assert self.microbatch == self.batch_size
            micro = batch
            micro_cond = cond
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], self.device)

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,  # [bs, ch, image_size, image_size]      # x_start, (2, 135, 1, 240)
                t,  # [bs](int) sampled timesteps
                model_kwargs=micro_cond,
                dataset='kit'
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)


    def ckpt_file_name(self):
        return f"model{(self.step+self.resume_step):09d}.pt"


    def save(self):
        def save_checkpoint(params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)

            # Do not save CLIP weights
            clip_weights = [e for e in state_dict.keys() if e.startswith('clip_model.')]
            for e in clip_weights:
                del state_dict[e]

            logger.log(f"saving model...")
            filename = self.ckpt_file_name()
            with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                torch.save(state_dict, f)

        save_checkpoint(self.mp_trainer.master_params)

        with bf.BlobFile(
            bf.join(self.save_dir, f"opt{(self.step+self.resume_step):09d}.pt"),
            "wb",
        ) as f:
            torch.save(self.opt.state_dict(), f)


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
