'''
阶段1：自动编码器(Autoencoder)
    在你自己的数据集上进行训练可以帮助你获得更好的令牌，从而为你的领域获得更好的图像。
    在ImageNet上训练 KL 正则化自动编码器的配置在configs/autoencoder中提供。
    作者未完成:
        要训练 VQ 正则化模型，请参阅taming-transformers存储库： https://github.com/CompVis/taming-transformers
    
    运行: python wll_kl_utoencoder.py --base configs/autoencoder/autoencoder_kl_8x8x64.yaml --train True

阶段2：扩散模型(Diffusion Model)
    运行: python wll.py --base configs/latent-diffusion/txt2img/txt2img-sdv1.yaml --train True 



'''
import argparse, os, sys, datetime, glob
import numpy as np
import time
import torch
import torchvision
import pytorch_lightning as pl

from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from functools import partial
from PIL import Image

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info

from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config
# 
import types

def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-f",
        "--finetune",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="finetune from the public checkpoint",
    )

    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-x",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    return parser


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[worker_id * split_size:(worker_id + 1) * split_size]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False if is_iterable_dataset else True,
                          worker_init_fn=init_fn)

    def _val_dataloader(self, shuffle=False):
        if isinstance(self.datasets['validation'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          worker_init_fn=init_fn,
                          shuffle=shuffle)

    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)

        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle)

    def _predict_dataloader(self, shuffle=False):
        if isinstance(self.datasets['predict'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn)


class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_keyboard_interrupt(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            if "callbacks" in self.lightning_config:
                if 'metrics_over_trainsteps_checkpoint' in self.lightning_config['callbacks']:
                    os.makedirs(os.path.join(self.ckptdir, 'trainstep_checkpoints'), exist_ok=True)
            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass


class ImageLogger(Callback):
    def __init__(self,
                 batch_frequency, # Frequency of batches on which to log images
                 max_images,# Maximum number of images to log
                 clamp=True, # Whether to clamp pixel values to [-1,1]
                 increase_log_steps=True, # Whether to increase frequency of log steps exponentially
                 rescale=True,  # Whether to rescale pixel values to [0,1]
                 disabled=False, # Whether to disable logging
                 log_on_batch_idx=False, # Whether to log on batch index instead of global step
                 log_first_step=False,  # Whether to log on the first step
                 log_images_kwargs=None): # Additional keyword arguments to pass to log_images method

        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.TestTubeLogger: self._testtube,
        }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir,
                           split,
                           images,
                           pl_module.global_step,
                           pl_module.current_epoch,
                           batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
                check_idx > 0 or self.log_first_step):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                print(e)
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and pl_module.global_step > 0:
            self.log_img(pl_module, batch, batch_idx, split="val")
        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.root_gpu)
        torch.cuda.synchronize(trainer.root_gpu)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        torch.cuda.synchronize(trainer.root_gpu)
        max_memory = torch.cuda.max_memory_allocated(trainer.root_gpu) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass


if __name__ == "__main__":
    
    # 当前时间
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

   
    # parser = get_parser()
    # parser = Trainer.add_argparse_args(parser)
    # opt, unknown = parser.parse_known_args()    
    unknown=[]
    opt = types.SimpleNamespace()
    opt.accelerator=None
    opt.accumulate_grad_batches=1
    opt.amp_backend='native'
    opt.amp_level='O2'
    opt.auto_lr_find=False
    opt.auto_scale_batch_size=False
    opt.auto_select_gpus=False
    opt.base=['configs/autoencoder/autoencoder_kl_8x8x64.yaml']
    opt.benchmark=False
    opt.check_val_every_n_epoch=1
    opt.checkpoint_callback=True
    opt.debug=False
    opt.default_root_dir=None
    opt.deterministic=False
    opt.devices=None
    opt.distributed_backend=None
    opt.fast_dev_run=False
    opt.finetune=''
    opt.flush_logs_every_n_steps=100
    opt.gpus=None
    opt.gradient_clip_algorithm='norm'
    opt.gradient_clip_val=0.0
    opt.ipus=None
    opt.limit_predict_batches=1.0
    opt.limit_test_batches=1.0
    opt.limit_train_batches=1.0
    opt.limit_val_batches=1.0
    opt.log_every_n_steps=50
    opt.log_gpu_memory=None
    opt.logdir='logs'
    opt.logger=True
    opt.max_epochs=None
    opt.max_steps=None
    opt.max_time=None
    opt.min_epochs=None
    opt.min_steps=None
    opt.move_metrics_to_cpu=False
    opt.multiple_trainloader_mode='max_size_cycle'
    opt.name=''
    opt.no_test=False
    opt.num_nodes=1
    opt.num_processes=1
    opt.num_sanity_val_steps=2
    opt.overfit_batches=0.0
    opt.plugins=None
    opt.postfix=''
    opt.precision=32
    opt.prepare_data_per_node=True
    opt.process_position=0
    opt.profiler=None
    opt.progress_bar_refresh_rate=None
    opt.project=None
    opt.reload_dataloaders_every_epoch=False
    opt.reload_dataloaders_every_n_epochs=0
    opt.replace_sampler_ddp=True
    opt.resume=''
    opt.resume_from_checkpoint=None
    opt.scale_lr=True
    opt.seed=23
    opt.stochastic_weight_avg=False
    opt.sync_batchnorm=False
    opt.terminate_on_nan=False
    opt.tpu_cores=None
    opt.track_grad_norm=-1
    opt.train=True
    opt.truncated_bptt_steps=None
    opt.val_check_interval=1.0
    opt.weights_save_path=None
    opt.weights_summary='top'


    # 深度学习随机种子，seed_value可以是任一大于0的整数
    seed_everything(opt.seed)


    # 验证参数
    if opt.name and opt.resume:
        raise ValueError(
            "--name 和 --resume 不能同时指定。"
            "如果要在新的日志文件夹中继续训练：用 --name 和 --resume_from_checkpoint=logs/2024-07-09T17-24-01_autoencoder_kl_8x8x64/checkpoints/last.ckpt "
        )

    if opt.resume:#概述
        # 如果参数有，但是文件不存在
        if not os.path.exists(opt.resume):
            raise ValueError("没找到opt.resume= {}".format(opt.resume))

        # 如果参数有，文件存在
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")            
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume #
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt #logs/2024-07-09T17-24-01_autoencoder_kl_8x8x64/checkpoints/last.ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml"))) #logs/2024-07-09T17-24-01_autoencoder_kl_8x8x64/configs/*.yaml
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]

    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base: #当前代码执行这里
            
            # print('=========+++++++++++++++:',os.path.split(opt.base[0]) ) #('configs/autoencoder', 'autoencoder_kl_8x8x64.yaml')            
            cfg_fname = os.path.split(opt.base[0])[-1] #autoencoder_kl_8x8x64.yaml
            
            # print('=========+++++++++++++++:',os.path.splitext(cfg_fname) ) # ('autoencoder_kl_8x8x64', '.yaml')            
            cfg_name = os.path.splitext(cfg_fname)[0] #autoencoder_kl_8x8x64
            name = "_" + cfg_name #_autoencoder_kl_8x8x64            

        else:
            name = ""


        nowname = now + name + opt.postfix #2024-07-09T17-24-01_autoencoder_kl_8x8x64
        logdir = os.path.join(opt.logdir, nowname)

    # print('---+++++++++++++++++++++++-----',logdir) #logs\2024-07-09T17-24-01_autoencoder_kl_8x8x64   
    ckptdir = os.path.join(logdir, "checkpoints") #logs\2024-07-09T17-24-01_autoencoder_kl_8x8x64/checkpoints
    cfgdir = os.path.join(logdir, "configs")      #logs\2024-07-09T17-24-01_autoencoder_kl_8x8x64/configs
    
    ###########################################################################
 

    # 读取.yaml配置文件
    '''
     [{'model': {'base_learning_rate': 4.5e-06, 'target': 'ldm.models.autoencoder.AutoencoderKL', 'params': {'monitor': 'val/rec_loss', 'embed_dim': 64, 'lossconfig': {'target': 'ldm.modules.losses.LPIPSWithDiscriminator', 'params': {'disc_start': 50001, 'kl_weight': 1e-06, 'disc_weight': 0.5}}, 'ddconfig': {'double_z': True, 'z_channels': 64, 'resolution': 256, 'in_channels': 3, 'out_ch': 3, 'ch': 128, 'ch_mult': [1, 1, 2, 2, 4, 4], 'num_res_blocks': 2, 'attn_resolutions': [16, 8], 'dropout': 0.0}}}, 'data': {'target': 'main.DataModuleFromConfig', 'params': {'batch_size': 2, 'wrap': True, 'train': {'target': 'ldm.data.imagenet.ImageNetSRTrain', 'params': {'size': 256, 'degradation': 'pil_nearest'}}, 'validation': {'target': 'ldm.data.imagenet.ImageNetSRValidation', 'params': {'size': 256, 'degradation': 'pil_nearest'}}}}, 'lightning': {'callbacks': {'image_logger': {'target': 'main.ImageLogger', 'params': {'batch_frequency': 1000, 'max_images': 8, 'increase_log_steps': True}}}, 'trainer': {'benchmark': True, 'accumulate_grad_batches': 2}}}]
    '''
    configs = [OmegaConf.load(cfg) for cfg in opt.base] #opt.base=['configs/autoencoder/autoencoder_kl_8x8x64.yaml']

    # print('----------------------:',configs)
    # exit()

    # 合并参数
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    # 弹出 - 只获取 lightning
    #{'callbacks': {'image_logger': {'target': 'main.ImageLogger', 'params': {'batch_frequency': 1000, 'max_images': 8, 'increase_log_steps': True}}}, 'trainer': {'benchmark': True, 'accumulate_grad_batches': 2}}
    lightning_config = config.pop("lightning", OmegaConf.create())

    # {'benchmark': True, 'accumulate_grad_batches': 2}  - merge trainer cli with config
    trainer_config = lightning_config.get("trainer", OmegaConf.create()) #其实这样就可以：trainer_config = lightning_config.get("trainer")
    
    # 添加 ddp
    trainer_config["accelerator"] = "ddp" #{'benchmark': True, 'accumulate_grad_batches': 2, 'accelerator': 'ddp'}
        
    # 空 [] - 没有用到
    for k in nondefault_trainer_args(opt):
        trainer_config[k] = getattr(opt, k)
    
    if not "gpus" in trainer_config: #cpu执行这里
        del trainer_config["accelerator"]
        cpu = True        
    else: #gpu执行这里
        gpuinfo = trainer_config["gpus"]
        print(f"Running on GPUs {gpuinfo}")
        cpu = False

    # ----------------------------------------------------------------------
    trainer_opt = trainer_config #argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config

    print('------------22----------:',trainer_config)
    exit()

    # model
    model = instantiate_from_config(config.model)

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         # params.append(param)
    #         # names.append(name)
    #         print(name)
    #         # print(param)


    # 官方提供的模型无法通过resume方式加载
    if opt.finetune:
        model.load_state_dict(torch.load(opt.finetune)['state_dict'])

    # trainer and callbacks
    trainer_kwargs = dict()

    # default logger configs
    default_logger_cfgs = {
        "wandb": {
            "target": "pytorch_lightning.loggers.WandbLogger",
            "params": {
                "name": nowname,
                "save_dir": logdir,
                "offline": opt.debug,
                "id": nowname,
            }
        },
        "testtube": {
            "target": "pytorch_lightning.loggers.TestTubeLogger",
            "params": {
                "name": "testtube",
                "save_dir": logdir,
            }
        },
    }

    default_logger_cfg = default_logger_cfgs["testtube"]
    if "logger" in lightning_config:
        logger_cfg = lightning_config.logger
    else:
        logger_cfg = OmegaConf.create()
    logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
    trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

    # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
    # specify which metric is used to determine best pre_trained_models
    default_modelckpt_cfg = {
        "target": "pytorch_lightning.callbacks.ModelCheckpoint",
        "params": {
            "dirpath": ckptdir,
            "filename": "{epoch:06}",
            "verbose": True,
            "save_last": True,
        }
    }
    if hasattr(model, "monitor"):
        print(f"Monitoring {model.monitor} as checkpoint metric.")
        default_modelckpt_cfg["params"]["monitor"] = model.monitor
        default_modelckpt_cfg["params"]["save_top_k"] = 3

    if "modelcheckpoint" in lightning_config:
        modelckpt_cfg = lightning_config.modelcheckpoint
    else:
        modelckpt_cfg = OmegaConf.create()
    modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
    print(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")
    if version.parse(pl.__version__) < version.parse('1.4.0'):
        trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

    # add callback which sets up log directory
    default_callbacks_cfg = {
        "setup_callback": {
            "target": "main.SetupCallback",
            "params": {
                "resume": opt.resume,
                "now": now,
                "logdir": logdir,
                "ckptdir": ckptdir,
                "cfgdir": cfgdir,
                "config": config,
                "lightning_config": lightning_config,
            }
        },
        "image_logger": {
            "target": "main.ImageLogger",
            "params": {
                "batch_frequency": 750,
                "max_images": 4,
                "clamp": True
            }
        },
        "learning_rate_logger": {
            "target": "main.LearningRateMonitor",
            "params": {
                "logging_interval": "step",
                # "log_momentum": True
            }
        },
        "cuda_callback": {
            "target": "main.CUDACallback"
        },
    }
    if version.parse(pl.__version__) >= version.parse('1.4.0'):
        default_callbacks_cfg.update({'checkpoint_callback': modelckpt_cfg})

    if "callbacks" in lightning_config:
        callbacks_cfg = lightning_config.callbacks
    else:
        callbacks_cfg = OmegaConf.create()

    if 'metrics_over_trainsteps_checkpoint' in callbacks_cfg:
        print(
            'Caution: Saving checkpoints every n train steps without deleting. This might require some free space.')
        default_metrics_over_trainsteps_ckpt_dict = {
            'metrics_over_trainsteps_checkpoint':
                {"target": 'pytorch_lightning.callbacks.ModelCheckpoint',
                 'params': {
                     "dirpath": os.path.join(ckptdir, 'trainstep_checkpoints'),
                     "filename": "{epoch:06}-{step:09}",
                     "verbose": True,
                     'save_top_k': -1,
                     'every_n_train_steps': 10000,
                     'save_weights_only': True
                 }
                 }
        }
        default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
    if 'ignore_keys_callback' in callbacks_cfg and hasattr(trainer_opt, 'resume_from_checkpoint'):
        callbacks_cfg.ignore_keys_callback.params['ckpt_path'] = trainer_opt.resume_from_checkpoint
    elif 'ignore_keys_callback' in callbacks_cfg:
        del callbacks_cfg['ignore_keys_callback']

    trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

    trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
    trainer.logdir = logdir

    # data
    data = instantiate_from_config(config.data)
    # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
    # calling these ourselves should not be necessary but it is.
    # lightning still takes care of proper multiprocessing though
    print('----------end1111---------',data)
    # exit()
    data.prepare_data()
    data.setup()

    print("#### Data #####")
    for k in data.datasets:
        print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

    # configure learning rate
    bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
    if not cpu:
        print('---if not cpu --')
        ngpu = len(lightning_config.trainer.gpus.strip(",").split(','))
    else:
        print('---ngpu = 1 --')
        ngpu = 1

    if 'accumulate_grad_batches' in lightning_config.trainer:
        accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
    else:
        accumulate_grad_batches = 1
        
    print(f"accumulate_grad_batches = {accumulate_grad_batches}")
    lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
    if opt.scale_lr:
        model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
        print(
            "-----------Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
    else:
        model.learning_rate = base_lr
        print("++++ NOT USING LR SCALING ++++")
        print(f"Setting learning rate to {model.learning_rate:.2e}")


    # allow checkpointing via USR1
    def melk(*args, **kwargs):
        # run all checkpoint hooks
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)


    def divein(*args, **kwargs):
        if trainer.global_rank == 0:
            import pudb;
            pudb.set_trace()

    

    # # wll 修改 windows没有这玩意
    # import signal
    # signal.signal(signal.SIGUSR1, melk)
    # signal.signal(signal.SIGUSR2, divein)


    # run
    if opt.train:
        try:
            trainer.fit(model, data)
        except Exception:
            melk()
            raise
    if not opt.no_test and not trainer.interrupted:
        trainer.test(model, data)
    
       
        