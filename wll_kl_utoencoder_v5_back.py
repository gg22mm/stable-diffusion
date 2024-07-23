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
import os,  datetime, glob
import numpy as np
import time
import torch
import torchvision

from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import  DataLoader, Dataset
from functools import partial
from PIL import Image

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info
# 
import types
import importlib
from ldm.data.base import Txt2ImgIterableBaseDataset

# 根据.yaml来加载模型
def instantiate_from_config(config):
    
    # 验证是否在模型路径配置
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("必须要配置模型路径在.yaml中的 target 字段 ")

    print('----------加载模型，并传入模型配置中的.yaml参数：params---------')    
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1) #ldm.models.autoencoder.AutoencoderKL 变成： module=ldm.models.autoencoder , cls=AutoencoderKL
        
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)

    # 加载模型 - 载入类中的方法
    return getattr( importlib.import_module(module, package=None) , cls ) #getattr(对象=ldm.models.autoencoder,方法=AutoencoderKL)


# 数据集方法1
class WrappedDataset(Dataset):
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# 数据集方法2
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

# 数据集方法3
class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = 0 # num_workers if num_workers is not None else batch_size * 2
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
                          num_workers=0,#self.num_workers,
                          shuffle=False if is_iterable_dataset else True,
                          worker_init_fn=init_fn)

    def _val_dataloader(self, shuffle=False):
        if isinstance(self.datasets['validation'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=0,#self.num_workers,
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
                          num_workers=0,#self.num_workers,
                          worker_init_fn=init_fn, shuffle=shuffle)

    def _predict_dataloader(self, shuffle=False):
        if isinstance(self.datasets['predict'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          num_workers=0,#self.num_workers, 
                          worker_init_fn=init_fn)


#################################################################################
    
# 当前时间
now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
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

# 继续训练
opt.resume=''                   # true
opt.resume_from_checkpoint=None # 模型目录： ./logs/2024-07-09T17-24-01_autoencoder_kl_8x8x64/    #不要后面的 这些 checkpoints/last.ckpt
opt.train=True                  # 是否开启训练

opt.scale_lr=True
opt.seed=23
opt.stochastic_weight_avg=False
opt.sync_batchnorm=False
opt.terminate_on_nan=False
opt.tpu_cores=None
opt.track_grad_norm=-1
opt.truncated_bptt_steps=None
opt.val_check_interval=1.0
opt.weights_save_path=None
opt.weights_summary='top'

# 深度学习随机种子，seed_value可以是任一大于0的整数
seed_everything(opt.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 验证参数
if opt.name and opt.resume:
    print('--name 和 --resume 不能同时指定。')
    exit()
    

# 官方提供的模型无法通过 resume 方式加载
if opt.resume:#继续训练为:true , 配合： resume_from_checkpoint = ./myModelDir/    

    ckpt = os.path.join(opt.resume_from_checkpoint, "checkpoints", "last.ckpt")  #logs/2024-07-09T17-24-01_autoencoder_kl_8x8x64/checkpoints/last.ckpt  
    opt.resume_from_checkpoint = ckpt  #靠这个

    base_configs = sorted(glob.glob(os.path.join(opt.resume_from_checkpoint, "configs/*.yaml"))) #logs/2024-07-09T17-24-01_autoencoder_kl_8x8x64/configs/*.yaml
    opt.base = base_configs + opt.base #靠这个   
    nowname = './logs/resume_output/'  #继续训练模型输出

else: #一般都是运行这里，直接训练
    
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

# print('---+++++++++++++++++++++++-----',logdir)   #logs\2024-07-09T17-24-01_autoencoder_kl_8x8x64   
ckptdir = os.path.join(logdir, "checkpoints")       #logs\2024-07-09T17-24-01_autoencoder_kl_8x8x64/checkpoints
cfgdir = os.path.join(logdir, "configs")            #logs\2024-07-09T17-24-01_autoencoder_kl_8x8x64/configs

############################### 一、读取.yaml ###############################


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
#所有是这样的：{'callbacks': {'image_logger': {'target': 'main.ImageLogger', 'params': {'batch_frequency': 1000, 'max_images': 8, 'increase_log_steps': True}}}, 'trainer': {'benchmark': True, 'accumulate_grad_batches': 2}}
lightning_config = config.pop("lightning", OmegaConf.create()) #OmegaConf.create() 是一个废话:{}


# {'benchmark': True, 'accumulate_grad_batches': 2}  - merge trainer cli with config
trainer_opt = lightning_config.get("trainer", OmegaConf.create()) #其实这样就可以： trainer_opt = lightning_config.get("trainer")

# 为 cuda 添加 ddp
trainer_opt["accelerator"] = "ddp" #{'benchmark': True, 'accumulate_grad_batches': 2, 'accelerator': 'ddp'}

if device=='cpu': #cpu执行这里,不要accelerator
    del trainer_opt["accelerator"]
    
lightning_config.trainer = trainer_opt #从新覆盖.yaml中的配置 , 以后用： trainer_opt 就是正确的


############################### 四、读取数据集 ###############################

# 手动方式数据集合
data = getattr( importlib.import_module('wll_kl_utoencoder', package=None) , 'DataModuleFromConfig' )( **config.data.get("params", dict()) ) #也可以写死这样 
data.prepare_data() #没什么用
data.setup()        #没什么用


print("#### Data #####")
'''
train, WrappedDataset, 48627
validation, WrappedDataset, 48627
'''
for k in data.datasets:
    print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")


############################### 二、创建模型 ###############################

# model
model = getattr( importlib.import_module('ldm.models.autoencoder', package=None) , 'AutoencoderKL' )( **config.model.get("params", dict()) ) #也可以写死这样    
# print(model)
# exit()

# 官方提供的模型无法通过 resume 方式加载
if opt.finetune:
    model.load_state_dict(torch.load(opt.finetune)['state_dict'])


# 模型学习率
bs = config.data.params.batch_size          # 数据批次
base_lr = config.model.base_learning_rate   # rl
accumulate_grad_batches=lightning_config.trainer.accumulate_grad_batches #缯的什么数
ngpu = 1 #cpu核数 或 cuda数 , 根据实际情况更改
if opt.scale_lr: #执行这里
    model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr # accumulate_grad_batches2 * ngpu1 * bs2 * base_lr_4.5e-6 #1.80e-05 = 2 (accumulate_grad_batches) * 1 (num_gpus) * 2 (batchsize) * 4.50e-06 (base_lr)    
    print("----模型当前的学习率： {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format( model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
else:
    model.learning_rate = base_lr   
    print(f"----自动设置模型当前的学习率： scale_lr :  {model.learning_rate:.2e}")


############################### 三、附加 训练参数放到：pytorch_lightning的 Trainer 中 ############################### 

# trainer and callbacks - 训练 和 回调
trainer_kwargs = dict()

# 1、日志配置,不放到.yaml中
default_logger_cfgs = {
    "wandb": {
        "target": "pytorch_lightning.loggers.WandbLogger",
        "params": {
            "name": nowname, #2024-07-09T17-24-01_autoencoder_kl_8x8x64
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
# 手动方式,可以换成 wandb 方式
trainer_kwargs["logger"] = getattr( importlib.import_module('pytorch_lightning.loggers', package=None) , 'TestTubeLogger' )( **default_logger_cfgs["testtube"].get("params", dict()) ) #也可以写死这样   


trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs) #pytorch_lightning的 Trainer 参数： {'benchmark': True, 'accumulate_grad_batches': 2, 'accelerator': 'ddp'} , trainer_kwargs
trainer.logdir = logdir


############################### 五、训练 ###############################


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

   
        