'''
阶段2：扩散模型主体(Diffusion Model) CLIP+UNet+阶段1:自动编码器(autoencoder),扩散模型 压缩与解压图片到 latent 空间
    运行: python wll.py --base configs/latent-diffusion/txt2img/txt2img-sdv1.yaml --train True 

Downloading: "https://github.com/DagnyT/hardnet/raw/master/pretrained/train_liberty_with_aug/checkpoint_liberty_with_aug.pth" to C:/Users/Administrator/.cache/torch/hub/checkpoints/checkpoint_liberty_with_aug.pth
pip install ftfy -i https://pypi.tuna.tsinghua.edu.cn/simple

'''
import os, glob
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import  DataLoader, Dataset
from functools import partial
from PIL import Image
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
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
opt = types.SimpleNamespace()
opt.base=['configs/latent-diffusion/txt2img/txt2img-sdv1.yaml']
opt.finetune=''
opt.scale_lr=True


# 深度学习随机种子，seed_value可以是任一大于0的整数
seed_everything(23)

# 读取.yaml配置文件
config = OmegaConf.load(opt.base[0]) #opt.base=['configs/autoencoder/autoencoder_kl_8x8x64.yaml']


############################### 四、读取数据集 ###############################

# 手动方式数据集合
data=DataModuleFromConfig( **config.data.get("params", dict()) ) # **表示这是一个可变关键字参数 会以字典的形式提供给函数参数，字典的键是参数的名字，值是传递给参数的值。
data.prepare_data() #声明完对象：要第一个运行
data.setup()        #声明完对象：要第二个运行


print("#### Data #####")
'''
train, WrappedDataset, 48627
validation, WrappedDataset, 48627
'''
for k in data.datasets:
    print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")


############################### 二、创建模型 ###############################

# # model
# model = getattr( importlib.import_module('ldm.models.diffusion.ddpm', package=None) , 'LatentDiffusion' )( **config.model.get("params", dict()) ) #也可以写死这样
from ldm.models.diffusion.ddpm import  LatentDiffusion
model = LatentDiffusion( **config.model.get("params", dict()) ) #也可以写死这样    


# 官方提供的模型无法通过这种方式加载
if opt.finetune:
    model.load_state_dict(torch.load(opt.finetune)['state_dict'])


# 模型学习率
bs = config.data.params.batch_size          # 数据批次
base_lr = config.model.base_learning_rate   # rl
accumulate_grad_batches=2 #
ngpu = 1 #cpu核数 或 cuda数 , 根据实际情况更改
if opt.scale_lr: #执行这里
    model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr # accumulate_grad_batches2 * ngpu1 * bs2 * base_lr_4.5e-6 #1.80e-05 = 2 (accumulate_grad_batches) * 1 (num_gpus) * 2 (batchsize) * 4.50e-06 (base_lr)    
    print("----模型当前的学习率： {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format( model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
else:
    model.learning_rate = base_lr   
    print(f"----自动设置模型当前的学习率： scale_lr :  {model.learning_rate:.2e}")


############################### 五、训练 ###############################

# 训练模型  5 个 epoch，并在每训练 60 步（batch）时保存一个 checkpoint。ModelCheckpoint 回调函数的 save_top_k 参数为 -1，表示保存所有 checkpoint。
# 利用 save_top_k 设置保存所有的模型，因为不是通过 monitor 的形式保存的，所以 save_top_k 只能设置为 -1，0，1，分别表示保存所有的模型，不保存模型和保存最后一个模型
# 认会保存在 lightning_logs 下，可以通过 Trainer 的参数配置保存路径（可以去文档查一下，具体忘了参数名叫什么）
trainer = pl.Trainer(
    # gpus=1,
    max_epochs=5,
    callbacks=[pl.callbacks.ModelCheckpoint(every_n_train_steps=60, save_top_k=-1)]
)
trainer.fit(model, data)
'''
def forward(self, x, c, *args, **kwargs):
        # # print('模型x',x.shape) #torch.Size([1, 4, 64, 64])
        # print('模型c',c) #模型c ['000000064898.jpg']
        # exit()

DDPM基类: 
    def training_step(self, batch, batch_idx):
            loss, loss_dict = self.shared_step(batch)
            print('==========================99999999999999999==============',loss)
            exit()

    def shared_step(self, batch):
        x = self.get_input(batch=batch, k=self.first_stage_key)       
        loss, loss_dict = self(x) ##########  这里放到当前模型，当前模型是什么呢？ CLIP+UNet
        return loss, loss_dict

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, 'b h w c -> b c h w')
        if self.use_fp16:
            x = x.to(memory_format=torch.contiguous_format).half()
        else:
            x = x.to(memory_format=torch.contiguous_format).float()
        return x



其它：
    from transformers import CLIPTokenizer, CLIPTextModel
    class FrozenCLIPEmbedder(AbstractEncoder):
        """Uses the CLIP transformer encoder for text (from Hugging Face)"""
        def __init__(self, version="openai/clip-vit-large-patch14", device="cpu", max_length=77):
        # def __init__(self, version="/opt/data/private/latent-diffusion/pre_trained_models/openai/clip-vit-large-patch14", device="cpu", max_length=77):

            super().__init__()
            self.tokenizer = CLIPTokenizer.from_pretrained(version)
            self.transformer = CLIPTextModel.from_pretrained(version)
            self.device = device
            self.max_length = max_length
            self.freeze()

        def freeze(self):
            self.transformer = self.transformer.eval()
            for param in self.parameters():
                param.requires_grad = False

        def forward(self, text):
            batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                            return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
            tokens = batch_encoding["input_ids"].to(self.device)
            outputs = self.transformer(input_ids=tokens) #这个clip输入=图片了 哈哈 https://blog.csdn.net/FrenzyTechAI/article/details/132696135

            z = outputs.last_hidden_state
            return z

        def encode(self, text):
            return self(text)

'''



####################################################################################################
   
# # 加载已保存的模型
# # 当我们需要从保存的 checkpoint 恢复模型时，可以使用 Trainer 类的 resume_from_checkpoint 参数：
# trainer = pl.Trainer(
#     gpus=1,
#     max_epochs=5,
#     callbacks=[pl.callbacks.ModelCheckpoint(every_n_train_steps=60, save_top_k=-1)],
#     resume_from_checkpoint='path/to/checkpoint.ckpt'
# )
# model = Net(num_classes=10)
# trainer.fit(model, train_loader, val_loader)
