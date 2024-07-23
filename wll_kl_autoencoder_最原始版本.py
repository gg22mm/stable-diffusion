'''
阶段1：自动编码器(autoencoder),扩散模型 latent 空间
    训练原因: 在你自己的数据集上进行训练可以帮助你获得更好的令牌，从而为你的领域获得更好的图像。 （很少人训练这一步）
    在ImageNet上训练 KL 正则化自动编码器的配置在configs/autoencoder中提供。
    作者未完成:
        要训练 VQ 正则化模型，请参阅taming-transformers存储库： https://github.com/CompVis/taming-transformers

        KL正则化自动编码器和VQ正则化自动编码器都是一种用于图像压缩的自动编码器，但它们之间有一些区别。
        KL正则化自动编码器通过引入KL散度损失函数来约束生成的分布与原始分布之间的距离，从而使得生成的图像更加接近原始图像。这种方法可以帮助提高生成图像的质量，但需要大量的计算资源和时间。
        VQ正则化自动编码器则通过将原始图像编码为一组离散值（即向量），然后使用这些向量来重建原始图像。这种方法可以提高生成图像的质量，并且可以在不增加计算复杂度的情况下提高压缩率。但是，由于只能使用离散值进行编码，因此生成的图像可能会失去一些细节信息。
        总的来说，KL正则化自动编码器和VQ正则化自动编码器都是用于图像压缩的有效方法，但它们的应用场景和效果略有不同。
    

    SD模型的主体结构如下图所示，主要包括三个模型：        
            autoencoder 中的 encoder 与 decoder 都是用扩散的方式(生成图像的 latents )， encoder是压缩到 latent 空间, decoder 是解码成为图像；
            CLIP text encoder : 提取输入text的text embeddings，通过cross attention(与自注意力相比，cross-attention就更简单了，它用于处理两个不同模态序列之间的关联，在多模态场景中用于将文本和图像等不同类型的数据进行交互处理)方式送入扩散模型的UNet中作为condition；
            UNet:扩散模型的主体 - 用文本引导扩散模型来生成图像的 latents 。  UNet中引入text condition来实现基于文本生成图像


    例子：
            
        #autoencoder
            
        #在这个例子中，我们定义了一个自编码器，其中encoder部分用于从文本提示中提取latents，
        #decoder部分用于重构原始文本提示。然后我们用一个随机生成的1x10的向量作为文本提示，通过encoder网络得到对应的latents。
        #请注意，这只是一个简化的例子，实际的自编码器或GAN模型会更复杂，并且可能需要额外的训练步骤来优化模型参数。 
        
        import torch
        import torch.nn as nn
         
        # 假设这是一个简单的自编码器模型
        class AutoEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(10, 5),  # 假设输入文本特征维度是10
                    nn.ReLU(),
                    nn.Linear(5, 3)  # 假设我们想要生成3个latents
                )
                self.decoder = nn.Sequential(
                    nn.Linear(3, 5),
                    nn.ReLU(),
                    nn.Linear(5, 10)
                )
         
            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return encoded, decoded
         
        # 实例化模型并生成latents
        model = AutoEncoder()
        text_prompt = torch.randn(1, 10)  # 假设这是一个1x10的文本提示向量
        latents = model.encoder(text_prompt)     
        print(latents)

        myAutoEncoder = model(text_prompt)
        print(myAutoEncoder)


阶段2：扩散模型主体(Diffusion Model) CLIP+UNet
    运行: python wll.py --base configs/latent-diffusion/txt2img/txt2img-sdv1.yaml --train True 


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


# 数据集方法1 - 按条件把训练数据集 和 测试数据集 进行格式化，如果数据集合没有标准的 "Dataset" 格式就需要执行这个
class WrappedDataset(Dataset):
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


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
        
        # 训练与测试只能同时都有值
        if train is not None:
            #初始化模型时指定使用类配置
            self.dataset_configs["train"] = train #{'target': 'ldm.data.imagenet.ImageNetSRTrain', 'params': {'size': 256, 'degradation': 'pil_nearest'}} 
            #调用上面初始化的类来 生成数据集
            self.train_dataloader = self._train_dataloader #这里以后可以自己写死
        
        if validation is not None:
            #初始化模型时指定使用类配置
            self.dataset_configs["validation"] = validation #初始化模型时指定使用类配置 #{'target': 'ldm.data.imagenet.ImageNetSRValidation', 'params': {'size': 256, 'degradation': 'pil_nearest'}}
            #调用上面初始化的类来 生成数据集 
            self.val_dataloader = self._val_dataloader #partial(self._val_dataloader, shuffle=shuffle_val_dataloader) #这里以后可以自己写死

        # 测试与预测可能同没有值
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = self._test_dataloader #partial(self._test_dataloader, shuffle=shuffle_test_loader) #这里以后可以自己写死
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader #这里以后可以自己写死

        # 是否要修正数据格式： 如果不是标准的格式需要在 .yaml中传进来的 True | False
        self.wrap = wrap

    
    # 声明完对象：要第一个运行,注:根据初始化模型时来导入对象 - 初始化模型时指定使用类配置
    def prepare_data(self):
        # 导入训练集对象与参数 和 测试集对象与参数
        # dict_values([{'target': 'ldm.data.imagenet.ImageNetSRTrain', 'params': {'size': 256, 'degradation': 'pil_nearest'}}, {'target': 'ldm.data.imagenet.ImageNetSRValidation', 'params': {'size': 256, 'degradation': 'pil_nearest'}}])
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    # 声明完对象：要第二个运行
    def setup(self, stage=None):        
        # {'train': <ldm.data.imagenet.ImageNetSRTrain object at 0x00000199FDD5C160>, 'validation': <ldm.data.imagenet.ImageNetSRValidation object at 0x00000199FDD52EE0>}
        self.datasets = dict(    (k, instantiate_from_config(self.dataset_configs[k]))   for k in self.dataset_configs  ) #k=train 和 validation       

        # 按条件把训练数据集 和 测试数据集 进行格式化，如果数据集合没有标准的 "Dataset" 格式就需要执行这个 def __init__(self, dataset): def __len__(self): def __getitem__(self, idx): 有这些才是标准的格式
        if self.wrap:# True
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k]) #{'train': <__main__.WrappedDataset object at 0x000001BC2A7478B0>, 'validation': <__main__.WrappedDataset object at 0x000001BC2A747A90>}



    # 训练数据获取
    def _train_dataloader(self): 
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,num_workers=0,worker_init_fn=None,shuffle=True)

    def _val_dataloader(self, shuffle=False):       
        return DataLoader(self.datasets["validation"], batch_size=self.batch_size,num_workers=0,worker_init_fn=None,shuffle=False)

    def _test_dataloader(self, shuffle=False):
        return DataLoader(self.datasets["test"], batch_size=self.batch_size,num_workers=0,worker_init_fn=None,shuffle=False)

    def _predict_dataloader(self, shuffle=False):        
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,num_workers=0,worker_init_fn=None,shuffle=False)


#################################################################################
    
# 当前时间
opt = types.SimpleNamespace()
opt.base=['configs/autoencoder/autoencoder_kl_8x8x64.yaml']
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


# print('---------------11-------------')
# exit()

############################### 二、创建模型 ###############################



# model
# model = getattr( importlib.import_module('ldm.models.autoencoder', package=None) , 'AutoencoderKL' )( **config.model.get("params", dict()) ) #也可以写死这样 

from ldm.models.autoencoder import  AutoencoderKL
model = AutoencoderKL( **config.model.get("params", dict()) ) #也可以写死这样    

# print(model)
# exit()

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
