import os, yaml, pickle, shutil, tarfile, glob
import cv2
import albumentations
import PIL
import numpy as np
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
from functools import partial
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, Subset

import ldm.data.utils as tdu
from ldm.util import download, retrieve
from ldm.data.base import ImagePaths

from ldm.modules.image_degradation import degradation_fn_bsr, degradation_fn_bsr_light


def synset2idx(path_to_yaml="data/index_synset.yaml"):
    with open(path_to_yaml) as f:
        di2s = yaml.load(f)
    return dict((v,k) for k,v in di2s.items())

def give_synsets_from_indices(indices, path_to_yaml="data/imagenet_idx_to_synset.yaml"):
    synsets = []
    with open(path_to_yaml) as f:
        di2s = yaml.load(f)
    for idx in indices:
        synsets.append(str(di2s[idx]))
    print("Using {} different synsets for construction of Restriced Imagenet.".format(len(synsets)))
    return synsets


def str_to_indices(string):
    """Expects a string in the format '32-123, 256, 280-321'"""
    assert not string.endswith(","), "provided string '{}' ends with a comma, pls remove it".format(string)
    subs = string.split(",")
    indices = []
    for sub in subs:
        subsubs = sub.split("-")
        assert len(subsubs) > 0
        if len(subsubs) == 1:
            indices.append(int(subsubs[0]))
        else:
            rang = [j for j in range(int(subsubs[0]), int(subsubs[1]))]
            indices.extend(rang)
    return sorted(indices)




class ImageNetBase(Dataset):
    def __init__(self, config=None):
        self.config = config or OmegaConf.create()
        if not type(self.config)==dict:
            self.config = OmegaConf.to_container(self.config)
        self.keep_orig_class_label = self.config.get("keep_orig_class_label", False)
        self.process_images = True  # if False we skip loading & processing images and self.data contains filepaths
        self._prepare()
        self._prepare_synset_to_human()
        self._prepare_idx_to_synset()
        self._prepare_human_to_integer_label()
        self._load()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def _prepare(self):
        raise NotImplementedError()

    def _filter_relpaths(self, relpaths):
        ignore = set([
            "n06596364_9591.JPEG",
        ])
        relpaths = [rpath for rpath in relpaths if not rpath.split("/")[-1] in ignore]
        if "sub_indices" in self.config:
            indices = str_to_indices(self.config["sub_indices"])
            synsets = give_synsets_from_indices(indices, path_to_yaml=self.idx2syn)  # returns a list of strings
            self.synset2idx = synset2idx(path_to_yaml=self.idx2syn)
            files = []
            for rpath in relpaths:
                syn = rpath.split("/")[0]
                if syn in synsets:
                    files.append(rpath)
            return files
        else:
            return relpaths

    def _prepare_synset_to_human(self):
        SIZE = 2655750
        URL = "https://heibox.uni-heidelberg.de/f/9f28e956cd304264bb82/?dl=1"
        self.human_dict = os.path.join(self.root, "synset_human.txt")
        if (not os.path.exists(self.human_dict) or
                not os.path.getsize(self.human_dict)==SIZE):
            download(URL, self.human_dict)

    def _prepare_idx_to_synset(self):
        URL = "https://heibox.uni-heidelberg.de/f/d835d5b6ceda4d3aa910/?dl=1"
        self.idx2syn = os.path.join(self.root, "index_synset.yaml")
        if (not os.path.exists(self.idx2syn)):
            download(URL, self.idx2syn)

    def _prepare_human_to_integer_label(self):
        URL = "https://heibox.uni-heidelberg.de/f/2362b797d5be43b883f6/?dl=1"
        self.human2integer = os.path.join(self.root, "imagenet1000_clsidx_to_labels.txt")
        if (not os.path.exists(self.human2integer)):
            download(URL, self.human2integer)
        with open(self.human2integer, "r") as f:
            lines = f.read().splitlines()
            assert len(lines) == 1000
            self.human2integer_dict = dict()
            for line in lines:
                value, key = line.split(":")
                self.human2integer_dict[key] = int(value)

    def _load(self):        
        with open(self.txt_filelist, "r") as f:
            self.relpaths = f.read().splitlines() #图片列表
            l1 = len(self.relpaths)

            self.relpaths = self._filter_relpaths(self.relpaths)
            print("-------------在筛选过程中从文件列表中删除了{}个文件".format(l1 - len(self.relpaths)))

        # print(self.relpaths)
        

        self.synsets = [p.split("/")[0] for p in self.relpaths]       
        self.abspaths = [os.path.join(self.datadir, p) for p in self.relpaths]
        # print(self.abspaths)
        # exit()

        unique_synsets = np.unique(self.synsets)
        class_dict = dict((synset, i) for i, synset in enumerate(unique_synsets))
        if not self.keep_orig_class_label:
            self.class_labels = [class_dict[s] for s in self.synsets]
        else:
            self.class_labels = [self.synset2idx[s] for s in self.synsets]

        with open(self.human_dict, "r") as f:
            human_dict = f.read().splitlines()
            human_dict = dict(line.split(maxsplit=1) for line in human_dict)

        self.human_labels = [human_dict[s] for s in self.synsets]

        '''
        {
            'relpath': array(['n01440764/ILSVRC2012_val_00000293.JPEG',
               'n01440764/ILSVRC2012_val_00002138.JPEG',
               'n01440764/ILSVRC2012_val_00003014.JPEG', ...,
               'n15075141/ILSVRC2012_val_00046353.JPEG',
               'n15075141/ILSVRC2012_val_00047144.JPEG',
               'n15075141/ILSVRC2012_val_00049174.JPEG'], dtype='<U38'),
            
            'synsets': array(['n01440764', 'n01440764', 'n01440764', ..., 'n15075141',
               'n15075141', 'n15075141'], dtype='<U9'),

            'class_label': array([  0,   0,   0, ..., 999, 999, 999]),

            'human_label': array(['tench, Tinca tinca', 'tench, Tinca tinca', 'tench, Tinca tinca',
               ..., 'toilet tissue, toilet paper, bathroom tissue',
               'toilet tissue, toilet paper, bathroom tissue',
               'toilet tissue, toilet paper, bathroom tissue'], dtype='<U121')
        }

        '''
        labels = {
            "relpath": np.array(self.relpaths),
            "synsets": np.array(self.synsets),
            "class_label": np.array(self.class_labels),
            "human_label": np.array(self.human_labels),
        }

        # print(labels)
        # exit()
        
        # labels = {
        #     "relpath": np.array(self.relpaths),
        #     "synsets": np.array([1]),           #21,841类别(synsets)
        #     "class_label": np.array([2]),
        #     "human_label": np.array([3]),
        # }

        if self.process_images:
            self.size = retrieve(self.config, "size", default=256)
            self.data = ImagePaths(self.abspaths,
                                   labels=labels,
                                   size=self.size,
                                   random_crop=self.random_crop,
                                   )
        else:
            self.data = self.abspaths


class ImageNetTrain(ImageNetBase):
    NAME = "ILSVRC2012_validation" #ILSVRC2012_train
    URL = "http://www.image-net.org/challenges/LSVRC/2012/"
    AT_HASH = "a306397ccf9c2ead27155983c254227c0fd938e2"
    FILES = [
        "ILSVRC2012_img_train.tar",
    ]
    SIZES = [
        147897477120,
    ]

    def __init__(self, process_images=True, data_root=None, **kwargs):
        self.process_images = process_images
        self.data_root = data_root
        super().__init__(**kwargs)

    def _prepare(self):
        
        if self.data_root:
            self.root = os.path.join(self.data_root, self.NAME)
            # print(self.root) #data/myimages/ILSVRC2012_validation
            # exit()
        else:
            # cachedir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
            # self.root = os.path.join(cachedir, "autoencoders/data", self.NAME)
            print('不要在线下载ILSVRC2012,太大了')
            exit()

        self.datadir = os.path.join(self.root, "data")
        # print(self.datadir) #data/myimages/ILSVRC2012_validation\data
        # exit()

        
        self.txt_filelist = os.path.join(self.root, "me_images.txt")  
        print('================',self.txt_filelist)  
        self.expected_length = 1281167
        self.random_crop = retrieve(self.config, "ImageNetTrain/random_crop",   default=True)
        

        if not tdu.is_prepared(self.root):

            # prep
            print("Preparing dataset {} in {}".format(self.NAME, self.root))


            datadir = self.datadir

            # if not os.path.exists(datadir):
            #     path = os.path.join(self.root, self.FILES[0])
            #     if not os.path.exists(path) or not os.path.getsize(path)==self.SIZES[0]:
            #         import academictorrents as at
            #         atpath = at.get(self.AT_HASH, datastore=self.root)
            #         assert atpath == path

            #     print("Extracting {} to {}".format(path, datadir))
            #     os.makedirs(datadir, exist_ok=True)
            #     with tarfile.open(path, "r:") as tar:
            #         tar.extractall(path=datadir)

            #     print("Extracting sub-tars.")
            #     subpaths = sorted(glob.glob(os.path.join(datadir, "*.tar")))
            #     for subpath in tqdm(subpaths):
            #         subdir = subpath[:-len(".tar")]
            #         os.makedirs(subdir, exist_ok=True)
            #         with tarfile.open(subpath, "r:") as tar:
            #             tar.extractall(path=subdir)

            # filelist = glob.glob(os.path.join(datadir, "**", "*.JPEG"))
            # filelist = glob.glob(os.path.join(datadir, "*.JPEG")) 
            
            filelist = glob.glob(os.path.join(datadir, "*", "*.JPEG"))
            filelist = [os.path.relpath(p, start=datadir) for p in filelist]
            filelist = sorted(filelist)
            filelist = "\n".join(filelist)+"\n"
            with open(self.txt_filelist, "w") as f:
                f.write(filelist)

        tdu.mark_prepared(self.root)


class ImageNetValidation(ImageNetBase):
    NAME = "ILSVRC2012_validation"
    URL = "http://www.image-net.org/challenges/LSVRC/2012/"
    AT_HASH = "5d6d0df7ed81efd49ca99ea4737e0ae5e3a5f2e5"
    VS_URL = "https://heibox.uni-heidelberg.de/f/3e0f6e9c624e45f2bd73/?dl=1"
    FILES = [
        "ILSVRC2012_img_val.tar",
        "validation_synset.txt",
    ]
    SIZES = [
        6744924160,
        1950000,
    ]

    def __init__(self, process_images=True, data_root=None, **kwargs):
        self.data_root = data_root
        self.process_images = process_images
        super().__init__(**kwargs)

    def _prepare(self):
        
        if self.data_root:
            self.root = os.path.join(self.data_root, self.NAME)
            # print(self.root) #data/myimages/ILSVRC2012_validation
            # exit()
        else:
            # cachedir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
            # self.root = os.path.join(cachedir, "autoencoders/data", self.NAME)
            print('不要在线下载ILSVRC2012,太大了')
            exit()

        self.datadir = os.path.join(self.root, "data")
        # print(self.datadir) #data/myimages/ILSVRC2012_validation\data
        # exit()

        
        self.txt_filelist = os.path.join(self.root, "me_images.txt")    
        print('================',self.txt_filelist)
        self.expected_length = 1281167
        self.random_crop = retrieve(self.config, "ImageNetTrain/random_crop",   default=True)
        

        if not tdu.is_prepared(self.root):

            # prep
            print("Preparing dataset {} in {}".format(self.NAME, self.root))


            datadir = self.datadir

            # if not os.path.exists(datadir):
            #     path = os.path.join(self.root, self.FILES[0])
            #     if not os.path.exists(path) or not os.path.getsize(path)==self.SIZES[0]:
            #         import academictorrents as at
            #         atpath = at.get(self.AT_HASH, datastore=self.root)
            #         assert atpath == path

            #     print("Extracting {} to {}".format(path, datadir))
            #     os.makedirs(datadir, exist_ok=True)
            #     with tarfile.open(path, "r:") as tar:
            #         tar.extractall(path=datadir)

            #     print("Extracting sub-tars.")
            #     subpaths = sorted(glob.glob(os.path.join(datadir, "*.tar")))
            #     for subpath in tqdm(subpaths):
            #         subdir = subpath[:-len(".tar")]
            #         os.makedirs(subdir, exist_ok=True)
            #         with tarfile.open(subpath, "r:") as tar:
            #             tar.extractall(path=subdir)

            # filelist = glob.glob(os.path.join(datadir, "**", "*.JPEG"))
            # filelist = glob.glob(os.path.join(datadir, "*.JPEG")) 
            
            filelist = glob.glob(os.path.join(datadir, "*", "*.JPEG"))
            filelist = [os.path.relpath(p, start=datadir) for p in filelist]
            filelist = sorted(filelist)
            filelist = "\n".join(filelist)+"\n"
            with open(self.txt_filelist, "w") as f:
                f.write(filelist)

        tdu.mark_prepared(self.root)



class ImageNetSR(Dataset):
    def __init__(self, size=None,
                 degradation=None,
                 downscale_f=4,
                 min_crop_f=0.5,
                 max_crop_f=1.,
                 random_crop=True):
        """
        Imagenet Superresolution Dataloader
        Performs following ops in order:
        1.  crops a crop of size s from image either as random or center crop
        2.  resizes crop to size with cv2.area_interpolation
        3.  degrades resized crop with degradation_fn

        :param size: resizing to size after cropping
        :param degradation: degradation_fn, e.g. cv_bicubic or bsrgan_light
        :param downscale_f: Low Resolution Downsample factor
        :param min_crop_f: determines crop size s,
          where s = c * min_img_side_len with c sampled from interval (min_crop_f, max_crop_f)
        :param max_crop_f: ""
        :param data_root:
        :param random_crop:
        """
        self.base = self.get_base()
        assert size
        assert (size / downscale_f).is_integer()
        self.size = size
        self.LR_size = int(size / downscale_f)
        self.min_crop_f = min_crop_f
        self.max_crop_f = max_crop_f
        assert(max_crop_f <= 1.)
        self.center_crop = not random_crop

        self.image_rescaler = albumentations.SmallestMaxSize(max_size=size, interpolation=cv2.INTER_AREA)

        self.pil_interpolation = False # gets reset later if incase interp_op is from pillow

        if degradation == "bsrgan":
            self.degradation_process = partial(degradation_fn_bsr, sf=downscale_f)

        elif degradation == "bsrgan_light":
            self.degradation_process = partial(degradation_fn_bsr_light, sf=downscale_f)

        else:
            interpolation_fn = {
            "cv_nearest": cv2.INTER_NEAREST,
            "cv_bilinear": cv2.INTER_LINEAR,
            "cv_bicubic": cv2.INTER_CUBIC,
            "cv_area": cv2.INTER_AREA,
            "cv_lanczos": cv2.INTER_LANCZOS4,
            "pil_nearest": PIL.Image.NEAREST,
            "pil_bilinear": PIL.Image.BILINEAR,
            "pil_bicubic": PIL.Image.BICUBIC,
            "pil_box": PIL.Image.BOX,
            "pil_hamming": PIL.Image.HAMMING,
            "pil_lanczos": PIL.Image.LANCZOS,
            }[degradation]

            self.pil_interpolation = degradation.startswith("pil_")

            if self.pil_interpolation:
                self.degradation_process = partial(TF.resize, size=self.LR_size, interpolation=interpolation_fn)

            else:
                self.degradation_process = albumentations.SmallestMaxSize(max_size=self.LR_size,
                                                                          interpolation=interpolation_fn)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        example = self.base[i]
        image = Image.open(example["file_path_"])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        image = np.array(image).astype(np.uint8)

        min_side_len = min(image.shape[:2])
        crop_side_len = min_side_len * np.random.uniform(self.min_crop_f, self.max_crop_f, size=None)
        crop_side_len = int(crop_side_len)

        if self.center_crop:
            self.cropper = albumentations.CenterCrop(height=crop_side_len, width=crop_side_len)

        else:
            self.cropper = albumentations.RandomCrop(height=crop_side_len, width=crop_side_len)

        image = self.cropper(image=image)["image"]
        image = self.image_rescaler(image=image)["image"]

        if self.pil_interpolation:
            image_pil = PIL.Image.fromarray(image)
            LR_image = self.degradation_process(image_pil)
            LR_image = np.array(LR_image).astype(np.uint8)

        else:
            LR_image = self.degradation_process(image=image)["image"]

        example["image"] = (image/127.5 - 1.0).astype(np.float32)
        example["LR_image"] = (LR_image/127.5 - 1.0).astype(np.float32)

        return example


class ImageNetSRTrain(ImageNetSR):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_base(self):
        with open("data/imagenet_val_hr_indices.p", "rb") as f: #按索引来取数据Dataset 
            indices = pickle.load(f) #subset_indices = [1, 2, 5, 7]      

        dset = ImageNetTrain(process_images=False,data_root='data/myimages/')

        # print(list( Subset(dset, indices) ))
        # exit()
        return Subset(dset, indices)


class ImageNetSRValidation(ImageNetSR):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_base(self):
        with open("data/imagenet_val_hr_indices.p", "rb") as f:
            indices = pickle.load(f)
        dset = ImageNetValidation(process_images=False,data_root='data/myimages/')
        return Subset(dset, indices)
