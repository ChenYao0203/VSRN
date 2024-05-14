import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
import json as jsonmod
import clip


def get_paths(path, name='coco', use_restval=False):
    """
    Returns paths to images and annotations for the given datasets. For MSCOCO
    indices are also returned to control the data split being used.
    The indices are extracted from the Karpathy et al. splits using this
    snippet:

    >>> import json
    >>> dataset=json.load(open('dataset_coco.json','r'))
    >>> A=[]
    >>> for i in range(len(D['images'])):
    ...   if D['images'][i]['split'] == 'val':
    ...     A+=D['images'][i]['sentids'][:5]
    ...

    :param name: Dataset names
    :param use_restval: If True, the the `restval` data is included in train.
    """
    roots = {}
    ids = {}
    imgdir = os.path.join(path, 'images')
    cap = os.path.join(path, 'dataset_flickr30k.json')
    roots['train'] = {'img': imgdir, 'cap': cap}
    roots['val'] = {'img': imgdir, 'cap': cap}
    roots['test'] = {'img': imgdir, 'cap': cap}
    ids = {'train': None, 'val': None, 'test': None}

    return roots, ids

def get_clip_tokenize(prompt_caption):
    model, preprocess = clip.load("ViT-B/32")
    prompt_caption_token = preprocess(prompt_caption)
    return prompt_caption_token


class FlickrDataset(data.Dataset):
    """
    Dataset loader for Flickr30k and Flickr8k full datasets.
    """

    def __init__(self, root, json, split):
        self.root = root
        self.split = split
        self.dataset = jsonmod.load(open(json, 'r'))['images']
        self.ids = []
        for i, d in enumerate(self.dataset):
            if d['split'] == split:
                self.ids += [(i, x) for x in range(len(d['sentences']))] #获取的id是每一个图像的id，每一个id对应了五个标题
    #get_item获取的image是预处理过的，标题是tokenize过的，但是长度不一样，在这里可以直接调用CLIP的token_ize，把CLIP的tokenize封装为get_tokenize的函数
    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        root = self.root
        ann_id = self.ids[index]
        img_id = ann_id[0]
        caption = self.dataset[img_id]['sentences'][ann_id[1]]['raw'] #获得原始标题文本,只获取了图像对应的一个标题
        prompt_caption = " " + caption #变换提示
        path = self.dataset[img_id]['filename']

        image = Image.open(os.path.join(root, path)).convert('RGB') #打开原始图像，只获取了一张图像
        image = clip.transform(image)

        # Convert caption (string) to word ids. 将标题转换为 word id
        
        prompt_caption_token = get_clip_tokenize(prompt_caption)
        return image, prompt_caption_token, index, img_id #每一条数据包括，预处理之后的一个图像、tokenize之后的caption，这条数据对应的index，以及图像对应的id
    #获取的是所有标题的数量
    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256). 
            - caption: torch tensor of shape (?); length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # ids是每一条数据的index, img_ids是图像的id,后面两个不知道怎么来的
    images, captions, ids, img_ids = zip(*data) 

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)
    prompt_captions =torch.stack(captions,0)
    return images, prompt_captions, ids


def get_loader_single(data_name, split, root, json, vocab, transform,
                      batch_size=100, shuffle=True,
                      num_workers=2, ids=None, collate_fn=collate_fn):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dataset = FlickrDataset(root=root,
                                split=split,
                                json=json,
                                vocab=vocab,
                                transform=transform)

    # Data loader
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader






def get_loaders(data_name, vocab, crop_size, batch_size, workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
  
    # Build Dataset Loader
    roots, ids = get_paths(dpath, data_name, opt.use_restval)

    transform = get_transform(data_name, 'train', opt)
    train_loader = get_loader_single(opt.data_name, 'train',
                                     roots['train']['img'],
                                     roots['train']['cap'],
                                     vocab, transform, ids=ids['train'],
                                     batch_size=batch_size, shuffle=True,
                                     num_workers=workers,
                                     collate_fn=collate_fn)

    transform = get_transform(data_name, 'val', opt)
    val_loader = get_loader_single(opt.data_name, 'val',
                                   roots['val']['img'],
                                   roots['val']['cap'],
                                   vocab, transform, ids=ids['val'],
                                   batch_size=batch_size, shuffle=False,
                                   num_workers=workers,
                                   collate_fn=collate_fn)

    return train_loader, val_loader


def get_test_loader(split_name, data_name, vocab, crop_size, batch_size,
                    workers, opt):
    dpath = os.path.join(opt.data_path, data_name)

    # Build Dataset Loader
    roots, ids = get_paths(dpath, data_name, opt.use_restval)

    transform = get_transform(data_name, split_name, opt)
    test_loader = get_loader_single(opt.data_name, split_name,
                                    roots[split_name]['img'],
                                    roots[split_name]['cap'],
                                    vocab, transform, ids=ids[split_name],
                                    batch_size=batch_size, shuffle=False,
                                    num_workers=workers,
                                    collate_fn=collate_fn)

    return test_loader
