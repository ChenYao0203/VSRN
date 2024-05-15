from __future__ import print_function
import os
import pickle
import clip
import torch
import numpy
from data import get_test_loader
import time
import numpy as np
from vocab import Vocabulary  # NOQA
from model import VSRN, order_sim
from collections import OrderedDict




class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.iteritems()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.iteritems():
            tb_logger.log_value(prefix + k, v.val, step=step)

#这里调用CLIP
def encode_data(model, data_loader, log_step=10, logging=print):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model, preprocess = clip.load("ViT-B/32")
    end = time.time()

    # numpy array to keep all the embeddings
    img_embs = None
    cap_embs = None
    for i, (images, captions, lengths, ids, caption_labels, caption_masks) in enumerate(data_loader):
        # make sure val logger is used
        model.logger = val_logger

        # compute the embeddings
        img_emb, cap_emb, fc_img_emd = model.forward_emb(images, captions, lengths,
                                             volatile=True)

        # initialize the numpy arrays given the size of the embeddings
        if img_embs is None:
            img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)))
            cap_embs = np.zeros((len(data_loader.dataset), cap_emb.size(1)))

        # preserve the embeddings by copying from gpu and converting to numpy
        img_embs[ids] = img_emb.data.cpu().numpy().copy()
        cap_embs[ids] = cap_emb.data.cpu().numpy().copy()


        del images, captions

    return img_embs, cap_embs




#测试函数
def evalrank(model_path, model_path2, data_path=None, split='dev'):
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """
    # load model and options
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']

    checkpoint2 = torch.load(model_path2)
    opt2 = checkpoint2['opt']

    if data_path is not None:
        opt.data_path = data_path

    # load vocabulary used by the model
    with open(os.path.join(opt.vocab_path,
                           '%s_vocab.pkl' % opt.data_name), 'rb') as f:
        vocab = pickle.load(f)
    opt.vocab_size = len(vocab)



    model = VSRN(opt)
    model2 = VSRN(opt2)

    # load model state
    model.load_state_dict(checkpoint['model'])

    model2.load_state_dict(checkpoint2['model'])

    print('Loading dataset')
    data_loader = get_test_loader()

    print('Computing results...')
    img_embs, cap_embs = encode_data(model, data_loader)
    img_embs2, cap_embs2 = encode_data(model2, data_loader)
    
    #测试的图像的数量和标题的数量
    print('Images: %d, Captions: %d' %
          (img_embs.shape[0] / 5, cap_embs.shape[0]))

    # no cross-validation, full evaluation
    r, rt = i2t(img_embs, cap_embs, img_embs2, cap_embs2, measure=opt.measure, return_ranks=True)
    ar = (r[0] + r[1] + r[2]) / 3
    print("Average i2t Recall: %.1f" % ar)
    print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
    torch.save({'rt': rt}, 'ranks.pth.tar')


def i2t(images, captions, images2, captions2, npts=None, measure='cosine', return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = images.shape[0] // 5
    index_list = []
    
    #ranks、top1的长度和图像数量保持一致
    ranks = numpy.zeros(npts)
    top1 = numpy.zeros(npts)
    for index in range(npts):

        # Get query image
        im = images[5 * index].reshape(1, images.shape[1])
        im_2 = images2[5 * index].reshape(1, images2.shape[1])
        # Compute scores
        if measure == 'order':
            bs = 100
            if index % bs == 0:
                mx = min(images.shape[0], 5 * (index + bs))
                im2 = images[5 * index:mx:5]
                d2 = order_sim(torch.Tensor(im2).cuda(),
                               torch.Tensor(captions).cuda())
                d2 = d2.cpu().numpy()
            d = d2[index % bs]
        else:
            h = numpy.dot(im, captions.T).flatten()
            d2 = numpy.dot(im_2, captions2.T).flatten()
            d = (d + d2)/2
            
        inds = numpy.argsort(d)[::-1]
        index_list.append(inds[0])

        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = numpy.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank #图像的label标题在检索排名中最靠前的排名
        top1[index] = inds[0] #图像对应的检索排名第一的标题index

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)
if __name__='__main__':
    print('Evaluation on Flickr30K:')
    evalrank("pretrain_model/flickr/model_fliker_1.pth.tar", "pretrain_model/flickr/model_fliker_2.pth.tar", data_path='data/', split="test", fold5=False)

