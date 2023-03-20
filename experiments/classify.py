import former
# from former import util
# from former.util import d, here

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

# from torchtext import data, datasets, vocab
from torchtext.legacy import data, datasets, vocab
import pickle

import numpy as np

from argparse import ArgumentParser
import random, tqdm, sys, math, gzip
from torch.utils.tensorboard import SummaryWriter
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import os
import pathlib

from get_data import get_dataset, device
from utils_train import train_CrossEntropy, testing,prepare_loader

# Used for converting between nats and bits
LOG2E = math.log2(math.e)
TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
LABEL = data.Field(sequential=False)
NUM_CLS = 2



def go(arg):
    """
    Creates and trains a basic transformer for the IMDB sentiment classification task.
    """
    tbw = SummaryWriter(log_dir=arg.tb_dir) # Tensorboard logging 

    trainset, testset = get_dataset(arg)
    train_iter = torch.utils.data.DataLoader(trainset, batch_size=arg.batch_size, \
                                                shuffle=True, \
                                                pin_memory=True, \
                                                drop_last = True)
    test_iter = torch.utils.data.DataLoader(testset, batch_size=arg.batch_size, shuffle=True,pin_memory=True)
    
    print(f'- nr. of training examples {len(train_iter)}')
    print(f'- nr. of {"test" if arg.final else "validation"} examples {len(test_iter)}')

    if arg.max_length < 0:
        mx = max([input.text[0].size(1) for input in train_iter])
        mx = mx * 2
        print(f'- maximum sequence length: {mx}')
    else:
        mx = arg.max_length

    # create the model
    model = former.CTransformer(emb=arg.embedding_size, heads=arg.num_heads, depth=arg.depth, seq_length=mx, num_tokens=arg.vocab_size, num_classes=NUM_CLS, max_pool=arg.max_pool)
    if torch.cuda.is_available():
        model.cuda()

    model = nn.DataParallel(model, device_ids=[0, 1, 2])
    # opt = torch.optim.Adam(lr=arg.lr, params=model.parameters())
    opt = torch.optim.SGD(lr=arg.lr,params=model.parameters(), momentum=arg.momentum)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / (arg.lr_warmup / arg.batch_size), 1.0))

    # training loop
    seen = 0
    
    loss_train_epoch = []
    loss_val_epoch = []
    acc_train_per_epoch = []
    acc_val_per_epoch = []
    entropies = []
    
    # res_path = os.path.join('./', 'metrics' + '_{0}'.format(arg.method))
    res_path = get_path(arg)

    for e in range(arg.num_epochs):
        train_iter = prepare_loader(arg, train_iter, e)

        path = get_path(arg)
        loss_per_epoch, _, top1_train_ac, entropy = train_CrossEntropy(arg, model, device, \
                                                        train_iter, opt, sch, e, path = path)
        
        entropies += [entropy]
        loss_train_epoch += [loss_per_epoch]

        print('######  testing')
        loss_per_epoch, acc_val_per_epoch_i = testing(arg, model, device, test_iter)
        loss_val_epoch += [loss_per_epoch]
        
        acc_train_per_epoch += [top1_train_ac]
        acc_val_per_epoch += acc_val_per_epoch_i

        # Save losses:
        np.save(res_path + '/' + 'LOSS_epoch_train.npy', np.asarray(loss_train_epoch))
        np.save(res_path + '/' + 'LOSS_epoch_val.npy', np.asarray(loss_val_epoch))

        # Save accuracies:
        np.save(res_path + '/' + 'accuracy_per_epoch_train.npy', np.asarray(acc_train_per_epoch))
        np.save(res_path + '/' + 'accuracy_per_epoch_val.npy', np.asarray(acc_val_per_epoch))

        # Save entropies:
        np.save(res_path + '/' + 'entropy_per_epoch.npy', np.asarray(entropies))


        # Save model every 5 epochs:
        if e % 5 == 4:
            save_model(arg, model, e)
    
    return model
        
def save_model(options, model, epoch):
    # check if directory exists
    path = get_path(options)

    print('saving model ..')
    with open(f'{path}/e={epoch}-model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print('model saved')

def get_path(options):
    path = f'm={options.method}-lr={options.lr}-ew={options.entropy_weight}'
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    return path

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-m","--method",
                        dest="method",
                        help="type of sgd",
                        default='unif-SGD', type=str)
    
    parser.add_argument('--c_sgd_warmup', 
                        help="Number of ecpochs with random sampling for p-SGD andd c-SGD",
                        default=0, type=int)

    parser.add_argument('--budget',
                        help='Percentage of buget to use',
                        default=1.0,type=float)

    parser.add_argument("-e", "--num-epochs",
                        dest="num_epochs",
                        help="Number of epochs.",
                        default=30, type=int)

    parser.add_argument("-b", "--batch-size",
                        dest="batch_size",
                        help="The batch size.",
                        default=4, type=int)

    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate",
                        default=0.01, type=float)

    parser.add_argument("-T", "--tb_dir", dest="tb_dir",
                        help="Tensorboard logging directory",
                        default='./runs')

    parser.add_argument("-f", "--final", dest="final",
                        help="Whether to run on the real test set (if not included, the validation set is used).",
                        action="store_true")

    parser.add_argument("--max-pool", dest="max_pool",
                        help="Use max pooling in the final classification layer.",
                        action="store_true")

    parser.add_argument("-E", "--embedding", dest="embedding_size",
                        help="Size of the character embeddings.",
                        default=256, type=int)

    parser.add_argument("-V", "--vocab-size", dest="vocab_size",
                        help="Number of words in the vocabulary.",
                        default=101000, type=int)

    parser.add_argument("-M", "--max", dest="max_length",
                        help="Max sequence length. Longer sequences are clipped (-1 for no limit).",
                        default=512, type=int)

    parser.add_argument("-H", "--heads", dest="num_heads",
                        help="Number of attention heads.",
                        default=8, type=int)

    parser.add_argument("-d", "--depth", dest="depth",
                        help="Depth of the network (nr. of self-attention layers)",
                        default=6, type=int)

    parser.add_argument("-r", "--random-seed",
                        dest="seed",
                        help="RNG seed. Negative for random",
                        default=1, type=int)

    parser.add_argument("--lr-warmup",
                        dest="lr_warmup",
                        help="Learning rate warmup.",
                        default=10_000, type=int)

    parser.add_argument("--gradient-clipping",
                        dest="gradient_clipping",
                        help="Gradient clipping.",
                        default=1.0, type=float)
    
    parser.add_argument("--momentum",
                        dest="momentum",
                        help="momentum for SGD",
                        default=0.9, type=float)
    
    parser.add_argument("--entropy_weight",
                        dest="entropy_weight",
                        help="weight for entropy loss",
                        default=0.1, type=float)

    options = parser.parse_args()


    print('OPTIONS ', options)

    model = go(options)

    # time = datetime.now()
    # save_model(options, model)
