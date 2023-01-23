from torchtext import data



class MyIterator(data.Iterator):

    def __init__(self, *arg , **kwarg) -> None:
        super().__init__(*arg, **kwarg)
        




    def __iter__(self):
        pass

def select_samples(args, train_loader, epoch):
    '''Gives back the indexes that correspond to the samples to be used for training'''
    if args.method == "unif-SGD":
        curr_prob = np.ones(len(train_loader.dataset.labels))
    elif args.method == "p-SGD":
        curr_prob = train_loader.dataset.avg_probs.copy()
        # This is for the initial epochs, to force the moded to see all the samples:
        if curr_prob.max() == -1:
            curr_prob *= -1
        max_prob = curr_prob.max()
        curr_prob[curr_prob==-1] = max_prob
    elif args.method == "c-SGD":
        curr_prob = train_loader.dataset.avg_probs.copy()
        # This is for the initial epochs, to force the moded to see all the samples:
        if curr_prob.max() == -1:
            curr_prob *= -1
        max_prob = curr_prob.max()
        curr_prob[curr_prob==-1] = max_prob
        # Use the confusion instaed of the probability:
        curr_prob = curr_prob * (1 - curr_prob)
    
    # Random sampling warmup for baselines without budget restrictions
    if epoch < args.c_sgd_warmup:
        len_curr = len(curr_prob)
        curr_prob = np.ones(len_curr)

    # Smoothness constant
    c = curr_prob.mean()
    curr_prob = curr_prob + c

    # Probability normalization
    y = curr_prob
    if y.sum() == 0:
        y = y+1e-10
    curr_prob = (y)/(y).sum()

    # Select the samples to be used:
    samples_to_keep = int(len(curr_prob)*args.budget)
    try:
        curr_samples_idx = np.random.choice(len(curr_prob), (samples_to_keep), p = curr_prob, replace = False)
    except:
        curr_prob[curr_prob == 0] = 1e-10
        curr_samples_idx = np.random.choice(len(curr_prob), (samples_to_keep), p = curr_prob/curr_prob.sum(), replace = False)

    return curr_samples_idx


def prepare_loader(args, train_loader, epoch):
    '''Prepares the dataset with the samples to be used in the following epochs'''
    curr_samples_idx = select_samples(args, train_loader, epoch)

    dataset_sampler = torch.utils.data.SubsetRandomSampler(curr_samples_idx)
    train_loader.dataset.train_samples_idx = curr_samples_idx
    train_loader = torch.utils.data.DataLoader(train_loader.dataset, sampler=dataset_sampler, \
                                            batch_size=args.batch_size, \
                                            num_workers=args.num_workers, \
                                            pin_memory=True, \
                                            drop_last = True)

    return train_loader

