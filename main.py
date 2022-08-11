import random

from model import CCA_SSG, LogReg
from aug import random_aug
from dataset import load

import numpy as np
import torch as th
import torch.nn as nn
import torch

import warnings

warnings.filterwarnings('ignore')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    ## cuda 问题
    torch.cuda.current_device()
    torch.cuda._initialized = True


def neg_hscore(f, g, args):
    """
    compute the negative h-score
     """
    f0 = f - torch.mean(f, 0)  # zero-mean
    g0 = g - torch.mean(g, 0)
    corr = torch.mean(torch.sum(f0 * g0, 1))
    cov_f = torch.t(f0) @ f0 / (f0.size()[0] - 1.)
    cov_g = torch.t(g0) @ g0 / (g0.size()[0] - 1.)
    # loss = - corr + args.lambd * torch.trace(cov_f @ cov_g)
    loss = - corr + args.lambd * torch.sum(cov_f * cov_g, dim=(-2, -1))
    return loss


def cca_loss(h1, h2, args):
    z1 = (h1 - h1.mean(0)) / h1.std(0)
    z2 = (h2 - h2.mean(0)) / h2.std(0)
    c = th.mm(z1.T, z2)
    c1 = th.mm(z1.T, z1)
    c2 = th.mm(z2.T, z2)
    N = h1.shape[0]
    c = c / N
    c1 = c1 / N
    c2 = c2 / N

    loss_inv = -th.diagonal(c).sum()
    iden = th.tensor(np.eye(c.shape[0])).to(args.device)
    loss_dec1 = (iden - c1).pow(2).sum()
    loss_dec2 = (iden - c2).pow(2).sum()
    loss = loss_inv + args.lambd * (loss_dec1 + loss_dec2)
    return loss


def _main():
    # print(args)
    graph, feat, labels, num_class, train_idx, val_idx, test_idx = load(args.dataname)
    in_dim = feat.shape[1]

    model = CCA_SSG(in_dim, args.hid_dim, args.out_dim, args.n_layers, args.use_mlp)
    model = model.to(args.device)

    optimizer = th.optim.Adam(model.parameters(), lr=args.lr1, weight_decay=args.wd1)

    N = graph.number_of_nodes()

    print('self_sup_type:', args.self_sup_type)
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        graph1, feat1 = random_aug(graph, feat, args.dfr, args.der)
        graph2, feat2 = random_aug(graph, feat, args.dfr, args.der)

        graph1 = graph1.add_self_loop()
        graph2 = graph2.add_self_loop()

        graph1 = graph1.to(args.device)
        graph2 = graph2.to(args.device)

        feat1 = feat1.to(args.device)
        feat2 = feat2.to(args.device)

        h1, h2 = model(graph1, feat1, graph2, feat2)

        if args.self_sup_type == 'cca':

            loss = cca_loss(h1, h2, args)
        else:
            loss = neg_hscore(h1, h2, args)

        # c = th.mm(z1.T, z2)
        # c1 = th.mm(z1.T, z1)
        # c2 = th.mm(z2.T, z2)
        #
        # c = c / N
        # c1 = c1 / N
        # c2 = c2 / N
        #
        # loss_inv = -th.diagonal(c).sum()
        # iden = th.tensor(np.eye(c.shape[0])).to(args.device)
        # loss_dec1 = (iden - c1).pow(2).sum()
        # loss_dec2 = (iden - c2).pow(2).sum()
        #
        # loss = loss_inv + args.lambd * (loss_dec1 + loss_dec2)

        loss.backward()
        optimizer.step()

        print('Epoch={:03d}, loss={:.4f}'.format(epoch, loss.item()))

    print("=== Evaluation ===")
    graph = graph.to(args.device)
    graph = graph.remove_self_loop().add_self_loop()
    feat = feat.to(args.device)

    embeds = model.get_embedding(graph, feat)

    train_embs = embeds[train_idx]
    val_embs = embeds[val_idx]
    test_embs = embeds[test_idx]

    label = labels.to(args.device)

    train_labels = label[train_idx]
    val_labels = label[val_idx]
    test_labels = label[test_idx]

    train_feat = feat[train_idx]
    val_feat = feat[val_idx]
    test_feat = feat[test_idx]

    ''' Linear Evaluation '''
    logreg = LogReg(train_embs.shape[1], num_class)
    opt = th.optim.Adam(logreg.parameters(), lr=args.lr2, weight_decay=args.wd2)

    logreg = logreg.to(args.device)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0
    eval_acc = 0

    for epoch in range(2000):
        logreg.train()
        opt.zero_grad()
        logits = logreg(train_embs)
        preds = th.argmax(logits, dim=1)
        train_acc = th.sum(preds == train_labels).float() / train_labels.shape[0]
        loss = loss_fn(logits, train_labels)
        loss.backward()
        opt.step()

        logreg.eval()
        with th.no_grad():
            val_logits = logreg(val_embs)
            test_logits = logreg(test_embs)

            val_preds = th.argmax(val_logits, dim=1)
            test_preds = th.argmax(test_logits, dim=1)

            val_acc = th.sum(val_preds == val_labels).float() / val_labels.shape[0]
            test_acc = th.sum(test_preds == test_labels).float() / test_labels.shape[0]

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                if test_acc > eval_acc:
                    eval_acc = test_acc

            print(
                'Epoch:{}, train_acc:{:.4f}, val_acc:{:4f}, test_acc:{:4f}'.format(epoch, train_acc, val_acc, test_acc))

    print('Linear evaluation accuracy:{:.4f}'.format(eval_acc))
    return float(test_acc)


if __name__ == '__main__':
    from parser import get_parser

    args = get_parser()
    print(args)
    # args.debug = False
    # args.debug = False
    results, elapsed_times = [], []
    if args.debug:
        seed_num = [args.seed]

    seed_nums = [i for i in range(10)]
    for seed_num in seed_nums:
        print(f'Seed {seed_num}/{max(seed_nums)}')
        setup_seed(seed_num)
        test_acc = _main()
        results.append(test_acc)
        print(f'Seed {seed_num}/{max(seed_nums)} Acc array: ', results)
    print(args)
    results = np.array(results)
    elapsed_times = np.array(elapsed_times)
    print(f'avg_test_acc={results.mean()*100:.5f}')
    print(f"avg_test_acc={results.mean() * 100:.3f}$\pm${results.std() * 100:.3f}")
