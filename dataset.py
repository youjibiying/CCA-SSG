import numpy as np
import torch as th

from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.data import AmazonCoBuyPhotoDataset, AmazonCoBuyComputerDataset
from dgl.data import CoauthorCSDataset, CoauthorPhysicsDataset

def load(name):
    data_path = '/apdcephfs/share_1364275/jiyingzhang/data/dgl'
    if name == 'cora':
        dataset = CoraGraphDataset(raw_dir=data_path)
    elif name == 'citeseer':
        dataset = CiteseerGraphDataset(raw_dir=data_path)
    elif name == 'pubmed':
        dataset = PubmedGraphDataset(raw_dir=data_path)
    elif name == 'photo':
        dataset = AmazonCoBuyPhotoDataset(raw_dir=data_path)
    elif name == 'comp':
        dataset = AmazonCoBuyComputerDataset(raw_dir=data_path)
    elif name == 'cs':
        dataset = CoauthorCSDataset(raw_dir=data_path)
    elif name == 'physics':
        dataset = CoauthorPhysicsDataset(raw_dir=data_path)

    graph = dataset[0]
    citegraph = ['cora', 'citeseer', 'pubmed']
    cograph = ['photo', 'comp', 'cs', 'physics']

    if name in citegraph:
        train_mask = graph.ndata.pop('train_mask')
        val_mask = graph.ndata.pop('val_mask')
        test_mask = graph.ndata.pop('test_mask')

        train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
        val_idx = th.nonzero(val_mask, as_tuple=False).squeeze()
        test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()

    if name in cograph:
        train_ratio = 0.1
        val_ratio = 0.1
        test_ratio = 0.8

        N = graph.number_of_nodes()
        train_num = int(N * train_ratio)
        val_num = int(N * (train_ratio + val_ratio))

        idx = np.arange(N)
        np.random.shuffle(idx)

        train_idx = idx[:train_num]
        val_idx = idx[train_num:val_num]
        test_idx = idx[val_num:]

        train_idx = th.tensor(train_idx)
        val_idx = th.tensor(val_idx)
        test_idx = th.tensor(test_idx)

    num_class = dataset.num_classes
    feat = graph.ndata.pop('feat')
    labels = graph.ndata.pop('label')

    return graph, feat, labels, num_class, train_idx, val_idx, test_idx
