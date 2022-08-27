#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries

# In[3]:


### Author: Yuvasri Raghavan (2022)


# In[4]:


# argparse for user friendly parsing of arguments
import argparse

#machine learning python framework
import torch

#PyTorch library for graph networks
from torch_geometric.nn import Node2Vec

# OGB Dataloader to load Drug Interaction Network Data
from ogb.linkproppred import PygLinkPropPredDataset


# ## Node2Vec

# ### Saving Embeddings

# In[2]:


#Creating a function to save embeddings in Node2Vec

def save_embedding_todevice(model):
    torch.save(model.embedding.weight.data.cpu(), 'embedding.pt')


# ### Load Data

# In[2]:


#load OGB DDI Dataset
dataset = PygLinkPropPredDataset(name='ogbl-ddi')
data = dataset[0]

#Split the data based on a protein split
split_edge = dataset.get_edge_split()
idx = torch.randperm(split_edge['train']['edge'].size(0))
idx = idx[:split_edge['valid']['edge'].size(0)]
split_edge['eval_train'] = {'edge': split_edge['train']['edge'][idx]}


# ### Node2Vec Embedding Model

# In[3]:


def main():
    
    #Parser for arguments for Node2Vec
    
    parser = argparse.ArgumentParser(description='OGBL-DDI (Node2Vec)')
    parser.add_argument('-f')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--walk_length', type=int, default=40)
    parser.add_argument('--context_size', type=int, default=20)
    parser.add_argument('--walks_per_node', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--log_steps', type=int, default=1)
    args = parser.parse_args()

    #Use GPU if available
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    #load OGB DDI Dataset
    dataset = PygLinkPropPredDataset(name='ogbl-ddi')
    data = dataset[0]

    #Defining the Node@Vec Model
    model = Node2Vec(data.edge_index, args.embedding_dim, args.walk_length,
                     args.context_size, args.walks_per_node,
                     sparse=True).to(device)

    loader = model.loader(batch_size=args.batch_size, shuffle=True,
                          num_workers=4)
    
    # Using SparseAdam optimizer for parameterization
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=args.lr)

    #Train Node2Vec model
    model.train()
    for epoch in range(1, args.epochs + 1):
        for i, (pos_rw, neg_rw) in enumerate(loader):
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()

            if (i + 1) % args.log_steps == 0:
                print(f'Epoch: {epoch:02d}, Step: {i+1:03d}/{len(loader)}, '
                      f'Loss: {loss:.4f}')

            if (i + 1) % 100 == 0:  # Save model every 100 steps.
                save_embedding_todevice(model)
        save_embedding_todevice(model)


if __name__ == "__main__":
    main()


# ## Tracker Function to Record Run details

# In[ ]:


# Logger Code referred from: https://github.com/snap-stanford/ogb/blob/master/examples/linkproppred/ddi/logger.py


# In[2]:


class Logger_Models(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')


# ## Multilayer Perceptron

# In[5]:


import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator


# In[13]:


from ann_visualizer.visualize import ann_viz;

ann_viz(model, title="My first neural network")


# In[6]:


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


def train(predictor, x, edge_index, split_edge, optimizer, batch_size):
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].to(x.device)

    total_loss = total_examples = 0
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True):
        optimizer.zero_grad()

        edge = pos_train_edge[perm].t()

        pos_out = predictor(x[edge[0]], x[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        edge = negative_sampling(edge_index, num_nodes=x.size(0),
                                 num_neg_samples=perm.size(0), method='dense')

        neg_out = predictor(x[edge[0]], x[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(predictor, x, split_edge, evaluator, batch_size):
    predictor.eval()

    pos_train_edge = split_edge['eval_train']['edge'].to(x.device)
    pos_valid_edge = split_edge['valid']['edge'].to(x.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(x.device)
    pos_test_edge = split_edge['test']['edge'].to(x.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(x.device)

    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
        edge = pos_train_edge[perm].t()
        pos_train_preds += [predictor(x[edge[0]], x[edge[1]]).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(x[edge[0]], x[edge[1]]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(x[edge[0]], x[edge[1]]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(x[edge[0]], x[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(x[edge[0]], x[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    results = {}
    for K in [10, 20, 30]:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)

    return results


def main():
    parser = argparse.ArgumentParser(description='OGBL-DDI (MLP)')
    parser.add_argument('-f')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=64 * 1024)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=10)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygLinkPropPredDataset(name='ogbl-ddi')
    data = dataset[0]
    split_edge = dataset.get_edge_split()

    # We randomly pick some training samples that we want to evaluate on:
    torch.manual_seed(12345)
    idx = torch.randperm(split_edge['train']['edge'].size(0))
    idx = idx[:split_edge['valid']['edge'].size(0)]
    split_edge['eval_train'] = {'edge': split_edge['train']['edge'][idx]}

    x = torch.load('embedding.pt', map_location='cpu').to(device)

    predictor = LinkPredictor(x.size(-1), args.hidden_channels, 1,
                              args.num_layers, args.dropout).to(device)

    evaluator = Evaluator(name='ogbl-ddi')
    Logger_Models_Models = {
        'Hits@10': Logger_Models_Model(args.runs, args),
        'Hits@20': Logger_Models_Model(args.runs, args),
        'Hits@30': Logger_Models_Model(args.runs, args),
    }

    for run in range(args.runs):
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(predictor.parameters(), lr=args.lr)

        for epoch in range(1, 1 + args.epochs):
            loss = train(predictor, x, data.edge_index, split_edge, optimizer,
                         args.batch_size)

            if epoch % args.eval_steps == 0:
                results = test(predictor, x, split_edge, evaluator,
                               args.batch_size)
                for key, result in results.items():
                    Logger_Models_Models[key].add_result(run, result)

                if epoch % args.log_steps == 0:
                    for key, result in results.items():
                        train_hits, valid_hits, test_hits = result
                        print(key)
                        print(f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'Train: {100 * train_hits:.2f}%, '
                              f'Valid: {100 * valid_hits:.2f}%, '
                              f'Test: {100 * test_hits:.2f}%')
                    print('---')

        for key in Logger_Models_Models.keys():
            print(key)
            Logger_Models_Models[key].print_statistics(run)

    for key in Logger_Models_Models.keys():
        print(key)
        Logger_Models_Models[key].print_statistics()


if __name__ == "__main__":
    main()


# ## Matrix Factorization

# In[7]:


import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator


# In[8]:


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


def train(predictor, x, edge_index, split_edge, optimizer, batch_size):
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].to(x.device)

    total_loss = total_examples = 0
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True):
        optimizer.zero_grad()

        edge = pos_train_edge[perm].t()

        pos_out = predictor(x[edge[0]], x[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        edge = negative_sampling(edge_index, num_nodes=x.size(0),
                                 num_neg_samples=perm.size(0), method='dense')

        neg_out = predictor(x[edge[0]], x[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(predictor, x, split_edge, evaluator, batch_size):
    predictor.eval()

    pos_train_edge = split_edge['eval_train']['edge'].to(x.device)
    pos_valid_edge = split_edge['valid']['edge'].to(x.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(x.device)
    pos_test_edge = split_edge['test']['edge'].to(x.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(x.device)

    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
        edge = pos_train_edge[perm].t()
        pos_train_preds += [predictor(x[edge[0]], x[edge[1]]).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(x[edge[0]], x[edge[1]]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(x[edge[0]], x[edge[1]]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(x[edge[0]], x[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(x[edge[0]], x[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    results = {}
    for K in [10, 20, 30]:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)

    return results


def main():
    parser = argparse.ArgumentParser(description='OGBL-DDI (MF)')
    parser.add_argument('-f')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=64 * 1024)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=10)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygLinkPropPredDataset(name='ogbl-ddi')
    data = dataset[0]
    split_edge = dataset.get_edge_split()

    # We randomly pick some training samples that we want to evaluate on:
    torch.manual_seed(12345)
    idx = torch.randperm(split_edge['train']['edge'].size(0))
    idx = idx[:split_edge['valid']['edge'].size(0)]
    split_edge['eval_train'] = {'edge': split_edge['train']['edge'][idx]}

    emb = torch.nn.Embedding(data.num_nodes, args.hidden_channels).to(device)
    predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1,
                              args.num_layers, args.dropout).to(device)

    evaluator = Evaluator(name='ogbl-ddi')
    Logger_Models_Models = {
        'Hits@10': Logger_Models_Model(args.runs, args),
        'Hits@20': Logger_Models_Model(args.runs, args),
        'Hits@30': Logger_Models_Model(args.runs, args),
    }

    for run in range(args.runs):
        emb.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(
            list(emb.parameters()) + list(predictor.parameters()), lr=args.lr)

        for epoch in range(1, 1 + args.epochs):
            loss = train(predictor, emb.weight, data.edge_index, split_edge,
                         optimizer, args.batch_size)

            if epoch % args.eval_steps == 0:
                results = test(predictor, emb.weight, split_edge, evaluator,
                               args.batch_size)
                for key, result in results.items():
                    Logger_Models_Models[key].add_result(run, result)

                if epoch % args.log_steps == 0:
                    for key, result in results.items():
                        train_hits, valid_hits, test_hits = result
                        print(key)
                        print(f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'Train: {100 * train_hits:.2f}%, '
                              f'Valid: {100 * valid_hits:.2f}%, '
                              f'Test: {100 * test_hits:.2f}%')
                    print('---')

        for key in Logger_Models_Models.keys():
            print(key)
            Logger_Models_Models[key].print_statistics(run)

    for key in Logger_Models_Models.keys():
        print(key)
        Logger_Models_Models[key].print_statistics()


if __name__ == "__main__":
    main()


# ## GNN

# In[9]:


import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator


# In[10]:


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


def train(model, predictor, x, adj_t, split_edge, optimizer, batch_size):

    row, col, _ = adj_t.coo()
    edge_index = torch.stack([col, row], dim=0)

    model.train()
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].to(x.device)

    total_loss = total_examples = 0
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True):
        optimizer.zero_grad()

        h = model(x, adj_t)

        edge = pos_train_edge[perm].t()

        pos_out = predictor(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        edge = negative_sampling(edge_index, num_nodes=x.size(0),
                                 num_neg_samples=perm.size(0), method='dense')

        neg_out = predictor(h[edge[0]], h[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(x, 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(model, predictor, x, adj_t, split_edge, evaluator, batch_size):
    model.eval()
    predictor.eval()

    h = model(x, adj_t)

    pos_train_edge = split_edge['eval_train']['edge'].to(x.device)
    pos_valid_edge = split_edge['valid']['edge'].to(x.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(x.device)
    pos_test_edge = split_edge['test']['edge'].to(x.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(x.device)

    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
        edge = pos_train_edge[perm].t()
        pos_train_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    results = {}
    for K in [10, 20, 30]:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)

    return results


def main():
    parser = argparse.ArgumentParser(description='OGBL-DDI (GNN)')
    parser.add_argument('-f')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=64 * 1024)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=10)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygLinkPropPredDataset(name='ogbl-ddi',
                                     transform=T.ToSparseTensor())
    data = dataset[0]
    adj_t = data.adj_t.to(device)

    split_edge = dataset.get_edge_split()

    # We randomly pick some training samples that we want to evaluate on:
    torch.manual_seed(12345)
    idx = torch.randperm(split_edge['train']['edge'].size(0))
    idx = idx[:split_edge['valid']['edge'].size(0)]
    split_edge['eval_train'] = {'edge': split_edge['train']['edge'][idx]}

    if args.use_sage:
        model = SAGE(args.hidden_channels, args.hidden_channels,
                     args.hidden_channels, args.num_layers,
                     args.dropout).to(device)
    else:
        model = GCN(args.hidden_channels, args.hidden_channels,
                    args.hidden_channels, args.num_layers,
                    args.dropout).to(device)

    emb = torch.nn.Embedding(data.adj_t.size(0),
                             args.hidden_channels).to(device)
    predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1,
                              args.num_layers, args.dropout).to(device)

    evaluator = Evaluator(name='ogbl-ddi')
    Logger_Models_Models = {
        'Hits@10': Logger_Models_Model(args.runs, args),
        'Hits@20': Logger_Models_Model(args.runs, args),
        'Hits@30': Logger_Models_Model(args.runs, args),
    }

    for run in range(args.runs):
        torch.nn.init.xavier_uniform_(emb.weight)
        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(emb.parameters()) +
            list(predictor.parameters()), lr=args.lr)

        for epoch in range(1, 1 + args.epochs):
            loss = train(model, predictor, emb.weight, adj_t, split_edge,
                         optimizer, args.batch_size)

            if epoch % args.eval_steps == 0:
                results = test(model, predictor, emb.weight, adj_t, split_edge,
                               evaluator, args.batch_size)
                for key, result in results.items():
                    Logger_Models_Models[key].add_result(run, result)

                if epoch % args.log_steps == 0:
                    for key, result in results.items():
                        train_hits, valid_hits, test_hits = result
                        print(key)
                        print(f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'Train: {100 * train_hits:.2f}%, '
                              f'Valid: {100 * valid_hits:.2f}%, '
                              f'Test: {100 * test_hits:.2f}%')
                    print('---')

        for key in Logger_Models_Models.keys():
            print(key)
            Logger_Models_Models[key].print_statistics(run)

    for key in Logger_Models_Models.keys():
        print(key)
        Logger_Models_Models[key].print_statistics()


if __name__ == "__main__":
    main()


# ## Graph Sage

# In[11]:


import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator


# In[12]:


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)



def train(model, predictor, x, adj_t, split_edge, optimizer, batch_size):

    row, col, _ = adj_t.coo()
    edge_index = torch.stack([col, row], dim=0)

    model.train()
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].to(x.device)

    total_loss = total_examples = 0
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True):
        optimizer.zero_grad()

        h = model(x, adj_t)

        edge = pos_train_edge[perm].t()

        pos_out = predictor(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        edge = negative_sampling(edge_index, num_nodes=x.size(0),
                                 num_neg_samples=perm.size(0), method='dense')

        neg_out = predictor(h[edge[0]], h[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(x, 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(model, predictor, x, adj_t, split_edge, evaluator, batch_size):
    model.eval()
    predictor.eval()

    h = model(x, adj_t)

    pos_train_edge = split_edge['eval_train']['edge'].to(x.device)
    pos_valid_edge = split_edge['valid']['edge'].to(x.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(x.device)
    pos_test_edge = split_edge['test']['edge'].to(x.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(x.device)

    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
        edge = pos_train_edge[perm].t()
        pos_train_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    results = {}
    for K in [10, 20, 30]:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)

    return results


def main():
    parser = argparse.ArgumentParser(description='OGBL-DDI (GNN)')
    parser.add_argument('-f')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', default = True)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=64 * 1024)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=10)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygLinkPropPredDataset(name='ogbl-ddi',
                                     transform=T.ToSparseTensor())
    data = dataset[0]
    adj_t = data.adj_t.to(device)

    split_edge = dataset.get_edge_split()

    # We randomly pick some training samples that we want to evaluate on:
    torch.manual_seed(12345)
    idx = torch.randperm(split_edge['train']['edge'].size(0))
    idx = idx[:split_edge['valid']['edge'].size(0)]
    split_edge['eval_train'] = {'edge': split_edge['train']['edge'][idx]}

    if args.use_sage:
        model = SAGE(args.hidden_channels, args.hidden_channels,
                     args.hidden_channels, args.num_layers,
                     args.dropout).to(device)
    else:
        model = GCN(args.hidden_channels, args.hidden_channels,
                    args.hidden_channels, args.num_layers,
                    args.dropout).to(device)

    emb = torch.nn.Embedding(data.adj_t.size(0),
                             args.hidden_channels).to(device)
    predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1,
                              args.num_layers, args.dropout).to(device)

    evaluator = Evaluator(name='ogbl-ddi')
    Logger_Models_Models = {
        'Hits@10': Logger_Models_Model(args.runs, args),
        'Hits@20': Logger_Models_Model(args.runs, args),
        'Hits@30': Logger_Models_Model(args.runs, args),
    }

    for run in range(args.runs):
        torch.nn.init.xavier_uniform_(emb.weight)
        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(emb.parameters()) +
            list(predictor.parameters()), lr=args.lr)

        for epoch in range(1, 1 + args.epochs):
            loss = train(model, predictor, emb.weight, adj_t, split_edge,
                         optimizer, args.batch_size)

            if epoch % args.eval_steps == 0:
                results = test(model, predictor, emb.weight, adj_t, split_edge,
                               evaluator, args.batch_size)
                for key, result in results.items():
                    Logger_Models_Models[key].add_result(run, result)

                if epoch % args.log_steps == 0:
                    for key, result in results.items():
                        train_hits, valid_hits, test_hits = result
                        print(key)
                        print(f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'Train: {100 * train_hits:.2f}%, '
                              f'Valid: {100 * valid_hits:.2f}%, '
                              f'Test: {100 * test_hits:.2f}%')
                    print('---')

        for key in Logger_Models_Models.keys():
            print(key)
            Logger_Models_Models[key].print_statistics(run)

    for key in Logger_Models_Models.keys():
        print(key)
        Logger_Models_Models[key].print_statistics()


if __name__ == "__main__":
    main()


# ## Graph Sage with Nueral Link Predictor

# In[13]:


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class NeuralLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(NeuralLinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x).squeeze()


def train(model, predictor, x, adj_t, split_edge, optimizer, batch_size):

    row, col, _ = adj_t.coo()
    edge_index = torch.stack([col, row], dim=0)

    model.train()
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].to(x.device)

    total_loss = total_examples = 0
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True):
        optimizer.zero_grad()

        h = model(x, adj_t)

        edge = pos_train_edge[perm].t()

        pos_out = predictor(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        edge = negative_sampling(edge_index, num_nodes=x.size(0),
                                 num_neg_samples=perm.size(0), method='dense')

        neg_out = predictor(h[edge[0]], h[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(x, 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(model, predictor, x, adj_t, split_edge, evaluator, batch_size):
    model.eval()
    predictor.eval()

    h = model(x, adj_t)

    pos_train_edge = split_edge['eval_train']['edge'].to(x.device)
    pos_valid_edge = split_edge['valid']['edge'].to(x.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(x.device)
    pos_test_edge = split_edge['test']['edge'].to(x.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(x.device)

    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
        edge = pos_train_edge[perm].t()
        pos_train_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    results = {}
    for K in [10, 20, 30]:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)

    return results


def main():
    parser = argparse.ArgumentParser(description='OGBL-DDI (GNN)')
    parser.add_argument('-f')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', default = True)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=64 * 1024)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=10)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygLinkPropPredDataset(name='ogbl-ddi',
                                     transform=T.ToSparseTensor())
    data = dataset[0]
    adj_t = data.adj_t.to(device)

    split_edge = dataset.get_edge_split()

    # We randomly pick some training samples that we want to evaluate on:
    torch.manual_seed(12345)
    idx = torch.randperm(split_edge['train']['edge'].size(0))
    idx = idx[:split_edge['valid']['edge'].size(0)]
    split_edge['eval_train'] = {'edge': split_edge['train']['edge'][idx]}

    if args.use_sage:
        model = SAGE(args.hidden_channels, args.hidden_channels,
                     args.hidden_channels, args.num_layers,
                     args.dropout).to(device)
    else:
        model = GCN(args.hidden_channels, args.hidden_channels,
                    args.hidden_channels, args.num_layers,
                    args.dropout).to(device)

    emb = torch.nn.Embedding(data.adj_t.size(0),
                             args.hidden_channels).to(device)
    predictor = NeuralLinkPredictor(args.hidden_channels, args.hidden_channels, 1,
                              args.num_layers, args.dropout).to(device)

    evaluator = Evaluator(name='ogbl-ddi')
    Logger_Models_Models = {
        'Hits@10': Logger_Models_Model(args.runs, args),
        'Hits@20': Logger_Models_Model(args.runs, args),
        'Hits@30': Logger_Models_Model(args.runs, args),
    }

    for run in range(args.runs):
        torch.nn.init.xavier_uniform_(emb.weight)
        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(emb.parameters()) +
            list(predictor.parameters()), lr=args.lr)

        for epoch in range(1, 1 + args.epochs):
            loss = train(model, predictor, emb.weight, adj_t, split_edge,
                         optimizer, args.batch_size)

            if epoch % args.eval_steps == 0:
                results = test(model, predictor, emb.weight, adj_t, split_edge,
                               evaluator, args.batch_size)
                for key, result in results.items():
                    Logger_Models_Models[key].add_result(run, result)

                if epoch % args.log_steps == 0:
                    for key, result in results.items():
                        train_hits, valid_hits, test_hits = result
                        print(key)
                        print(f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'Train: {100 * train_hits:.2f}%, '
                              f'Valid: {100 * valid_hits:.2f}%, '
                              f'Test: {100 * test_hits:.2f}%')
                    print('---')

        for key in Logger_Models_Models.keys():
            print(key)
            Logger_Models_Models[key].print_statistics(run)

    for key in Logger_Models_Models.keys():
        print(key)
        Logger_Models_Models[key].print_statistics()


if __name__ == "__main__":
    main()


# ## EDA on Graph

# In[ ]:


import pandas as pd
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
import networkx as nx
import random
import torch_geometric.transforms as T

from torch import Tensor
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling, convert, to_dense_adj
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.nn.conv import MessagePassing
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from sklearn.preprocessing import MinMaxScaler


# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sweetviz as sv
import spacy
import warnings
warnings.filterwarnings('ignore')
from termcolor import colored
import re
import random
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import en_core_web_sm
import seaborn as sns
import xlsxwriter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import en_core_web_sm
nlp = en_core_web_sm.load()
import gensim
from gensim import corpora
import gensim, spacy, logging, warnings
import gensim.corpora as corpora
#from gensim.utils import lemmatize, simple_preprocess
from gensim.models import CoherenceModel
import pyLDAvis


# In[ ]:


node_description = pd.read_csv('dataset/ogbl_ddi/mapping/nodeidx2drugid.csv')
node_info_df = node_description.copy()


# In[ ]:


drug_description = pd.read_csv('dataset/ogbl_ddi/mapping/ddi_description.csv')
drug_interaction_info_df = drug_description.copy()


# In[ ]:


node_list_data = node_description['node idx'].drop_duplicates().to_list()


# In[ ]:


from ogb.linkproppred import PygLinkPropPredDataset

dataset = PygLinkPropPredDataset(name = 'ogbl-ddi') 

split_edge = dataset.get_edge_split()
train_edge, valid_edge, test_edge = split_edge["train"], split_edge["valid"], split_edge["test"]
data = dataset[0]


# ### Drug Names

# In[ ]:


drug_names_1 = drug_description[['first drug id','first drug name']].drop_duplicates()
drug_names_1 = drug_names_1.rename(columns = {'first drug id':'drug id','first drug name':'drug name'})
drug_names_2 = drug_description[['second drug id','second drug name']].drop_duplicates()
drug_names_2 = drug_names_2.rename(columns = {'second drug id':'drug id','second drug name':'drug name'})
total_drug_names = pd.concat([drug_names_1,drug_names_2], axis = 0).drop_duplicates()


# In[ ]:


node_info_df_with_names = node_info_df.merge(total_drug_names, on = 'drug id', how = 'left')
node_info_df_with_names.head()
node_info_df_with_names = node_info_df_with_names.rename(columns = {'node idx': 'first node','drug id':'first drug id','drug name':'first drug name'})
dataset = PygLinkPropPredDataset(name='ogbl-ddi')
data = dataset[0]
G = convert.to_networkx(data, to_undirected=True)


# ### Node Neighbours

# In[ ]:


node_list = list(G.nodes)
nodes_neighbours = {}
for i in node_list:
    nodes_neighbours[i] = [n for n in G.neighbors(i)]  


# In[ ]:


node_degree = {}
for i in node_list:
    node_degree[i] = G.degree[i]


# In[ ]:


node_degree_data = pd.DataFrame()
node_degree_data['Nodes'] = node_list_data
node_degree_data['Degree'] = node_degree_data['Nodes'].map(node_degree)


# In[ ]:


node_degree_data.describe().to_csv('Degree_Stats.txt', sep = '\t')


# In[ ]:


node_degree_data.to_csv('Node_Degree_Data.txt', index = False, sep = '\t')


# In[ ]:


node_info_df_with_names


# ### Mapping node to neighbours

# In[ ]:


node_info_df_with_names['second node'] = node_info_df_with_names['first node'].map(nodes_neighbours)
node_info_df_with_names = (node_info_df_with_names.join(pd.DataFrame(node_info_df_with_names.pop('second node')
                            .values.tolist())
               .stack()
               .reset_index(level=1, drop=True)
               .rename('second node'))).reset_index(drop=True)
node_info_df_with_names['second node'] = node_info_df_with_names['second node'].astype(int)


# In[ ]:


adding_second_drug_id = node_info_df_with_names.merge(node_description, left_on = 'second node', right_on = 'node idx')
adding_second_drug_id = adding_second_drug_id.drop(['node idx'], axis = 1)
adding_second_drug_name = adding_second_drug_id.merge(total_drug_names, on = 'drug id', how = 'left')
adding_second_drug_name = adding_second_drug_name.rename(columns = {'drug id':'second drug id','drug name':'second drug name'})


# In[ ]:


complete_node_interaction_df = adding_second_drug_name.merge(drug_interaction_info_df, 
                              on = ['first drug id','first drug name','second drug id','second drug name'], 
                             how = 'left')
complete_node_interaction_df = complete_node_interaction_df.rename(columns = {'description':'polypharmacy side effect'})


# In[ ]:


empty_data = complete_node_interaction_df[complete_node_interaction_df['polypharmacy side effect'].isna()]


# In[ ]:


def assign_reverse_combos(row):
    first_node = row[1]
    second_node = row[4]
    
    global complete_node_interaction_df
    
    get_val = complete_node_interaction_df[(complete_node_interaction_df['first drug id']==second_node) & (complete_node_interaction_df['second drug id']==first_node) ]['polypharmacy side effect']
    get_val = get_val.tolist()[0]
    
    complete_node_interaction_df.loc[(complete_node_interaction_df['second drug id']==second_node) & (complete_node_interaction_df['first drug id']==first_node),'polypharmacy side effect' ] = get_val
    
empty_data.apply(assign_reverse_combos,axis=1)


# In[ ]:


mismatch_df = pd.DataFrame()
def check_value(row):
    first_node = row[1]
    second_node = row[4]
    
    global mismatch_df
    
    combination_1 = complete_node_interaction_df[(complete_node_interaction_df['first drug id']==second_node) & (complete_node_interaction_df['second drug id']==first_node) ]['polypharmacy side effect']
    combination_1 = combination_1.tolist()[0]
    
    combination_2 = complete_node_interaction_df[(complete_node_interaction_df['second drug id']==second_node) & (complete_node_interaction_df['first drug id']==first_node) ]['polypharmacy side effect']
    combination_2 = combination_2.tolist()[0]
    
    if combination_1 != combination_2:
        mismatch_df.append(row)
        
complete_node_interaction_df.apply(check_value,axis=1)


# In[ ]:





# ### Embedding Visualizations

# In[ ]:


# Randomly sample 2k training edges 
tsne_edges_train = split_edge['train']['edge']
tsne_edges_train = tsne_edges_train.T
train_edge_tuples = list(zip(tsne_edges_train[0], tsne_edges_train[1]))
tsne_tuples_train = random.sample(train_edge_tuples, 3500)

# Randomly sample 2k test edges 
tsne_edges_test = split_edge['test']['edge']
tsne_edges_test = tsne_edges_test.T
test_edge_tuples = list(zip(tsne_edges_test[0], tsne_edges_test[1]))
tsne_tuples_test = random.sample(test_edge_tuples, 1000)


# In[ ]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# If you use GPU, the device should be cuda
print('Device: {}'.format(device))


# In[ ]:


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random


# In[ ]:



def plot_tsne(emb_filepath, emb_model_name, color):
  '''
    Generate 2D tSNE representation of node embeddings as specified in 
    emb_filepath. Generate 3 plots: one with just the node embeddings, another
    with node embeddings and 2k randomly sampled edges from the train set, and
    a third with node embeddings and 2k randomly sampled edges from the test set. 
  '''
  node_emb = torch.load(filepath, map_location='cpu').to(device)
  cpu_emb = node_emb.cpu().data.numpy() # move to cpu, convert to numpy array

  # Apply t-SNE transformation on node embeddings
  tsne = TSNE(n_components=2)
  node_embeddings_2d = tsne.fit_transform(cpu_emb)  


  # Define subplots
  f, axs = plt.subplots(1,3,figsize=(18,5), dpi=80)

  # Plot train set embeddings
  emb_color = '#EF8A5A'
  alpha = 0.2
  axs[0].scatter(
      node_embeddings_2d[:, 0],
      node_embeddings_2d[:, 1],
      s=100,
      c=emb_color,
      alpha=alpha,
  )

  plot_title = f'Node Embeddings'
  axs[0].set_title(plot_title)

  # Plot embeddings with randomly sampled train set edges
  train_color = '#F6B53D'
  axs[1].scatter(
      node_embeddings_2d[:, 0],
      node_embeddings_2d[:, 1],
      s=100,
      c=train_color, 
      alpha=alpha,
  )
  for x, y in tsne_tuples_train:
      i, j = x.item(), y.item()
      x_i, x_j = node_embeddings_2d[i, 0], node_embeddings_2d[j, 0]
      y_i, y_j = node_embeddings_2d[i, 1], node_embeddings_2d[j, 1]
      axs[1].plot([x_i,x_j],[y_i,y_j],'k-', linewidth=0.10)

  plot_title = f'Node Embeddings: Train Edges'
  axs[1].set_title(plot_title)
  # Plot embeddings with randomly sampled test set edges
  test_color = '#15CAB6' #hot pink

  axs[2].scatter(
      node_embeddings_2d[:, 0],
      node_embeddings_2d[:, 1],
      s=100,
      c=test_color, 
      alpha=alpha,
  )
  for x, y in tsne_tuples_test:
      i, j = x.item(), y.item()
      x_i, x_j = node_embeddings_2d[i, 0], node_embeddings_2d[j, 0]
      y_i, y_j = node_embeddings_2d[i, 1], node_embeddings_2d[j, 1]
      axs[2].plot([x_i,x_j],[y_i,y_j],'k-', linewidth=0.10)
  
  plot_title = f'Node Embeddings: Test Edges'
  axs[2].set_title(plot_title)

  sup_title = f't-SNE Visualization of Train and Test Edge Embeddings using GCN'
  f.suptitle(sup_title)
  figure_path = f'tsne_{emb_model_name}.png'
  plt.savefig(figure_path)


# In[ ]:


model_name = 'graph_sage'
run = 0
filepath = 'C:/Users/yuvas/Documents/MSc Course/Dissertation/OGB - DDI/Output/training_outputs/'+f'{model_name}_final_emb_{run}.pt'
color = '#f794e9'
plot_tsne(filepath, model_name, color)


# In[ ]:


model_name = 'mf'
run = 0
filepath = 'C:/Users/yuvas/Documents/MSc Course/Dissertation/OGB - DDI/Output/training_outputs/'+f'{model_name}_final_emb_{run}.pt'
color = '#f794e9'
plot_tsne(filepath, model_name, color)


# In[ ]:


filepath = 'C:/Users/yuvas/Documents/MSc Course/Dissertation/OGB - DDI/Output/training_outputs/' + 'embedding.pt' # Plot tSNE for Node2Vec embeddings
color = '#f794e9'
plot_tsne(filepath, 'Node2Vec_256_dim', color)


# In[ ]:


model_name = 'gnn'
run = 0
filepath = f'{model_name}_final_emb_{run}.pt'
color = '#f794e9'
plot_tsne(filepath, model_name, color)


# In[ ]:





# In[ ]:




