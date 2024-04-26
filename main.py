import argparse
import numpy as np
import os
import random
import torch
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader, Subset, random_split
from models.TactileGAT import TactileGAT
from datasets.time_dataset import TimeDataset
from utils.device import get_device, set_device
from utils.preprocessing import build_loc_net, construct_data, get_feature_map, get_fc_graph_struc, get_tactile_graph_struc
from test import test
from train import train

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class Main:
    def __init__(self, config):
        setup_seed(config['seed'])
        self.datestr = None
        self.config = config
        set_device(config['device'])
        self.device = get_device()
        self.setup_data()
        self.setup_model()

    def setup_data(self):
        """Load and prepare data."""
        feature_map = get_feature_map(self.config['dataset'], self.config['signal_name'])
        self.feature_map = feature_map
        train_data, test_data = construct_data(self.config['dataset'], self.config['signal_name'])

        train_dataset = TimeDataset(train_data, mode='train', config=self.config)
        test_dataset = TimeDataset(test_data, mode='test', config=self.config)

        val_size = int(self.config['val_ratio'] * len(train_dataset))
        train_set, val_set = random_split(train_dataset, [len(train_dataset)-val_size, val_size])
        self.train_dataloader = DataLoader(train_set, batch_size=self.config['batch'], shuffle=True)
        self.val_dataloader = DataLoader(val_set, batch_size=self.config['batch'], shuffle=False)

        self.test_dataloader = DataLoader(test_dataset, batch_size=self.config['batch'], shuffle=False)

    def setup_model(self):
        """Initialize the model based on graph structure."""
        edge_index_sets = []
        fc_struc = get_fc_graph_struc(self.feature_map)
        tactile_struc = get_tactile_graph_struc(self.feature_map)
        fc_edge_index = build_loc_net(fc_struc, self.feature_map)
        tactile_edge_index = build_loc_net(tactile_struc, self.feature_map)
        fc_edge_index = torch.tensor(fc_edge_index, dtype=torch.long, device=self.device)
        tactile_edge_index = torch.tensor(tactile_edge_index, dtype=torch.long, device=self.device)
        if self.config['graph_name'] == 'g':
            edge_index_sets.append(fc_edge_index)
        elif self.config['graph_name'] == 'g+':
            edge_index_sets.append(tactile_edge_index)
        else:
            edge_index_sets.extend([fc_edge_index, tactile_edge_index])

        self.model = TactileGAT(edge_index_sets, len(self.feature_map),
                         graph_name=self.config['graph_name'],
                         dim=self.config['dim'],
                         input_dim=self.config['slide_win'],
                         out_layer_num=self.config['out_layer_num'],
                         out_layer_inter_dim=self.config['out_layer_inter_dim'],
                         topk=self.config['topk']).to(self.device)

    def run(self):
        """Run training and testing."""
        self.config['model_save_path'], self.config['result_save_path'] = self.get_save_path()
        train(self.model, self.config, self.train_dataloader, self.val_dataloader)
        self.model.load_state_dict(torch.load(self.config['model_save_path']))
        test_result, avg_loss, accuracy_rate, precision, recall, f1 = test(self.model, self.test_dataloader)
        learned_graph = self.model.get_learned_graph()
        self.save_results(self.config['result_save_path'], accuracy_rate, test_result, learned_graph)

    def get_save_path(self, feature_name=''):
        dir_path = self.config['save_path']
        
        if self.datestr is None:
            now = datetime.now()
            self.datestr = now.strftime('%m-%d_%H-%M-%S')
        datestr = self.datestr
        paths = [
            f'./pretrained/{dir_path}/best_{datestr}.pt',
            f'./results/{dir_path}/{datestr}',
        ]
        for path in paths:
            dirname = os.path.dirname(path)
            Path(dirname).mkdir(parents=True, exist_ok=True)
        return paths[0], paths[1]
    
    def save_results(self, result_save_path, accuracy, test_result, learned_graph):
        """Save test results and learned graph."""
        np_name = f"{result_save_path}_{self.config['topk']}_{self.config['graph_name']}_{accuracy:.4f}.npy"
        np.save(np_name, {
            'accuracy_rate': accuracy / 100,
            'test_result': test_result,
            'learned_graph': learned_graph
        })

def parse_args():
    parser = argparse.ArgumentParser(description="Main script parameters")
    parser.add_argument('-batch', help='Batch size', type = int, default=256)
    parser.add_argument('-epoch', help='Number of training epochs', type = int, default=50)
    parser.add_argument('-signal_name', help='signal name', type = str, default='')
    parser.add_argument('-graph_name', help='graph name(g or g+ or null for both)', type = str, default='')
    parser.add_argument('-slide_win', help='slide_win', type = int, default=200)
    parser.add_argument('-dim', help='dimension', type = int, default=64)#64
    parser.add_argument('-slide_stride', help='slide_stride', type = int, default=200)
    parser.add_argument('-save_path_pattern', help='save path pattern', type = str, default='tactile')
    parser.add_argument('-model_save_path', help='save model path', type = str, default='')
    parser.add_argument('-result_save_path', help='save result path', type = str, default='')
    parser.add_argument('-dataset', help='Name of Dataset', type = str, default='tactile')
    parser.add_argument('-device', help='cuda / cpu', type = str, default='cuda')
    parser.add_argument('-random_seed', help='random seed', type = int, default=0)
    parser.add_argument('-comment', help='experiment comment', type = str, default='')
    parser.add_argument('-out_layer_num', help='outlayer num', type = int, default=1)
    parser.add_argument('-out_layer_inter_dim', help='out_layer_inter_dim', type = int, default=256)
    parser.add_argument('-decay', help='decay', type = float, default=0)
    parser.add_argument('-lr', help='learning rat', type = float, default=0.01)
    parser.add_argument('-val_ratio', help='val ratio', type = float, default=0.1)
    parser.add_argument('-topk', help='topk num', type = int, default=6)
    parser.add_argument('-report', help='best / val', type = str, default='best')
    parser.add_argument('-load_model_path', help='trained model path', type = str, default='')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config = {
        'batch': args.batch,
        'epoch': args.epoch,
        'signal_name': args.signal_name,
        'graph_name': args.graph_name,
        'slide_win': args.slide_win,
        'dim': args.dim,
        'slide_stride': args.slide_stride,
        'comment': args.comment,
        'seed': args.random_seed,
        'out_layer_num': args.out_layer_num,
        'out_layer_inter_dim': args.out_layer_inter_dim,
        'decay': args.decay,
        'val_ratio': args.val_ratio,
        'topk': args.topk,
        'lr': args.lr,
        'save_path': args.save_path_pattern,
        'model_save_path': args.model_save_path,
        'result_save_path': args.result_save_path,
        'dataset': args.dataset,
        'report': args.report,
        'device': args.device,
        'load_model_path': args.load_model_path
    }
    main = Main(config)
    main.run()
