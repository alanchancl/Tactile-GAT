import os
import re
import numpy as np
import pandas as pd

def load_labels(file_path):
    with open(file_path, 'r') as file:
        labels = [line.strip() for line in file.readlines() if line.strip()]
    return labels

def create_label_dict(labels):
    return {label: idx for idx, label in enumerate(labels)}

class LabelExtractor:
    def __init__(self, label_file_path):
        self.labels = load_labels(label_file_path)
        self.label_dict = create_label_dict(self.labels)
        self.label_pattern = self.build_label_pattern(self.labels)

    @staticmethod
    def build_label_pattern(labels):
        escaped_labels = [re.escape(label) for label in labels]
        pattern = '|'.join(escaped_labels)
        return re.compile(pattern, re.IGNORECASE)

    def get_label_from_filename(self, filename):
        match = self.label_pattern.search(filename)
        if match:
            return self.label_dict[match.group()]
        return None
    
def get_most_common_features(target, all_features, max = 3, min = 3):
    res = []
    main_keys = target.split('_')

    for feature in all_features:
        if target == feature:
            continue

        f_keys = feature.split('_')
        common_key_num = len(list(set(f_keys) & set(main_keys)))

        if common_key_num >= min and common_key_num <= max:
            res.append(feature)

    return res

def build_net(target, all_features):
    # get edge_indexes, and index_feature_map
    main_keys = target.split('_')
    edge_indexes = [
        [],
        []
    ]
    index_feature_map = [target]

    # find closest features(nodes):
    parent_list = [target]
    graph_map = {}
    depth = 2
    
    for i in range(depth):        
        for feature in parent_list:
            children = get_most_common_features(feature, all_features)

            if feature not in graph_map:
                graph_map[feature] = []
            
            # exclude parent
            pure_children = []
            for child in children:
                if child not in graph_map:
                    pure_children.append(child)

            graph_map[feature] = pure_children

            if feature not in index_feature_map:
                index_feature_map.append(feature)
            p_index = index_feature_map.index(feature)
            for child in pure_children:
                if child not in index_feature_map:
                    index_feature_map.append(child)
                c_index = index_feature_map.index(child)

                edge_indexes[1].append(p_index)
                edge_indexes[0].append(c_index)

        parent_list = pure_children

    return edge_indexes, index_feature_map

def load_data(npy_folder_path, signal_name, extractor):
    data_list = []
    for filePath in os.listdir(npy_folder_path):
        full_path = os.path.join(npy_folder_path, filePath)
        label = extractor.get_label_from_filename(filePath)
        if label is None:
            continue
        data = np.load(full_path)
        res = [list(data.T[i]) for i in range(get_range(signal_name))]
        labels = list(np.repeat(label, data.shape[0]))
        res.append(labels)
        data_list.append(res)
    return data_list

def get_range(signal_name):
    if signal_name == 'press':
        return 16
    elif signal_name == 'flex':
        return range(16, 22)
    return 22

def construct_data(dataset, signal_name):
    # npy_path = r'C:/Users/alanc/Desktop/GAT-Tactile/data/tactile/TactileData_npy/'
    extractor = LabelExtractor(f'./data/{dataset}/label.txt')
    data_path = f'./data/{dataset}/'
    train_data_path = os.path.join(data_path, 'train')
    test_data_path = os.path.join(data_path, 'test')

    train_data = load_data(train_data_path, signal_name, extractor)
    test_data = load_data(test_data_path, signal_name, extractor)

    return train_data, test_data

def build_loc_net(struc, feature_map):

    index_feature_map = feature_map
    edge_indexes = [
        [],
        []
    ]
    for node_name, node_list in struc.items():
        if node_name not in feature_map:
            continue

        if node_name not in index_feature_map:
            index_feature_map.append(node_name)
        
        p_index = index_feature_map.index(node_name)
        for child in node_list:
            if child not in feature_map:
                continue

            if child not in index_feature_map:
                print(f'error: {child} not in index_feature_map')
                # index_feature_map.append(child)

            c_index = index_feature_map.index(child)
            # edge_indexes[0].append(p_index)
            # edge_indexes[1].append(c_index)
            edge_indexes[0].append(c_index)
            edge_indexes[1].append(p_index)
    
    return edge_indexes

def get_feature_map(dataset, signal_name):
    """ Retrieves a list of features based on the signal type. """
    with open(f'./data/{dataset}/list.txt', 'r') as feature_file:
        feature_list = [ft.strip() for ft in feature_file]

    if signal_name == 'press':
        return feature_list[0:16]
    elif signal_name == 'flex':
        return feature_list[16:22]
    return feature_list
    

def get_tactile_graph_struc(feature_list):
    """ Generates a graph structure from adjacency matrix for tactile features. """
    adjacency_matrix = pd.read_csv('data/tactile/graph_struc.csv', index_col='Name')
    struc_map = {ft: adjacency_matrix.columns[adjacency_matrix.loc[ft] == 1].tolist() for ft in feature_list}
    return struc_map

def get_fc_graph_struc(feature_list):
    """ Generates a fully connected graph structure. """
    return {ft: [other_ft for other_ft in feature_list if other_ft != ft] for ft in feature_list}