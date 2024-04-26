import torch
from torch.utils.data import Dataset

class TimeDataset(Dataset):
    def __init__(self, raw_data, mode='train', config=None):
        """
        Initializes the TimeDataset with raw data, operation mode, and configuration.
        
        Parameters:
        - raw_data (list): List of raw data entries.
        - mode (str): Indicates the mode of operation, 'train' or 'test'.
        - config (dict): Configuration dict that includes 'slide_win' and 'slide_stride'.
        """
        self.raw_data = raw_data
        self.mode = mode
        self.config = config
        self.x, self.y, self.labels = self.process(raw_data)
    
    def __len__(self):
        return len(self.x)

    def process(self, all_raw_data):
        """
        Processes raw data into features (x), targets (y), and labels.
        
        Returns:
        - tuple of (x, y, labels): Processed features, targets, and labels tensors.
        """
        x_list, y_list, labels_list = [], [], []

        slide_win = self.config['slide_win']
        slide_stride = self.config['slide_stride']

        for data_entry in all_raw_data:
            x_data = torch.tensor(data_entry[:-1], dtype=torch.float32)
            labels = torch.tensor(data_entry[-1], dtype=torch.float32)

            node_num, total_time_len = x_data.shape

            range_step = range(slide_win, total_time_len, slide_stride) if self.mode == 'train' else range(slide_win, total_time_len)
            
            for i in range_step:
                features = x_data[:, i-slide_win:i]
                target = x_data[:, i]

                x_list.append(features)
                y_list.append(target)
                labels_list.append(labels[i])

        return torch.stack(x_list), torch.stack(y_list), torch.tensor(labels_list, dtype=torch.float32)

    def __getitem__(self, idx):
        """
        Retrieves an item at the specified index.
        
        Parameters:
        - idx (int): Index of the item to retrieve.
        
        Returns:
        - tuple: (feature, target, label) tensors for the given index.
        """
        return self.x[idx], self.y[idx], self.labels[idx]
