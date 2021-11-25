from torch.utils.data import Dataset
import numpy as np

# TODO: Build class custom_dataset 
# att: train, test, transform

class TripletData(Dataset):
    def __init__(self, custom_dataset):
        self.dataset = custom_dataset
        self.train = self.custom_dataset.train
        self.transform = self.custom_dataset.transform

        if self.train:
            self.train_labels = self.custom_dataset.train_labels
            self.train_data = self.custom_dataset.train_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0] for label in self.labels_set}
        
        else:
            self.test_labels = self.custom_dataset.test_labels
            self.test_data = self.custom_dataset.test_data
            self.labels_set = set(self.test_data.numpy())
            self.label_to_indices = {label: np.where(self.test_data.numpy() == label)[0] for label in self.labels_set}      

        random_state = np.random.RandomState(11)
        
        # Generate fixed triplets for testing
        
        triplets = [[i, 
        random_state.choice(self.label_to_indices[self.test_data[i].item()]),
        random_state.choice(self.label_to_indices[np.random.choice(list(self.labels_set - set([self.test_data[i].item()])))])] for i in range(len(self.test_data))]

        self.test_triplets = triplets
    
    def __getitem__(self, index):
        if self.train:
            audio1, label1 = self.train_data[index], self.train_labels[index].items()
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set)-set([label1]))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            audio2 = self.train_data[positive_index]
            audio3 = self.train_data[negative_index]
        else:
            audio1 = self.test_data[self.test_triplets[index][0]]
            audio2 = self.test_data[self.test_triplets[index][1]]
            audio3 = self.test_data[self.test_triplets[index][2]]
        
        if self.transform is not None:
            audio1 = self.transform(audio1)
            audio2 = self.transform(audio2)
            audio3 = self.transform(audio3)

        return (audio1, audio2, audio3), []
    
    def __len__(self):
        return len(self.dataset)