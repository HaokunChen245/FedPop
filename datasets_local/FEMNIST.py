from utils.dataset_utils import *
from datasets.base import BaseDataset
ImageFile.LOAD_TRUNCATED_IMAGES = True
import json

class FEMNIST(BaseDataset):     
    def __init__(self, dataset_root_dir, mode, domain, img_size=28):
        BaseDataset.__init__(self, dataset_root_dir, mode, domain, img_size)
        if mode=="val":
            data_dir = os.path.join(dataset_root_dir, "test")
        else:
            data_dir = os.path.join(dataset_root_dir, mode)
        self.get_data(data_dir, domain)

    def get_data(self, data_dir, domain):
        files = os.listdir(data_dir)
        files = [f for f in files if f.endswith('.json')]
        clients = []
        data = {}        
        for f in files:
            file_path = os.path.join(data_dir, f)
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
            clients += cdata['users']
            data.update(cdata['user_data'])

        self.data = []
        k = sorted(list(data.keys()))[domain]
        for idx, x in enumerate(data[k]["x"]):
            x = torch.Tensor(x).view([1,28,28])
            y = torch.LongTensor([data[k]["y"][idx]]).squeeze()
            self.data.append([x, y])       

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):     
        img, label = self.data[index]
        return img.cuda(), label.cuda()


