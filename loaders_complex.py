import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as tfs
import pickle
from torch.utils.data import DataLoader, random_split
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import ConcatDataset, random_split

class BaseDataset(data.Dataset):
    def __init__(self, dataset_root_dir, mode, domain, img_size):
        self.root_dir = dataset_root_dir
        self.imgs = []
        self.labels = []
        self.domain = domain
        self.mode = mode
        if mode == 'test' or mode == 'val':
            self.transforms = tfs.Compose([tfs.Resize((img_size, img_size)),
                                           tfs.ToTensor(),
                                           tfs.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])
        elif mode == 'train':
            self.transforms = tfs.Compose([
                tfs.RandomResizedCrop(img_size, scale=(0.7, 1.0)), 
                tfs.RandomHorizontalFlip(),
                tfs.ColorJitter(0.3, 0.3, 0.3, 0.3),
                tfs.RandomGrayscale(),
                tfs.ToTensor(),
                tfs.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        
    def __len__(self):
        return len(self.imgs)

    def collate_fn(self, batch):
        imgs, labels = zip(*batch)
        imgs = torch.stack(imgs).cuda()
        labels = torch.stack(labels).cuda()
        return imgs, labels

class OfficeCaltech(BaseDataset):
    def __init__(self, dataset_root_dir, mode, domain, img_size=224):
        BaseDataset.__init__(self, dataset_root_dir, mode, domain, img_size)
        if mode=='train':
            self.paths, self.text_labels = np.load(os.path.join(dataset_root_dir, '{}_train.pkl'.format(domain)), allow_pickle=True)
            self.perm = list(range(len(self.paths)))
            random.Random(1).shuffle(self.perm)
            self.perm = self.perm[:60]
        else:
            self.paths, self.text_labels = np.load(os.path.join(dataset_root_dir, '{}_test.pkl'.format(domain)), allow_pickle=True)
            self.perm = list(range(len(self.paths)))
            random.Random(1).shuffle(self.perm)
            self.perm = self.perm[:24]        
            
        label_dict={'back_pack':0, 'bike':1, 'calculator':2, 'headphones':3, 'keyboard':4, 'laptop_computer':5, 'monitor':6, 'mouse':7, 'mug':8, 'projector':9}
        self.labels = [label_dict[text] for text in self.text_labels]
        self.CLASSES = range(10)

    def __len__(self):
        return len(self.perm)

    def __getitem__(self, idx):
        idx = self.perm[idx]
        img_path = os.path.join(self.root_dir, self.paths[idx])
        label = self.labels[idx]
        image = Image.open(img_path.replace('office_caltech_10/', ''))
        if len(image.split()) != 3:
            image = tfs.Grayscale(num_output_channels=3)(image)

        return self.transforms(image), label

class DomainNet(BaseDataset):   
    def __init__(self, dataset_root_dir, mode, domain, img_size=224):
        #Code took from FedBN
        if 'DomainNet' in dataset_root_dir:
            dataset_root_dir = dataset_root_dir.replace('/DomainNet', '')
        BaseDataset.__init__(self, dataset_root_dir, mode, domain, img_size)        
        
        self.source_domains = ['real', 'clipart', 'painting', 'sketch', 'infograph', 'quickdraw']
        self.CLASSES = ['bird', 'feather', 'headphones', 'ice_cream', 'teapot', 'tiger', 'whale', 'windmill', 'wine_glass', 'zebra']  
        
        #follow FedBN
        if mode=='train':
            self.split_images('train')
        if mode=='val':
            self.split_images('test_val')  
        if mode=='test':
            self.split_images('test')  
        
    def split_images(self, mode):         
        split_file = os.path.join(
            self.root_dir, f'DomainNet/FedBN_split/{self.domain}_{mode.split("_")[0]}.pkl')
        
        with open(split_file, "rb") as f:
            self.imgs = pickle.load(f)[0]
        random.Random(1).shuffle(self.imgs)    
        if mode=='test':
            self.imgs = self.imgs[:int(0.5 * len(self.imgs))]
        elif mode=='test_val':
            self.imgs = self.imgs[int(0.5 * len(self.imgs)):]
        self.imgs = self.imgs[:int(0.1 * len(self.imgs))]

    def __getitem__(self, index):     
        imgs = []
        labels = []   
        p = self.imgs[index]
        img = self.transforms(Image.open(os.path.join(self.root_dir,p)).convert('RGB'))
        label = torch.tensor(self.CLASSES.index(p.split('/')[2]))
        
        return img, label       

class PACS(BaseDataset):   
    def __init__(self, dataset_root_dir, mode, domain, img_size=224):
        BaseDataset.__init__(self, dataset_root_dir, mode, domain, img_size)
        self.CLASSES = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
        self.mode = mode
        self.split_images()        
        self.source_domains = ['art_painting', 'cartoon', 'photo', 'sketch']
        
    def split_images(self):    
        split_dir = self.root_dir + '/splits/'
        for p in os.listdir(split_dir):
            if 'test' in p and self.domain in p:
                with open(split_dir + p) as f:                    
                    for l in f.readlines():
                        self.imgs.append(l.split(' ')[0])    

        N = len(self.imgs) 
        random.Random(1).shuffle(self.imgs) 
        if self.mode=='train':
            self.imgs = self.imgs[:int(0.8*N)]
        elif self.mode=='val':
            self.imgs = self.imgs[int(0.8*N):int(0.9*N)]
        elif self.mode=='test':
            self.imgs = self.imgs[int(0.9*N):]

        self.imgs = self.imgs[:int(0.1*len(self.imgs))]
        
    def __getitem__(self, index):     
        imgs = []
        labels = []   
        p = self.imgs[index]
        img = self.transforms(Image.open(self.root_dir + '/images/kfold/' + p).convert('RGB'))
        tag = p.split('/')[1]
        label = torch.tensor(self.CLASSES.index(tag))
        return img, label  

class rxrx1(BaseDataset):   
    def __init__(self, dataset_root_dir, mode, domain, img_size=256):
        BaseDataset.__init__(self, dataset_root_dir, mode, domain, img_size)
        self.CLASSES = list(range(20))
        self.client_id = domain * 4
        self.mode = mode
        self.dataset_root_dir = dataset_root_dir
        self.imgs, self.gts = [], []
        PATH = dataset_root_dir
        for d in os.listdir(PATH):
            for cid in range(12):
                if cid%4==domain:
                    if not (f'client_{cid}_' in d and mode in d and 'img' in d): continue
                # if f'client_{self.client_id + delta}_' in d and mode in d and 'img' in d:
                    gt = torch.load(os.path.join(PATH, d.replace('img', 'gt')))
                    if not (gt<=11 or gt>=40): continue
                    self.imgs.append(os.path.join(PATH, d))
                    self.gts.append(os.path.join(PATH, d.replace('img', 'gt')))
                    
    def __getitem__(self, index):
        img = torch.load(self.imgs[index])
        label = torch.load(self.gts[index])
        if label>11:
            label -= 28
        return img, torch.LongTensor([label]).squeeze()

class OfficeHome(BaseDataset):
    def __init__(self, dataset_root_dir, mode, domain, img_size=224):
        BaseDataset.__init__(self, dataset_root_dir, mode, domain, img_size)
        
        CLASSES = 'Alarm Clock, Backpack, Batteries, Bed, Bike, Bottle, Bucket, Calculator, Calendar, Candles, Chair, Clipboards, Computer, Couch, Curtains, Desk Lamp, Drill, Eraser, Exit Sign, Fan, File Cabinet, Flipflops, Flowers, Folder, Fork, Glasses, Hammer, Helmet, Kettle, Keyboard, Knives, Lamp Shade, Laptop, Marker, Monitor, Mop, Mouse, Mug, Notebook, Oven, Pan, Paper Clip, Pen, Pencil, Postit Notes, Printer, Push Pin, Radio, Refrigerator, ruler, Scissors, Screwdriver, Shelf, Sink, Sneakers, Soda, Speaker, Spoon, Table, Telephone, Toothbrush, Toys, Trash Can, TV, Webcam'
        self.CLASSES = [c.upper() for c in CLASSES.split(', ')]
        self.source_domains = ['Art', 'Clipart', 'Product', 'Real_World']
        
        imgs = {}
        f = os.path.join(self.root_dir, self.domain)
        for cls in os.listdir(f):
            imgs[cls] = []
            for fff in os.listdir(os.path.join(f, cls)):
                imgs[cls].append(os.path.join(self.domain, cls, fff))                
        
        self.imgs = []
        for cls in imgs.keys():
            random.Random(1).shuffle(imgs[cls]) 
            if self.mode=='train':
                self.imgs += imgs[cls][:8]
            elif self.mode=='val':
                self.imgs += imgs[cls][8:10]
            elif self.mode=='test':
                self.imgs += imgs[cls][10:12]
            
    def __getitem__(self, index):
        imgs = []
        labels = []
        p = self.imgs[index]
        img = self.transforms(Image.open(
            os.path.join(self.root_dir, p)).convert('RGB'))
        tag = p.split('/')[1].replace('_', ' ').upper()
        label = torch.tensor(self.CLASSES.index(tag))   
        
        return img, label

def get_source_domain_names(dataset):
    if dataset=='rxrx1':
        return [0, 1, 2, 3]
    elif dataset=='PACS':
        return ['art_painting', 'cartoon', 'photo', 'sketch']
    elif dataset=='DomainNet':
        return ['real', 'clipart', 'painting', 'sketch', 'infograph', 'quickdraw']
    elif dataset=='OfficeHome':
        return ['Art', 'Clipart', 'Product', 'Real_World']
    elif dataset=='OfficeCaltech':
        return ['amazon', 'caltech', 'dslr', 'webcam']

def init_loaders_complex(
        config
    ):
        loaders = []
        for d in get_source_domain_names(config['dataset']):
            dataset = {
                'rxrx1': rxrx1,
                'PACS': PACS,
                'DomainNet': DomainNet,
                'OfficeHome': OfficeHome,
                'OfficeCaltech': OfficeCaltech,
            }[config['dataset']]
            trainset = dataset(config['dataset_dir'], mode = 'train', domain=d)
            valset = dataset(config['dataset_dir'], mode = 'val', domain=d)
            if config['dataset'] in ['rxrx1', 'OfficeCaltech']:
                valset, testset = torch.utils.data.random_split(valset, [len(valset)//2, len(valset)-len(valset)//2],
                    generator=torch.Generator().manual_seed(1))
            else:
                testset = dataset(config['dataset_dir'], mode = 'test', domain=d)
            print(f"{d}: trainset length: {len(trainset)} || valset length: {len(valset)} || testset length: {len(testset)}")

            def get_loader(trainset, valset, testset):
                data = {}
                data['train'] = DataLoader(dataset=trainset, batch_size=len(trainset), shuffle=True)
                data['val'] = DataLoader(dataset=valset, batch_size=len(valset))
                data['test'] = DataLoader(dataset=testset, batch_size=len(testset))

                def loader(*args):
                    output = []
                    for arg in args:
                        Xarg, Yarg = next(iter(data[arg]))
                        output.append(Xarg.cuda(non_blocking=True))
                        output.append(Yarg.cuda(non_blocking=True))
                    return output
                return loader
                
            loaders.append(get_loader(trainset, valset, testset))

        config['num_users'] = len(loaders)
        config['num_classes'] = len(trainset.CLASSES)
        return loaders
