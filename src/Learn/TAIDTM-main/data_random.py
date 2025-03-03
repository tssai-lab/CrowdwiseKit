import numpy as np
import torch.utils.data as Data
from PIL import Image

import tools
import torch
import sys
    
class svhn_dataset(Data.Dataset):
    def __init__(self, root,train=True, transform=None, target_transform=None, noise_rate=0.2, split_percentage=0.9, seed=1, num_classes=10, feature_size=3*32*32, norm_std=0.1, num_worker=300, expert_type_num=30,redun=2):
            
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.class_num=num_classes
        
        original_images = np.load(root+'train_images.npy')
        original_labels = np.load(root+'train_labels.npy')
        data = torch.from_numpy(original_images).float()
        targets = torch.from_numpy(original_labels)

        new_labels=[]
        for i in range(expert_type_num):
            dataset = zip(data, targets)
            new_labels.append(tools.get_instance_noisy_label(noise_rate, dataset, targets, num_classes, feature_size, norm_std, seed+i,int(num_worker/expert_type_num)))
        new_labels=torch.Tensor(np.concatenate(new_labels,axis=1))
        print(new_labels.shape)
        rand=torch.from_numpy(np.random.rand(new_labels.shape[0],new_labels.shape[1]))
        values,indices=rand.topk(1,dim=1)
        mask= rand>=values[:,-1,None]

        self.right_data=self.get_right_data( new_labels[mask].view(-1,1).long().cpu().numpy(), indices.long().cpu().numpy(),num_worker)
        if(redun>1):
            label_pro=(redun-1)/float(num_worker)
            all_indices =torch.nonzero(rand<label_pro).long()
            print((rand<label_pro).shape)
            print(all_indices.shape)
            for i,j in zip(all_indices[:,0],all_indices[:,1]):
                self.right_data[i,j,new_labels[i,j].long()]=1
            print(len(indices)/len(self.right_data), label_pro)
        print(np.sum(self.right_data)/len(self.right_data))
        #print(self.right_data[0])
        self.train_data, self.val_data, self.train_labels, self.val_labels, train_true_labels, val_true_labels = tools.data_split(original_images, self.right_data, targets, split_percentage,seed)
        if self.train:
            self.left_data = self.train_data.reshape((-1,3,32,32)).transpose((0, 2, 3, 1))
            self.right_data = self.train_labels
            self.true_label = train_true_labels
        else:
            self.left_data = self.val_data.reshape((-1,3,32,32)).transpose((0, 2, 3, 1))
            self.right_data = self.val_labels
            self.true_label = val_true_labels
        self.label_initial()

    def __getitem__(self, index):
           
        if self.train:
            img, label = self.train_data[index], self.train_labels[index]
            
        else:
            img, label = self.val_data[index], self.val_labels[index]
            
        img = Image.fromarray(img)
           
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
     
        return img, label
    def __len__(self):
            
        if self.train:
            return len(self.train_data)
        
        else:
            return len(self.val_data)
        
class svhn_test_dataset(Data.Dataset):
    def __init__(self, root, transform=None, target_transform=None):
            
        self.transform = transform
        self.target_transform = target_transform
           
        self.test_data = np.load(root+'test_images.npy')
        self.test_labels = np.load(root+'test_labels.npy')
        self.test_data = self.test_data.reshape((-1,3,32,32))
        self.test_data = self.test_data.transpose((0, 2, 3, 1)) 

    def __getitem__(self, index):
        
        img, label = self.test_data[index], self.test_labels[index]
        
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
     
        return img, label
    
    def __len__(self):
        return len(self.test_data)
    
class fashionmnist_dataset(Data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, noise_rate=0.2, split_percentage=0.9, seed=1, num_classes=10, feature_size=784, norm_std=0.1, num_worker=30,expert_type_num=3,redun=2):
            
        self.transform = transform
        self.target_transform = target_transform
        self.train = train 
        self.class_num=num_classes
        
        original_images = np.load(root+'train_images.npy')
        original_labels = np.load(root+'train_labels.npy')
        data = torch.from_numpy(original_images).float()
        targets = torch.from_numpy(original_labels)

        new_labels=[]
        for i in range(expert_type_num):
            dataset = zip(data, targets)
            new_labels.append(tools.get_instance_noisy_label(noise_rate, dataset, targets, num_classes, feature_size, norm_std, seed+i,int(num_worker/expert_type_num)))
        new_labels=torch.Tensor(np.concatenate(new_labels,axis=1))
        print(new_labels.shape)
        rand=torch.from_numpy(np.random.rand(new_labels.shape[0],new_labels.shape[1]))
        values,indices=rand.topk(1,dim=1)
        mask= rand>=values[:,-1,None]
        #mask=torch.zeros(new_labels.shape[0],new_labels.shape[1])
        #new_labels[mask!=1]= -1
        #print(new_labels[mask==1][:100])
        #print(indices)
        # print(indices.shape)
        # print(indices[:10])
        # print(indices[0])
        # print(new_labels[indices][0])
        # print(new_labels[0])
        # sys.exit()
        
        #print(new_labels[:3])
        #print(new_labels[mask].view(-1,1).long().cpu().numpy()[:3])
        #print(indices.long().cpu().numpy()[:3])
        self.right_data=self.get_right_data( new_labels[mask].view(-1,1).long().cpu().numpy(), indices.long().cpu().numpy(),num_worker)
        if(redun>1):
            label_pro=(redun-1)/float(num_worker)
            all_indices =torch.nonzero(rand<label_pro).long()
            print((rand<label_pro).shape)
            print(all_indices.shape)            
            for i,j in zip(all_indices[:,0],all_indices[:,1]):
                self.right_data[i,j,new_labels[i,j].long()]=1
            print(len(indices)/len(self.right_data), label_pro)
        print(np.sum(self.right_data)/len(self.right_data))
        #print(self.right_data[:3])
        self.train_data, self.val_data, self.train_labels, self.val_labels, train_true_labels, val_true_labels = tools.data_split(original_images, self.right_data, targets, split_percentage,seed)
        if self.train:
            self.left_data = self.train_data
            self.right_data =self.train_labels
            self.true_label=train_true_labels
        else:
            self.left_data = self.val_data
            self.right_data =self.val_labels
            self.true_label=val_true_labels
        self.label_initial()
            
    def __getitem__(self, index):
            
        img = Image.fromarray(self.left_data[index])
           
        if self.transform is not None:
            img = self.transform(img)
           
        left, right, label, true_label = img, self.right_data[index], self.label[index], self.true_label[index]
        return left, right, label, true_label,index
        
    def __len__(self):
            
        return len(self.left_data)

    def label_initial(self):

        linear_sum = torch.sum(torch.tensor(self.right_data), dim=1)
        linear_sum /= torch.sum(linear_sum,1).unsqueeze(1)
        print(linear_sum.shape)
        _,self.mv_label = torch.max(linear_sum,-1)
        self.label =  self.mv_label
        
        one_hot_labels=torch.zeros(len(self.true_label), self.class_num).scatter_(1, self.true_label.view(-1,1), 1)
        
        #print(self.label[:100])
        print('mv noise rate', sum(self.mv_label!=self.true_label)/len(self.true_label))
        print('all noise rate', 1-torch.sum(one_hot_labels*linear_sum)/len(self.true_label))
        #sys.exit()

    def get_right_data(self, anwsers, workers,num_worker):
        print(workers.shape)
        print(anwsers.shape)
        right_data=np.zeros((len(anwsers),num_worker,self.class_num))
        for i in range(len(anwsers)):
            for j,worker_id in zip(anwsers[i],workers[i]):
                right_data[i,worker_id,j]=1
        #print(anwsers[0])
        #print(workers[0])
        #print(right_data[0])
        
        # temp=torch.from_numpy(right_data)
        # labels=torch.nonzero(temp).numpy()
        # np.savetxt("cifar10n.csv", labels, fmt='%d',delimiter=',') 
        #print(np.sum(right_data))        
        return right_data
        
    def label_update(self, new_label):
        index = (new_label>=0)
        self.left_data, self.right_data, self.label,self.true_label= self.left_data[index], self.right_data[index], new_label[index],self.true_label[index]
        print(len(self.left_data))
        class_num= []
        for i in range(self.class_num):
            temp = (i==self.label).sum()
            class_num.append(temp)
        # weight = 1/torch.Tensor(class_num)
        # weight = weight/sum(weight)
        # samples_weight = np.array([weight[t.long()] for t in self.label])
        # samples_weight = torch.from_numpy(samples_weight)
        print('class_num',class_num)
        return class_num

    def get_worker_data(self, worker_id):
        temp = torch.from_numpy(np.sum(self.right_data[:,worker_id], axis=-1))
        index_list=torch.nonzero(temp).squeeze().long()
        batch_img=[]
        batch_right=[]
        batch_label=[]
        batch_true_label=[]
        right_data = torch.from_numpy(self.right_data)
        true_label = self.true_label
        #print(index_list.shape)
        if(index_list.shape==torch.Size([])):
            return [],[],[],[]
        if(len(index_list)<5):
            return [],[],[],[]
        for index in index_list:
            img = Image.fromarray(self.left_data[index])

            img = self.transform(img)
        
            batch_img.append(img)
            batch_right.append(right_data[index,worker_id])
            batch_label.append(self.label[index])
            batch_true_label.append(true_label[index])           
            
            #left, right, label, true_label = img, self.right_data[index,worker_id], self.label[index], self.true_label[index]
        left = torch.stack(batch_img, 0)
        right = torch.stack(batch_right, 0)
        label = torch.stack(batch_label, 0)
        true_label = torch.stack(batch_true_label, 0)
        
        
        # print(left.shape)
        # print(right.shape)
        # print(label.shape)
        # print(true_label.shape)
        
        # sys.exit()
        return left, right, label, true_label
        
class fashionmnist_test_dataset(Data.Dataset):
    def __init__(self, root, transform=None, target_transform=None):
            
        self.transform = transform
        self.target_transform = target_transform
           
        self.test_data = np.load(root+'test_images.npy')
        self.test_labels = np.load(root+'test_labels.npy')


    def __getitem__(self, index):
        
        img, label = self.test_data[index], self.test_labels[index]
        
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
     
        return img, label, index
    
    def __len__(self):
        return len(self.test_data)

class fashionmnist_train_dataset(Data.Dataset):
    def __init__(self, root, transform=None, target_transform=None):
            
        self.transform = transform
        self.target_transform = target_transform
           
        self.train_data = np.load(root+'train_images.npy')
        self.train_labels = np.load(root+'train_labels.npy')


    def __getitem__(self, index):
        
        img, label = self.train_data[index], self.train_labels[index]
        
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
     
        return img, label, index
    
    def __len__(self):
        return len(self.train_data)
    

