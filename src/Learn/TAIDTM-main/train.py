import torch.nn.functional as F
from model_fmnist import *
from data_random import *
from torch.autograd import Variable
from sklearn.mixture import GaussianMixture
from common_DNNTM import *
import copy
import time
from sklearn.decomposition import KernelPCA as PCA #PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tools import *
import networkx as nx
import random
from torch_cluster import knn_graph
import sys
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score

def adjust_learning_rate(optimizer, epoch):
    epoch = epoch+ args.warmup_epoch
    lr = args.warmup_lr  * (0.1 ** (epoch > 40)) * (0.1 ** (epoch > 55) )
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

torch.set_printoptions(profile="full")
def warmup_train(epoch,net,net_optimizer,train_loader):
    net.train()
    begain = time.time()
    for batch_idx , (left, right, distilled_label, true_label, index) in enumerate(train_loader):
    
        ep = Variable(right).long().cuda()

        images = Variable(left).float().cuda()

        # distilled_label= distilled_label.long()
        # distilled_label= torch.zeros(len(distilled_label), Config.num_classes).scatter_(1, distilled_label.view(-1,1), 1).cuda()    
        
        net_optimizer.zero_grad()
        cls_out = net(images)
        
        # loss = -torch.log(cls_out+1e-20)*distilled_label
        # loss= loss.mean()
        out = torch.log(cls_out+1e-20).view(ep.size()[0],1,ep.size()[2]).repeat(1,ep.size()[1],1)
        vec=torch.sum(-out*ep,dim=-1)
        mask=torch.max(ep,dim=2)[0]
        zeromap = torch.zeros((ep.size()[0],ep.size()[1])).cuda()
        vec = torch.where(mask == 0,zeromap,vec)
        loss = torch.sum(vec)/torch.sum(mask)

        prior = torch.ones(Config.num_classes)/Config.num_classes
        prior = prior.cuda()
        pred_mean = cls_out.mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean)) 
        loss+= penalty
        
        loss.backward()
        net_optimizer.step()
    print('warmup epoch time', time.time()-begain)
    return 

def train_matrix_emb_training(epoch, label_emb_model, left_optimizer_emb,train_loader_select): 
    label_emb_model.train()
    avg_loss= 0
    total =0
    begain = time.time()
    for batch_idx , (left, right, distilled_label, true_label, index) in enumerate(train_loader_select):

        ep = Variable(right).float().cuda()

        left_optimizer_emb.zero_grad()  
        images = Variable(left).float().cuda()
        
        distilled_label= distilled_label.long()
        distilled_label= torch.zeros(len(distilled_label), Config.num_classes).scatter_(1, distilled_label.view(-1,1), 1).cuda()      
        noise_matrices_for_x = label_emb_model(images)
        out = torch.einsum("ij,irjk->irk",(distilled_label,noise_matrices_for_x))
        
        out = torch.log(out+1e-20)
        vec=torch.sum(-out*ep,dim=-1)
        mask=torch.max(ep,dim=2)[0]
        zeromap = torch.zeros((ep.size()[0],ep.size()[1])).cuda()
        vec = torch.where(mask == 0,zeromap,vec)
        loss = torch.sum(vec)/torch.sum(mask)        

        
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(label_emb_model.parameters(), max_norm=1, norm_type=2)
        left_optimizer_emb.step()
        avg_loss +=loss.item()
        total = batch_idx +1
    print('loss', avg_loss/total)
    print('train_matrix_emb epoch time', time.time()-begain)
    
def update_distilled_labels(history_model,train_loader_select):
    
    temp_labels = torch.zeros(len(train_loader_select.dataset),Config.num_classes)
    for static_para in history_model:
        new_model = left_neural_net().cuda()
        new_model.load_state_dict(static_para)
        new_model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, right_data, label, true_label, index) in enumerate(train_loader_select):
                inputs =inputs.cuda()
                outputs = new_model(inputs)
                temp_labels[index]+= outputs.cpu().data
    temp_labels = temp_labels/len(history_model)
    index=torch.nonzero(temp_labels>0.5*(1+args.th))
    distilled_labels=torch.ones(len(train_loader_select.dataset))*(-1)
    distilled_labels[index[:,0]]=index[:,1].float()
    class_num=train_loader_select.dataset.label_update(distilled_labels)            
    distilled_acc=torch.mean(torch.eq(index[:,1],train_loader_select.dataset.true_label).float())
    print('distilled_acc',distilled_acc)
    return train_loader_select,class_num
    
def train_transition_matrix(epoch, left_model_aux,  left_optimizer_aux,train_loader_select):
    left_model_aux.train()
    avg_loss= 0
    total =0
    begain = time.time()
    for batch_idx , (left, right, distilled_label, true_label, index) in enumerate(train_loader_select):
  
        ep = Variable(right).long().cuda()

        left_optimizer_aux.zero_grad()
        images = Variable(left).float().cuda()
        
        distilled_label= distilled_label.long()
        distilled_label= torch.zeros(len(distilled_label), Config.num_classes).scatter_(1, distilled_label.view(-1,1), 1).cuda()  
        noise_matrices_for_x = left_model_aux(images)

        out = torch.einsum("ij,ijk->ik",(distilled_label,noise_matrices_for_x))
        
        out = torch.log(out+1e-20).view(ep.size()[0],1,ep.size()[2]).repeat(1,ep.size()[1],1)
        vec=torch.sum(-out*ep,dim=-1)
        mask=torch.max(ep,dim=2)[0]
        zeromap = torch.zeros((ep.size()[0],ep.size()[1])).cuda()
        vec = torch.where(mask == 0,zeromap,vec)
        loss = torch.sum(vec)/torch.sum(mask)

        loss.backward()
        #torch.nn.utils.clip_grad_norm_(left_model_aux.parameters(), max_norm=1, norm_type=2)
        left_optimizer_aux.step()
        avg_loss +=loss.item()
        total = batch_idx +1     
    print('loss', avg_loss/total)
    print('TM epoch time', time.time()-begain)

def val_correction(epoch, left_model_aux,val_loader):
    left_model_aux.eval()
    avg_loss= 0
    total =0
    begain = time.time()
    for batch_idx , (left, right, _, true_label, index) in enumerate(val_loader):

        ep = Variable(right).long().cuda()

        images = Variable(left).float().cuda()
        
        distilled_label= true_label.long()
        distilled_label= torch.zeros(len(distilled_label), Config.num_classes).scatter_(1, distilled_label.view(-1,1), 1).cuda()  
        noise_matrices_for_x = left_model_aux(images)

        out = torch.einsum("ij,ijk->ik",(distilled_label,noise_matrices_for_x))
        
        out = torch.log(out+1e-20).view(ep.size()[0],1,ep.size()[2]).repeat(1,ep.size()[1],1)
        vec=torch.sum(-out*ep,dim=-1)
        mask=torch.max(ep,dim=2)[0]
        zeromap = torch.zeros((ep.size()[0],ep.size()[1])).cuda()
        vec = torch.where(mask == 0,zeromap,vec)
        loss = torch.sum(vec)/torch.sum(mask)        
        
        #loss = F.nll_loss(torch.log(out+1e-20),ep)
        #torch.nn.utils.clip_grad_norm_(left_model_aux.parameters(), max_norm=1, norm_type=2)
        avg_loss +=loss.item()
        total = batch_idx +1
    
    val_loss = avg_loss/total
    print('val loss', avg_loss/total)
    return val_loss

def true_train_correction(epoch, left_model_aux,val_loader):
    left_model_aux.eval()
    avg_loss= 0
    total =0
    begain = time.time()
    for batch_idx , (left, right, _, true_label, index) in enumerate(val_loader):

        ep = Variable(right).long().cuda()

        images = Variable(left).float().cuda()
        
        distilled_label= true_label.long()
        distilled_label= torch.zeros(len(distilled_label), Config.num_classes).scatter_(1, distilled_label.view(-1,1), 1).cuda()  
        noise_matrices_for_x = left_model_aux(images)

        out = torch.einsum("ij,ijk->ik",(distilled_label,noise_matrices_for_x))
        
        out = torch.log(out+1e-20).view(ep.size()[0],1,ep.size()[2]).repeat(1,ep.size()[1],1)
        vec=torch.sum(-out*ep,dim=-1)
        mask=torch.max(ep,dim=2)[0]
        zeromap = torch.zeros((ep.size()[0],ep.size()[1])).cuda()
        vec = torch.where(mask == 0,zeromap,vec)
        loss = torch.sum(vec)/torch.sum(mask)        
        
        #loss = F.nll_loss(torch.log(out+1e-20),ep)
        #torch.nn.utils.clip_grad_norm_(left_model_aux.parameters(), max_norm=1, norm_type=2)
        avg_loss +=loss.item()
        total = batch_idx +1
    
    true_train_loss = avg_loss/total
    print('true_train_loss', avg_loss/total)
    return true_train_loss
    
def save_global_TM(left_model_aux,train_loader):
    left_model_aux.eval()
    begain = time.time()
    global_TM = torch.zeros(len(train_loader.dataset),Config.num_classes,Config.num_classes)
    with torch.no_grad():
        for batch_idx , (left, right, distilled_label, true_label, index) in enumerate(train_loader):

            images_raw = Variable(left).float().cuda()

            noise_matrices_for_x = left_model_aux(images_raw).data.cpu()
            
            global_TM[index]=noise_matrices_for_x
    print('save TM time', time.time()-begain)
    return global_TM  
    
def train_matrix_multi_source_transfer_training(epoch, multi_source_label_model, left_optimizer_multi_source,label_model_multi_source_init,train_loader):
    multi_source_label_model.train()
    label_model_multi_source_init.eval()
    avg_loss= 0
    avg_matrice_loss=0
    total =0
    begain = time.time()
    skip_worker = []
    for worker_id in range(Config.expert_num):
        left, right, distilled_label, true_label = train_loader.dataset.get_worker_data(worker_id)
        if (len(left)<30):
            skip_worker.append(worker_id)
            #print('worker_id',worker_id,'num', len(left))
            continue
        
        ep = Variable(right).float().cuda()

        left_optimizer_multi_source.zero_grad()  
        images = Variable(left).float().cuda()
        
        distilled_label= distilled_label.long()
        distilled_label= torch.zeros(len(distilled_label), Config.num_classes).scatter_(1, distilled_label.view(-1,1), 1).cuda()      
        noise_matrices_for_x = multi_source_label_model(images, worker_id=worker_id)
        out = torch.einsum("ij,ijk->ik",(distilled_label,noise_matrices_for_x))
        
        out = torch.log(out+1e-20)
        vec=torch.sum(-out*ep,dim=-1)
        loss = torch.sum(vec)/len(vec)
        
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(multi_source_label_model.parameters(), max_norm=1, norm_type=2)
        left_optimizer_multi_source.step()
        avg_loss +=loss.item()
        total +=1
    print('skip_worker',skip_worker)
    print('loss', avg_loss/total)
    print('individual NT epoch time', time.time()-begain)
    return skip_worker

def val_correction_multi_source(epoch, left_model_aux,val_loader):
    left_model_aux.eval()
    avg_loss= 0
    total =0
    begain = time.time()
    for batch_idx , (left, right, _, true_label, index) in enumerate(val_loader):

        ep = Variable(right).long().cuda()

        images = Variable(left).float().cuda()
        
        distilled_label= true_label.long()
        distilled_label= torch.zeros(len(distilled_label), Config.num_classes).scatter_(1, distilled_label.view(-1,1), 1).cuda()  
        noise_matrices_for_x = left_model_aux(images)

        out = torch.einsum("ij,irjk->irk",(distilled_label,noise_matrices_for_x))
        
        out = torch.log(out+1e-20)
        vec=torch.sum(-out*ep,dim=-1)
        mask=torch.max(ep,dim=2)[0]
        zeromap = torch.zeros((ep.size()[0],ep.size()[1])).cuda()
        vec = torch.where(mask == 0,zeromap,vec)
        loss = torch.sum(vec)/torch.sum(mask)        
        
        avg_loss +=loss.item()
        total = batch_idx +1
             
    val_loss = avg_loss/total
    print('val loss', avg_loss/total)
    return val_loss

def true_training_correction_multi_source(epoch, left_model_aux,val_loader):
    left_model_aux.eval()
    avg_loss= 0
    total =0
    begain = time.time()
    for batch_idx , (left, right, _, true_label, index) in enumerate(val_loader):

        ep = Variable(right).long().cuda()

        images = Variable(left).float().cuda()
        
        distilled_label= true_label.long()
        distilled_label= torch.zeros(len(distilled_label), Config.num_classes).scatter_(1, distilled_label.view(-1,1), 1).cuda()  
        noise_matrices_for_x = left_model_aux(images)

        out = torch.einsum("ij,irjk->irk",(distilled_label,noise_matrices_for_x))
        
        out = torch.log(out+1e-20)
        vec=torch.sum(-out*ep,dim=-1)
        mask=torch.max(ep,dim=2)[0]
        zeromap = torch.zeros((ep.size()[0],ep.size()[1])).cuda()
        vec = torch.where(mask == 0,zeromap,vec)
        loss = torch.sum(vec)/torch.sum(mask)
        
        avg_loss +=loss.item()
        total = batch_idx +1
             
    true_training_loss = avg_loss/total
    print('true_training_loss', avg_loss/total)
    return true_training_loss

def train_matrix_multi_source(epoch, left_model_aux, net_optimizer,train_loader):
    left_model_aux.train()
    avg_loss= 0
    total =0
    begain = time.time()
    for batch_idx , (left, right, distilled_label, true_label, index) in enumerate(train_loader):

        ep = Variable(right).long().cuda()

        images = Variable(left).float().cuda()

        distilled_label= distilled_label.long()
        distilled_label= torch.zeros(len(distilled_label), Config.num_classes).scatter_(1, distilled_label.view(-1,1), 1).cuda()  
        
        left_model_aux.zero_grad()  
        noise_matrices_for_x = left_model_aux(images,no_grad=args.no_grad) #no_grad=False
        
        out = torch.einsum("ij,irjk->irk",(distilled_label,noise_matrices_for_x))

        out = torch.log(out+1e-20)
        vec=torch.sum(-out*ep,dim=-1)
        mask=torch.max(ep,dim=2)[0]
        zeromap = torch.zeros((ep.size()[0],ep.size()[1])).cuda()
        vec = torch.where(mask == 0,zeromap,vec)
        loss = torch.sum(vec)/torch.sum(mask)
        
        avg_loss +=loss.item()
        total = batch_idx +1

        loss.backward()
        net_optimizer.step()
        
    training_loss = avg_loss/total
    print('training_loss', avg_loss/total)
    return training_loss



def train_one_source(epoch,left_model, global_TM, left_optimizer,train_loader):
    left_model.train()
    avg_loss= 0
    total =0
    begain = time.time()
    for batch_idx , (left, right,distilled_label, true_label, index) in enumerate(train_loader):

        ep = Variable(right).long().cuda()

        images = Variable(left).float().cuda()
        
        left_optimizer.zero_grad()
        cls_out = left_model(images)

        noise_matrices_for_x = global_TM[index].cuda()
        
        out = torch.einsum("ij,ijk->ik",(cls_out,noise_matrices_for_x))

        
        out = torch.log(out+1e-20).view(ep.size()[0],1,ep.size()[2]).repeat(1,ep.size()[1],1)
        vec=torch.sum(-out*ep,dim=-1)
        mask=torch.max(ep,dim=2)[0]
        zeromap = torch.zeros((ep.size()[0],ep.size()[1])).cuda()
        vec = torch.where(mask == 0,zeromap,vec)
        loss = torch.sum(vec)/torch.sum(mask)        
        #loss = F.nll_loss(torch.log(out+1e-20),ep)
       
        
        loss.backward()
        left_optimizer.step()
        avg_loss +=loss.item()
        total = batch_idx +1
    print('loss', avg_loss/total)
    print('Global training epoch time', time.time()-begain)

def save_individual_TM(left_model_aux,train_loader):
    left_model_aux.eval()
    begain = time.time()
    individual_TM = torch.zeros(len(train_loader.dataset),Config.expert_num,Config.num_classes,Config.num_classes)
    with torch.no_grad():
        for batch_idx , (left, right, _, true_label, index) in enumerate(train_loader):

            ep = Variable(right).float().cuda()

            images = Variable(left).float().cuda()

            noise_matrices_for_x = left_model_aux(images).data.cpu()
            
            individual_TM[index]=noise_matrices_for_x
    print('individual TM time', time.time()-begain)
    return individual_TM

def train_one_source_revision(epoch, left_model, label_model, left_optimizer,left_optimizer_aux, train_loader):
    left_model.train()
    label_model.train()
    avg_loss= 0
    total =0
    begain = time.time()
    for batch_idx , (left, right,distilled_label, true_label, index) in enumerate(train_loader):

        ep = Variable(right).long().cuda()

        images = Variable(left).float().cuda()
        
        left_optimizer.zero_grad()
        left_optimizer_aux.zero_grad()
        cls_out = left_model(images)

        noise_matrices_for_x = label_model(images)
        
        out = torch.einsum("ij,ijk->ik",(cls_out,noise_matrices_for_x))

        out = torch.log(out+1e-20).view(ep.size()[0],1,ep.size()[2]).repeat(1,ep.size()[1],1)
        vec=torch.sum(-out*ep,dim=-1)
        mask=torch.max(ep,dim=2)[0]
        zeromap = torch.zeros((ep.size()[0],ep.size()[1])).cuda()
        vec = torch.where(mask == 0,zeromap,vec)
        loss = torch.sum(vec)/torch.sum(mask) 
       
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(label_model.parameters(), max_norm=1, norm_type=2)
        left_optimizer.step()
        left_optimizer_aux.step()
        avg_loss +=loss.item()
        total = batch_idx +1
    print('loss', avg_loss/total)
    print('training epoch time', time.time()-begain)
    
def train_multi_source(epoch, left_model, individual_TM, left_optimizer,train_loader):
    left_model.train()
    avg_loss= 0
    total =0
    begain = time.time() #left, right,distilled_label, true_label, index
    for batch_idx , (left, right, _, true_label, index) in enumerate(train_loader):

        ep = Variable(right).float().cuda()

        images = Variable(left).float().cuda()


        left_optimizer.zero_grad()
        cls_out = left_model(images)
        
        noise_matrices_for_x=individual_TM[index].cuda()
        
        out = torch.einsum("ij,irjk->irk",(cls_out,noise_matrices_for_x))
        out = torch.log(out+1e-20)
        vec=torch.sum(-out*ep,dim=-1)
        mask=torch.max(ep,dim=2)[0]
        zeromap = torch.zeros((ep.size()[0],ep.size()[1])).cuda()
        vec = torch.where(mask == 0,zeromap,vec)
        loss = torch.sum(vec)/torch.sum(mask) 
        
        loss.backward()
        left_optimizer.step()
        avg_loss +=loss.item()
        total = batch_idx +1
    print('loss', avg_loss/total) 
    print('individual training epoch time', time.time()-begain)
    
def train_multi_source_revision( epoch, left_model, label_model, left_optimizer,left_optimizer_aux, train_loader):
    left_model.train()
    label_model.train()
    avg_loss= 0
    total =0
    begain = time.time()
    for batch_idx , (left, right,distilled_label, true_label, index) in enumerate(train_loader):

        ep = Variable(right).long().cuda()

        images = Variable(left).float().cuda()
        
        left_optimizer.zero_grad()
        left_optimizer_aux.zero_grad()
        cls_out = left_model(images)

        noise_matrices_for_x = label_model(images)
        
        out = torch.einsum("ij,irjk->irk",(cls_out,noise_matrices_for_x))

        out = torch.log(out+1e-20)
        vec=torch.sum(-out*ep,dim=-1)
        mask=torch.max(ep,dim=2)[0]
        zeromap = torch.zeros((ep.size()[0],ep.size()[1])).cuda()
        vec = torch.where(mask == 0,zeromap,vec)
        loss = torch.sum(vec)/torch.sum(mask) 
       
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(label_model.parameters(), max_norm=1, norm_type=2)
        left_optimizer.step()
        left_optimizer_aux.step()
        avg_loss +=loss.item()
        total = batch_idx +1
    print('loss', avg_loss/total)
    print('individual training epoch time', time.time()-begain)






def val(epoch,left_model,test_loader) :

    left_model.eval() 
    
    total_correct = 0
    total_sample =0
    for images, ep, mv_labels, true_label, index in test_loader:
        images = Variable(images).float().cuda()
        ep = ep.cuda()
        total_sample += images.size()[0]

        outputs = left_model(images)
        _, predicts = torch.max(outputs.data, 1)
        one_hot_labels=torch.zeros(len(predicts), Config.num_classes).cuda().scatter_(1, predicts.view(-1,1), 1)

        linear_sum = torch.sum(ep, dim=1)
        linear_sum /= torch.sum(linear_sum,1).unsqueeze(1)
        total_correct += torch.sum(one_hot_labels*linear_sum)
    #print(predicts, labels)
    acc = float(total_correct) / float(total_sample)

    return acc
    
def test(epoch,left_model,test_loader) :

    left_model.eval() 
    
    total_correct = 0
    total_sample =0
    for images, ep, index in test_loader:
        images = Variable(images).float().cuda()
        labels = ep.cuda()
        total_sample += images.size()[0]

        outputs = left_model(images)
        _, predicts = torch.max(outputs.data, 1)
        total_correct += torch.sum(predicts == labels)
        
    acc = float(total_correct) / float(total_sample)

    return acc

def bulid_data_model():
    
    data_model = left_neural_net().cuda()

    left_optimizer =  torch.optim.SGD(data_model.parameters(), lr = args.lr,momentum=0.9,weight_decay=1e-4)#torch.optim.Adam(data_model.parameters(), lr = Config.left_learning_rate,weight_decay=1e-4)
    #torch.optim.SGD(data_model.parameters(), lr = args.warmup_lr,momentum=0.9,weight_decay=1e-4)
    
    return data_model, left_optimizer
    
def bulid_warmup_data_model():
    
    warmup_data_model = left_neural_net().cuda()

    left_optimizer = torch.optim.SGD(warmup_data_model.parameters(), lr = args.warmup_lr,momentum=0.9)
    
    return warmup_data_model, left_optimizer
    
def bulid_label_model_one_source():

    label_model = left_neural_aux_net_one_source().cuda()
    
    left_optimizer_aux= torch.optim.SGD(label_model.parameters(), lr = args.TM_lr,momentum=0.9)
    
    return label_model, left_optimizer_aux

def bulid_label_model_multi_source():

    label_model = left_neural_aux_net_multi_source_with_fea_to_NT_layers().cuda()
    
    left_optimizer_aux= torch.optim.SGD(label_model.parameters(), lr = args.TM_lr,momentum=0.9)
    
    return label_model, left_optimizer_aux

def bulid_label_model_multi_source_gcn(emb, adj):

    label_model = left_neural_aux_net_multi_source_with_gcn(emb, adj).cuda()
    
    left_optimizer_aux= torch.optim.SGD(label_model.parameters(), lr = args.TM_lr,momentum=0.9)
    
    return label_model, left_optimizer_aux


    
    return estimator, optimizer_adj, optimizer_l1, optimizer_nuclear, label_model, left_optimizer_aux

def truncatedSVD(data, k=10):
        """Truncated SVD on input data.

        Parameters
        ----------
        data :
            input matrix to be decomposed
        k : int
            number of singular values and vectors to compute.

        Returns
        -------
        numpy.array
            reconstructed matrix.
        """
        print('=== GCN-SVD: rank={} ==='.format(k))
        if sp.issparse(data):
            data = data.asfptype()
            U, S, V = sp.linalg.svds(data, k=k)
            print("rank_after = {}".format(len(S.nonzero()[0])))
            diag_S = np.diag(S)
        else:
            U, S, V = np.linalg.svd(data)
            U = U[:, :k]
            S = S[:k]
            V = V[:k, :]
            print("rank_before = {}".format(len(S.nonzero()[0])))
            diag_S = np.diag(S)
            print("rank_after = {}".format(len(diag_S.nonzero()[0])))

        return np.clip(U @ diag_S @ V,0,1)

def train_one_source_tuning(epoch, left_model, label_model, left_optimizer,left_optimizer_aux, train_loader):
    left_model.train()
    label_model.train()
    avg_loss= 0
    total =0
    begain = time.time()
    for batch_idx , (left, right,distilled_label, true_label, index) in enumerate(train_loader):

        ep = Variable(right).long().cuda()

        images = Variable(left).float().cuda()
        
        left_optimizer.zero_grad()
        left_optimizer_aux.zero_grad()
        cls_out = left_model(images)

        noise_matrices_for_x = label_model(images)
        
        out = torch.einsum("ij,ijk->ik",(cls_out,noise_matrices_for_x))

        out = torch.log(out+1e-20).view(ep.size()[0],1,ep.size()[2]).repeat(1,ep.size()[1],1)
        vec=torch.sum(-out*ep,dim=-1)
        mask=torch.max(ep,dim=2)[0]
        zeromap = torch.zeros((ep.size()[0],ep.size()[1])).cuda()
        vec = torch.where(mask == 0,zeromap,vec)
        loss = torch.sum(vec)/torch.sum(mask) 
       
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(label_model.parameters(), max_norm=1, norm_type=2)
        left_optimizer.step()
        left_optimizer_aux.step()
        avg_loss +=loss.item()
        total = batch_idx +1
    print('loss', avg_loss/total)
    print('training epoch time', time.time()-begain)

def train_multi_source_revision_tuning( epoch, left_model, label_model, left_optimizer,left_optimizer_aux, train_loader):
    left_model.train()
    label_model.train()
    avg_loss= 0
    total =0
    begain = time.time()
    for batch_idx , (left, right,distilled_label, true_label, index) in enumerate(train_loader):

        ep = Variable(right).long().cuda()

        images = Variable(left).float().cuda()
        
        left_optimizer.zero_grad()
        left_optimizer_aux.zero_grad()
        cls_out = left_model(images)

        noise_matrices_for_x = label_model(images,no_grad=False)
        
        out = torch.einsum("ij,irjk->irk",(cls_out,noise_matrices_for_x))

        out = torch.log(out+1e-20)
        vec=torch.sum(-out*ep,dim=-1)
        mask=torch.max(ep,dim=2)[0]
        zeromap = torch.zeros((ep.size()[0],ep.size()[1])).cuda()
        vec = torch.where(mask == 0,zeromap,vec)
        loss = torch.sum(vec)/torch.sum(mask) 
       
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(label_model.parameters(), max_norm=1, norm_type=2)
        left_optimizer.step()
        left_optimizer_aux.step()
        avg_loss +=loss.item()
        total = batch_idx +1
    print('loss', avg_loss/total)
    print('individual training epoch time', time.time()-begain)
    
if __name__ == '__main__':
    test_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1904, ),(0.3475, )),])
                                    
    train_transform = test_transform
    
    train_dataset = fashionmnist_dataset( root=Config.data_root,  train=True, transform=train_transform, noise_rate=Config.noise_rate, num_worker=Config.expert_num,expert_type_num=Config.expert_type_num, seed=args.seed,redun=args.redun)
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = Config.batch_size, shuffle = True)
    train_loader_norandom = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = Config.batch_size, shuffle = False, drop_last=False)

    val_dataset = fashionmnist_dataset( root=Config.data_root,  train=False, transform=test_transform, noise_rate=Config.noise_rate, num_worker=Config.expert_num,expert_type_num=Config.expert_type_num, seed=args.seed,redun=args.redun)
    val_loader = torch.utils.data.DataLoader(dataset = val_dataset, batch_size = Config.batch_size, shuffle = False, drop_last=False)
    
    test_dataset = fashionmnist_test_dataset(  root=Config.data_root, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = Config.batch_size, shuffle = False)
    
    for x in range(10):
        best_acc = 0
        best_test_acc = 0
        data_model, left_optimizer = bulid_warmup_data_model()
        train_loader_warmup = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = args.batch_size, shuffle = True)
        history_model=[]
        for epoch in range(args.warmup_epoch):
            print('warmup train epoch = %d' % epoch)
            warmup_train(epoch,data_model,left_optimizer,train_loader)
            acc= test(epoch,data_model,test_loader)
            val_acc= val(epoch,data_model,val_loader)
            if(val_acc>best_acc):
                best_acc = max(best_acc,val_acc)
                best_test_acc = acc
                best_state_dict=copy.deepcopy(data_model.state_dict())
            #if(epoch>0):
                #history_model.append(copy.deepcopy(data_model.state_dict()))
            print('train epoch =', epoch,"warmup test Acc:", acc,"val Acc:", val_acc,'best_test_acc',best_test_acc)
            log_file.flush()
        data_model.load_state_dict(best_state_dict)
        history_model.append(copy.deepcopy(data_model.state_dict()))
        print('--------------------')
        
        train_dataset_select = fashionmnist_dataset( root=Config.data_root,  train=True, transform=train_transform, noise_rate=Config.noise_rate, num_worker=Config.expert_num,expert_type_num=Config.expert_type_num, seed=args.seed,redun=args.redun)
        train_loader_select = torch.utils.data.DataLoader(dataset = train_dataset_select, batch_size = args.batch_size, shuffle = True, drop_last=False)  
        train_loader_select, class_num = update_distilled_labels(history_model,train_loader_select)
        if(min(class_num)>5):
            break

    
    history_label_model = []
    label_model, left_optimizer_aux = bulid_label_model_one_source()
    for epoch in range(args.TM_epoch):
        print('Global matrix train epoch = %d' % epoch)
        log_file.flush()
        train_transition_matrix(epoch, label_model,  left_optimizer_aux,train_loader_select)
        val_loss=val_correction(epoch, label_model, val_loader)
        true_train_correction(epoch, label_model,train_loader)
    history_label_model.append(copy.deepcopy(label_model.state_dict()))
    print('--------------------')

#-------------------------------------

    #global_TM=save_global_TM(label_model,train_loader_norandom)
    left_optimizer_aux= torch.optim.SGD(label_model.parameters(), lr = args.TM_lr,momentum=0.9,weight_decay=1e-4)
    best_acc = 0
    best_test_acc =0 
    data_model, left_optimizer = bulid_data_model()
    data_model.load_state_dict(best_state_dict)     
    for epoch in range(args.epoch):
        log_file.flush()            
        adjust_learning_rate(left_optimizer, epoch)
        adjust_learning_rate(left_optimizer_aux, epoch) 
        #train_one_source(epoch, data_model, global_TM, left_optimizer,train_loader)
        train_one_source_tuning(epoch, data_model, label_model, left_optimizer,left_optimizer_aux, train_loader)
        acc= test(epoch,data_model,test_loader)
        val_acc= val(epoch,data_model,val_loader)
        if(val_acc>best_acc):
            best_acc = max(best_acc,val_acc)
            best_test_acc = acc
            history_label_model=[copy.deepcopy(label_model.state_dict())]
        print('train epoch =', epoch, "lr1e-2 Global Loss Correction test Acc:", acc,"val Acc:", val_acc,'best_test_acc',best_test_acc)
    print('--------------------------')
    #sys.exit()
#-------------------------------------
    label_model.load_state_dict(history_label_model[-1])
    label_model_multi_source, left_optimizer_aux_multi_source = bulid_label_model_multi_source()
    label_model_multi_source.copy_backbone(label_model.backbone_for_NT.state_dict(), label_model.linear_1.state_dict())
    label_model_multi_source.copy_NT_layer(label_model.fea_to_NT_layer.state_dict())
    #label_model_multi_source.copy_bias(copy.deepcopy(label_model.bias.data))
    label_model_multi_source_init = copy.deepcopy(label_model_multi_source)
    val_correction_multi_source(epoch, label_model_multi_source,val_loader)
    
    history_label_model_multi_source = []
    for epoch in range(args.FT_TM_epoch):
        print('Individual matrix train epoch = %d' % epoch)
        log_file.flush()
        skip_worker=train_matrix_multi_source_transfer_training(epoch, label_model_multi_source, left_optimizer_aux_multi_source,label_model_multi_source_init,train_loader_select)
        val_loss=val_correction_multi_source(epoch, label_model_multi_source,val_loader)
        true_training_correction_multi_source(epoch, label_model_multi_source,train_loader_norandom)
    history_label_model_multi_source.append(copy.deepcopy(label_model_multi_source.state_dict()))
    print('--------------------')

    #individual_TM=save_individual_TM(label_model_multi_source,train_loader_norandom)
    left_optimizer_aux= torch.optim.SGD(label_model_multi_source.parameters(), lr = args.TM_lr,momentum=0.9,weight_decay=1e-4)
    best_acc = 0
    best_test_acc = 0
    data_model, left_optimizer = bulid_data_model()
    data_model.load_state_dict(best_state_dict)
    for epoch in range(args.epoch):
        log_file.flush() 
        adjust_learning_rate(left_optimizer, epoch)
        adjust_learning_rate(left_optimizer_aux, epoch)         
        #train_multi_source(epoch, data_model, individual_TM, left_optimizer,train_loader)
        train_multi_source_revision_tuning( epoch, data_model, label_model_multi_source, left_optimizer,left_optimizer_aux, train_loader)
        acc= test(epoch,data_model,test_loader)
        val_acc= val(epoch,data_model,val_loader)
        if(val_acc>best_acc):
            best_acc = max(best_acc,val_acc)
            best_test_acc = acc
        print('train epoch =', epoch,"lr1e-2 Individual FT Loss Correction test Acc:", acc,"val Acc:", val_acc,'best_test_acc',best_test_acc)
    print('--------------------')    
    
    #sys.exit()
#-------------------------------------
    #label_model_multi_source.load_state_dict(history_label_model_multi_source[-1])
    base_vec_1=label_model.fea_to_NT_layer.to_NT.weight.data.view(-1)
    base_vec_2=label_model.fea_to_NT_layer.to_NT.bias.data.view(-1)
    base_vec= torch.cat([base_vec_1,base_vec_2])
    vec=torch.zeros(Config.expert_num,len(base_vec)).cuda()
    for r in range(Config.expert_num):
        vec_1=label_model_multi_source.get_NT_layer(r).to_NT.weight.data.view(-1)
        vec_2=label_model_multi_source.get_NT_layer(r).to_NT.bias.data.view(-1)
        vec_r=torch.cat([vec_1,vec_2])
        vec[r]=(base_vec-vec_r)
    pca=PCA(n_components=2, kernel='linear')
    educed_vec=torch.from_numpy(pca.fit_transform(vec.cpu().data.numpy()))
    one_type_num=int(Config.expert_num/Config.expert_type_num)
    labels = np.array([0]*one_type_num+[1]*one_type_num+[2]*one_type_num)
    label_set=[0,1,2]
    plt.figure(figsize=(8,8))
    for label in label_set:
        indices, = np.where(labels == label)
        xs, ys = educed_vec.cpu().data.numpy()[indices].T
        plt.scatter(xs, ys, label=label, marker=".",)
        plt.axis('off')
    plt.legend(fontsize=24,loc=2)
    plt.savefig("plot_{}.jpg".format(args.experiment_name),dpi=72)
    plt.close()
    
    #np.save('./node_vec_1.npy', vec.data.cpu().numpy())

    x = vec
    batch = torch.zeros(len(x)).long()
    edge_index = knn_graph(x.cuda(), k=args.k, batch=batch.cuda(),loop=False,cosine=True).cpu().numpy()

    A=np.zeros((Config.expert_num,Config.expert_num))
    for i in range(Config.expert_type_num):
        A[one_type_num*i:one_type_num*(i+1),one_type_num*i:one_type_num*(i+1)]=1
    right_edges=[]
    wrong_edges=[]
    for i,j in zip(edge_index[0],edge_index[1]):
        if(A[i,j]==1): 
            right_edges.append((i,j))
        else:
            wrong_edges.append((i,j))
        
    G=nx.DiGraph()
    nodes=np.arange(Config.expert_num).tolist()
    #print(nodes)
    G.add_nodes_from(nodes)
    #print(right_edges+wrong_edges)
    G.add_edges_from(right_edges+wrong_edges)
    B=nx.convert_matrix.to_numpy_array(G, nodelist=nodes)
    precision=sum((B==1)&(A==B))/sum(B==1)
    print('precision_mean',np.mean(precision),'precision_min',min(precision))
    
    print("Graph has %d nodes with %d edges" %(G.number_of_nodes(),  G.number_of_edges()))
    
    print('---------------------------------------------')
    C=truncatedSVD(B,k=args.svd_k)
    print(C)
    D= C>0.1
    precision=sum((D==1)&(A==D))/sum(D==1)
    print('precision_mean',np.mean(precision),'precision_min',min(precision))
    
    C=np.clip(C,0,1)
    skip_worker=torch.Tensor(skip_worker).long()
    C[skip_worker,skip_worker.unsqueeze(-1)]=1
    C[np.eye(Config.expert_num)==1]=1
    B[np.eye(Config.expert_num)==1]=1


#-------------------------------------
    
    label_model_multi_source, left_optimizer_aux_multi_source = bulid_label_model_multi_source_gcn(emb=torch.from_numpy(np.eye(Config.expert_num)).cuda(), adj=torch.from_numpy(C).cuda())
    label_model_multi_source.copy_backbone(label_model.backbone_for_NT.state_dict(), label_model.linear_1.state_dict())
    label_model_multi_source.copy_NT_layer(label_model.fea_to_NT_layer.state_dict())
    val_correction_multi_source(epoch, label_model_multi_source,val_loader)
    
    for epoch in range(args.GCN_TM_epoch):
        print('Individual matrix train epoch = %d' % epoch)
        log_file.flush()
        train_matrix_multi_source(epoch, label_model_multi_source, left_optimizer_aux_multi_source,train_loader_select)
        val_loss=val_correction_multi_source(epoch, label_model_multi_source,val_loader)
        true_training_correction_multi_source(epoch, label_model_multi_source,train_loader_norandom)
    print('--------------------')

    left_optimizer_aux= torch.optim.SGD(label_model_multi_source.parameters(), lr = args.TM_lr,momentum=0.9,weight_decay=1e-4)
    best_acc = 0
    best_test_acc = 0
    data_model, left_optimizer = bulid_data_model()
    data_model.load_state_dict(best_state_dict)
    for epoch in range(args.epoch):
        log_file.flush()     
        adjust_learning_rate(left_optimizer, epoch)
        adjust_learning_rate(left_optimizer_aux, epoch)         
        train_multi_source_revision_tuning( epoch, data_model, label_model_multi_source, left_optimizer,left_optimizer_aux, train_loader)
        acc= test(epoch,data_model,test_loader)
        val_acc= val(epoch,data_model,val_loader)
        if(val_acc>best_acc):
            best_acc = max(best_acc,val_acc)
            best_test_acc = acc
        print('train epoch =', epoch,"lr1e-2 Individual svd gcn Loss Correction test Acc:", acc,"val Acc:", val_acc,'best_test_acc',best_test_acc)
    print('--------------------')