import matplotlib.pyplot as plt
import torch

loss_save_path1 = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/results/loss/loss_alder.pkl'
loss_save_path2 = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/results/loss/loss_eucalyptus.pkl'
loss_save_path3 = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/results/loss/loss_maple.pkl'
loss_save_path4 = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/results/loss/loss_oak.pkl'
loss_save_path5 = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/results/loss/loss_biodeg.pkl'
loss_save_path6 = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/results/loss/loss_ionosphere.pkl'

loss_save_path7 = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/results/loss/loss_vehicle.pkl'
loss_save_path8 = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/results/loss/loss_waveform.pkl'

loss_list_1 = torch.load(loss_save_path1)
loss_list_2 = torch.load(loss_save_path2)
loss_list_3 = torch.load(loss_save_path3)
loss_list_4 = torch.load(loss_save_path4)
loss_list_5 = torch.load(loss_save_path5)
loss_list_6 = torch.load(loss_save_path6)

loss_list_7 = torch.load(loss_save_path7)
loss_list_8 = torch.load(loss_save_path8)

n = len(loss_list_1)
x = [i * 10 for i in range(n)]

plt.plot(x,loss_list_1,color='g',linestyle='-',label='alder')
plt.plot(x,loss_list_2,color='r',linestyle='--',label='eucalyptus')
plt.plot(x,loss_list_3,color='c',linestyle='-.',label='maple')
plt.plot(x,loss_list_4,color='k',linestyle=':',label='oak')
plt.plot(x,loss_list_5,color='b',marker='.',linestyle='-',label='biodeg')
plt.plot(x,loss_list_6,color='m',marker='.',linestyle='--',label='ionosphere')

plt.plot(x,loss_list_7,color='k',marker='p',linestyle='-.',label='vehicle')
plt.plot(x,loss_list_8,color='c',marker='o',linestyle='-',label='waveform')

plt.xlabel('Number of Epoches')
plt.ylabel('CrossEntropy Loss')

plt.legend(loc=2, bbox_to_anchor=(1.1,1.0),borderaxespad = 0.)
plt.show()
