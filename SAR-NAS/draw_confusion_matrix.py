# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 14:59:54 2019

@author: a
"""
# make confusion matrix


from sklearn.metrics import confusion_matrix    # 生成混淆矩阵函数
import matplotlib.pyplot as plt    # 绘图库
import numpy as np
import os

def show_confmat(confusion_mat, classes_name, set_name, out_dir):
    #归一化
    confusion_mat_2 = np.empty((len(classes_name),len(classes_name)))
    for i in range(len(classes_name)):
       confusion_mat_2[i,:]  = confusion_mat[i, :] / np.sum(confusion_mat[i,:])
    #print(confusion_mat_2)
    #confusion_mat_2 = "%.3f"%confusion_mat_2
    #round(confusion_mat_2, 3)
    confusion_mat_2 = np.around(confusion_mat_2,decimals=3,out=None)
    #获取颜色
    cmap = plt.cm.get_cmap('Blues')#plasma viridis Wistia inferno magma
    plt.imshow(confusion_mat_2, cmap=cmap)
    plt.colorbar()
    #设置文字
    #plt.axis([-0.5,11.5,-0.5,11.5])   #sysu
    #plt.axis([-0.5,15,-0.5,12.5])
    xlocations = np.array(range(len(classes_name)))
    #ylocations = np.linspace(-0.5,10.5,num = 12)
    #xlocations = np.array(range(len(classes_name)))
    plt.xticks(xlocations, classes_name, fontsize=2.5, rotation=90)
    plt.yticks(xlocations, classes_name, fontsize=4)
    #plt.xticks(xlocations,fontsize=0.1)
    #plt.xlabel('Predict label')  
    #plt.ylabel('True label')
    plt.title('Confusion_Matrix_' + set_name)
    #打印数字
    for i in range(confusion_mat_2.shape[0]):
        for j in range(confusion_mat_2.shape[1]):
            #if confusion_mat_2[i, j] > 0.03:
            if confusion_mat_2[i, j] != 0:
                 plt.text(x=j, y=i, s=confusion_mat_2[i, j], va='center', ha='center', color='black', fontsize=1.5)

    #保存
   # plt.gca().set_xticks(xlocations+0.5, minor=True)  
    #plt.show()
    #plt.savefig(os.path.join(out_dir, 'Confusion_Matrix_' + set_name + '.png'), dpi=300, bbox_inches = 'tight')
    plt.savefig(os.path.join(out_dir, 'Confusion_Matrix_' + set_name + '.pdf'), dpi=300, bbox_inches = 'tight')
    plt.close()
       
#cm_2 = cm_1/20
#                                SYSU          CLASSES
classes_name_sysu = ['drinking', 'pouring', 'calling phone', 'playing phone', 'wearing backpacks',
                'packing backpacks','sitting chair', 'moving chair', 'taking out wallet', 
                'taking from wallet', 'mopping', 'sweeping']



#                               UTD         CLASSES           
 
classes_name_utd = ['swipe_left', 'swipe_right', 'wave', 'clap', 'throw', 'arm_cross', 
                'basketball_shoo', 'draw_x', 'draw_circle_CW', 'draw_circle_CCW', 'draw_triangle', 'bowling','boxing', 'baseball_swing', 
                'tennis_swing', 'arm_curl', 'tennis_serve', 'push', 'knock', 'catch', 'pickup_throw', 'jog',
                'walk', 'sit2stand', 'stand2sit', 'lunge', 'squat']


#                                NTU          CLASSES
classes_name_ntu = ['drink water', 'eat meal', 'brush teeth', 'brush hair', 'drop', 'pick up', 
                    'throw', 'sit down', 'stand up', 'clapping', 'reading', 'writing', 
                    'tear up paper', 'put on jacket', 'take off jacket', 'put on a shoe', 'take off a shoe', 'put on glasses',
                    'take off glasses', 'put on a hat/cap', 'take off a hat/cap', 'cheer up', 'hand waving', 'kicking something',
                    'reach into pocket', 'hopping ', 'jump up', 'phone call', 'play with phone/tablet', 'type on a keyboard ', 
                    'point to something ', 'taking a selfie', 'check time (from watch)', 'rub two hands', 'nod head/bow ', 'shake head', 
                    'wipe face ', 'salute', 'put palms together ', 'cross hands in front ', 'sneeze/cough', 'staggering', 
                    'falling down', 'headache', 'chest pain', 'back pain', 'neck pain', 'nausea/vomiting ',
                    'fan self', 'punch/slap', 'kicking', 'pushing', 'pat on back', 'point finger',
                    'hugging', 'giving object', 'touch pocket', 'shaking hands', 'walking towards', 'walking apart', ]


test_label_path='/media/lab540/79eff75a-f78c-42f2-8902-9358e88bf654/lab540/Neura_auto_search/datasets/ntu112/ntu_cv/test.txt'
predict = np.load("/home/lab540/PycharmProjects/darts_skeleton/predicted_labels.npy")
save_dir = "/home/lab540/PycharmProjects/darts_skeleton"
#predict = predict.astype('int64')
#pred_list = np.empty((240))
#labels = np.empty((240, 1), dtype = "int")
Label = []
Pred = []
f = open(test_label_path)
lines = f.readlines()
for line in lines:
       tmp = [val for val in line.strip().split(' ')]
       label=int(tmp[1])
       Label.append(label)
# x_max = np.max(predict, axis=1)
# x_max_1 = np.expand_dims(x_max, axis=1)
# x_max_2 = np.repeat(x_max_1, len(classes_name_ntu), axis=1)
# x_max_index = np.where(predict == x_max_2)
pred_list = predict
for ii in pred_list:
    #print(ii)
    Pred.append(ii)
cm_ntu = confusion_matrix(Label, Pred)
cm_ntu_normalized = cm_ntu / cm_ntu.sum(axis=1)[:, np.newaxis]

show_confmat(cm_ntu_normalized, classes_name_ntu, 'NTU_CS1', save_dir)
#########################################UTD##################################################
# test_label_path='/media/ubuntu/DISK_IMG/UTD/test.txt'
# predict = np.load("/media/ubuntu/DISK_IMG/UTD/utd.npy")
# save_dir = "/media/ubuntu/DISK_IMG/UTD/"
# #predict = predict.astype('int64')
# #pred_list = np.empty((240))
# #labels = np.empty((240, 1), dtype = "int")
# Label = [ ]
# Pred = [ ]
# f = open(test_label_path)
# lines=f.readlines()
# for line in lines:
#        tmp = [val for val in line.strip().split(' ')]
#        label=int(tmp[1])
#        Label.append(label)
# x_max = np.max(predict, axis = 1)
# x_max_1 = np.expand_dims(x_max,axis=1)
# x_max_2 = np.repeat(x_max_1, len(classes_name_utd), axis=1)
# x_max_index = np.where(predict==x_max_2)
# pred_list = x_max_index[1]
# for ii in pred_list:
#     #print(ii)
#     Pred.append(ii)
# cm_utd = confusion_matrix(Label, Pred)
#
#
# show_confmat(cm_utd, classes_name_utd, 'UTD', save_dir)

#########################################SYSU##################################################
# =============================================================================
# test_label_path='/media/ubuntu/DISK_IMG/SYSU/test.txt'
# predict = np.load("/media/ubuntu/DISK_IMG/SYSU/sysu.npy")
# save_dir = "/media/ubuntu/DISK_IMG/SYSU/"
# #predict = predict.astype('int64')
# #pred_list = np.empty((240))
# #labels = np.empty((240, 1), dtype = "int")
# Label = [ ]
# Pred = [ ]
# f = open(test_label_path)
# lines=f.readlines()
# for line in lines:
#        tmp = [val for val in line.strip().split(' ')]
#        label=int(tmp[1])
#        Label.append(label)
# x_max = np.max(predict, axis = 1)
# x_max_1 = np.expand_dims(x_max,axis=1)
# x_max_2 = np.repeat(x_max_1, len(classes_name_sysu), axis=1)
# x_max_index = np.where(predict==x_max_2)
# pred_list = x_max_index[1]
# for ii in pred_list:
#     #print(ii)
#     Pred.append(ii)
# cm_sysu = confusion_matrix(Label, Pred)
# 
# 
# show_confmat(cm_sysu, classes_name_sysu, 'SYSU', save_dir)   
# =============================================================================


#########################################NTU_CS##################################################
# =============================================================================
# test_label_path='/media/ubuntu/new1/xry/my/data/label/ntu_cs/test_label.txt'
# predict = np.load("/media/ubuntu/new1/xry/my/g_weights/ntu_cs.npy")
# save_dir = "/media/ubuntu/DISK_IMG/NTU/"
# #predict = predict.astype('int64')
# #pred_list = np.empty((240))
# #labels = np.empty((240, 1), dtype = "int")
# Label = [ ]
# Pred = [ ]
# f = open(test_label_path)
# lines=f.readlines()
# for line in lines:
#        tmp = [val for val in line.strip().split(' ')]
#        label=int(tmp[1])
#        Label.append(label)
# x_max = np.max(predict, axis = 1)
# x_max_1 = np.expand_dims(x_max,axis=1)
# x_max_2 = np.repeat(x_max_1, len(classes_name_ntu), axis=1)
# x_max_index = np.where(predict==x_max_2)
# pred_list = x_max_index[1]
# for ii in pred_list:
#     #print(ii)
#     Pred.append(ii)
# cm_ntu = confusion_matrix(Label, Pred)
# 
# 
# show_confmat(cm_ntu, classes_name_ntu, 'NTU_CS', save_dir)     
# =============================================================================

#########################################NTU_CV##################################################
# =============================================================================

# =============================================================================
# model.eval()
#     test_samples = vid_seq_test.__len__()
#     print('Number of samples = {}'.format(test_samples))
#     print('Evaluating...')
#     Num_Corr = 0
#     true_labels = []
#     predicted_labels = []
#     with torch.no_grad():
#         for j, (inputs, targets) in enumerate(test_loader):
#             inputs, targets = inputs.to(device), targets.to(device)
#             output_label = model(inputs)
#             _, predicted = torch.max(output_label.data, 1)
#             Num_Corr += (predicted == targets).sum()
#             true_labels.append(targets)
#             predicted_labels.append(predicted)
#         test_accuracy = (Num_Corr.item() / test_samples) * 100
#         print('Test Accuracy = {}%'.format(test_accuracy))
#         with open(os.path.abspath(os.path.dirname(args.Model_path)+os.path.sep+".") + '/test_accuracy.txt', 'w') as f:
#             f.write('Test_Accuracy: {}%\n'.format(test_accuracy))
#         ticks = np.linspace(0, num_classes-1, num=num_classes)
#         '''Plot Confusion_Matrix'''
#         cnf_matrix = confusion_matrix(true_labels, predicted_labels).astype(float)
#         cnf_matrix_normalized = cnf_matrix / cnf_matrix.sum(axis=1)[:, np.newaxis]
#         plt.imshow(cnf_matrix_normalized, interpolation='none', cmap='Blues')
#         plt.colorbar()
#         plt.xticks(ticks, fontsize=3)
#         plt.yticks(ticks, fontsize=3)
#         plt.grid(True)
#         plt.clim(0, 1)
#         plt.savefig(args.Model_path + '-rgb.pdf', bbox_inches='tight', cmap='Blues')
#         plt.show()
# ########################################################################################################################
