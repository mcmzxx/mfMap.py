#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 08:41:45 2021

@author: zxx
"""

import torch
import numpy as np
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from earlystoping import Earlystopping
from sklearn import metrics
import tensorflow.compat.v1 as tf
import os
from lr_scheduler import ReduceLROnPlateau
os.chdir('/storage/zhang/mfMap.py')
import re

tf.app.flags.DEFINE_string('f', '', 'kernel')
flags=tf.app.flags
FLAGS=flags.FLAGS
flags.DEFINE_bool('parallel', True, 'Parallelism of model')
flags.DEFINE_bool('output_loss_record', True, 'Early stopping')
flags.DEFINE_bool('early_stopping', False, 'Early stopping')
flags.DEFINE_integer('random_seed', 42, 'Random seed for torch.')
flags.DEFINE_integer('batch_size', 32, 'Random seed for torch.')
flags.DEFINE_integer('latent_space_dim', 2, 'Number of dimensions in latent space dim.')
flags.DEFINE_float('learning_rate', 0.001152512, 'Initial learning rate.')
flags.DEFINE_integer('p1_epoch_num', 2, 'Number of epochs to train.')
flags.DEFINE_integer('p2_epoch_num', 3000, 'Number of epochs to train.')
flags.DEFINE_integer('level_2_dim_cnv', 1024, 'Number of dimensions in level 2.')
flags.DEFINE_integer('level_3_dim_cnv', 512, 'Number of dimensions in level 3.')
flags.DEFINE_integer('level_2_dim_expr', 1024, 'Number of dimensions in level 2.')
flags.DEFINE_integer('level_3_dim_expr', 512, 'Number of dimensions in level 3.')
flags.DEFINE_integer('level_4_dim', 256, 'Number of dimensions in level 4.')
flags.DEFINE_integer('classifier_1_dim', 128, 'Number of dimensions in classifier level 1.')
flags.DEFINE_integer('classifier_2_dim', 64, 'Number of dimensions in classifier level 2.')
flags.DEFINE_string('input_path', 'data',
                    'Data location of input.')
flags.DEFINE_string('organ', 'BRCA','organ')
flags.DEFINE_string('input1_fn', 'features_exp.txt', 'feature file name')
flags.DEFINE_string('input2_fn', 'features_mut_cnv_comb.txt', 'feature file name')
flags.DEFINE_string('label_fn', 'dataset_labels.txt', 'label file name') 
flags.DEFINE_integer('patience', 50, 'early stopping patience') 
flags.DEFINE_float('beta', 0.9, 'beta value')
flags.DEFINE_string('separate_testing', 'yes', 'Separate testing')
flags.DEFINE_string('use_cell','yes', 'if use cell line data')
FLAGS.input_path=os.path.join(FLAGS.input_path,FLAGS.organ)
torch.manual_seed(FLAGS.random_seed)
torch.cuda.manual_seed_all(FLAGS.random_seed)

#%% prepare input
def prepare_input_tum_cell(data_location,label_fn,input1_fn,input2_fn):
    fn=os.path.join(data_location,input1_fn)
    cnv_df_bak= pd.read_table(fn, index_col=0)
    fn=os.path.join(data_location,input2_fn)
    expr_df_bak= pd.read_table(fn, index_col=0)
    fn=os.path.join(data_location,label_fn)
    label_bak= pd.read_table(fn, index_col=None)
    cnv_df_bak=cnv_df_bak.clip(1e-3,0.999)
    expr_df_bak=expr_df_bak.clip(1e-3,0.99)
    ddsels=label_bak.barcode[[(label_bak['subtype'][x]=='NOLBL') and (label_bak['type'][x]=='tumor') for x in label_bak.index]].tolist()
    sels=label_bak.barcode[[(label_bak['subtype'][x]!='NOLBL') and (label_bak['type'][x]=='tumor') for x in label_bak.index]].tolist()
    dsels=label_bak.barcode[[(label_bak['subtype'][x]=='NOLBL') and (label_bak['type'][x]=='cell') for x in label_bak.index]].tolist()
    if FLAGS.organ=='COADREAD':
        cmsmapping={'CMS1': 0, 'CMS2': 1, 'CMS3': 2, 'CMS4': 3,'NOLBL':4}
    if FLAGS.organ=='BRCA':
        cmsmapping={"Luminal B": 0, "Luminal A": 1, "NOLBL": 4, "HER2-enriched": 2, "Basal-like": 3}
    if FLAGS.organ=='GBMLGG':
        cmsmapping={"G-CIMP-high": 0, "Codel": 1, "Mesenchymal-like": 2, "NOLBL": 7, "Classic-like": 3, 
"G-CIMP-low": 4, "PA-like": 5, "LGm6-GBM": 6}
    if FLAGS.organ=="ESCA":
        cmsmapping={"NOLBL":2, "ESCC":0, "AC":1}
    if FLAGS.organ=="HNSC":
        cmsmapping={"NOLBL":4, "Atypical":0, "Mesenchymal":1, "Basal":2, "Classical":3}
    if FLAGS.organ=="LUAD":
        cmsmapping={"NOLBL":3, "Terminal_respiratory_unit":0,"Proximal_inflammatory":1, "Proximal_proliferative":2}
    if FLAGS.organ=="LUSC":
        cmsmapping={"NOLBL":4, "basal":0, "secretory":1, "primitive":2, "classical":3}
    if FLAGS.organ=="PAAD":
        cmsmapping={"NOLBL":2, "Classical":0, "Basal":1}
    if FLAGS.organ=="SKCM":
        cmsmapping={"NOLBL":3, "keratin":0, "immune":1, "MITF_low":2}
    if FLAGS.organ=="UCEC":
        cmsmapping={"NOLBL":3, "Mitotic":0, "Immunoreactive":1, "Hormonal":2}
    label_bak['subtype']=[cmsmapping[i] for i in label_bak.subtype]
    dcnv_df=cnv_df_bak[dsels+ddsels]
    dexpr_df=expr_df_bak[dsels+ddsels]  
    dlabel=label_bak.loc[dsels+ddsels,] 
    cnv_df_bak=cnv_df_bak.drop(ddsels,axis=1)
    expr_df_bak=expr_df_bak.loc[:,list(cnv_df_bak.columns)]
    label_bak=label_bak.loc[list(cnv_df_bak.columns),:] 
    cnv_df=cnv_df_bak[sels]
    expr_df=expr_df_bak[sels]
    label=label_bak.loc[sels,]
  
    if FLAGS.separate_testing=='yes':
        valset_ratio=0.09
        if FLAGS.use_cell=='yes':
            train_index, val_index, train_label, val_label=train_test_split(label_bak['barcode'].values, label_bak['subtype'].values,                                                                                                           test_size=valset_ratio,                                                                                                                random_state=FLAGS.random_seed,                                                                                                              stratify=label_bak['subtype'].values)

            cnv_df_val=cnv_df_bak[val_index]
            cnv_df_train=cnv_df_bak[train_index]
            expr_df_val=expr_df_bak[val_index]
            expr_df_train=expr_df_bak[train_index]
            train_dataset=MultiOmiDataset(cnv_df=cnv_df_train,expr_df=expr_df_train, labels=train_label)
            train_loader=DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=6)
            val_dataset=MultiOmiDataset(cnv_df=cnv_df_val, expr_df=expr_df_val, labels=val_label)
            val_loader=DataLoader(val_dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=6)
            nolbl_dataset=MultiOmiDataset(cnv_df=dcnv_df, expr_df=dexpr_df,labels=dlabel['subtype'].values)
            nolbl_loader=DataLoader(nolbl_dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=6)
        else:
            train_index, val_index, train_label, val_label=train_test_split(label['barcode'].values, label['subtype'].values,                                                                                                           test_size=valset_ratio,                                                                                                                random_state=FLAGS.random_seed,                                                                                                              stratify=label['subtype'].values)

            cnv_df_val=cnv_df[val_index]
            cnv_df_train=cnv_df[train_index]
            expr_df_val=expr_df[val_index]
            expr_df_train=expr_df[train_index]
            train_dataset=MultiOmiDataset(cnv_df=cnv_df_train,expr_df=expr_df_train, labels=train_label)
            train_loader=DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=6)
            val_dataset=MultiOmiDataset(cnv_df=cnv_df_val, expr_df=expr_df_val, labels=val_label)
            val_loader=DataLoader(val_dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=6)
            nolbl_dataset=MultiOmiDataset(cnv_df=dcnv_df, expr_df=dexpr_df,labels=dlabel['subtype'].values)
            nolbl_loader=DataLoader(nolbl_dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=6)
    else:
        train_dataset=MultiOmiDataset(cnv_df=cnv_df_bak, expr_df=expr_df_bak,labels=label_bak['subtype'].values)
        train_loader=DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=6)
        nolbl_dataset=MultiOmiDataset(cnv_df=dcnv_df, expr_df=dexpr_df,labels=dlabel['subtype'].values)
        nolbl_loader=DataLoader(nolbl_dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=6)

    full_dataset=MultiOmiDataset(cnv_df=cnv_df_bak, expr_df=expr_df_bak,labels=label_bak['subtype'].values)
    full_loader=DataLoader(full_dataset, batch_size=FLAGS.batch_size, num_workers=6)
    if FLAGS.separate_testing=='yes':
        return nolbl_dataset, nolbl_loader, full_dataset, full_loader, train_dataset, train_loader, val_dataset, val_loader, cnv_df_bak, expr_df_bak, label_bak
    else:
        return nolbl_dataset, nolbl_loader, full_dataset, full_loader, train_dataset, train_loader, cnv_df_bak, expr_df_bak, label_bak
    
class MultiOmiDataset():
    def __init__(self, cnv_df,expr_df, labels):
        self.cnv_df=cnv_df
        self.expr_df=expr_df
        self.labels=labels

    def __len__(self):
        return self.expr_df.shape[1]

    def __getitem__(self, index):
        omics_data=[]
        cnv_line=self.cnv_df.iloc[:, index].values
        cnv_line_tensor=torch.Tensor(cnv_line)
        omics_data.append(cnv_line_tensor)
        expr_line=self.expr_df.iloc[:, index].values
        expr_line_tensor=torch.Tensor(expr_line)
        omics_data.append(expr_line_tensor)       
        label=self.labels[index]
        return [omics_data, label]
#%% fully-connected layer block
def fc_layer(in_dim, out_dim, activation=1, dropout=False, dropout_p=0.5):
    if activation==0:
        layer=nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim))
    elif activation==2:
        layer=nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.Sigmoid())
    else:
        if dropout:
            layer=nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.BatchNorm1d(out_dim),
                    nn.ReLU(),
                    nn.Dropout(p=dropout_p))
        else:
            layer=nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.BatchNorm1d(out_dim),
                    nn.ReLU())
    return layer
#%%
class mfMAP(nn.Module):
    def __init__(self,input_dim_cnv, input_dim_expr,classifier_out_dim):
        super(mfMAP, self).__init__()
        self.e_fc1_cnv=fc_layer(input_dim_cnv, FLAGS.level_2_dim_cnv)
        self.e_fc1_expr=fc_layer(input_dim_expr, FLAGS.level_2_dim_expr)
        self.e_fc2_cnv=fc_layer(FLAGS.level_2_dim_cnv, FLAGS.level_3_dim_cnv)
        self.e_fc2_expr=fc_layer(FLAGS.level_2_dim_expr, FLAGS.level_3_dim_expr)
        self.e_fc3=fc_layer(FLAGS.level_3_dim_cnv+FLAGS.level_3_dim_expr, FLAGS.level_4_dim)
        self.e_fc4_mean=fc_layer(FLAGS.level_4_dim, FLAGS.latent_space_dim, activation=0)
        self.e_fc4_log_var=fc_layer(FLAGS.level_4_dim, FLAGS.latent_space_dim, activation=0)
        self.d_fc4=fc_layer(FLAGS.latent_space_dim, FLAGS.level_4_dim)
        self.d_fc3=fc_layer(FLAGS.level_4_dim, FLAGS.level_3_dim_cnv+FLAGS.level_3_dim_expr)
        self.d_fc2_cnv=fc_layer(FLAGS.level_3_dim_cnv, FLAGS.level_2_dim_cnv)
        self.d_fc2_expr=fc_layer(FLAGS.level_3_dim_expr, FLAGS.level_2_dim_expr)
        self.d_fc1_cnv= fc_layer(FLAGS.level_2_dim_cnv, input_dim_cnv, activation=2)
        self.d_fc1_expr=fc_layer(FLAGS.level_2_dim_expr, input_dim_expr, activation=2)
        self.c_fc1=fc_layer(FLAGS.latent_space_dim, FLAGS.classifier_1_dim)
        self.c_fc2=fc_layer(FLAGS.classifier_1_dim, FLAGS.classifier_2_dim)
        self.c_fc3=fc_layer(FLAGS.classifier_2_dim, classifier_out_dim, activation=0)
            
    def encode(self, x):        
        cnv_level2_layer=self.e_fc1_cnv(x[0])
        expr_level2_layer=self.e_fc1_expr(x[1])
        cnv_level3_layer=self.e_fc2_cnv(cnv_level2_layer)
        expr_level3_layer=self.e_fc2_expr(expr_level2_layer)
        level_3_layer=torch.cat((cnv_level3_layer, expr_level3_layer), 1)
        level_4_layer=self.e_fc3(level_3_layer)
        latent_mean=self.e_fc4_mean(level_4_layer)
        latent_log_var=self.e_fc4_log_var(level_4_layer)
        return latent_mean, latent_log_var

    def reparameterize(self, mean, log_var):
        sigma=torch.exp(0.5 * log_var)
        eps=torch.randn_like(sigma)
        return mean+eps * sigma

    def decode(self, z):
        level_4_layer=self.d_fc4(z)
        level_3_layer=self.d_fc3(level_4_layer)
        cnv_level3_layer=level_3_layer.narrow(1, 0, FLAGS.level_3_dim_cnv)
        expr_level3_layer=level_3_layer.narrow(1, FLAGS.level_3_dim_cnv, FLAGS.level_3_dim_expr)
        cnv_level2_layer=self.d_fc2_cnv(cnv_level3_layer)
        expr_level2_layer=self.d_fc2_expr(expr_level3_layer)
        recon_x0=self.d_fc1_cnv(cnv_level2_layer)
        recon_x1=self.d_fc1_expr(expr_level2_layer)
        return [recon_x0, recon_x1]
    
    def classifier(self, mean):
        level_1_layer=self.c_fc1(mean)
        level_2_layer=self.c_fc2(level_1_layer)
        output_layer=self.c_fc3(level_2_layer)
        return output_layer

    def forward(self, x,y):
        mean, log_var=self.encode(x)
        z=self.reparameterize(mean, log_var)
        #tmpidx=[y_ for y_ in range(len(y)) if y[y_]!=4]
        classifier_x=mean#[tmpidx]
        recon_x=self.decode(z)
        pred_y=self.classifier(classifier_x)
        return z, recon_x, mean, log_var, pred_y
#%%
def cnv_recon_loss(recon_x, x):
    loss=F.binary_cross_entropy(recon_x[0], x[0], reduction='sum')
    return loss

def expr_recon_loss(recon_x, x):
    loss=F.binary_cross_entropy(recon_x[1], x[1], reduction='sum')
    return loss

def classifier_er(pred_y, y,beta=0.5):
    bootstrap=- (1.0 - beta) * torch.sum(F.softmax(pred_y, dim=1) * F.log_softmax(pred_y, dim=1), dim=1)
    return torch.sum(bootstrap) 


def kl_loss(mean, log_var):
    loss=-0.5 * torch.sum(1+log_var - mean.pow(2) - log_var.exp())
    return loss

def classifier_loss1(pred_y, y):
    loss=F.cross_entropy(pred_y, y, reduction='sum')
    return loss

def classifier_loss(pred_y, y,tmpidx,beta=0.5):
    return beta*F.cross_entropy(pred_y[tmpidx], y[tmpidx], reduction='sum')
#%%
def classifier_sb_loss(pred_y, y,tmpidx,beta=0.5):
    # cross_entropy=- t * log(p)
    utmpidx=[y_ for y_ in range(len(y)) if y[y_] not in tmpidx]
    #if (len(utmpidx)>0):
    _, z=torch.max(F.softmax(pred_y[utmpidx], dim=1), dim=1)
    z=z.view(-1, 1)
    bootstrap=F.log_softmax(pred_y[utmpidx], dim=1).gather(1, z).view(-1)
    # second term=(1 - beta) * z * log(p)
    bootstrap=- (1.0 - beta) * bootstrap
    loss=beta*F.cross_entropy(pred_y[tmpidx], y[tmpidx], reduction='sum')+torch.sum(bootstrap)
    return loss
    #else:
        #return beta*F.cross_entropy(pred_y[tmpidx], y[tmpidx], reduction='sum')

def classifier_hb_loss(pred_y, y,tmpidx,beta=0.5):
    bootstrap=- (1.0 - beta) * torch.sum(F.softmax(pred_y, dim=1) * F.log_softmax(pred_y, dim=1), dim=1)
    return beta*F.cross_entropy(pred_y[tmpidx], y[tmpidx], reduction='sum')+torch.sum(bootstrap) 

def run_train():
    if FLAGS.separate_testing=='yes':
        nolbl_dataset, nolbl_loader, full_dataset, full_loader, train_dataset, train_loader, val_dataset, val_loader, cnv_df_bak, expr_df_bak, label_bak=prepare_input_tum_cell(FLAGS.input_path,FLAGS.label_fn,FLAGS.input1_fn,FLAGS.input2_fn)
    else:
        nolbl_dataset, nolbl_loader, full_dataset, full_loader, train_dataset, train_loader, cnv_df_bak, expr_df_bak, label_bak=prepare_input_tum_cell(FLAGS.input_path,FLAGS.label_fn,FLAGS.input1_fn,FLAGS.input2_fn)

    input_dim_cnv=cnv_df_bak.shape[0]
    input_dim_expr=expr_df_bak.shape[0]
    iuidx=max(label_bak['subtype'].values)
    classifier_out_dim=len(label_bak.subtype.unique())-1
    device='cpu'
    FLAGS.parallel=torch.cuda.device_count() > 1 and FLAGS.parallel
    input_path_name=[FLAGS.input1_fn,FLAGS.input2_fn,'wait',str(FLAGS.patience),'st',str(FLAGS.separate_testing),'uc',str(FLAGS.use_cell),'p1',str(FLAGS.p1_epoch_num),'bt',str(FLAGS.beta)]
    input_path_name =re.sub("features_", "", '_'.join(input_path_name))
    input_path_name =re.sub(".txt", "", input_path_name)
    input_path_name='2v_hb_{}'.format(input_path_name)
    out_path_name='results/{}/'.format(FLAGS.organ)
    mfMAP_model=mfMAP(input_dim_cnv,input_dim_expr, classifier_out_dim).to(device)
    if FLAGS.early_stopping:
        early_stop_ob=Earlystopping(number=FLAGS.patience,path='ssd/{}/{}_{}D_checkpoint.pt'.format(FLAGS.organ,input_path_name,str(FLAGS.latent_space_dim)))
    total_params=sum(params.numel() for params in mfMAP_model.parameters())
    print('Number of parameters: {}'.format(total_params))
    optimizer=optim.Adam(mfMAP_model.parameters(), lr=FLAGS.learning_rate)
    scheduler=ReduceLROnPlateau(optimizer, mode='min', patience=50,min_lr=1e-5)
    train_writer=SummaryWriter(log_dir='logs/train')
    val_writer=SummaryWriter(log_dir='logs/val')
    loss_array=np.zeros(shape=(15, FLAGS.p1_epoch_num+FLAGS.p2_epoch_num+1))
    metrics_array=np.zeros(shape=(4, FLAGS.p1_epoch_num+FLAGS.p2_epoch_num+1))
    def train(e_index, e_num,k_cnv_recon, k_expr_recon, k_kl, k_hbc):
        mfMAP_model.train()
        train_cnv_recon=0
        train_expr_recon=0
        train_kl=0
        train_classifier=0
        train_correct_num=0
        train_total_loss=0
        train_cls_er=0
        for batch_index, sample in enumerate(train_loader):
            data=sample[0]
            y=sample[1]
            y=torch.tensor(y)
            for i in range(2):
                data[i]=data[i].to(device)
            y=y.to(device)
            optimizer.zero_grad()
            _, recon_data, mean, log_var, pred_y=mfMAP_model(data,y)
            cnv_recon = cnv_recon_loss(recon_data, data)/data[0].shape[1]
            expr_recon = expr_recon_loss(recon_data, data)/data[1].shape[1]
            kl=kl_loss(mean, log_var)
            tmpidx=[y_ for y_ in range(len(y)) if y[y_]!=iuidx]
            class_er=classifier_er(pred_y, y,beta=FLAGS.beta)
            class_hb_loss =classifier_hb_loss(pred_y, y,tmpidx,beta=FLAGS.beta)
            loss=k_cnv_recon * cnv_recon +k_expr_recon * expr_recon+k_kl * kl+k_hbc * class_hb_loss
            loss.backward()
            with torch.no_grad():
                pred_y_softmax=F.softmax(pred_y, dim=1)
                _, predicted=torch.max(pred_y_softmax, 1)
                correct=(predicted[tmpidx]==y[tmpidx]).sum().item()
                train_cnv_recon += cnv_recon.item()
                train_expr_recon += expr_recon.item()
                train_kl += kl.item()
                train_classifier += class_hb_loss.item()
                train_correct_num += correct
                train_total_loss += loss.item()
                train_cls_er += class_er.item()
            optimizer.step()
        train_cnv_recon_ave=train_cnv_recon / len(train_dataset)
        train_expr_recon_ave=train_expr_recon / len(train_dataset)
        train_kl_ave=train_kl / len(train_dataset)
        train_classifier_ave=train_classifier / len(train_dataset)
        train_classifier_er_ave=train_cls_er / len(train_dataset)
        all_len=len(set(train_dataset.cnv_df.columns.tolist())-set(nolbl_dataset.cnv_df.columns.tolist()))
        train_accuracy=train_correct_num/all_len*100                              
        train_total_loss_ave=train_total_loss / len(train_dataset)

        print('Epoch {:3d}/{:3d}\n'
              'Training\n'
              'First View Recon Loss: {:.2f}   Second View Recon Loss: {:.2f}   KL Loss: {:.2f}   '
              'Classification Loss: {:.2f}\nACC: {:.2f}%'.
              format(e_index+1, e_num, train_cnv_recon_ave, train_expr_recon_ave, train_kl_ave,
                     train_classifier_ave, train_accuracy))
        loss_array[0, e_index]=train_cnv_recon_ave
        loss_array[1, e_index]=train_expr_recon_ave
        loss_array[2, e_index]=train_kl_ave
        loss_array[3, e_index]=train_classifier_ave
        loss_array[4, e_index]=train_accuracy
        loss_array[11, e_index]=train_total_loss_ave
        loss_array[13, e_index]=train_classifier_er_ave

        train_writer.add_scalar('Total loss', train_total_loss_ave, e_index)
        train_writer.add_scalar('First recon loss', train_cnv_recon_ave, e_index)
        train_writer.add_scalar('Second recon loss', train_expr_recon_ave, e_index)
        train_writer.add_scalar('KL loss', train_kl_ave, e_index)
        train_writer.add_scalar('Classification loss', train_classifier_ave, e_index)
        train_writer.add_scalar('Accuracy', train_accuracy, e_index)
        return train_accuracy,train_total_loss_ave,train_classifier_er_ave
 
    if FLAGS.separate_testing=='yes':
        def val(e_index, k_cnv_recon, k_expr_recon, k_kl, k_hbc,get_metrics=True):
            mfMAP_model.eval()
            val_cnv_recon=0
            val_expr_recon=0
            val_kl=0
            val_classifier=0
            val_correct_num=0
            val_total_loss=0
            val_cls_er=0
            y_store=torch.tensor([0])
            predicted_store=torch.tensor([0])
            yl_store=torch.tensor([0])
            predictedl_store=torch.tensor([0])
            with torch.no_grad():
                for batch_index, sample in enumerate(val_loader):
                    data=sample[0]
                    y=sample[1]
                    for i in range(2):
                        data[i]=data[i].to(device)
                    y=y.to(device)
                    _, recon_data, mean, log_var, pred_y=mfMAP_model(data,y)
                    
                    cnv_recon=cnv_recon_loss(recon_data, data)
                    expr_recon=expr_recon_loss(recon_data, data)
                    kl=kl_loss(mean, log_var)
                    tmpidx=[y_ for y_ in range(len(y)) if y[y_]!=iuidx]
                    class_er=classifier_er(pred_y, y,beta=FLAGS.beta)
                    class_hb_loss =classifier_hb_loss(pred_y, y,tmpidx,beta=FLAGS.beta)
                    loss=k_cnv_recon * cnv_recon +k_expr_recon * expr_recon+k_kl * kl+k_hbc * class_hb_loss

                    pred_y_softmax=F.softmax(pred_y, dim=1)
                    _, predicted=torch.max(pred_y_softmax, 1)
                    correct=(predicted[tmpidx]==y[tmpidx]).sum().item()

                    y_store=torch.cat((y_store, y.cpu()))
                    predicted_store=torch.cat((predicted_store, predicted.cpu()))
                    yl_store=torch.cat((yl_store, y[tmpidx].cpu()))
                    predictedl_store=torch.cat((predictedl_store, predicted[tmpidx].cpu()))                    
                    val_cnv_recon += cnv_recon.item()
                    val_expr_recon += expr_recon.item()
                    val_kl += kl.item()
                    val_classifier += class_hb_loss.item()
                    val_cls_er += class_er.item()
                    val_correct_num += correct
                    val_total_loss += loss.item()
                    

            output_y=yl_store[1:].numpy()
            output_pred_y=predictedl_store[1:].numpy()

            if get_metrics:
                metrics_array[0, e_index]=metrics.accuracy_score(output_y, output_pred_y)
                metrics_array[1, e_index]=metrics.precision_score(output_y, output_pred_y, average='weighted')
                metrics_array[2, e_index]=metrics.recall_score(output_y, output_pred_y, average='weighted')
                metrics_array[3, e_index]=metrics.f1_score(output_y, output_pred_y, average='weighted')
            
            val_cnv_recon_ave=val_cnv_recon / len(val_dataset)
            val_expr_recon_ave=val_expr_recon / len(val_dataset)
            val_kl_ave=val_kl / len(val_dataset)
            val_classifier_ave=val_classifier / len(val_dataset)
            val_classifier_er_ave=val_cls_er / len(val_dataset)
            all_val_len=len(set(val_dataset.cnv_df.columns.tolist())-set(nolbl_dataset.cnv_df.columns.tolist()))
            val_accuracy=val_correct_num /all_val_len*100
            val_total_loss_ave=val_total_loss / len(val_dataset)


            print('Validation\n'
                  'First View Recon Loss: {:.2f}   Second View Recon Loss: {:.2f}   KL Loss: {:.2f}   Classification Loss: {:.2f}'
                  '\nACC: {:.2f}%\n'.
                  format(val_cnv_recon_ave, val_expr_recon_ave, val_kl_ave, val_classifier_ave, val_accuracy))
            loss_array[5, e_index]=val_cnv_recon_ave
            loss_array[6, e_index]=val_expr_recon_ave
            loss_array[7, e_index]=val_kl_ave
            loss_array[8, e_index]=val_classifier_ave
            loss_array[9, e_index]=val_accuracy
            loss_array[12, e_index]=val_total_loss_ave
            loss_array[14, e_index]=val_classifier_er_ave

            # TB
            val_writer.add_scalar('Total loss', val_total_loss_ave, e_index)
            val_writer.add_scalar('First recon loss', val_cnv_recon_ave, e_index)
            val_writer.add_scalar('Second recon loss', val_expr_recon_ave, e_index)
            val_writer.add_scalar('KL loss', val_kl_ave, e_index)
            val_writer.add_scalar('Classification loss', val_classifier_ave, e_index)
            val_writer.add_scalar('Accuracy', val_accuracy, e_index)

            return val_accuracy, output_pred_y,val_classifier_ave,val_total_loss_ave 

    #print('\nUNSUPERVISED PHASE\n')
    best_train_accuracy=0
    best_val_accuracy=0
    best_train_er=math.inf
    # unsupervised phase
    c1,c2=1,0
    for epoch_index in range(FLAGS.p1_epoch_num):
        train_accuracy,_,_=train(e_index=epoch_index, e_num=FLAGS.p1_epoch_num+FLAGS.p2_epoch_num, k_cnv_recon=c1,k_expr_recon=c1, k_kl=c1,k_hbc=c2)
        if FLAGS.separate_testing=='yes':
            _, out_pred_y,_,val_total_loss_ave= val(epoch_index,k_cnv_recon=c1,k_expr_recon=c1, k_kl=c1,k_hbc=c2)

    print('\nSUPERVISED\n')
    epoch_number=FLAGS.p1_epoch_num
    c1,c2=1,1
    for epoch_index in range(FLAGS.p1_epoch_num, FLAGS.p1_epoch_num+FLAGS.p2_epoch_num):
        print((c1,c2))
        print(FLAGS.learning_rate)
        epoch_number += 1
        print('best train acc:{}'.format(best_train_accuracy))
        print('best train er:{}'.format(best_train_er))
        print('best validation acc:{}'.format(best_val_accuracy))
        train_accuracy,train_total_loss_ave,train_classifier_er_ave=train(e_index=epoch_index, e_num=FLAGS.p1_epoch_num+FLAGS.p2_epoch_num,k_cnv_recon=c1,k_expr_recon=c1, k_kl=c1,k_hbc=c2)
        if best_train_accuracy<train_accuracy:
            best_train_accuracy=train_accuracy
            torch.save(mfMAP_model.state_dict(), 'ssd/{}/{}_{}D_cpt_best_train_acc.pt'.format(FLAGS.organ,input_path_name,str(FLAGS.latent_space_dim)))
        if best_train_er>train_classifier_er_ave:
            best_train_er=train_classifier_er_ave
            if train_accuracy>=best_train_accuracy:
                torch.save(mfMAP_model.state_dict(), 'ssd/{}/{}_{}D_cpt_best_train_acc.pt'.format(FLAGS.organ,input_path_name,str(FLAGS.latent_space_dim)))


        if FLAGS.separate_testing=='yes':
            if epoch_index==FLAGS.p1_epoch_num+FLAGS.p2_epoch_num-1:
                    val_classification_acc, out_pred_y,val_classification_loss,val_total_loss_ave=val(epoch_index, k_cnv_recon=c1,k_expr_recon=c1, k_kl=c1,k_hbc=c2,get_metrics=True)
            else:
                    val_classification_acc, out_pred_y,val_classification_loss,val_total_loss_ave=val(epoch_index, k_cnv_recon=c1,k_expr_recon=c1, k_kl=c1,k_hbc=c2)

            if best_val_accuracy<val_classification_acc:
                best_val_accuracy=val_classification_acc
                torch.save(mfMAP_model.state_dict(), 'ssd/{}/{}_{}D_cpt_best_val_acc.pt'.format(FLAGS.organ,input_path_name,str(FLAGS.latent_space_dim)))

            if FLAGS.early_stopping:
                early_stop_ob(mfMAP_model, val_classification_acc)
                if early_stop_ob.stop_now:
                    print('Early stopping\n')
                    break
    
            #if scheduler.step(val_classification_loss) and best_train_accuracy>=95:
                #FLAGS.learning_rate= max(FLAGS.learning_rate * scheduler.factor, scheduler.min_lr)
        if scheduler.step(train_total_loss_ave):
            if best_train_accuracy>=100:
                break
            #elif best_val_accuracy>=90:
                #c1=0
                #c2=1
            #else:
                #c1=max(c1*scheduler.factor,1e-3)
                #c2=min(c2/scheduler.factor, 8)
                
    def get_decoder_weight(dataset=full_dataset):
        mfMAP_model.eval()
        param_dict={name:param.data.cpu().numpy().T for name, param in mfMAP_model.named_parameters() if re.search(r'd_fc\S*\.0\.weight',name) }
        w_f1=param_dict['d_fc4.0.weight']
        for i in ['d_fc3.0.weight','d_fc2_cnv.0.weight','d_fc1_cnv.0.weight']:
            if i=='d_fc3.0.weight':
                tmp=param_dict['d_fc3.0.weight'][:,0:param_dict['d_fc2_cnv.0.weight'].shape[0]]     
            else:
                tmp=param_dict[i]
            w_f1=np.dot(w_f1,tmp)

        w_f2=param_dict['d_fc4.0.weight']
        for i in ['d_fc3.0.weight','d_fc2_expr.0.weight','d_fc1_expr.0.weight']:
            if i=='d_fc3.0.weight':
                tmp=param_dict['d_fc3.0.weight'][:,param_dict['d_fc2_cnv.0.weight'].shape[0]:]     
            else:
                tmp=param_dict[i]
            w_f2=np.dot(w_f2,tmp)

    
        w_f1=pd.DataFrame(w_f1,columns=dataset.__dict__['cnv_df'].index)
        w_f2=pd.DataFrame(w_f2,columns=dataset.__dict__['expr_df'].index)
        return w_f1,w_f2

    def save_output(prefix,data_loader=full_loader,dataset=full_dataset):
        mfMAP_model.eval()
        predicted_store=torch.tensor([0]).to(device)
        d_z_store=torch.zeros(1, FLAGS.latent_space_dim).to(device)
        recon_cnv_store=torch.zeros(1,dataset.__dict__['cnv_df'].shape[0]).to(device)
        recon_expr_store=torch.zeros(1,dataset.__dict__['expr_df'].shape[0]).to(device)
        with torch.no_grad():
            for batch_index, sample in enumerate(data_loader):
                d=sample[0]
                y=sample[1].to(device)
                for j in range(2):
                    d[j]=d[j].to(device)
                _, recon_data, d_z, log_var, pred_y=mfMAP_model(d,y)
                for i in range(2):
                    recon_data[i]=recon_data[i].to(device)
                pred_y=pred_y.to(device)
                pred_y_softmax=F.softmax(pred_y, dim=1)
                _, predicted=torch.max(pred_y_softmax, 1)
                d_z_store=torch.cat((d_z_store, d_z), 0)
                recon_cnv_store=torch.cat((recon_cnv_store,recon_data[0]), 0)
                recon_expr_store=torch.cat((recon_expr_store,recon_data[1]), 0)
                predicted_store=torch.cat((predicted_store, predicted))
        all_data_z=d_z_store[1:]
        all_data_z_df=pd.DataFrame(all_data_z.cpu().numpy(), index=dataset.__dict__['cnv_df'].columns)
        all_pred_y=predicted_store[1:]
        all_pred_y_df=pd.DataFrame(all_pred_y.cpu().numpy(),index=dataset.__dict__['cnv_df'].columns)
        all_recon_cnv=recon_cnv_store[1:]
        all_recon_cnv_df=pd.DataFrame(all_recon_cnv.cpu().numpy().T,index=dataset.__dict__['cnv_df'].index,columns=dataset.__dict__['cnv_df'].columns)
        all_recon_expr=recon_expr_store[1:]
        all_recon_expr_df=pd.DataFrame(all_recon_expr.cpu().numpy().T,index=dataset.__dict__['expr_df'].index,columns=dataset.__dict__['expr_df'].columns)
        w_f1,w_f2=get_decoder_weight(dataset)
        prefix_name=out_path_name+input_path_name+"_{}_".format(prefix)+str(FLAGS.latent_space_dim)
        # Output file
        print('Preparing the output files... ')
        if not os.path.exists(out_path_name):
            os.mkdir(out_path_name)
        latent_space_path=prefix_name+'D_latent_space.tsv'
        all_data_z_df.to_csv(latent_space_path, sep='\t')
        str_fn1=FLAGS.input1_fn
        str_fn2=FLAGS.input2_fn
        for r in (("features_", ""), (".txt", "")):
            str_fn1=str_fn1.replace(*r)
            str_fn2=str_fn2.replace(*r)
        recon_fn1_path=prefix_name+'D_recon_{}.tsv'.format(str_fn1)
        recon_fn2_path=prefix_name+'D_recon_{}.tsv'.format(str_fn2)
        all_recon_cnv_df.to_csv(recon_fn1_path, sep='\t')
        all_recon_expr_df.to_csv(recon_fn2_path, sep='\t')
        dw_fn1_path=prefix_name+'D_decoder_w_{}.tsv'.format(str_fn1)
        dw_fn2_path=prefix_name+'D_decoder_w_{}.tsv'.format(str_fn2)
        w_f1.to_csv(dw_fn1_path, sep='\t')
        w_f2.to_csv(dw_fn2_path, sep='\t')
        pred_y_path=prefix_name+'D_pred_y.tsv'
        all_pred_y_df.to_csv(pred_y_path, sep='\t')
        if FLAGS.separate_testing=='yes':
            metrics_record_path=prefix_name+'D_metrics.tsv'
            np.savetxt(metrics_record_path, metrics_array, delimiter='\t')
            val_index_path=prefix_name+'D_val_index.tsv'
            np.savetxt(val_index_path,np.array(val_dataset.cnv_df.columns.tolist()), delimiter='\t', fmt="%s")

        if FLAGS.output_loss_record:
            loss_record_path=prefix_name+'D_loss_record.tsv'
            np.savetxt(loss_record_path, loss_array, delimiter='\t')
    print('Encoding data into latent representation...')
    print('separate testing:{}'.format(str(FLAGS.separate_testing)))
    print('use_cell:{}'.format(str(FLAGS.use_cell)))
    print(input_path_name)
    if FLAGS.separate_testing=='yes':
        if FLAGS.early_stopping:
            best_epoch=FLAGS.p1_epoch_num+early_stop_ob.best_epoch_num
            loss_array[10, 0]=best_epoch
            print('Load model of Epoch {:d}'.format(best_epoch))
            mfMAP_model.load_state_dict(torch.load('ssd/{}/{}_{}D_checkpoint.pt'.format(FLAGS.organ,input_path_name,str(FLAGS.latent_space_dim))))
            save_output('early_stop_full',full_loader, full_dataset)
            save_output('early_stop_nolbl',nolbl_loader,nolbl_dataset)
        else:
            print('Load model of best validation acc')
            mfMAP_model.load_state_dict(torch.load('ssd/{}/{}_{}D_cpt_best_val_acc.pt'.format(FLAGS.organ,input_path_name,str(FLAGS.latent_space_dim))))
            save_output('best_val_acc_full',full_loader, full_dataset)
            save_output('best_val_acc_nolbl',nolbl_loader,nolbl_dataset)
            print('Load model of best train acc')
            mfMAP_model.load_state_dict(torch.load('ssd/{}/{}_{}D_cpt_best_train_acc.pt'.format(FLAGS.organ,input_path_name,str(FLAGS.latent_space_dim))))
            save_output('best_train_acc_full',full_loader, full_dataset)
            save_output('best_train_nolbl',nolbl_loader,nolbl_dataset)
    else:
        print('Load model of best train acc')
        mfMAP_model.load_state_dict(torch.load('ssd/{}/{}_{}D_cpt_best_train_acc.pt'.format(FLAGS.organ,input_path_name,str(FLAGS.latent_space_dim))))
        save_output('best_train_acc_full',full_loader, full_dataset)
        save_output('best_train_acc_nolbl',nolbl_loader,nolbl_dataset)
#%%
run_train()





