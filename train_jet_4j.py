from __future__ import division
import sys, os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'

import math
import numpy as np
from keras import backend as K
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Activation, Dropout, add, LSTM, Concatenate, MaxPooling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv1D, Conv2D
from array import array
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from keras.utils.np_utils import to_categorical
from keras import optimizers
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from keras.layers.core import Reshape
from keras.utils.vis_utils import plot_model
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--e', required=False, default=1000, help='layers')

args = parser.parse_args()
epochs = int(args.e) #Epoch
dropout = 0.1 #Dropout
njets = 4

# input files
input_2016 = "/home/juhee5819/cheer/ttbb/mydnn/ntuples/arrays/v3_4j/ttbb_nano_2016_4j.h5"
input_2017 = "/home/juhee5819/cheer/ttbb/mydnn/ntuples/arrays/v3_4j/ttbb_nano_2017_4j.h5"
input_2018 = "/home/juhee5819/cheer/ttbb/mydnn/ntuples/arrays/v3_4j/ttbb_nano_2018_4j.h5"

data_2016 = pd.read_hdf( input_2016 )
data_2017 = pd.read_hdf( input_2017 )
data_2018 = pd.read_hdf( input_2018 )
data = data_2016.append( data_2017, ignore_index=True ).append( data_2018, ignore_index=True )

nperm = len( data['jet_perm'].unique() )-1
data = data.drop(data[data['jet_perm']==nperm].index)

import re
resultDir = "/home/juhee5819/cheer/ttbb/mydnn/train/results/v3_4j/"
ver = "try1"

newDir = resultDir+ver
if os.path.exists( newDir ):
    string = re.split(r'(\d+)', ver)[0]
    num = int( re.split(r'(\d+)', ver)[1] )
    while os.path.exists( newDir ):
        num = num+1
        newDir = resultDir+string+str(num)

print 'The path of directory is ', newDir
os.makedirs( newDir )

# 160 var
event_var = ['nbjets_m', 'ncjets_m', 'ngoodjets', 'Ht', 'lepton_pt', 'lepton_eta', 'lepton_phi', 'lepton_m', 'MET_pt', 'MET_phi', "dEta12", "dEta13", "dEta14", "dEta23", "dEta24", "dEta34", "dPhi12", "dPhi13", "dPhi14", "dPhi23", "dPhi24", "dPhi34", "invm12", "invm13", "invm14", "invm23", "invm24", "invm34", "dRnulep12", "dRnulep13", "dRnulep14", "dRnulep23", "dRnulep24", "dRnulep34"]
jet_var = ["jet1_pt", "jet1_eta", "jet1_m", "jet1_btag", "jet1_cvsl", "dRlep1", "dRnu1", "invmlep1", "jet2_pt", "jet2_eta", "jet2_m", "jet2_btag", "jet2_cvsl", "dRlep2", "dRnu2", "invmlep2", "jet3_pt", "jet3_eta", "jet3_m", "jet3_btag", "jet3_cvsl", "dRlep3", "dRnu3", "invmlep3", "jet4_pt", "jet4_eta", "jet4_m", "jet4_btag", "jet4_cvsl", "dRlep4", "dRnu4", "invmlep4"]
nvar = len( event_var ) + len( jet_var )

weights = compute_class_weight( class_weight='balanced', classes=np.unique(data['jet_perm']), y=data['jet_perm'])
dic_weights = dict(enumerate(weights))
print ' weight', weights

# split train valid set
def split_train_test():
    pd_out = data.filter(items = ['event_category', 'jet_perm'])
    pd_input = data.filter(items = event_var+jet_var)
    np_out = np.array( pd_out )
    np_input = np.array( pd_input )
    # split
    test_size = 0.3
    train_input, valid_input, train_out, valid_out = train_test_split( np_input, np_out, test_size=test_size )

    # train set
    pd_train_input = pd.DataFrame( train_input,columns= event_var+jet_var )
    pd_train_out = pd.DataFrame( train_out, columns=['event_category', 'jet_perm'] )
    # event info
    train_event_input = pd_train_input.filter( items=event_var )
    train_event_input = np.array( train_event_input )
    train_event_out = pd_train_out.filter( items=['event_category'] )
    train_event_out = to_categorical( train_event_out )
    # jet info
    train_jet_input = pd_train_input.filter( items=jet_var )
    train_jet_input = np.array( train_jet_input )
    train_jet_input = train_jet_input.reshape( train_jet_input.shape[0], 4, -1 )
    train_jet_out = pd_train_out.filter( items=['jet_perm'] )
    train_jet_out = to_categorical( train_jet_out )

    # valid set
    pd_valid_input = pd.DataFrame( valid_input,columns= event_var+jet_var )
    pd_valid_out = pd.DataFrame( valid_out, columns=['event_category', 'jet_perm'] )
    # event info
    valid_event_input = pd_valid_input.filter( items=event_var )
    valid_event_input = np.array( valid_event_input )
    valid_event_out = pd_valid_out.filter( items=['event_category'] )
    valid_event_out = to_categorical( valid_event_out )
    # jet info
    valid_jet_input = pd_valid_input.filter( items=jet_var )
    valid_jet_input = np.array( valid_jet_input )
    valid_jet_input = valid_jet_input.reshape( valid_jet_input.shape[0], 4, -1 )
    valid_jet_out = pd_valid_out.filter( items=['jet_perm'] )
    valid_jet_out = to_categorical( valid_jet_out )
    return train_event_input, train_event_out, train_jet_input, train_jet_out, valid_event_input, valid_event_out, valid_jet_input, valid_jet_out

train_event_input, train_event_out, train_jet_input, train_jet_out, valid_event_input, valid_event_out, valid_jet_input, valid_jet_out = split_train_test()
print( 'Split has done!' )

def jet_model():
    Inputs = [ Input( shape=(train_event_input.shape[1],) ), Input( shape=(train_jet_input.shape[1], train_jet_input.shape[2]), ) ]

    dropout = 0.1
    nodes = 50
    # BatchNormalization
    event_info = BatchNormalization( name = 'event_input_batchnorm' )(Inputs[0])
    jets = BatchNormalization( name = 'jet_input_batchnorm' )(Inputs[1])

    # Dense for event
    event_info = Dense(nodes, activation='relu', name='event_layer1')(event_info)
    event_info = Dropout(dropout)(event_info)
    event_info = Dense(nodes, activation='relu', name='event_layer2')(event_info)
    event_info = Dropout(dropout)(event_info)
    event_info = Dense(nodes, activation='relu', name='event_layer3')(event_info)
    event_info = Dropout(dropout)(event_info)

    # CNN for jet
    jets = Conv1D( 128, 1, kernel_initializer='lecun_uniform',  activation='relu', name='jets_conv0')(jets)
    jets = Dropout(dropout)(jets)
    jets = Conv1D( 64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='jets_conv1')(jets)
    jets = Dropout(dropout)(jets)
    jets = Conv1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='jets_conv2')(jets)
    jets = Dropout(dropout)(jets)
    jets = Conv1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='jets_conv3')(jets)
    jets = Dropout(dropout)(jets)
    jets = Conv1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='jets_conv4')(jets)
    jets = Dropout(dropout)(jets)
    jets = LSTM(20, go_backwards=True, implementation=2, name='jets_lstm')(jets)

    # Concatenate
    x = Concatenate()( [event_info, jets] )
    x = Dense(10, activation='relu',kernel_initializer='lecun_uniform', name='concat_layer')(x)

    perm_pred = Dense( int(nperm), activation='softmax', kernel_initializer='lecun_uniform', name='jet_prediction' )(x)
    model = Model(inputs=Inputs, outputs= perm_pred)
    model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
    return model

model = jet_model()
model_path = '/home/juhee5819/cheer/ttbb/mydnn/train/model/ttbb_v3_4j_model'
model.save(model_path)
print ' This model is saved in ', model_path

earlystop = EarlyStopping(monitor='val_loss', patience=20)
filename = os.path.join(newDir, 'best_model.h5')
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
batch_size = 1024

print '\ntraining with weight\n'
hist = model.fit([train_event_input,train_jet_input], train_jet_out, batch_size=batch_size, epochs=epochs, validation_data=([valid_event_input, valid_jet_input],  valid_jet_out), callbacks=[earlystop, checkpoint], class_weight=dic_weights)
model.summary()
# find best epoch
check_loss = model.history.history['val_loss']
bestepoch = np.argmin( check_loss )+1

model.load_weights( filename )
pred = model.predict( [valid_event_input, valid_jet_input] )
# pred jet & real jet
pred_jet = pred
pred_jet = np.argmax( pred_jet, axis=1 )
real_jet = np.argmax( valid_jet_out, axis=1) #???
real_event = np.argmax( valid_event_out, axis=1 )

from sklearn.metrics import confusion_matrix
val_result = pd.DataFrame( {'real_event':real_event, 'real_jet':real_jet, 'pred_jet':pred_jet} )
conf_jet   = confusion_matrix( val_result['real_jet'], val_result['pred_jet'] )
correct_jet   = conf_jet.trace()
sum_jet   = conf_jet.sum(axis=1)[:, np.newaxis]
acc_jet = correct_jet/len( valid_jet_out ) * 100

print 'writing results...'
with open('ttbb_des.txt', "a") as f_log:
    f_log.write("\ntrainInput "+input_2018)
    f_log.write(newDir)
    f_log.write('nvar: '+str(nvar)+'\n')
    f_log.write('training samples '+str(len(train_jet_out))+'   validation samples '+str(len(valid_jet_out))+'\n')
    f_log.write('best epoch: '+str(bestepoch)+'  jet accuracy: '+str(correct_jet)+'/'+str(len(valid_jet_out))+'='+str(acc_jet)+'\n')

with open(newDir+'/des.txt', "a") as f_log:
    f_log.write("\n\ntrainInput "+input_2018)
    f_log.write(newDir+'\n')
    f_log.write('nvar: '+str(nvar)+'\n')
    f_log.write('training samples '+str(len(train_jet_out))+'   validation samples '+str(len(valid_jet_out))+'\n')
    f_log.write('best epoch: '+str(bestepoch)+'   accuracy: '+str(correct_jet)+'/'+str(len(valid_jet_out))+'='+str(acc_jet)+'\n')
    f_log.write('conf jet: \n{}\n'.format(np.array2string(conf_jet)))

plotDir = newDir+'/'
print("Plotting scores")
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['Train','Test'], loc='lower right')
plt.savefig(os.path.join(plotDir, 'accuracy.pdf'), bbox_inches='tight')
plt.gcf().clear()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Train','Test'],loc='upper right')
plt.savefig(os.path.join(plotDir, 'loss.pdf'), bbox_inches='tight')
plt.gcf().clear()
#
##Heatmap
plt.rcParams['figure.figsize'] = [7.5, 6]
cfmt = lambda x,pos: '{:.0%}'.format(x)
heatmap = sns.heatmap(conf_jet/sum_jet, annot=True, cmap='YlGnBu', fmt='.0%', annot_kws={"size":9}, vmax=1, cbar_kws={'format': FuncFormatter(cfmt)} )
plt.title('Jet Permutation', fontsize=15)
plt.xlabel('pred.', fontsize=12)
plt.ylabel('real', fontsize=12)
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)
plt.savefig(os.path.join(plotDir, 'heatmap.pdf'), bbox_inches='tight')
plt.gcf().clear()
#
