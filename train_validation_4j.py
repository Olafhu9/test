import re, os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from keras.utils.np_utils import to_categorical
from ROOT import *
from array import array
import math
from sklearn.metrics import confusion_matrix

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--y', required=True, default=1000)
parser.add_argument('--vfp', required=False)
args = parser.parse_args()
year = args.y
#if year == 2016:
#   year = str(year)+vfp

input_ttbb = "/home/juhee5819/cheer/ttbb/mydnn/ntuples/arrays/v3_4j/ttbb_nano_"+str(year)+"_4j.h5"
data = pd.read_hdf( input_ttbb )
outfilename = 'ttbb_nano_'+str(year)+'_validation.root'

print( input_ttbb )
print('Data is loaded')

# input variables
event_var = ['nbjets_m', 'ncjets_m', 'ngoodjets', 'Ht', 'lepton_pt', 'lepton_eta', 'lepton_phi', 'lepton_m', 'MET_pt', 'MET_phi', "dEta12", "dEta13", "dEta14", "dEta23", "dEta24", "dEta34", "dPhi12", "dPhi13", "dPhi14", "dPhi23", "dPhi24", "dPhi34", "invm12", "invm13", "invm14", "invm23", "invm24", "invm34", "dRnulep12", "dRnulep13", "dRnulep14", "dRnulep23", "dRnulep24", "dRnulep34"]
jet_var = ["jet1_pt", "jet1_eta", "jet1_m", "jet1_btag", "jet1_cvsl", "dRlep1", "dRnu1", "invmlep1", "jet2_pt", "jet2_eta", "jet2_m", "jet2_btag", "jet2_cvsl", "dRlep2", "dRnu2", "invmlep2", "jet3_pt", "jet3_eta", "jet3_m", "jet3_btag", "jet3_cvsl", "dRlep3", "dRnu3", "invmlep3", "jet4_pt", "jet4_eta", "jet4_m", "jet4_btag", "jet4_cvsl", "dRlep4", "dRnu4", "invmlep4"]
nvar = len(event_var)+len(jet_var)
print(str(nvar))

# split data
def process_full():
    # event info    
    full_event_input = data.filter( items=event_var )
    full_event_input = np.array( full_event_input )
    full_event_out = data.filter( items=['event_category'] )
    full_event_out = to_categorical( full_event_out )
    # jet info
    full_jet_input = data.filter( items=jet_var )
    full_jet_input = np.array( full_jet_input )
    full_jet_input = full_jet_input.reshape( full_jet_input.shape[0], 4, -1 )
    full_jet_out = data.filter( items=['jet_perm'] )
    full_jet_out = to_categorical( full_jet_out )
    return full_event_input, full_event_out, full_jet_input, full_jet_out

full_event_input, full_event_out,full_jet_input, full_jet_out = process_full()

filename = '/home/juhee5819/cheer/ttbb/mydnn/train/results/v3_4j/try1/best_model.h5'

print( year )
print( filename )

model_path = '/home/juhee5819/cheer/ttbb/mydnn/train/model/ttbb_v3_4j_model'
model = tf.keras.models.load_model(model_path)
model.load_weights( filename )
print 'Model weights is loaded!'

pred = model.predict( [full_event_input, full_jet_input] )
pred_jet = np.argmax( pred, axis=1 )
real_jet = np.argmax( full_jet_out, axis=1) #???
print('Prediction has done')

# efficiencies
conf_jet = confusion_matrix( real_jet, pred_jet )
correct_jet = conf_jet.trace()
sum_jet = conf_jet.sum( axis=1 )[:, np.newaxis]

mateff = float(correct_jet)/float(len( full_jet_out )) * 100
recoeff = float(correct_jet)/ float((real_jet<6).sum()) * 100

print 'writing results...'
with open('ttbb_matnreco.txt', "a") as f_log:
    f_log.write("\ntrainInput "+input_ttbb)
    f_log.write("\nmatching eff: "+str(correct_jet)+"/"+str(len(full_jet_out))+"="+str(mateff))
    f_log.write("\nreco eff: "+str(correct_jet)+"/"+str( (real_jet<6).sum() )+"="+str(recoeff)+"\n")

data['pred_jet'] = pred_jet

perm = ['12', '13', '14', '23', '24', '34']

new_data = pd.DataFrame()
for i in range(len(perm)):
    d = data.loc[ data['pred_jet'] == i ].copy()
    d['dnn_dRbb'] = d['dR'+perm[i]]
    d['dnn_mbb'] = d['invm'+perm[i]]
    new_data = new_data.append( d )

new_data = new_data.sort_index()

outfile = TFile.Open(outfilename, 'RECREATE')

h_dnn_mbb = TH1D('h_dnn_mbb','',4,array('d',[0.0,60.0,100.0,170.0,400.0]))
h_dnn_mbb.GetXaxis().SetTitle("Reco. m_{b#bar{b}}(GeV)")
h_dnn_mbb.GetYaxis().SetTitle("Entries")
h_dnn_mbb.Sumw2()

h_dnn_dRbb = TH1D('h_dnn_dRbb','',4,array('d',[0.4,0.6,1.0,2.0,4.0]))
h_dnn_dRbb.GetXaxis().SetTitle("Reco. #DeltaR_{b#bar{b}}")
h_dnn_dRbb.GetYaxis().SetTitle("Entries")
h_dnn_dRbb.Sumw2()

h_gen_mbb = TH1D('h_gen_mbb','',4,array('d',[0.0,60.0,100.0,170.0,400.0]))
h_gen_mbb.GetXaxis().SetTitle("Gen. m_{b#bar{b}}(GeV)")
h_gen_mbb.GetYaxis().SetTitle("Entries")
h_gen_mbb.Sumw2()

h_gen_dRbb = TH1D('h_gen_dRbb','',4,array('d',[0.4,0.6,1.0,2.0,4.0]))
h_gen_dRbb.GetXaxis().SetTitle("Gen. #DeltaR_{b#bar{b}}")
h_gen_dRbb.GetYaxis().SetTitle("Entries")
h_gen_dRbb.Sumw2()
h_responseMatrix_mbb = TH2D('h_responseMatrix_mbb','',4,array('d',[0.0,60.0,100.0,170.0,400.0]),4,array('d',[0.0,60.0,100.0,170.0,400.0]))
h_responseMatrix_mbb.GetXaxis().SetTitle("Reco. m_{b#bar{b}}(GeV)")
h_responseMatrix_mbb.GetYaxis().SetTitle("Gen. m_{b#bar{b}}(GeV)")
h_responseMatrix_mbb.Sumw2()

h_responseMatrix_dRbb = TH2D('h_responseMatrix_dRbb','',4,array('d',[0.4,0.6,1.0,2.0,4.0]),4,array('d',[0.4,0.6,1.0,2.0,4.0]))
h_responseMatrix_dRbb.GetXaxis().SetTitle("Reco. #DeltaR_{b#bar{b}}")
h_responseMatrix_dRbb.GetYaxis().SetTitle("Gen. #DeltaR_{b#bar{b}}")
h_responseMatrix_dRbb.Sumw2()


for index, event in new_data.iterrows():
    dnn_mbb = event['dnn_mbb']
    dnn_dRbb = event['dnn_dRbb']
    gen_mbb = event['gen_mbb']
    gen_dRbb = event['gen_dRbb']
    h_dnn_mbb.Fill(dnn_mbb)
    h_dnn_dRbb.Fill(dnn_dRbb)
    h_gen_mbb.Fill(gen_mbb)
    h_gen_dRbb.Fill(gen_dRbb)
    h_responseMatrix_mbb.Fill(dnn_mbb, gen_mbb)
    h_responseMatrix_dRbb.Fill(dnn_dRbb, gen_dRbb)

h_purity_dRbb = h_responseMatrix_dRbb.ProjectionX()
h_purity_dRbb.SetName("h_purity_dRbb")
h_purity_dRbb.GetXaxis().SetTitle("Reco. #DeltaR_{b#bar{b}}")
h_purity_dRbb.GetYaxis().SetTitle("Purity (%)")

h_stability_dRbb = h_responseMatrix_dRbb.ProjectionY()
h_stability_dRbb.SetName("h_stability_dRbb")
h_stability_dRbb.GetXaxis().SetTitle("Gen. #DeltaR_{b#bar{b}}")
h_stability_dRbb.GetYaxis().SetTitle("Stability (%)")

h_purity_mbb = h_responseMatrix_mbb.ProjectionX()
h_purity_mbb.SetName("h_purity_mbb")
h_purity_mbb.GetXaxis().SetTitle("Reco. m_{b#bar{b}}(GeV)")
h_purity_mbb.GetYaxis().SetTitle("Purity (%)")

h_stability_mbb = h_responseMatrix_mbb.ProjectionY()
h_stability_mbb.SetName("h_stability_mbb")
h_stability_mbb.GetXaxis().SetTitle("Gen. m_{b#bar{b}}(GeV)")
h_stability_mbb.GetYaxis().SetTitle("Stability (%)")

h_stability_mbb = h_responseMatrix_mbb.ProjectionY()
h_stability_mbb.SetName("h_stability_mbb")
h_stability_mbb.GetXaxis().SetTitle("Gen. m_{b#bar{b}}(GeV)")
h_stability_mbb.GetYaxis().SetTitle("Stability (%)")

for i in range( h_purity_dRbb.GetNbinsX() ):
    den = h_purity_dRbb.GetBinContent(i+1)
    num = h_responseMatrix_dRbb.GetBinContent(i+1, i+1)
    if num*den > 0:
        purity = num/den*100
        h_purity_dRbb.SetBinContent(i+1, purity)
        h_purity_dRbb.SetBinError(i+1, abs(purity)*math.sqrt(pow(math.sqrt(den)/den,2)+pow(math.sqrt(num)/num,2)))

for i in range( h_stability_dRbb.GetNbinsX() ):
    den = h_stability_dRbb.GetBinContent(i+1)
    num = h_responseMatrix_dRbb.GetBinContent(i+1, i+1)
    if num*den > 0:
        stability = float(num/den*100)
        h_stability_dRbb.SetBinContent(i+1, stability)
        h_stability_dRbb.SetBinError(i+1, abs(stability)*math.sqrt(pow(math.sqrt(den)/den,2)+pow(math.sqrt(num)/num,2)))

for i in range( h_purity_mbb.GetNbinsX() ):
    den = h_purity_mbb.GetBinContent(i+1)
    num = h_responseMatrix_mbb.GetBinContent(i+1, i+1)
    if num*den > 0:
        purity = num/den*100
        h_purity_mbb.SetBinContent(i+1, purity)
        h_purity_mbb.SetBinError(i+1, abs(purity)*math.sqrt(pow(math.sqrt(den)/den,2)+pow(math.sqrt(num)/num,2)))

for i in range( h_stability_mbb.GetNbinsX() ):
    den = h_stability_mbb.GetBinContent(i+1)
    num = h_responseMatrix_mbb.GetBinContent(i+1, i+1)
    if num*den > 0:
        stability = float(num/den*100)
        h_stability_mbb.SetBinContent(i+1, stability)
        h_stability_mbb.SetBinError(i+1, abs(stability)*math.sqrt(pow(math.sqrt(den)/den,2)+pow(math.sqrt(num)/num,2)))

outfile.Write()
outfile.Close()
print('Saved')
