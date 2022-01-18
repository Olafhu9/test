#include "ROOT/RDataFrame.hxx"
#include "ROOT/RVec.hxx"
#include "ROOT/RDF/RInterface.hxx"
#include "TCanvas.h"
#include "TH1D.h"
#include "TLatex.h"
#include "TLegend.h"
#include "Math/Vector4Dfwd.h"
#include "TStyle.h"
#include <vector>
#include <algorithm>
#include <functional>
#include <ROOT/RSnapshotOptions.hxx>
#include "TMath.h"
#include <string>
#include <iostream>

using namespace ROOT::VecOps;
using namespace std;
using rvec_f = RVec<float>;
using rvec_b = RVec<bool>;
using rvec_i = RVec<int>;
using Impl_i = vector<int, ROOT::Detail::VecOps::RAdoptAllocator<int>>;

//********** sort jets by btag & pT
vector<int> sortbtag(rvec_f jet_pt, rvec_f jet_eta, rvec_f btag){
    auto sort_btag = Reverse(Argsort(btag));
    vector<int> bjet_idx;
    vector<int> pt_idx;
    vector<int> sortjet_idx;

    // sort jets by b-discriminant
    for( size_t i = 0; i < jet_pt.size(); i++ ){
        if( jet_pt[sort_btag[i]] > 20 && abs(jet_eta[sort_btag[i]]) < 2.4 ) bjet_idx.push_back(sort_btag[i]);
    }

    return bjet_idx;
}

// ********** jet permutation categorize
int jet_perm_cat (vector<TLorentzVector> Jets, TLorentzVector addjet1, TLorentzVector addjet2){

    float dR1, dR2;
    vector<int> idx1, idx2, idx3;
    // *** matching
    cout << endl << "       New event" <<endl;
    for( int i = 0; i < Jets.size(); i++){
        dR1 = Jets[i].DeltaR(addjet1);
        dR2 = Jets[i].DeltaR(addjet2);

        cout << i << " dR1: " << dR1 << " dR2: " << dR2 << endl;
        cout << Jets[i].Pt() << endl;

        if( dR1 < 0.4 && dR2 < 0.4 ) idx3.push_back( i );
        else if( dR1 < 0.4 ) idx1.push_back( i );
        else if( dR2 < 0.4 ) idx2.push_back( i );
    }
    // *** choose one jet in idx1, idx2
    int idx1_jet = 100, idx2_jet = 100;
    sort( idx1.begin(), idx1.end(), less<int>() );
    sort( idx2.begin(), idx2.end(), less<int>() );

    if( idx1.size() > 0 ) idx1_jet = idx1[0];
    if( idx2.size() > 0 ) idx2_jet = idx2[0];

    vector<int> match_idx;

    match_idx.insert( match_idx.begin(), idx3.begin(), idx3.end() );
    match_idx.push_back( idx1_jet );
    match_idx.push_back( idx2_jet );

    sort( match_idx.begin(), match_idx.end(), less<int>() );

    //debug
    cout << "mat idx ";
    for( int i=0; i<match_idx.size();i++ ){
        cout << match_idx[i] << " ";
    }
    cout << endl;

    int signal_num = 0;
    if( match_idx[0]+match_idx[1] < 12 ) signal_num = TMath::Power(2, match_idx[0]) + TMath::Power(2, match_idx[1]);

    int cat = 6;
    switch(signal_num){
        case 3: cat = 0; break;
        case 5: cat = 1; break;
        case 9: cat = 2; break;
        case 6: cat = 3; break;
        case 10: cat = 4; break;
        case 12: cat = 5; break;
        default: cat = 6; break;
    }
    cout << "sig num " << signal_num << " cat " << cat << endl;
    return cat;
}

//********** set TLorentzVector
TLorentzVector set_lorentz (float pt, float eta, float phi, float m){
    TLorentzVector p;
    p.SetPtEtaPhiM(pt, eta, phi, m);
    return p;
}

//********** set TLorentzVector
TLorentzVector set_lorentzw (int pt, int eta, int phi, int m){
    TLorentzVector p;
    p.SetPtEtaPhiM(pt, eta, phi, m);
    return p;
}

//********** set TLorentzVector of MET
TLorentzVector set_MET(float pt, float phi){
    TLorentzVector MET;
    MET.SetPtEtaPhiE(pt, 0.0f, phi, pt);
    return MET;
}

//********** save 6 sorted jets as vector
vector<TLorentzVector> vecJets (vector<int> sortjet_idx, rvec_f jet_pt, rvec_f jet_eta, rvec_f jet_phi, rvec_f jet_m, rvec_f btag){
    int nJets = 4;
    //*** TLorentzVector & vector
    vector<TLorentzVector> Jets;
    TLorentzVector jet;
    for (int i = 0; i < nJets; i++){
        jet.SetPtEtaPhiM(jet_pt[sortjet_idx[i]], jet_eta[sortjet_idx[i]], jet_phi[sortjet_idx[i]], jet_m[sortjet_idx[i]]);
        Jets.push_back(jet);
   }
    return Jets;
}

//********** delta R between particles
float deltaR (TLorentzVector p1, TLorentzVector p2){
    float dR = p1.DeltaR(p2);
    return dR;
}

//********** delta R between MET+lep and jet
float deltaR_nulj (TLorentzVector nu, TLorentzVector lep, TLorentzVector jet){
    float Wleta = (nu+lep).Eta();
    float Wlphi = (nu+lep).Phi();
    float jet_eta = jet.Eta();
    float jet_phi = jet.Phi();
    float dR = DeltaR(Wleta, jet_eta, Wlphi, jet_phi);
    return dR;
}

//********** delta R between MET+lep and jet combi
float deltaR_nuljj (TLorentzVector nu, TLorentzVector lep, TLorentzVector jet1, TLorentzVector jet2){
    float Wleta = (nu+lep).Eta();
    float Wlphi = (nu+lep).Phi();
    float jet_eta = (jet1+jet2).Eta();
    float jet_phi = (jet1+jet2).Phi();
    float dR = DeltaR(Wleta, jet_eta, Wlphi, jet_phi);
    return dR;
}

//**********  pT
float pt_sum (TLorentzVector p1, TLorentzVector p2){
    float pt = (p1+p2).Pt();
    return pt;
}

//**********  mass
float mass(TLorentzVector p){
    float m = p.M();
    return m;
}
//********** invariant mass between paritcles
float invmass (TLorentzVector p1, TLorentzVector p2){
    float invariantmass = (p1+p2).M();
    return invariantmass;
}

//********** delta Phi between particles
float deltaPhi (TLorentzVector p1, TLorentzVector p2){
    float dPhi = abs(p1.DeltaPhi(p2));
    return dPhi;
}

//********** delta Eta between particles
float deltaEta (TLorentzVector p1, TLorentzVector p2){
    float dEta = abs(p1.Eta() - p2.Eta());
    return dEta;
}

//********** Ht of 6 jets
float setHt (TLorentzVector jet1, TLorentzVector jet2, TLorentzVector jet3, TLorentzVector jet4){
    float pt1 = jet1.Pt();
    float pt2 = jet2.Pt();
    float pt3 = jet3.Pt();
    float pt4 = jet4.Pt();
    float Hts = pt1 + pt2 + pt3 + pt4;
    return Hts;
}

//********** St
float setSt (TLorentzVector p1, TLorentzVector p2, TLorentzVector p3, TLorentzVector p4, TLorentzVector p5, TLorentzVector p6){
    float pt1 = p1.Pt();
    float pt2 = p2.Pt();
    float pt3 = p3.Pt();
    float pt4 = p4.Pt();
    float pt5 = p5.Pt();
    float pt6 = p6.Pt();
    float Sts = pt1 + pt2 + pt3 + pt4 + pt5 + pt6;
    return Sts;
}

// root -l ana.C'(2018, "ttbb")'
void ana_KIT(int year, string process, string VFP=""){
    string inputfile, outputfile;
    float btag_m, cvsb_m, cvsl_m;
    //const char *inputFile, const char *outputFile, int jcut, int bcut
    if( year == 2016 ){
        btag_m = 0.3093;
        cvsb_m = 0.29;
        cvsl_m = 0.085;
    }
    else if( year == 2017 ){
        btag_m = 0.3033;
        cvsb_m = 0.29;
        cvsl_m = 0.144;
    }
    else if( year == 2018 ){
        btag_m = 0.277;
        cvsb_m = 0.29;
        cvsl_m = 0.085;
    }
    //inputfile = "/data/users/juhee5819/ntuple/nanoaod/ttbb_ntuple_v3/"+to_string(year)+"/TTToSemiLeptonic_ttbb.root";
    inputfile = "root://cms-t2-se01.sdfarm.kr:1094//store/user/vanderli/topnanoaodv9/ul/2018/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/ttbb2018HYUv3/220114_144329/0000/tree_1.root";
    outputfile = "ttbb_nano_"+to_string(year)+VFP+"_4j.root";
    cout << endl << "using " << to_string(year) << VFP << " " << process << " samples" << endl;
    cout << "input file " << inputfile << endl << "output file " << outputfile << endl;

    // HF tagging
    auto btag = [btag_m](rvec_f jet_pt, rvec_f jet_eta, rvec_f jet_deepJet) { return jet_pt > 20 && abs(jet_eta) < 2.4 && jet_deepJet > btag_m; };
    auto ctag = [cvsb_m, cvsl_m](rvec_f jet_pt, rvec_f jet_eta, rvec_f jet_cvsb, rvec_f jet_cvsl){ return jet_pt > 20 && abs(jet_eta) < 2.4 && jet_cvsb > cvsb_m && jet_cvsl > cvsl_m; };
    // event category
    auto event_cat = [process](){ int cat; if( process == "ttbb" ) cat = 0; else if( process == "ttbj" ) cat = 1; else if( process == "ttcc" ) cat = 2; else if(process == "ttLF") cat = 3; else if(process == "ttHbb") cat = 4; return cat; };

    ROOT::RDataFrame df("Events", inputfile);

    cout << "start" << endl;
    auto df_ = df.Define("goodlepton", "Lep_Pt >30 && abs(Lep_Eta) < 2.4")
                 .Define("ngoodlep", "Sum(goodlepton)");
    cout << "end lep" << endl;
    //auto df_goodlepton = df.Filter("Lep_Pt > 30 && abs(Lep_Eta) < 2.4")
    auto df_goodlepton = df_.Filter("ngoodlep==1")
                           .Define("goodjets","Jet_Pt_nom > 20 && abs(Jet_Eta_nom) < 2.4")
                           .Define("ngoodjets", "Sum(goodjets)")
                           .Define("bjets_m", btag, {"Jet_Pt_nom", "Jet_Eta_nom", "Jet_btagValue_nom"})
                           .Define("cjets_m", ctag, {"Jet_Pt_nom", "Jet_Eta_nom", "Jet_CvB_nom", "Jet_CvL_nom"})
                           .Define("ncjets_m", "Sum(cjets_m)")
                           .Define("nbjets_m", "Sum(bjets_m)");
    cout << "1" << endl;
    auto df_goodjet = df_goodlepton.Filter("ngoodjets >=6 ", "Events with at least 4 goodjets");
    auto df_bjet = df_goodjet.Filter("nbjets_m >= 4", "Events with at least 2 medium b-jets")

                             //.Define("addbjet1", set_lorentz, {"addbjet1_pt", "addbjet1_eta", "addbjet1_phi", "addbjet1_mass"})
                             //.Define("addbjet2", set_lorentz, {"addbjet2_pt", "addbjet2_eta", "addbjet2_phi", "addbjet2_mass"})
                             //.Define("gen_dRbb", deltaR, {"addbjet1", "addbjet2"})
                             //.Define("gen_mbb", invmass, {"addbjet1", "addbjet2"})

                             // reco particles
                             .Define("lepton_pt", "Lep_Pt[0]")
                             .Define("lepton_eta", "Lep_Eta[0]")
                             .Define("lepton_phi", "Lep_Phi[0]")
                             .Define("lepton_m", "Lep_M[0]")

                             //.Define("lepton", set_lorentz, {"Lep_Pt[0]", "Lep_Eta[0]", "Lep_Phi[0]", "Lep_M[0]"})
                             .Define("lepton", set_lorentz, {"lepton_pt", "lepton_eta", "lepton_phi", "lepton_m"})
                             .Define("nutrino", set_MET, {"MET_T1_Pt_nom","MET_T1_Phi_nom"})

                             .Define("sortjets", sortbtag, {"Jet_Pt_nom", "Jet_Eta_nom", "Jet_btagValue_nom"})
                             .Define("VecJets", vecJets, {"sortjets", "Jet_Pt_nom", "Jet_Eta_nom", "Jet_Phi_nom", "Jet_M_nom", "Jet_btagValue_nom"})

                             //*** categorize
                             //.Define("event_category", event_cat)
                             //.Define("jet_perm", jet_perm_cat, {"VecJets", "addbjet1", "addbjet2"})

                             .Define("jet1_pt", "Jet_Pt_nom[sortjets[0]]")
                             .Define("jet1_eta", "Jet_Eta_nom[sortjets[0]]")
                             .Define("jet1_phi", "Jet_Phi_nom[sortjets[0]]")
                             .Define("jet1_m", "Jet_M_nom[sortjets[0]]")
                             .Define("jet1_btag", "Jet_btagValue_nom[sortjets[0]]")
                             .Define("jet1_cvsb", "Jet_CvB_nom[sortjets[0]]")
                             .Define("jet1_cvsl", "Jet_CvL_nom[sortjets[0]]")

                             //*** jet2
                             .Define("jet2_pt", "Jet_Pt_nom[sortjets[1]]")
                             .Define("jet2_eta", "Jet_Eta_nom[sortjets[1]]")
                             .Define("jet2_phi", "Jet_Phi_nom[sortjets[1]]")
                             .Define("jet2_m", "Jet_M_nom[sortjets[1]]")
                             .Define("jet2_btag", "Jet_btagValue_nom[sortjets[1]]")
                             .Define("jet2_cvsb", "Jet_CvB_nom[sortjets[1]]")
                             .Define("jet2_cvsl", "Jet_CvL_nom[sortjets[1]]")

                             //*** jet3
                             .Define("jet3_pt", "Jet_Pt_nom[sortjets[2]]")
                             .Define("jet3_eta", "Jet_Eta_nom[sortjets[2]]")
                             .Define("jet3_phi", "Jet_Phi_nom[sortjets[2]]")
                             .Define("jet3_m", "Jet_M_nom[sortjets[2]]")
                             .Define("jet3_btag", "Jet_btagValue_nom[sortjets[2]]")
                             .Define("jet3_cvsb", "Jet_CvB_nom[sortjets[2]]")
                             .Define("jet3_cvsl", "Jet_CvL_nom[sortjets[2]]")

                             //*** jet4
                             .Define("jet4_pt", "Jet_Pt_nom[sortjets[3]]")
                             .Define("jet4_eta", "Jet_Eta_nom[sortjets[3]]")
                             .Define("jet4_phi", "Jet_Phi_nom[sortjets[3]]")
                             .Define("jet4_m", "Jet_M_nom[sortjets[3]]")
                             .Define("jet4_btag", "Jet_btagValue_nom[sortjets[3]]")
                             .Define("jet4_cvsb", "Jet_CvB_nom[sortjets[3]]")
                             .Define("jet4_cvsl", "Jet_CvL_nom[sortjets[3]]")

                             //*** TLorentzVector jets
                             .Define("jet1", set_lorentz, {"jet1_pt", "jet1_eta", "jet1_phi", "jet1_m"})
                             .Define("jet2", set_lorentz, {"jet2_pt", "jet2_eta", "jet2_phi", "jet2_m"})
                             .Define("jet3", set_lorentz, {"jet3_pt", "jet3_eta", "jet3_phi", "jet3_m"})
                             .Define("jet4", set_lorentz, {"jet4_pt", "jet4_eta", "jet4_phi", "jet4_m"})

                             //*** Pt of leptonically decaying W
                             .Define("nulep_pt", pt_sum, {"nutrino", "lepton"})

                             .Define("Ht", setHt, {"jet1", "jet2", "jet3", "jet4"})
                             .Define("St", setSt, {"jet1", "jet2", "jet3", "jet4", "nutrino", "lepton"})

                             //*** dR
                             .Define("dR12", deltaR, {"jet1", "jet2"})
                             .Define("dR13", deltaR, {"jet1", "jet3"})
                             .Define("dR14", deltaR, {"jet1", "jet4"})
                             .Define("dR23", deltaR, {"jet2", "jet3"})
                             .Define("dR24", deltaR, {"jet2", "jet4"})
                             .Define("dR34", deltaR, {"jet3", "jet4"})

                             .Define("dRlep1", deltaR, {"jet1", "lepton"})
                             .Define("dRlep2", deltaR, {"jet2", "lepton"})
                             .Define("dRlep3", deltaR, {"jet3", "lepton"})
                             .Define("dRlep4", deltaR, {"jet4", "lepton"})

                             .Define("dRnu1", deltaR, {"jet1", "nutrino"})
                             .Define("dRnu2", deltaR, {"jet2", "nutrino"})
                             .Define("dRnu3", deltaR, {"jet3", "nutrino"})
                             .Define("dRnu4", deltaR, {"jet4", "nutrino"})

                             .Define("dRnulep1", deltaR_nulj, {"nutrino", "lepton", "jet1"})
                             .Define("dRnulep2", deltaR_nulj, {"nutrino", "lepton", "jet2"})
                             .Define("dRnulep3", deltaR_nulj, {"nutrino", "lepton", "jet3"})
                             .Define("dRnulep4", deltaR_nulj, {"nutrino", "lepton", "jet4"})

                             .Define("dRnulep12", deltaR_nuljj, {"nutrino", "lepton", "jet1", "jet2"})
                             .Define("dRnulep13", deltaR_nuljj, {"nutrino", "lepton", "jet1", "jet3"})
                             .Define("dRnulep14", deltaR_nuljj, {"nutrino", "lepton", "jet1", "jet4"})
                             .Define("dRnulep23", deltaR_nuljj, {"nutrino", "lepton", "jet2", "jet3"})
                             .Define("dRnulep24", deltaR_nuljj, {"nutrino", "lepton", "jet2", "jet4"})
                             .Define("dRnulep34", deltaR_nuljj, {"nutrino", "lepton", "jet3", "jet4"})

                             //*** dEta
                             .Define("dEta12", deltaEta, {"jet1", "jet2"})
                             .Define("dEta13", deltaEta, {"jet1", "jet3"})
                             .Define("dEta14", deltaEta, {"jet1", "jet4"})
                             .Define("dEta23", deltaEta, {"jet2", "jet3"})
                             .Define("dEta24", deltaEta, {"jet2", "jet4"})
                             .Define("dEta34", deltaEta, {"jet3", "jet4"})

                             //*** dPhi
                             .Define("dPhi12", deltaPhi, {"jet1", "jet2"})
                             .Define("dPhi13", deltaPhi, {"jet1", "jet3"})
                             .Define("dPhi14", deltaPhi, {"jet1", "jet4"})
                             .Define("dPhi23", deltaPhi, {"jet2", "jet3"})
                             .Define("dPhi24", deltaPhi, {"jet2", "jet4"})
                             .Define("dPhi34", deltaPhi, {"jet3", "jet4"})
                             //*** invmass
                             .Define("invm12", invmass, {"jet1", "jet2"})
                             .Define("invm13", invmass, {"jet1", "jet3"})
                             .Define("invm14", invmass, {"jet1", "jet4"})
                             .Define("invm23", invmass, {"jet2", "jet3"})
                             .Define("invm24", invmass, {"jet2", "jet4"})
                             .Define("invm34", invmass, {"jet3", "jet4"})

                             .Define("invmlep1", invmass, {"jet1", "lepton"})
                             .Define("invmlep2", invmass, {"jet2", "lepton"})
                             .Define("invmlep3", invmass, {"jet3", "lepton"})
                             .Define("invmlep4", invmass, {"jet4", "lepton"})

                             .Define("invmnu1", invmass, {"jet1", "nutrino"})
                             .Define("invmnu2", invmass, {"jet2", "nutrino"})
                             .Define("invmnu3", invmass, {"jet3", "nutrino"})
                             .Define("invmnu4", invmass, {"jet4", "nutrino"})


;
    cout << endl << " good " << endl;

    df_bjet.Snapshot("tree", outputfile, {
//          // category
            //"event_category", "jet_perm",
            //*** Global var
            "ngoodjets", "nulep_pt", "St", "Ht", "nbjets_m", "ncjets_m", "lepton_Pt", "lepton_Eta", "lepton_Phi", "lepton_M", "MET_T1_Pt_nom", "MET_T1_Phi_nom",
            //*** low level var of jets
            "jet1_pt", "jet1_eta", "jet1_m", "jet1_btag", "jet1_cvsb", "jet1_cvsl", "dRlep1", "dRnu1", "dRnulep1", "invmlep1", "invmnu1",
            "jet2_pt", "jet2_eta", "jet2_m", "jet2_btag", "jet2_cvsb", "jet2_cvsl", "dRlep2", "dRnu2", "dRnulep2", "invmlep2", "invmnu2",
            "jet3_pt", "jet3_eta", "jet3_m", "jet3_btag", "jet3_cvsb", "jet3_cvsl", "dRlep3", "dRnu3", "dRnulep3", "invmlep3", "invmnu3",
            "jet4_pt", "jet4_eta", "jet4_m", "jet4_btag", "jet4_cvsb", "jet4_cvsl", "dRlep4", "dRnu4", "dRnulep4", "invmlep4", "invmnu4",

            //*** dR, dEta, dPhi, invariantmass of jets
            "dR12", "dR13", "dR14", "dR23", "dR24", "dR34",
            "dEta12", "dEta13", "dEta14", "dEta23", "dEta24", "dEta34",
            "dPhi12", "dPhi13", "dPhi14", "dPhi23", "dPhi24", "dPhi34",
            "invm12", "invm13", "invm14", "invm23", "invm24", "invm34",
            //*** dR, invariantmass between jet(s) & lep(MET)
            "dRnulep12", "dRnulep13", "dRnulep14", "dRnulep23", "dRnulep24", "dRnulep34",

            //"gen_dRbb", "gen_mbb"
            });
}
