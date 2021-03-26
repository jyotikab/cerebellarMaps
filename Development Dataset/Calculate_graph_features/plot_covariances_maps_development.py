import os
import glob
import numpy as np
import pylab as pl
import scipy.io as sio
# for_Jyotika.m
from copy import copy, deepcopy
import pickle
import matplotlib.cm as cm
import pdb
import h5py
import pandas as pd
import bct
from collections import Counter 
import matplotlib.cm as cm
import analyze as anal
import sys
import graph_prop_funcs_analyze as graph_anal


data_dir = "/home/bahuguna/Work/Isope_data/Isope_data_cerebellar_maps/"
data_target_dir = "/home/bahuguna/Work/Isope_data/"
fig_target_dir = "/home/bahuguna/Work/Isope_data/Isope_data_analysis/figs/"

development = "DEVELOPMENT"
days = os.listdir(data_dir+development)
data_2d = pickle.load(open(data_target_dir+"data_2d_maps_days.pickle","rb"))
data = pd.read_csv(data_target_dir+"meta_data_days.csv")
cov_2d_dict = deepcopy(data_2d)
gammas = np.round(np.arange(0.0,1.5,0.17),2)

graph_properties = dict()

gamma_re_arrange = 0.34
gamma_re_arrange_ind = np.where(gammas == gamma_re_arrange)[0][0]

# zones
zone_names = ["B_contra","AX_contra","Alat_contra","Amed_contra","Amed_ipsi","Alat_ipsi","AX_ipsi","B_ipsi"]
#zone_lims = [(-233,-133),(-133,-108),(-108,-58),(-58,0),(0,50),(50,100),(100,125),(125,330)]
zone_lims = [(-233,-133),(-133,-108),(-108,-58),(-58,0),(0,50),(50,100),(100,125),(125,235)]

zone_binning = np.arange(-235,235,5)

mat_type = "norm"

#for mat_type in ["norm", "threshold"]:
participation_pos_all = []
participation_neg_all = []
mdz_all = []

for st in days:
    data_slice = data.loc[data["days"]==st]
    num_subfigs = len(data_slice)
    fig = pl.figure(figsize=(20,20))
    fig1 = pl.figure(figsize=(20,20))
    fig2 = pl.figure(figsize=(20,20))
    rows = int(np.round(np.sqrt(num_subfigs)))
    cols = rows
    print(st)
    #print(num_subfigs)
    #print(rows,cols)
    if rows*cols < num_subfigs:
        rows = rows+1
    subfig_hands = []
    subfig_hands1 = []
    subfig_hands2 = []
    
    graph_properties[st] = dict()
    graph_properties[st]["modularity"] = dict()
    graph_properties[st]["indices"] = []
    graph_properties[st]["names"] = []
    fig.suptitle("Days:"+st,fontsize=15,fontweight='bold') 
    fig1.suptitle("Days:"+st,fontsize=15,fontweight='bold') 
    fig2.suptitle("Days:"+st+" rearranged, gamma = "+str(gamma_re_arrange),fontsize=15,fontweight='bold') 
    
    for i,(rn,cn) in enumerate(zip(data_slice["rat_num"],data_slice["cell_num"])):

        #if st == "LC" and i == 8:
        #    pdb.set_trace()

        subfig_hands.append(fig.add_subplot(rows,cols,i+1))
        subfig_hands1.append(fig1.add_subplot(rows,cols,i+1))
        subfig_hands2.append(fig2.add_subplot(rows,cols,i+1))
        graph_properties[st]["modularity"][i] = dict()
        #pdb.set_trace()
        graph_properties[st]["names"].append(str(rn)+"-"+str(cn))
        if str(cn) in list(data_2d[st][str(rn)].keys()):


            cov_2d = np.cov(data_2d[st][str(rn)][str(cn)]["map"].T)
            tot_amplitude = np.nansum(data_2d[st][str(rn)][str(cn)]["map_nz"])
            avg_amplitude = np.nanmean(data_2d[st][str(rn)][str(cn)]["map_nz"])
            nz_dim = np.shape(data_2d[st][str(rn)][str(cn)]["map"])
            active_sites = (len(np.where(data_2d[st][str(rn)][str(cn)]["map"]  > 3.0)[0])/(nz_dim[0]*nz_dim[1]))*100 # % active sites

            corr_2d = np.corrcoef(data_2d[st][str(rn)][str(cn)]["map"].T,data_2d[st][str(rn)][str(cn)]["map"].T)[:len(cov_2d),:len(cov_2d)]
            ind_nan = np.where(np.isnan(corr_2d)==True)
            #print(ind_nan[0])
            #if st == "ES" and rn == "050617":
            #    pdb.set_trace()
            if len(ind_nan[0]) > 0:
                ind_nonan = np.where(np.isnan(corr_2d)==False)
                xlim = (np.min(np.unique(ind_nonan[0])),np.max(np.unique(ind_nonan[0])))
                ylim = (np.min(np.unique(ind_nonan[1])),np.max(np.unique(ind_nonan[1])))
                corr_2d = corr_2d[xlim[0]:xlim[1],ylim[0]:ylim[1]]


            cov_2d_dict[st][str(rn)][str(cn)] = dict()
            cov_2d_dict[st][str(rn)][str(cn)]["cov"] = cov_2d #+np.nanmin(cov_2d)  # To remove -ve values so that louvain community algorithm can be used
            if mat_type == "norm":
                corr_2d = corr_2d
            #    corr_2d = corr_2d - np.eye(np.shape(corr_2d)[0],np.shape(corr_2d)[1]) # remove the self loops 
            elif mat_type == "threshold":
                thresh = 0.4
                corr_2d = anal.binarize(thresh,corr_2d)
                #corr_2d = corr_2d - np.eye(np.shape(corr_2d)[0],np.shape(corr_2d)[1])

            cov_2d_dict[st][str(rn)][str(cn)]["corr"] = corr_2d #+np.nanmin(corr_2d)
            
            # Find modularity index
            gammas,num_mods_cov, mod_index_cov,ci_list_cov = graph_anal.calc_modularity(cov_2d)
            _,num_mods_corr, mod_index_corr,ci_list_corr = graph_anal.calc_modularity(corr_2d)
           
            #pdb.set_trace()
            #binned_pos = np.histogram(data_2d[st][rn][str(cn)]["pos_centered"],bins=zone_binning)


            diversity_pos = []
            diversity_neg = []
            node_strengths = []
            node_strengths_ipsi_contra = []
            gateway_coef_pos = []
            gateway_coef_neg = []
            zscore = []
            for ci in ci_list_corr:
                hpos,hneg = graph_anal.calc_diversity_coeff(corr_2d,ci)
                diversity_pos.append(hpos)
                diversity_neg.append(hneg)

                gpos,gneg = graph_anal.calc_gateway_coef_sign(corr_2d,ci)
                gateway_coef_pos.append(gpos)
                gateway_coef_neg.append(gneg)

                zs = graph_anal.calc_module_degree_zscore(corr_2d,ci,True,False)
                zscore.append(zs)
                #rc = anal.calc_rich_club_wu(ci) # Rich club also gave nan
                #rich_club.append(rc)
            # independent of gammas
            mdz_all.append(zscore)
            clus_coef_pos, clus_coef_neg = graph_anal.calc_clustering_coef(corr_2d)
            part_pos, part_neg = graph_anal.calc_participation_coef_sign(corr_2d,ci_list_corr,False,True)
            participation_pos_all.append(part_pos)
            participation_neg_all.append(part_neg)

            # re arranging the correlation matroices
            list_nodes = [ bct.modularity.ci2ls(x1) for x1 in ci_list_corr]
            spos,sneg,vpos,vneg = graph_anal.calc_strengths_und_sign(corr_2d) # nodal strengths of positive, nodal strengths of negative, total positive weight, total negative weight
            node_strengths.append((np.median(spos),np.median(sneg),vpos,vneg))
            node_strengths_ipsi_contra.append((spos,sneg,vpos,vneg))

            trans = graph_anal.calc_transitivity(corr_2d)


            re_arranged_corr = graph_anal.get_re_arranged_matrix(ci_list_corr[gamma_re_arrange_ind],corr_2d) 
            #pdb.set_trace()
            distances = anal.calc_proxy_within_between_module_distance(list_nodes,gammas)
            #pdb.set_trace()
            loc_assort_pos, loc_assort_neg = graph_anal.calc_local_assortativity_sign(corr_2d)

            #graph_properties[st]["modularity"][str(i)+"_cov"] = (mod_index_cov,num_mods_cov)
            graph_properties[st]["modularity"][i]["cov"] = (mod_index_cov,num_mods_cov)
            #graph_properties[st]["modularity"][str(i)+"_corr"] = (mod_index_corr,num_mods_corr)
            graph_properties[st]["modularity"][i]["corr"] = (mod_index_corr,num_mods_corr)
            graph_properties[st]["modularity"][i]["rearranged_corr"] = re_arranged_corr 
            #graph_properties[st]["modularity"][str(i)+"_norm"] = np.linalg.norm(corr_2d) 
            graph_properties[st]["modularity"][i]["norm"] = np.linalg.norm(corr_2d) 
            graph_properties[st]["modularity"][i]["total_amplitude"] = np.abs(tot_amplitude)*0.01 # pA
            graph_properties[st]["modularity"][i]["average_amplitude"] = np.abs(avg_amplitude)*0.01 # pA
            graph_properties[st]["modularity"][i]["percentage_active_sites"] = active_sites


            graph_properties[st]["modularity"][i]["distances"] = distances
            graph_properties[st]["modularity"][i]["clustering_coef_whole"] = (np.median(clus_coef_pos),np.median(clus_coef_neg))
           
            # align node numbers with position in the binned_pos
            if len(ind_nan[0]) > 0:
                ind_zone_bins = np.digitize(data_2d[st][str(rn)][str(cn)]["pos_centered"][xlim[0]:xlim[1]],bins=zone_binning)
                graph_properties[st]["modularity"][i]["ind_zone_bins_node_num_mapping"] = (ind_zone_bins,np.arange(xlim[0],xlim[1]))
            else:
                ind_zone_bins = np.digitize(data_2d[st][str(rn)][str(cn)]["pos_centered"],bins=zone_binning)
                graph_properties[st]["modularity"][i]["ind_zone_bins_node_num_mapping"] = (ind_zone_bins,np.arange(0,np.shape(data_2d[st][str(rn)][str(cn)]["map"])[1]))
            graph_properties[st]["modularity"][i]["clustering_coef_pos_zone"] = clus_coef_pos
            graph_properties[st]["modularity"][i]["clustering_coef_neg_zone"] = clus_coef_neg
            graph_properties[st]["modularity"][i]["participation_pos_zone"] = part_pos
            graph_properties[st]["modularity"][i]["participation_neg_zone"] = part_neg
            graph_properties[st]["modularity"][i]["gateway_coef_pos_zone"] = gateway_coef_pos
            graph_properties[st]["modularity"][i]["gateway_coef_neg_zone"] = gateway_coef_neg
            graph_properties[st]["modularity"][i]["diversity_pos_zone"] = diversity_pos
            graph_properties[st]["modularity"][i]["diversity_neg_zone"] = diversity_neg
            graph_properties[st]["modularity"][i]["module_degree_zscore_zone"] = zscore
            graph_properties[st]["modularity"][i]["median_strengths_pos_zone"] = spos
            graph_properties[st]["modularity"][i]["median_strengths_neg_zone"] = sneg
            graph_properties[st]["modularity"][i]["local_assortativity_pos_whole_zone"] = loc_assort_pos







             


            if len(data_2d[st][str(rn)][str(cn)]["ind_ipsi"]) > 0:
                clus_coef_pos_ipsi = clus_coef_pos[np.min(data_2d[st][str(rn)][str(cn)]["ind_ipsi"]):np.max(data_2d[st][str(rn)][str(cn)]["ind_ipsi"])]
                clus_coef_neg_ipsi = clus_coef_neg[np.min(data_2d[st][str(rn)][str(cn)]["ind_ipsi"]):np.max(data_2d[st][str(rn)][str(cn)]["ind_ipsi"])]
                
                graph_properties[st]["modularity"][i]["clustering_coef_ipsi"] = (np.median(clus_coef_pos_ipsi),np.median(clus_coef_neg_ipsi))

                


            if len(data_2d[st][str(rn)][str(cn)]["ind_contra"]) > 0:
                clus_coef_pos_contra = clus_coef_pos[np.min(data_2d[st][str(rn)][str(cn)]["ind_contra"]):np.max(data_2d[st][str(rn)][str(cn)]["ind_contra"])]
                clus_coef_neg_contra = clus_coef_neg[np.min(data_2d[st][str(rn)][str(cn)]["ind_contra"]):np.max(data_2d[st][str(rn)][str(cn)]["ind_contra"])]
                
                graph_properties[st]["modularity"][i]["clustering_coef_contra"] = (np.median(clus_coef_pos_contra),np.median(clus_coef_neg_contra))


            graph_properties[st]["modularity"][i]["participation_whole"] = (np.median(part_pos,axis=1),np.median(part_neg,axis=1))

            if len(data_2d[st][str(rn)][str(cn)]["ind_ipsi"]) > 0:
                part_pos_ipsi = np.array(part_pos)[:,np.min(data_2d[st][str(rn)][str(cn)]["ind_ipsi"]):np.max(data_2d[st][str(rn)][str(cn)]["ind_ipsi"])]
                part_neg_ipsi = np.array(part_neg)[:,np.min(data_2d[st][str(rn)][str(cn)]["ind_ipsi"]):np.max(data_2d[st][str(rn)][str(cn)]["ind_ipsi"])]
                graph_properties[st]["modularity"][i]["participation_ipsi"] = (np.median(part_pos_ipsi,axis=1),np.median(part_neg_ipsi,axis=1))
            if len(data_2d[st][str(rn)][str(cn)]["ind_contra"]) > 0:
                part_pos_contra = np.array(part_pos)[:,np.min(data_2d[st][str(rn)][str(cn)]["ind_contra"]):np.max(data_2d[st][str(rn)][str(cn)]["ind_contra"])]
                part_neg_contra = np.array(part_neg)[:,np.min(data_2d[st][str(rn)][str(cn)]["ind_contra"]):np.max(data_2d[st][str(rn)][str(cn)]["ind_contra"])]
                graph_properties[st]["modularity"][i]["participation_contra"] = (np.median(part_pos_contra,axis=1),np.median(part_neg_contra,axis=1))




            graph_properties[st]["modularity"][i]["diversity_whole"] = (np.median(diversity_pos,axis=1),np.median(diversity_neg,axis=1))
            if len(data_2d[st][str(rn)][str(cn)]["ind_ipsi"]) > 0:
                div_pos_ipsi = np.array(diversity_pos)[:,np.min(data_2d[st][str(rn)][str(cn)]["ind_ipsi"]):np.max(data_2d[st][str(rn)][str(cn)]["ind_ipsi"])]
                div_neg_ipsi = np.array(diversity_neg)[:,np.min(data_2d[st][str(rn)][str(cn)]["ind_ipsi"]):np.max(data_2d[st][str(rn)][str(cn)]["ind_ipsi"])]
                graph_properties[st]["modularity"][i]["diversity_ipsi"] = (np.median(div_pos_ipsi,axis=1),np.median(div_neg_ipsi,axis=1))
            if len(data_2d[st][str(rn)][str(cn)]["ind_contra"]) > 0:
                div_pos_contra = np.array(diversity_pos)[:,np.min(data_2d[st][str(rn)][str(cn)]["ind_contra"]):np.max(data_2d[st][str(rn)][str(cn)]["ind_contra"])]
                div_neg_contra = np.array(diversity_neg)[:,np.min(data_2d[st][str(rn)][str(cn)]["ind_contra"]):np.max(data_2d[st][str(rn)][str(cn)]["ind_contra"])]
                graph_properties[st]["modularity"][i]["diversity_contra"] = (np.median(div_pos_contra,axis=1),np.median(div_neg_contra,axis=1))
                

            graph_properties[st]["modularity"][i]["gateway_coef_whole"] = (np.median(gateway_coef_pos,axis=1),np.median(gateway_coef_neg,axis=1))
            if len(data_2d[st][str(rn)][str(cn)]["ind_ipsi"]) > 0:
                gateway_coef_pos_ipsi = np.array(gateway_coef_pos)[:,np.min(data_2d[st][str(rn)][str(cn)]["ind_ipsi"]):np.max(data_2d[st][str(rn)][str(cn)]["ind_ipsi"])]
                gateway_coef_neg_ipsi = np.array(gateway_coef_neg)[:,np.min(data_2d[st][str(rn)][str(cn)]["ind_ipsi"]):np.max(data_2d[st][str(rn)][str(cn)]["ind_ipsi"])]
                graph_properties[st]["modularity"][i]["gateway_coef_ipsi"] = (np.median(gateway_coef_pos_ipsi,axis=1),np.median(gateway_coef_neg_ipsi,axis=1))

            if len(data_2d[st][str(rn)][str(cn)]["ind_contra"]) > 0:
                gateway_coef_pos_contra = np.array(gateway_coef_pos)[:,np.min(data_2d[st][str(rn)][str(cn)]["ind_contra"]):np.max(data_2d[st][str(rn)][str(cn)]["ind_contra"])]
                gateway_coef_neg_contra = np.array(gateway_coef_neg)[:,np.min(data_2d[st][str(rn)][str(cn)]["ind_contra"]):np.max(data_2d[st][str(rn)][str(cn)]["ind_contra"])]
                graph_properties[st]["modularity"][i]["gateway_coef_contra"] = (np.median(gateway_coef_pos_contra,axis=1),np.median(gateway_coef_neg_contra,axis=1))




            #graph_properties[st]["modularity"][i]["local_assortativity"] = (np.median(loc_assort_pos),np.median(loc_assort_neg))
            # loc_assort_neg gives back nan 
            graph_properties[st]["modularity"][i]["local_assortativity_whole"] = (np.median(loc_assort_pos))
            if len(data_2d[st][str(rn)][str(cn)]["ind_ipsi"]) > 0:
                lim1 = np.min([len(loc_assort_pos),np.min(data_2d[st][str(rn)][str(cn)]["ind_ipsi"])])
                lim2 = np.min([len(loc_assort_pos),np.max(data_2d[st][str(rn)][str(cn)]["ind_ipsi"])])
                graph_properties[st]["modularity"][i]["local_assortativity_ipsi"] = (np.median(loc_assort_pos[lim1:lim2]))
            if len(data_2d[st][str(rn)][str(cn)]["ind_contra"]) > 0:
                lim1 = np.min([len(loc_assort_pos),np.min(data_2d[st][str(rn)][str(cn)]["ind_contra"])])
                lim2 = np.min([len(loc_assort_pos),np.max(data_2d[st][str(rn)][str(cn)]["ind_contra"])])
                graph_properties[st]["modularity"][i]["local_assortativity_contra"] = (np.median(loc_assort_pos[lim1:lim2]))
            graph_properties[st]["modularity"][i]["node_strengths"] = node_strengths



            graph_properties[st]["modularity"][i]["module_degree_zscore_whole"] = np.median(zscore,axis=1)
            if len(data_2d[st][str(rn)][str(cn)]["ind_ipsi"]) > 0:
                zscore_ipsi = np.array(zscore)[:,np.min(data_2d[st][str(rn)][str(cn)]["ind_ipsi"]):np.max(data_2d[st][str(rn)][str(cn)]["ind_ipsi"])]
                graph_properties[st]["modularity"][i]["module_degree_zscore_ipsi"] = np.median(zscore_ipsi,axis=1)


            if len(data_2d[st][str(rn)][str(cn)]["ind_contra"]) > 0:
                zscore_contra = np.array(zscore)[:,np.min(data_2d[st][str(rn)][str(cn)]["ind_contra"]):np.max(data_2d[st][str(rn)][str(cn)]["ind_contra"])]
                graph_properties[st]["modularity"][i]["module_degree_zscore_contra"] = np.median(zscore_contra,axis=1)



            graph_properties[st]["modularity"][i]["transitivity"] = trans
            #graph_properties[st]["modularity"][i]["global_efficiency"] = anal.calc_efficiency_wei(corr_2d,False)
            #graph_properties[st]["names"][i] = rn+"-"+cn
            #graph_properties[st]["modularity"][i]["local_efficiency"] = anal.calc_efficiency_wei(corr_2d,True)
              
            graph_properties[st]["indices"].append(i)

            vmin = np.nanmin(cov_2d)/2.
            vmax = np.nanmax(cov_2d)/2.
            subfig_hands[-1].pcolor(cov_2d,cmap=cm.hot,vmin=vmin,vmax=vmax)

            vmin = np.nanmin(corr_2d)/2.
            vmax = np.nanmax(corr_2d)/2.
            print(vmin,vmax)
            #subfig_hands1[-1].pcolor(corr_2d,cmap=cm.hot,vmin=vmin,vmax=vmax)
            subfig_hands1[-1].pcolor(corr_2d,cmap=cm.coolwarm,vmin=vmin,vmax=vmax)
            subfig_hands2[-1].pcolor(re_arranged_corr,cmap=cm.coolwarm,vmin=vmin,vmax=vmax)
           
            subfig_hands[-1].set_aspect('equal')
            subfig_hands1[-1].set_aspect('equal')
            subfig_hands2[-1].set_aspect('equal')


            #subfig_hands[-1].set_aspect(5)
        subfig_hands[-1].set_title("rat num:"+str(rn)+",cell num:"+str(cn),fontsize=12,fontweight='bold')
        subfig_hands1[-1].set_title("rat num:"+str(rn)+",cell num:"+str(cn),fontsize=12,fontweight='bold')
        subfig_hands2[-1].set_title("rat num:"+str(rn)+",cell num:"+str(cn),fontsize=12,fontweight='bold')
        #subfig_hands[-1].set_aspect('equal')
        if i < (num_subfigs-2):
            subfig_hands[-1].set_xticklabels([])
            subfig_hands1[-1].set_xticklabels([])
            subfig_hands2[-1].set_xticklabels([])

    graph_properties[st]["gammas"] = gammas


    fig.subplots_adjust(left = 0.05,right=0.96,wspace=0.2,hspace=0.2,bottom=0.06,top=0.95)
    fig1.subplots_adjust(left = 0.05,right=0.96,wspace=0.2,hspace=0.2,bottom=0.06,top=0.95)
    fig2.subplots_adjust(left = 0.05,right=0.96,wspace=0.2,hspace=0.2,bottom=0.06,top=0.95)
    fig.savefig(fig_target_dir+"cov_maps_"+st+"_"+mat_type+".png")
    fig1.savefig(fig_target_dir+"corr_maps_"+st+"_"+mat_type+".png")
    fig2.savefig(fig_target_dir+"corr_maps_rearranged_"+st+"_"+mat_type+".png")




fig = pl.figure(figsize=(20,20))
t1 = fig.add_subplot(131)
t2 = fig.add_subplot(132)
t3 = fig.add_subplot(133)

t1.hist(np.hstack(np.hstack(participation_pos_all)),bins=np.linspace(0,1,10),density=True)
t1.set_title("Participation positive",fontsize=15,fontweight='bold')

t2.hist(np.hstack(np.hstack(participation_neg_all)),bins=np.linspace(0,1,10),density=True)
t2.set_title("Participation negative",fontsize=15,fontweight='bold')

t3.hist(np.hstack(np.hstack(mdz_all)),bins=np.linspace(-2,8,10),density=True)
t3.set_title("Module degree zscore",fontsize=15,fontweight='bold')

fig.savefig(fig_target_dir+"Distributions_PZ_days_"+mat_type+".png")

pickle.dump(cov_2d_dict,open(data_dir+"covariance_maps_days_"+mat_type+".pickle","wb"))
pickle.dump(graph_properties,open(data_dir+"graph_properties_days_"+mat_type+".pickle","wb"))




