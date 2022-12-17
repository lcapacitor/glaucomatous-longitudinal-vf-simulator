#!/usr/bin/env python3
'''
Simulation model for longitudinal Visual Field tests in glaucoma

This is part of the paper "A Data-driven Model for Simulating Longitudinal Visual Field Tests in Glaucoma"

Author: Yan Li

Version: v1.0

Date: 2022-12

License: MIT
'''

import os
import joblib
import string
import pickle
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from scipy import ndimage       #from scipy.ndimage import measurements # deprecated
from scipy import stats
from scipy import interpolate

from utils import vfData_Padding, data_vec_2_mat, show_sim_vf, grayscale_pattern
from constants import NORM_SLOPE, NORM_INTERCEPT, TD_WEIGHTS, LOC_24_2, EMPTY_INDEX_OD_72, MATRIX_784x54,\
                      GHT_PD_THRES_p05, GH_MAP_52, PROBA_WITH_DEFECTS, PROBA_NO_DEFECTS


class Longitudinal_VF_simulator:
    def __init__(self):
        super(Longitudinal_VF_simulator, self).__init__()

    def __get_empirical_prog_stats(self):
        progress_path = './data/cluster_progression_rate_distributions.pkl'
        noise_path    = './data/residual_distribution_and_noise_template.pkl'
        if not os.path.exists(progress_path):
            raise FileNotFoundError("No such file or directory:", progress_path)
        else:
            prog_rates_distributions = joblib.load(progress_path)
        if not os.path.exists(noise_path):
            raise FileNotFoundError("No such file or directory:", empirical_fpath)
        else:
            noise_dict = joblib.load(noise_path)
        return prog_rates_distributions, noise_dict

    def __vf_gray_plot_mat(self, vf):
        """
        # vf: array like with length of 52 or 54
        """
        # Prepare data
        cur_vf_mat = vfData_Padding(vf, 'OD', 50)
        cur_vf54   = np.delete(cur_vf_mat, EMPTY_INDEX_OD_72)
        vf_mat_28by28 = np.matmul(MATRIX_784x54, cur_vf54).reshape(28, 28)
        vf_gray_mat   = np.zeros((28*24, 28*24))
        for p_row in range(28):
            for p_col in range(28):
                idx   = p_row*28 + p_col
                row_s = p_row*24
                row_e = row_s+24
                col_s = p_col*24
                col_e = col_s+24
                interp_dB = vf_mat_28by28[p_row, p_col]
                im_block  = grayscale_pattern(interp_dB)
                vf_gray_mat[row_s:row_e, col_s:col_e] = im_block
        return vf_gray_mat

    def __find_cluster_defect(self, base_data, display=False):
        """
        # base_data: (52*3+2,)
        # 3 contiguous depressed points in either hemifield that exceed p<0.05 level
        """
        num_defect_points = 3
        base_vf  = base_data[:52]
        base_td  = base_data[52:52*2]
        base_pd  = base_data[52*2:52*3]
        base_age = base_data[52*3]
        base_md  = base_data[52*3+1]
        #--------------------------------------------------------
        # Find cluster of defects in baseline
        #--------------------------------------------------------
        base_defect_mat_p05 = vfData_Padding((base_pd < GHT_PD_THRES_p05).astype(int), 'OD', 0)
        base_vf_mat_bi_sup = base_defect_mat_p05[:4,:]
        base_vf_mat_bi_inf = base_defect_mat_p05[4:,:]
        feat_map_sup, num_sup = ndimage.label(base_vf_mat_bi_sup)       #measurements.label(base_vf_mat_bi_sup)
        feat_map_inf, num_inf = ndimage.label(base_vf_mat_bi_inf)       #measurements.label(base_vf_mat_bi_inf)
        areas_sup = ndimage.sum_labels(base_vf_mat_bi_sup, feat_map_sup, index=np.arange(feat_map_sup.max() + 1))  #measurements.sum(base_vf_mat_bi_sup, feat_map_sup, index=np.arange(feat_map_sup.max() + 1))
        areas_inf = ndimage.sum_labels(base_vf_mat_bi_inf, feat_map_inf, index=np.arange(feat_map_inf.max() + 1))  #measurements.sum(base_vf_mat_bi_inf, feat_map_inf, index=np.arange(feat_map_inf.max() + 1))
        hemi_f_c = max(np.max(areas_sup), np.max(areas_inf))
        #--------------------------------------------------------
        # Get all qualified defected points in each hemifield
        #--------------------------------------------------------
        thres_num_defect   = 1
        feat_map_list      = [feat_map_sup, feat_map_inf]
        feat_area_list     = [areas_sup, areas_inf]
        cluster_index_list = []
        for fi in range(len(feat_map_list)):
            cur_area = feat_area_list[fi]
            cur_feat_map = feat_map_list[fi]
            # If current hemifield contain clusters of defects
            if np.max(cur_area)>=thres_num_defect:
                labels = np.arange(cur_feat_map.max()+1)
                tar_label = labels[cur_area>=thres_num_defect]
                for c_lab in tar_label:
                    tar_index_2d = np.argwhere(cur_feat_map==c_lab)
                    tar_index_52 = []
                    for p in range(tar_index_2d.shape[0]):
                        idx72 = (tar_index_2d[p, 0] + 4*fi) * 9 + tar_index_2d[p, 1]
                        idx52 = LOC_24_2.index(idx72)
                        tar_index_52.append(idx52)
                    cluster_index_list.append(tar_index_52)
        #--------------------------------------------------------
        # Get the result of baseline defect
        #--------------------------------------------------------
        cluster_index_list = sorted(cluster_index_list, key=len, reverse=True)     # Sort nested list by length
        if len(cluster_index_list)==0:
            contain_base_scotoma = False
        elif len(cluster_index_list[0])>=num_defect_points:
            contain_base_scotoma= True
        else:
            contain_base_scotoma = False
        #--------------------------------------------------------
        # Visualize
        #--------------------------------------------------------
        if display>1:
            base_defect_map = np.zeros(52)
            num_cluster = len(cluster_index_list_sort)
            for i, cluster in enumerate(cluster_index_list_sort):
                base_defect_map[cluster] = num_cluster - i
            fontsize = 10
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            vf_gray_mat = self.__vf_gray_plot_mat(base_vf)
            ax[0].imshow(vf_gray_mat, cmap='gray')
            ax[0].set_title(f'Age={base_age:.1f}, MD={base_md:.1f} dB', fontsize=fontsize)
            ax[0].set_axis_off()
            ax[1].imshow(vfData_Padding(base_pd, 'OD', 10), cmap='gray')
            ax[1].set_title(f'Baseline PD', fontsize=fontsize)
            ax[1].set_axis_off()
            ax[2].imshow(vfData_Padding(base_defect_map, 'OD', -1), cmap='coolwarm')
            ax[2].set_title(f'Baseline defects (n={num_cluster})', fontsize=fontsize)
            ax[2].set_axis_off()
            plt.show()
        return cluster_index_list, contain_base_scotoma

    def __simulate_one_sequence(self, base_data, stats_params, sim_len, test_interval, min_gh_dist=10, progress_rate='random', progress_range=[10,5],
                               noise_model='template', sim_data_type='random', progress_cluster='multi-cluster', verbose=0):
        """
        Generate a sequence of VF tests given the baseline data
        base_data: numpy array (52*3+2,)
        """
        #--------------------------------------------------------
        # Prepare data and stats
        #--------------------------------------------------------
        base_vf  = base_data[:52]
        base_td  = base_data[52:52*2]
        base_pd  = base_data[52*2:52*3]
        base_age = base_data[52*3]
        base_md  = base_data[52*3+1]
        prog_ub, prog_lb = progress_range
        prog_rates_distributions, noise_dict = stats_params
        # Progression rate distributions
        cluster_prog_rate_fast_dict = prog_rates_distributions['cluster_progression_rates_leq5degree']
        cluster_prog_rate_slow_dict = prog_rates_distributions['cluster_progression_rates_5to10degree']
        # Residual distribuion and Noise templates
        residual_distribution = noise_dict['Residual distributions given true DLS']   #freq_residuals
        noise_probability_map = noise_dict['Noise probability template']
        residual_cmf = noise_dict['Residual CMF']
        residual_histogram_dict = {key: None for key in residual_distribution.keys()}
        ceiling = np.max(list(residual_distribution.keys()))
        for dls in residual_histogram_dict.keys():
            res_distr = np.around(residual_distribution[dls],2)
            unique, counts = np.unique(res_distr, return_counts=True)
            freqs = counts/np.sum(counts)
            residual_histogram_dict[dls] = np.vstack((unique, freqs))
        #--------------------------------------------------------
        # Find cluster of defects in baseline
        #--------------------------------------------------------
        cluster_index_list, contain_base_scotoma = self.__find_cluster_defect(base_data)
        #--------------------------------------------------------
        # Sort points in each cluster based on PD values in ascending order
        #--------------------------------------------------------
        cluster_index_list_all_points = sum(cluster_index_list, [])
        pd_cluster= base_pd[cluster_index_list_all_points]
        sort_idx  = np.argsort(pd_cluster, kind='mergesort')
        all_base_scotoma_index = [cluster_index_list_all_points[x] for x in sort_idx]
        all_base_points_index_sorted  = all_base_scotoma_index + [x for x in np.arange(len(base_pd)) if x not in all_base_scotoma_index]
        assert len(all_base_points_index_sorted)==len(base_pd)
        #----------------------------------------
        # Determine the number of progressions
        #----------------------------------------
        cutoff_p = 0.01
        if contain_base_scotoma:
            # Define this base_defect_pattern array for display's use
            base_defect_pattern = np.zeros(52)
            num_cluster = len(cluster_index_list)
            for i, cluster in enumerate(cluster_index_list):
                base_defect_pattern[cluster] = 1 #num_cluster - i
            cum_proba_num_prog_center = PROBA_WITH_DEFECTS.copy()
            tag = 'W/'
        else:
            base_defect_pattern = None
            cum_proba_num_prog_center = PROBA_NO_DEFECTS.copy()
            tag = 'W/O'
        # Adjust the CMF if only simulate progressing eyes
        if sim_data_type=='progress':   
            cum_proba_num_prog_center = cum_proba_num_prog_center[1:] / cum_proba_num_prog_center[1]
        rand = np.random.random()
        diff = cum_proba_num_prog_center - rand
        sim_prog_num = np.argmin(diff[diff>=0])
        cutoff_num = np.argmin(cum_proba_num_prog_center[(cum_proba_num_prog_center-cutoff_p)>=0])
        sim_prog_num = min(sim_prog_num, cutoff_num)
        #----------------------------------------
        if sim_data_type=='stable':
            sim_prog_num = 0
        if progress_cluster=='single-cluster':
            sim_prog_num = 1 if sim_prog_num>0 else 0
        if type(progress_cluster)==int:
            sim_prog_num = progress_cluster
        if isinstance(progress_cluster, (np.ndarray, list)):
            low, high = min(progress_cluster), max(progress_cluster)
            assert low<high
            sim_prog_num = random.randint(low, high)
        #----------------------------------------
        if verbose>0:
            print ('\n\n')
            print ('-'*50)
            print ('Base PD field')
            print (vfData_Padding(base_pd, 'OD', 50).astype(int))
            print (f'Eye {tag} baseline defects: sim-{sim_data_type}, {progress_cluster}\nNumber of simulated progressions: {sim_prog_num}, random number={rand}, cutoff num={cutoff_num}')
            print ('Contain baseline defects?', contain_base_scotoma)
            print ('Baseline defect clusters:', cluster_index_list)
            print ('Baseline defect points:', all_base_scotoma_index)
            print ('All baseline points:', all_base_points_index_sorted)
        #-----------------------------------------------------
        # If sim progressiom=0, then simulate stable eyes
        #-----------------------------------------------------
        if sim_prog_num==0:
            sim_type = 0
            prog_rates = -0.1 * np.ones(52) # age-related progression
            if verbose>0:
                print (f'\tSimulating stable follow-up tests...')
        #----------------------------------------------------
        # If sim_prog_num>=1, then simulate progressing eyes
        #----------------------------------------------------
        else:
            if verbose>0:
                print (f'\tStart searching for {sim_prog_num} progression centers...')
            sim_type = 1
            #-----------------------------------------
            # Get the pool of progression centers 
            #-----------------------------------------
            # For the eye with baseline defect, we select the cluster with lowest average DLS as the candidate progress cluster 
            cluster_mean_list = []
            for pt in all_base_points_index_sorted:
                cur_gh = GH_MAP_52[pt+1]
                cluster_index = [k-1 for k, v in GH_MAP_52.items() if abs(v-cur_gh)<=prog_ub]
                cluster_mean  = np.mean(base_pd[cluster_index])
                cluster_mean_list.append(cluster_mean)
            cluster_mean_sort_index = np.argsort(cluster_mean_list, kind='mergesort')
            candidate_progression_centers = [all_base_points_index_sorted[x] for x in cluster_mean_sort_index]
            assert len(set(candidate_progression_centers))==52
            if verbose>0:
                print('Center pool:', candidate_progression_centers)
            #-------- Previous version -------- 
            #sorted_base_pd_locs = np.argsort(base_pd, kind='mergesort')
            #candidate_progression_centers = all_base_scotoma_index + [x for x in sorted_base_pd_locs if x not in all_base_scotoma_index]
            #assert len(set(candidate_progression_centers))==52
            #-----------------------------------------
            # Interatively select eiligible progression clusters
            #-----------------------------------------
            prog_centers  = []
            prog_clusters = []
            fast_prog_points = []
            slow_prog_points = []
            fast_prog_rate = []
            slow_prog_rate = []
            #-----------------------------------------
            for ci, pc in enumerate(candidate_progression_centers):
                # If all cluster has been found then stop
                if len(prog_clusters)>=sim_prog_num:
                    if verbose>0:
                        print (f'\tAll {sim_prog_num} qualified progression clusters have been found.')
                    break
                #-----------------------------------------
                # Check if the current center is in the selected cluster
                center_gh    = GH_MAP_52[pc+1]
                selected_ghs = [ GH_MAP_52[x+1] for x in sum(prog_clusters, []) ]
                is_pass_gh_check = True
                for s_gh in selected_ghs:
                    cur_diff = abs(center_gh - s_gh)
                    if cur_diff<min_gh_dist:
                        is_pass_gh_check = False
                        break
                if not is_pass_gh_check:
                    if verbose>0:
                        print (f'\tSkip center loc:{pc}, reason: current center within {min_gh_dist} degree from previous centers')
                    continue                
                #-----------------------------------------
                # Check other conditions
                center_gh      = GH_MAP_52[pc+1]     # because GH_MAP_52 is 1-indexing
                fast_prog_locs = [k-1 for k, v in GH_MAP_52.items() if abs(v-center_gh)<=prog_lb]
                slow_prog_locs = [k-1 for k, v in GH_MAP_52.items() if abs(v-center_gh)<=prog_ub and abs(v-center_gh)>prog_lb and k-1!=pc]
                prog_locs      = fast_prog_locs + slow_prog_locs
                non_prog_locs  = [x-1 for x in list(GH_MAP_52.keys()) if x-1 not in fast_prog_locs and x-1 not in slow_prog_locs and (x-1)!=pc]
                assert len(prog_locs+non_prog_locs)==52
                #-----------------------------------------
                # Check mean sensitivity of the selected cluster
                prog_mean_val = np.mean(base_vf[prog_locs]) 
                thres_val     = abs(-1*(sim_len-1)*test_interval)
                if prog_mean_val<thres_val: 
                    if verbose>0:
                        print (f'\tSkip center loc:{pc}, reason: mean sensitivity={prog_mean_val:.2f}, threshold={thres_val:.2f}')
                    continue
                # Check number of points in current cluster
                if len(prog_locs)<3:
                    if verbose>0:
                        print (f'\tSkip center loc:{pc}, reason: Num_prog={len(prog_locs)}')
                    continue
                #-----------------------------------------
                # Found progression center!
                prog_centers.append(pc)
                prog_clusters.append(prog_locs)
                fast_prog_points.append(fast_prog_locs)
                slow_prog_points.append(slow_prog_locs)
                if verbose>0:
                    print (f'\tFound {len(prog_centers)}/{sim_prog_num} progression center loc :{pc}, mean sensitivity={prog_mean_val:.2f}, threshold={thres_val:.2f}')
                    print (f'\t\tProgression   cluster:{prog_locs}')
                    print (f'\t\tFast     progress loc:{fast_prog_locs}')
                    print (f'\t\tModerate progress loc:{slow_prog_locs}')
            # Check if result lists match
            assert len(prog_centers)==len(prog_clusters)==len(fast_prog_points)==len(slow_prog_points)
            if len(prog_centers)==0:
                if verbose>0:
                    print (f'\t[Exception] Cannot find any qualified progression clusters, simulate stable eye...')
                sim_type = 0
                prog_rates = -0.1 * np.ones(52)    # age-related progression
            else:
                if verbose>0 and ci==51 and len(prog_clusters)<sim_prog_num:
                    print (f'\t[Exception] Cannot find {sim_prog_num} qualified progression clusters, use the present {len(prog_clusters)} clusters')
                if progress_rate=='random':
                    prog_rates_list = []
                    for pi in range(len(prog_clusters)):
                        cur_prog_rate = np.zeros(52)
                        cur_c = prog_centers[pi]
                        cur_fast = fast_prog_points[pi]
                        cur_slow = slow_prog_points[pi]
                        for d, cur_rate_dict in enumerate([cluster_prog_rate_fast_dict, cluster_prog_rate_slow_dict]):
                            cur_rates= np.around(cur_rate_dict[cur_c], 1)
                            cur_bins = np.arange(np.min(cur_rates), np.max(cur_rates)+0.01, 0.1)
                            hist, _  = np.histogram(cur_rates, cur_bins, density=False)
                            proba    = hist / np.sum(hist)
                            sIdx     = np.random.choice(np.arange(len(proba)), size=1, replace=False, p=proba)[0]
                            selected_rate = cur_rates[sIdx]
                            if d==0:
                                cur_prog_rate[cur_fast] = selected_rate
                            if d==1:
                                cur_prog_rate[cur_slow] = selected_rate
                        prog_rates_list.append(cur_prog_rate)
                    if len(prog_rates_list)==1:
                        prog_rates = prog_rates_list[0]
                    else:
                        prog_rates_arr = np.array(prog_rates_list)
                        # Take element-wise minimum
                        #prog_rates = np.amin(prog_rates_arr, axis=0)
                        # Take the sum
                        prog_rates = np.sum(prog_rates_arr, axis=0)
                    prog_rates[prog_rates==0] = -0.1 # age-related progression
                else:
                    if isinstance(progress_rate, (list, np.ndarray)):
                        if len(progress_rate)==1:
                            fast_rate = progress_rate[0]
                            slow_rate = progress_rate[0]
                        else:
                            fast_rate = progress_rate[0]
                            slow_rate = progress_rate[1]
                    elif isinstance(progress_rate, (int, float)):
                        fast_rate = progress_rate
                        slow_rate = progress_rate
                    else:
                        raise ValueError("Invalid progress_rate value, supported type: array_like or float or int")
                    prog_rates = -0.1 * np.ones(52)
                    for pi in range(len(prog_clusters)):
                        fast_prog = fast_prog_points[pi]
                        mild_prog = slow_prog_points[pi]
                        prog_rates[fast_prog] += fast_rate
                        prog_rates[mild_prog] += slow_rate
        #----------------------------------------------------
        # Start simulating
        #----------------------------------------------------
        simulated_eye = np.zeros((sim_len, 52*3+2))
        simulated_eye[0,:] = base_data
        num_sim_fu  = sim_len - 1
        sim_fu_ages = base_age + np.arange(1, sim_len)*test_interval
        fu_years    = np.arange(1, sim_len)*test_interval
        fu_years    = fu_years.reshape(-1, 1)
        sim_changes = fu_years*prog_rates
        sim_fu_vfs  = base_vf + sim_changes
        sim_fu_vfs  = np.clip(sim_fu_vfs, 0, ceiling)
        #----------------------------------------------------
        # Select a noise template for each follow-up tests
        #if base_md<-6:
        #    base_group = 'mild'
        #elif base_md>=-6 and base_md<-12:
        #    base_group = 'moderate'
        #else:
        #    base_group = 'severe'
        base_group= 'all'
        noise_idx = np.random.choice(np.arange(noise_probability_map[base_group].shape[0]), size=num_sim_fu)
        noise_proba_map = noise_probability_map[base_group][noise_idx].squeeze()
        noise_proba_map = np.around(noise_proba_map, 2)
        step_size = 0.1
        hist_bins = np.arange(-ceiling, ceiling+step_size, step_size)
        #----------------------------------------------------
        # Sample noise using noise templates
        for ns in range(num_sim_fu):
            if noise_model=='template':
                cur_sim_vf = np.around(sim_fu_vfs[ns]).astype(int)
                cur_noise_pmap = noise_proba_map[ns]
                cmf_for_vf = residual_cmf[cur_sim_vf]
                sampled_noise_arr = np.array([interpolate.interp1d(list(cmf_for_vf[i])+[1], hist_bins, assume_sorted=True)(cur_noise_pmap[i]) for i in range(len(cur_sim_vf))])
                sim_fu_vfs[ns] += sampled_noise_arr
            elif noise_model=='independent':
                cur_sim_vf = np.around(sim_fu_vfs[ns]).astype(int)
                for sn, true_dls in enumerate(cur_sim_vf):
                    res_list, res_proba = residual_histogram_dict[true_dls]
                    res_idx = np.random.choice(np.arange(len(res_list)), size=1, replace=False, p=res_proba)
                    cur_res = res_list[res_idx]
                    sim_fu_vfs[ns, sn] = true_dls + cur_res
            else:
                raise ValueError(f"Invalid noise_model: {noise_model}, supported value: template or independent.")
        #----------------------------------------------------
        # Calculate TD PD and MD, then update to container
        sim_fu_vfs = np.clip(sim_fu_vfs, 0, ceiling)
        sim_normal_value = sim_fu_ages.reshape(-1,1)*NORM_SLOPE.reshape(1, -1) + NORM_INTERCEPT.reshape(1, -1)
        sim_fu_td = sim_fu_vfs - sim_normal_value
        sim_td_8  = np.percentile(sim_fu_td, 85, axis=1)
        sim_fu_pd = sim_fu_td - sim_td_8.reshape(-1, 1)
        sim_fu_md = np.average(sim_fu_td, weights=TD_WEIGHTS, axis=1)
        sim_fu_data = np.hstack(( sim_fu_vfs, sim_fu_td, sim_fu_pd, sim_fu_ages.reshape(-1, 1), sim_fu_md.reshape(-1, 1) ))
        simulated_eye[1:, :] = sim_fu_data
        #----------------------------------------------------
        if verbose>1 and sim_type==0:
            is_save = True if verbose==3 else False
            sim_series_vec = np.expand_dims(simulated_eye,0)
            sim_series_mat = data_vec_2_mat(sim_series_vec)
            cur_time_mili  = str(datetime.datetime.now().timestamp()).split('.')[1]
            show_sim_vf(vf=sim_series_mat, d_set_name=f'Stable_eye #{cur_time_mili}', d_type_name='VF', field_type=0, 
                        num_flu=sim_len, is_grayscale=True, is_save=is_save)
        if verbose>1 and sim_type==1:
            is_save = True if verbose==3 else False
            abs_rates=np.abs(prog_rates)
            _vf_mat  = vfData_Padding(abs_rates, 'OD', -1).reshape(8,9)
            if verbose>0:
                print ('Progression rates mat')
                print (_vf_mat)
            sim_series_vec = np.expand_dims(simulated_eye,0)
            sim_series_mat = data_vec_2_mat(sim_series_vec)
            cur_time_mili  = str(datetime.datetime.now().timestamp()).split('.')[1]
            path_name = f'sim_prog_clusters/Prog_eye #{cur_time_mili}, inital MD={round(base_md,1)}dB, prog_centers: {prog_centers}, '
            show_sim_vf(vf=sim_series_mat, d_set_name=path_name, d_type_name='VF', field_type=0, prog_pat=(base_pd, base_defect_pattern, _vf_mat, prog_centers), 
                        num_flu=sim_len, is_grayscale=True, is_save=is_save) 
        return simulated_eye, sim_type

    def process_baseline(self, data_list, min_init_md=-20, base_index=[0,1]):
        processed_data = []
        if type(data_list)!=list:
            raise TypeError("The input VF data must in a list")
        else:
            for data_eye in data_list:
                fu_index = [x for x in range(data_eye.shape[0]) if x not in base_index]
                age = np.mean(data_eye[base_index, 52*3], axis=0).squeeze()
                vf  = np.mean(data_eye[base_index, :52],  axis=0).squeeze()
                norm= NORM_SLOPE*age + NORM_INTERCEPT
                td  = vf - norm
                pdev= td - np.percentile(td, 85)
                md  = np.average(td, weights=TD_WEIGHTS)
                if md < min_init_md:
                    continue
                baseline_data= np.hstack((vf, td, pdev, [age, md]))
                assert baseline_data.shape[0]==52*3+2
                followUp_data = data_eye[fu_index, :]
                all_data_new  = np.vstack((baseline_data, followUp_data))
                if all_data_new.shape[0] < 10:
                    continue
                else:
                    processed_data.append(all_data_new)
        if len(processed_data)==0:
            raise ValueError("No eligible baseline test found")
        else:
            return processed_data

    def simulate(self, vf_data, sim_len, test_interval, min_gh_dist=10, progress_rate='random', progress_range=[10, 5],
                 noise_model='template', sim_data_type='random', progress_cluster='multi-cluster', verbose=0):
        """
        baseline_list: arrays (n, 52*3+2), where n is the number of eyes
        """
        # Define containers
        simulated_stable_data   = np.zeros((0, sim_len, 52*3+2))
        simulated_progress_data = np.zeros((0, sim_len, 52*3+2))
        # Get statistics
        if verbose>0:
            print('[INFO] Start simulating...')
        stats_params = self.__get_empirical_prog_stats()
        # Loop over each baseline
        for e in tqdm(range(len(vf_data)), leave=False):
            data_eye  = vf_data[e]
            base_data = data_eye[0]
            sim_one_eye, sim_type = self.__simulate_one_sequence(base_data, stats_params, sim_len, test_interval, min_gh_dist, progress_rate, 
                                                                progress_range, noise_model, sim_data_type, progress_cluster, verbose)
            if sim_type==0:
                simulated_stable_data = np.vstack((simulated_stable_data, np.expand_dims(sim_one_eye, axis=0)))
            else:
                simulated_progress_data = np.vstack((simulated_progress_data, np.expand_dims(sim_one_eye, axis=0)))
        if verbose>0:
            print (f'Total simulated progressing eyes: {simulated_progress_data.shape}, stable eyes: {simulated_stable_data.shape}')
        return simulated_stable_data, simulated_progress_data

    def visualize_var_rates(self, vf_data, repeat_per_eye, sim_len=15, test_interval=0.5, selected_eye=None, min_gh_dist=10, progress_rate=['random', 'random', 'random', 'random'], 
                            progress_cluster=['stable', 'multi-cluster', 'multi-cluster', 'multi-cluster'], progress_range=[10, 5], noise_model=None, condition_on=True, 
                            save_path=None, save_fig_obj=False):
        assert len(progress_rate)==len(progress_cluster)
        if noise_model is None:
            noise_model = ['template']*len(progress_cluster)
        num_sim_seq= len(progress_rate)
        num_cols   = 8
        num_rows   = 1 + num_sim_seq
        show_gap   = 1
        show_years = np.arange(0, num_cols, show_gap)
        show_index = [x-1 for x in range(1,sim_len+1) if (x-1)*test_interval in show_years]
        panel_names= list(string.ascii_lowercase)[:num_rows]
        fontsize   = 8
        if condition_on:
            slp_diff_thres = 0.1
        #===============================
        # Get statistics
        #===============================
        stats_params = self.__get_empirical_prog_stats()
        #===============================
        # Get date for selected eyes
        #===============================
        if selected_eye is not None:
            selected_vf_data = [vf_data[i] for i in selected_eye]
        else:
            selected_vf_data = vf_data.copy()
        #===============================
        # Start simulating for each selected eye
        for e in tqdm(range(len(selected_eye)), leave=False):
            real_sequence = selected_vf_data[e]
            real_base_vf  = real_sequence[0]
            real_ages_arr = real_sequence[:, 52*3]
            real_show_idx = []
            for t in show_years:
                tar_year = real_ages_arr[0] + t
                tar_index_sort = np.argsort(np.abs(real_ages_arr - tar_year))
                for tidx in tar_index_sort:
                    if tidx in real_show_idx:
                        continue
                    else:
                        real_show_idx.append(tidx)
                        break
            real_sequence_selected = real_sequence[real_show_idx]
            real_seq_md = real_sequence_selected[:, 52*3+1]
            real_seq_age= real_sequence_selected[:, 52*3]
            real_seq_md_slp,_,_,_,_ = stats.linregress(real_seq_age, real_seq_md)

            # Create VF sequences for display
            for repeat in tqdm(range(repeat_per_eye), leave=False):
                demo_data_list = []
                demo_data_list.append(real_sequence_selected)
                sim_prog_list = []
                sim_prog_md_slope_list = []

                for i in range(num_sim_seq):
                    cur_noise = noise_model[i]
                    cur_prog_rate = progress_rate[i]
                    cur_prog_type = progress_cluster[i]
                    # Simulate stable VF sequence
                    if cur_prog_type=='stable':
                        sim_stable_data, _ = self.__simulate_one_sequence(real_base_vf, stats_params, sim_len, test_interval, min_gh_dist, noise_model=cur_noise, sim_data_type='stable')
                        sim_stable_selected= sim_stable_data[show_index]
                        demo_data_list.append(sim_stable_selected)
                        # Display VF sequence satisfy conditions: 1. stabel prog_rate should > -0.25
                        if condition_on:
                            sim_stable_seq_md  = sim_stable_selected[:, 52*3+1]
                            sim_stable_seq_age = sim_stable_selected[:, 52*3]
                            sim_stable_seq_md_slp,_,_,_,_ = stats.linregress(sim_stable_seq_age, sim_stable_seq_md)
                            if not abs(sim_stable_seq_md_slp)<slp_diff_thres:
                                break
                    # Simulate progressing VF sequence
                    else:
                        sim_prog_data, _ = self.__simulate_one_sequence(real_base_vf, stats_params, sim_len, test_interval, min_gh_dist, cur_prog_rate,
                                                                        noise_model=cur_noise, sim_data_type='progress', progress_cluster=cur_prog_type)
                        sim_prog_data_selected = sim_prog_data[show_index]
                        if not condition_on:
                            demo_data_list.append(sim_prog_data_selected)
                        else:
                            sim_prog_age= sim_prog_data_selected[:, 52*3]
                            sim_prog_md = sim_prog_data_selected[:, 52*3+1]
                            sim_md_slp,_,_,_,_ = stats.linregress(sim_prog_age, sim_prog_md)
                            sim_prog_list.append(sim_prog_data_selected)
                            sim_prog_md_slope_list.append(sim_md_slp)
                if condition_on and len(sim_prog_md_slope_list)>0:
                    # Select desired MD slopes
                    if len(sim_prog_md_slope_list)==3:
                        r1, r2, r3 = sim_prog_md_slope_list 
                        if not ( (abs(r1-real_seq_md_slp)<slp_diff_thres) and (abs(r2-(-1.0))<slp_diff_thres) and (abs(r3-(-1.5))<slp_diff_thres) ):
                            continue
                        print(sim_stable_seq_md_slp, r1, r2, r3)
                    else:
                        pass
                    # Add progressing VF sequences to the demo list
                    demo_data_list += sim_prog_list
                if len(demo_data_list)<num_sim_seq+1:
                    continue
                #===============================
                # Plot
                #===============================
                fig, ax = plt.subplots(num_rows, num_cols, figsize=(10, 6), sharex=True, sharey=True)
                col_gap_base = 1/num_cols
                for r in range(num_rows):
                    data_seq = demo_data_list[r]
                    cur_vf = data_seq[:, :52]
                    cur_md = data_seq[:, 52*3+1]
                    cur_age= data_seq[:, 52*3]
                    md_slp, _,_,_,_ = stats.linregress(cur_age, cur_md)
                    for c in range(num_cols):
                        cur_vf_mat = self.__vf_gray_plot_mat(cur_vf[c])
                        ax[r, c].imshow(cur_vf_mat, cmap='gray')
                        ax[r, c].set_axis_off()
                        if r==0:
                            if c==0:
                                text = f'Baseline: MD={cur_md[0]:.1f}dB'
                            elif c==num_cols-1:
                                text = f'year {show_years[c]} MD={cur_md[-1]:.1f}dB'
                            else:
                                text = f'year {show_years[c]}'
                            fig.text(col_gap_base*(c+0.5), 0.97, text, ha='center', fontsize=fontsize)
                        if c==0:
                            bias = 0.005 if r<2 else 0.02
                            fig.text(0.01, 1-(1/num_rows)*(r+1)+bias, f'({panel_names[r]}): {md_slp:.1f} dB/year', va='center', fontsize=fontsize)
                plt.tight_layout()
                if save_path is None:
                    plt.show()
                else:
                    eye_folder = os.path.join(save_path, f'sim_eye_{selected_eye[e]}')
                    if not os.path.exists(eye_folder):
                        os.makedirs(eye_folder)
                    fname = f'eye_{selected_eye[e]}_{repeat}.png'
                    plt.savefig(os.path.join(eye_folder, fname), dpi=300)
                    fig_folder = os.path.join(save_path, f'sim_eye_{selected_eye[e]}/figure_object/')
                    if save_fig_obj:
                        if not os.path.exists(fig_folder):
                            os.makedirs(fig_folder)
                        fname_fig = f'eye_{selected_eye[e]}_{repeat}_fig_obj.pkl'
                        pickle.dump(fig, open(os.path.join(fig_folder, fname_fig), 'wb'))
                    plt.close()



if __name__ == '__main__':
    #==========================================================
    # Usage Examples:
    #==========================================================
    from utils import load_all_rt_by_eyes

    # Load and process Rotterdam and Washington datasets
    data_list_rt  = load_all_rt_by_eyes(pt_list=None, ceiling=35, min_len=10, cut_type=None, fu_year=5, md_range=None, verbose=1)
    print(f'[INFO] RT data contains {len(data_list_rt)} eligible eyes, {np.sum([x.shape[0] for x in data_list_rt])} tests')

    # Process baseline VF tests
    vf_simulator = Longitudinal_VF_simulator()
    vf_data_list = vf_simulator.process_baseline(data_list_rt)
    print(f'[INFO] Total eligible eyes {len(vf_data_list)}')

    # Simulate stable and progressing VF sequences
    simulated_stable_data, simulated_progress_data = vf_simulator.simulate(vf_data_list, sim_len=15, test_interval=0.5, verbose=0)
    print (simulated_stable_data.shape, simulated_progress_data.shape)
    
    # Visualize VF sequences for the given eye 
    selected_eye = [218]
    vf_simulator.visualize_var_rates(vf_data_list, repeat_per_eye=1000, selected_eye=selected_eye, progress_rate=[0,-1.5,-1.6,-2.5], progress_cluster=['stable',3,6,8])