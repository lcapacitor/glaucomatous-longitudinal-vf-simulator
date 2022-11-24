#!/usr/bin/env python3
'''Simulation model for longitudinal Visual Field tests in glaucoma

This is part of the work of "A Data-driven Model for Simulating Longitudinal Visual Field Tests in Glaucoma"

Author: Yan Li

Version: v1.0

Date: 2022-11

License: MIT
'''

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
from scipy import stats
from tqdm import tqdm
from constants import BS_INDEX_OD_54, LOC_24_2, EMPTY_INDEX_OD_72, EMPTY_INDEX_OD_72_BS, VF_GRAY_PATTERN, MATRIX_784x54


#===================== VF data processing =====================
def pad52_54(v52, side, pad):
    v54 = np.zeros(54)
    if side == 0 or side == 'OD':
        v54[BS_INDEX_OD_54] = pad
        v54[:25] = v52[:25]
        v54[26:34] = v52[25:33]
        v54[35:] = v52[33:]
    if side == 1 or side == 'OS':
        v54[BS_INDEX_OS_54] = pad
        v54[:19] = v52[:19]
        v54[20:28] = v52[19:27]
        v54[29:] = v52[27:]
    return v54

def vfData_Padding(vf, site, p):
    if len(vf) == 52:
        vf = pad52_54(vf, site, p)

    vfMat = p * np.ones((8, 9))
    if site == 'OD' or site == 0:
        vfMat[0, 3:7] = vf[:4]
        vfMat[1, 2:8] = vf[4:10]
        vfMat[2, 1:9] = vf[10:18]
        vfMat[3, :]   = vf[18:27]
        vfMat[4, :]   = vf[27:36]
        vfMat[5, 1:9] = vf[36:44]
        vfMat[6, 2:8] = vf[44:50]
        vfMat[7, 3:7] = vf[50:]
    if site == 'OS' or site == 1:
        vfMat[0, 2:6] = vf[:4]
        vfMat[1, 1:7] = vf[4:10]
        vfMat[2, :8]  = vf[10:18]
        vfMat[3, :]   = vf[18:27]
        vfMat[4, :]   = vf[27:36]
        vfMat[5, :8]  = vf[36:44]
        vfMat[6, 1:7] = vf[44:50]
        vfMat[7, 2:6] = vf[50:]
    return vfMat

def data_vec_2_mat(data_vec, vf_pad=-1, td_pad=35, num_f=3, num_nf=2):
    '''
    Convert data_vec: [n, length, 52*5+2] to
            data_mat: [n, length, features, 8, 9]
    '''
    d0, d1 = data_vec.shape[0], data_vec.shape[1]
    d2 = num_f+num_nf
    data_mat = np.zeros((d0, d1, d2, 8, 9))
    for i in range(d2):
        if i < num_f:
            pad = vf_pad if (i==0 or i>2) else td_pad
            feat_f = data_vec[:, :, i*52:(i+1)*52]
            for j in EMPTY_INDEX_OD_72_BS:
                feat_f = np.insert(feat_f, j, pad, axis=2)
            feat_f = feat_f.reshape(d0, d1, 8, 9)
            data_mat[:, :, i, :, :] = feat_f
        else:
            feat_nf = data_vec[:, :, num_f*52+(i-num_f)].reshape((d0, d1, 1, 1))
            feat_nf = feat_nf * np.ones((d0, d1, 8, 9))
            data_mat[:, :, i, :, :] = feat_nf
    return data_mat


#===================== For visualization =====================
def grayscale_pattern(interp_dB):
    if interp_dB==interp_dB:
        if interp_dB<=1:
            v = VF_GRAY_PATTERN['pattern_v0']
        elif interp_dB<=6:
            v = VF_GRAY_PATTERN['pattern_v1']
        elif interp_dB<=11:
            v = VF_GRAY_PATTERN['pattern_v6']
        elif interp_dB<=16:
            v = VF_GRAY_PATTERN['pattern_v11']
        elif interp_dB<=21:
            v = VF_GRAY_PATTERN['pattern_v16']
        elif interp_dB<=26:
            v = VF_GRAY_PATTERN['pattern_v21']
        elif interp_dB<=31:
            v = VF_GRAY_PATTERN['pattern_v26']
        else:
            v = VF_GRAY_PATTERN['pattern_v31']
        im = Image.fromarray(v.to_numpy().astype(bool))
        im = im.resize((24, 24), Image.NEAREST)
        assert im.mode == '1'
        im = im.convert("LA")
        im_arr = np.asarray(im)[:,:,0]
    else:
        im_arr = np.ones((24,24))*255
    return im_arr

def show_sim_vf(vf, d_set_name, d_type_name, field_type=0, prog_pat=None, 
                eye_index=None, num_flu=15, is_grayscale=True, is_save=False):
    field_names = ['VF', 'TD', 'PD']
    f = field_type
    vmin, vmax = 0, 35
    for eye in range(vf.shape[0]):
        if num_flu<=5:
            col, row, fs = num_flu, 1, 3.0
        elif num_flu<6:
            col, row, fs = 3, 2, 3.0
        elif num_flu<8:
            col, row, fs = 4, 2, 2.0
        elif num_flu<9:
            col, row, fs = 3, 3, 3.0
        elif num_flu<10:
            col, row, fs = 5, 2, 2.0
        elif num_flu<12:
            col, row, fs = 4, 3, 2.0
        elif num_flu==15 and prog_pat is None:
            col, row, fs = 5, 3, 2.0
        elif num_flu<16 and prog_pat is None:
            col, row, fs = 4, 4, 2.0
        elif num_flu<20:
            col, row, fs = 5, 4, 2.0
        elif num_flu<24:
            col, row, fs = 6, 4, 2.0
        eps_x, eps_y = 0.4, 0.25
        row += 1
        cmap_name = 'gray'
        prog_lb, prog_ub = 5, 10

        ages = vf[eye, :, 3, 0, 0].squeeze()
        mds  = vf[eye, :, 4, 0, 0].squeeze()
        md_slope, md_b, _, md_pval,_ = stats.linregress(ages, mds)
        assert len(ages)==len(mds)==num_flu
        font_size = 8 if is_save else 11

        fig = plt.figure(figsize=(fs*col, fs*row))
        gs = fig.add_gridspec(row, col)
        for r in range(row):
            for c in range(col):
                if r<row-1:
                    n = r*col + c
                    if prog_pat is None:
                        if n>num_flu-1:
                            break
                    else:
                        if n>num_flu+1:
                            break
                    ax = fig.add_subplot(gs[r, c])
                    if prog_pat is None:
                        if is_grayscale:
                            cur_fvec72 = vf[eye, n, f].flatten()
                            cur_fvec54 = np.delete(cur_fvec72, EMPTY_INDEX_OD_72)
                            vf_mat_28by28 = np.matmul(MATRIX_784x54, cur_fvec54).reshape(28, 28)
                            vf_gray_mat = np.zeros((28*24, 28*24))
                            for p_row in range(28):
                                for p_col in range(28):
                                    idx = p_row*28 + p_col
                                    row_s = p_row*24
                                    row_e = row_s+24
                                    col_s = p_col*24
                                    col_e = col_s+24
                                    interp_dB = vf_mat_28by28[p_row, p_col]
                                    im_block  = grayscale_pattern(interp_dB)
                                    vf_gray_mat[row_s:row_e, col_s:col_e] = im_block
                            im = ax.imshow(vf_gray_mat, cmap=cmap_name)
                            ax.set_axis_off()
                            ax.set_title(f'#{n+1} age:{vf[eye, n, 3, 0, 0]:.1f} MD:{vf[eye, n, 4, 0, 0]:.1f}', fontsize=font_size)
                        else:
                            cur_fmat = vf[eye, n, f]
                            if cmap_name == 'gray':
                                cur_fvec = cur_fmat.flatten()
                                cur_fvec[EMPTY_INDEX_OD_72_BS] = vmax
                                cur_fmat = cur_fvec.reshape(8,9)
                            im = ax.imshow(cur_fmat, cmap=cmap_name, vmin=vmin, vmax=vmax)
                            ax.set_axis_off()
                            ax.set_title(f'#{n+1} age:{vf[eye, n, 3, 0, 0]:.1f} MD:{vf[eye, n, 4, 0, 0]:.1f}', fontsize=font_size)
                            for y in range(8):
                                for x in range(9):
                                    idx = 9 * y + x
                                    if idx in EMPTY_INDEX_OD_72_BS:
                                        continue
                                    thres_val = int(round(vf[eye,n,f,y,x],0)) #
                                    fcolor = 'white' if thres_val<15 else 'black'
                                    ax.text(x-eps_x, y+eps_y, thres_val, fontsize=6, color=fcolor)
                    else:
                        base_pd, base_defect_pattern, cur_fmat, prog_center_list = prog_pat
                        if n==0:
                            base_pd_mat = vfData_Padding(base_pd.astype(int), 'OD', -36) if base_defect_pattern is None else vfData_Padding(-1*base_defect_pattern.astype(int), 'OD', 1)
                            im = ax.imshow(base_pd_mat, cmap='gray')
                            for y in range(8):
                                for x in range(9):
                                    idx = 9 * y + x
                                    if idx in EMPTY_INDEX_OD_72_BS:
                                        continue
                                    idx52 = LOC_24_2.index(idx)
                                    text  = str(int(np.around(base_pd[idx52], 0)))
                                    color = 'black'
                                    if base_defect_pattern is not None:
                                        based_defect_loc = np.argwhere(base_defect_pattern>0).squeeze()
                                        if idx52 in based_defect_loc:
                                            text = str(int(np.around(base_pd[idx52], 0)))
                                            color= 'white'
                                    ax.text(x-eps_x, y+eps_y, text, color=color, fontsize=5)
                            ax.set_axis_off()
                            ax.set_title('Baseline PD', size=font_size)
                        elif n==1:
                            cur_fmat[cur_fmat>0.1]=1
                            cur_fmat[cur_fmat==0.1]=0
                            im = ax.imshow(cur_fmat, cmap='coolwarm')
                            for y in range(8):
                                for x in range(9):
                                    idx = 9 * y + x
                                    if idx in EMPTY_INDEX_OD_72_BS:
                                        continue
                                    text = ''
                                    loc_24_2_arr = np.array(LOC_24_2)
                                    if idx in loc_24_2_arr[prog_center_list]:
                                        text = f'c{list(loc_24_2_arr[prog_center_list]).index(idx)+1}'
                                    ax.text(x-eps_x, y+eps_y, text, fontsize=6)
                            ax.set_axis_off()
                            ax.set_title('Progression pattern', size=font_size)
                        else:
                            if is_grayscale:
                                fn = n-2
                                cur_fvec72 = vf[eye, fn, f].flatten()
                                cur_fvec54 = np.delete(cur_fvec72, EMPTY_INDEX_OD_72)
                                vf_mat_28by28 = np.matmul(MATRIX_784x54, cur_fvec54).reshape(28, 28)
                                vf_gray_mat = np.zeros((28*24, 28*24))
                                for p_row in range(28):
                                    for p_col in range(28):
                                        idx = p_row*28 + p_col
                                        row_s = p_row*24
                                        row_e = row_s+24
                                        col_s = p_col*24
                                        col_e = col_s+24
                                        interp_dB = vf_mat_28by28[p_row, p_col]
                                        im_block  = grayscale_pattern(interp_dB)
                                        vf_gray_mat[row_s:row_e, col_s:col_e] = im_block
                                im = ax.imshow(vf_gray_mat, cmap=cmap_name)
                                ax.set_axis_off()
                                ax.set_title(f'#{fn+1} age:{vf[eye, fn, 3, 0, 0]:.1f} MD:{vf[eye, fn, 4, 0, 0]:.1f}', fontsize=font_size)
                            else:
                                fn = n-2
                                cur_fmat = vf[eye, fn, f]
                                if cmap_name == 'gray':
                                    cur_fvec = cur_fmat.flatten()
                                    cur_fvec[EMPTY_INDEX_OD_72_BS] = vmax
                                    cur_fmat = cur_fvec.reshape(8,9)
                                im = ax.imshow(cur_fmat, cmap=cmap_name, vmin=vmin, vmax=vmax)
                                ax.set_axis_off()
                                ax.set_title(f'#{fn+1} age:{vf[eye, fn, 3, 0, 0]:.1f} MD:{vf[eye, fn, 4, 0, 0]:.1f}', fontsize=font_size)
                                for y in range(8):
                                    for x in range(9):
                                        idx = 9 * y + x
                                        if idx in EMPTY_INDEX_OD_72_BS:
                                            continue
                                        thres_val = int(round(vf[eye,fn,f,y,x],1))
                                        fcolor = 'white' if thres_val<15 else 'black'
                                        ax.text(x-eps_x, y+eps_y, thres_val, fontsize=6, color=fcolor)
                else:
                    ax = fig.add_subplot(gs[r, :])
                    ax.scatter(ages, mds, c='tab:blue', marker='x', alpha=0.8)
                    ax.set_xlabel('Age (years)', fontsize=font_size)
                    ax.set_ylabel('MD (dB)',     fontsize=font_size)
                    ax.tick_params(axis='both',  labelsize=font_size )
                    ax.plot([ages[0], ages[-1]], [ages[0]*md_slope+md_b, ages[-1]*md_slope+md_b], ls='--', c='tab:orange', alpha=0.7)
                    ax.grid(True, linestyle='--')

        if prog_pat is None:
            save_path  = f'./figures/simulated/{d_set_name}'
            fig.suptitle('{}, MD slope: {:.2f} dB/year, p={:.2f}'
                         .format(d_set_name+'_'+d_type_name if d_set_name is None else d_set_name, md_slope, md_pval))
        else:
            save_folder, fig_title = d_set_name.split('/')
            save_path = f'./figures/simulated/{save_folder}'
            eye_sn = fig_title.split(',')[0].split('#')[1]
            _,_,_, prog_center_list = prog_pat
            if len(prog_center_list)>1:
                fig_title_= fig_title+'\nMD slope: {:.2f} dB/year, p={:.2f}'.format(md_slope, md_pval)
            else:
                fig_title_= fig_title+'MD slope: {:.2f} dB/year, p={:.2f}'.format(md_slope, md_pval)
            fig.suptitle(fig_title_, fontsize=10)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        if is_save:
            if prog_pat is None:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                out_tmp_name = d_set_name.split('/')[1] if '/' in d_set_name else d_set_name
                fname = 'Eye-{}_{}_{}.png'.format(eye, out_tmp_name, d_type_name)
            else:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                fname = f'Eye-{eye_sn}.png'
            plt.savefig(os.path.join(save_path, fname), dpi=300)
            plt.close()
        else:
            plt.show()


#===================== Data loaders =====================
def load_all_rt_by_eyes(pt_list=None, ceiling=None, min_len=10, cut_type=None, fu_year=None, md_range=None, verbose=1):
    # Read CSV files
    df_vfPoints = pd.read_csv('data/Rotterdam/VFPoints.csv')
    df_vfIndex  = pd.read_csv('data/Rotterdam/VisualFields.csv')
    df_patient  = pd.read_csv('data/Rotterdam/Patients.csv')

    # Find all OS fileds then flip to OD
    OS_fields_list = df_vfIndex.loc[df_vfIndex['SITE']=='OS', 'FIELD_ID'].tolist()
    df_vfPoints.loc[df_vfPoints['FIELD_ID'].isin(OS_fields_list), 'X'] = -1 * df_vfPoints.loc[df_vfPoints['FIELD_ID'].isin(OS_fields_list), 'X'].values
    # Sort data frames
    df_vfIndex  = df_vfIndex.sort_values(by=['STUDY_ID', 'SITE', 'AGE'], ascending=[True, True, True])
    df_vfPoints = df_vfPoints.sort_values(by=['FIELD_ID', 'Y', 'X'], ascending=[True, False, True])

    out_data_list = []

    if pt_list is None:
        pt_list = df_vfIndex['STUDY_ID'].unique().tolist()
        if verbose>0:
            print ('[INFO] load_data_RT: Number of patients in RT:', len(pt_list))

    # Loop over each patient's VFs
    for pt in tqdm(pt_list, leave=False):
        for side in ['OS', 'OD']:
            df_vfs = df_vfIndex.loc[(df_vfIndex['STUDY_ID'] == pt) & (df_vfIndex['SITE'] == side)]
            num_tests = df_vfs.shape[0]

            # Skip the eye if less than min_len tests
            if num_tests<min_len:
                continue

            # Select different number of tests per eye based on cut_type
            if cut_type=='front':
                df_vfs = df_vfs.iloc[:min_len]
            elif cut_type=='back':
                df_vfs = df_vfs.iloc[-min_len:]
            elif cut_type is None:
                pass
            else:
                raise ValueError(f"Unknown cut_type: {cut_type}, allow value: 'front' or 'back'")

            # Get all avaiable tests
            filed_list = df_vfs['FIELD_ID'].tolist()
            age_arr = df_vfs.loc[df_vfs['FIELD_ID'].isin(filed_list), 'AGE'].values / 365.25
            md_arr  = df_vfs.loc[df_vfs['FIELD_ID'].isin(filed_list), 'MD'].values
            assert age_arr.shape==md_arr.shape

            # Select eyes satisfy follow-up years
            if fu_year is not None:
                fu_duration = age_arr[-1] - age_arr[0]
                if fu_duration<fu_year:
                    continue

            # Select eyes satisfy MD slope
            if md_range is not None:
                md_slope, md_intercept ,_,_,_ = stats.linregress(age_arr, md_arr)
                if type(md_range)==float or type(md_range)==int:
                    if md_slope<-abs(md_range) or md_slope>abs(md_range):
                        continue
                if type(md_range)==list:
                    if md_slope<md_range[0] or md_slope>md_range[1]:
                        continue

            vf_54 = df_vfPoints.loc[df_vfPoints['FIELD_ID'].isin(filed_list), 'THRESHOLD'].values.astype(float)
            if ceiling is not None:
                vf_54 = np.clip(vf_54, 0, ceiling)
            else:
                vf_54 = np.clip(vf_54, 0, None)
            td_54 = df_vfPoints.loc[df_vfPoints['FIELD_ID'].isin(filed_list), 'TOTAL_DEVIATION'].values.astype(float)
            assert len(vf_54)==len(td_54)

            vf_54_arr = vf_54.reshape(-1, 54)
            td_54_arr = td_54.reshape(-1, 54)
            vf_52_arr = np.delete(vf_54_arr, BS_INDEX_OD_54, axis=1)
            td_52_arr = np.delete(td_54_arr, BS_INDEX_OD_54, axis=1)
            td_85_pct = np.percentile(td_52_arr, 85, axis=1).reshape(-1, 1)
            pd_52_arr = td_52_arr - td_85_pct
            assert vf_52_arr.shape==td_52_arr.shape==pd_52_arr.shape

            # Construct output data
            data_eye = np.hstack((vf_52_arr, td_52_arr, pd_52_arr, age_arr.reshape(-1, 1), md_arr.reshape(-1, 1)))
            #print (pt, side, data_eye.shape)
            assert np.sum(td_52_arr)==np.sum(td_52_arr)     #no NaN
            assert data_eye.shape[0]>=min_len
            out_data_list.append(data_eye)
    return out_data_list


def load_all_uw_by_eyes(pt_list=None, ceiling=None, min_len=10, cut_type=None, fu_year=None, md_range=None, verbose=1):
    # Read csv file
    df_uw = pd.read_csv('./data/UW/VF_Data.csv')
    # Get patient list
    if pt_list is None:
        pt_list = df_uw['PatID'].unique().tolist()
        if verbose>0:
            print ('[INFO] load_data_UW: Number of patients in UW:', len(pt_list))

    # Initialize the VF data container
    out_data_list = []
    # Get target colunm names
    vf_col_names = [f'Sens_{x}' for x in range(1, 55) if x not in [26, 35]]
    td_col_names = [f'TD_{x}'   for x in range(1, 55) if x not in [26, 35]]
    pd_col_names = [f'PD_{x}'   for x in range(1, 55) if x not in [26, 35]]

    for pt in tqdm(pt_list, leave=False):
        for side in ['Right', 'Left']:
            df_vfs = df_uw.loc[(df_uw['PatID']==pt) & (df_uw['Eye']==side)]
            num_tests = df_vfs.shape[0]

            # Skip the eye if less than min_len tests
            if num_tests<min_len:
                continue

            # Select different number of tests per eye based on cut_type
            if cut_type=='front':
                df_vfs = df_vfs.iloc[:min_len]
            elif cut_type=='back':
                df_vfs = df_vfs.iloc[-min_len:]
            elif cut_type is None:
                pass
            else:
                raise ValueError(f"Unknown cut_type: {cut_type}, allow value: 'front' or 'back'")

            # Get data arrays
            age_arr= df_vfs['Age'].values
            vf_arr = df_vfs[vf_col_names].values
            vf_arr = np.clip(vf_arr, 0, ceiling)
            td_arr = df_vfs[td_col_names].values
            pd_arr = df_vfs[pd_col_names].values
            md_arr = np.average(td_arr, axis=1)
            assert vf_arr.shape==td_arr.shape==pd_arr.shape
            assert age_arr.shape==md_arr.shape

            # Select eyes satisfy follow-up years
            if fu_year is not None:
                fu_duration = age_arr[-1] - age_arr[0]
                if fu_duration<fu_year:
                    continue

            # Select eyes satisfy MD slope
            if md_range is not None:
                md_slope, _,_,_,_ = stats.linregress(age_arr, md_arr)
                if type(md_range)==float or type(md_range)==int:
                    if md_slope<-abs(md_range) or md_slope>abs(md_range):
                        continue
                if type(md_range)==list:
                    if md_slope<md_range[0] or md_slope>md_range[1]:
                        continue

            # Construct output data
            data_eye = np.hstack((vf_arr, td_arr, pd_arr, age_arr.reshape(-1, 1), md_arr.reshape(-1, 1)))
            #print (pt, side, data_eye.shape)
            assert np.sum(td_arr)==np.sum(td_arr)     #no NaN
            assert data_eye.shape[0]>=min_len
            out_data_list.append(data_eye)
    return out_data_list