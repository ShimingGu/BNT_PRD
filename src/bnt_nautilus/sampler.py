from scipy.stats import norm,multivariate_normal
from nautilus import Prior,Sampler
import time
import os
from pathlib import Path
import numpy as np
from .core import bnt_core,redshift_select_2,ell_select
import pyccl as ccl
from scipy.ndimage import gaussian_filter as gf
from scipy.stats import norm
from datetime import datetime as dt
import sys
import subprocess as sub
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
os.environ["OMP_NUM_THREADS"] = "1"

# The original analysis used paths tied to its HPC checkout. Keep all
# run-time products and data configurable, while defaulting to this checkout.
project_dir = Path(__file__).resolve().parents[2]
data_dir = Path(os.environ.get('BNT_DATA_DIR', project_dir / 'data'))
output_dir = Path(os.environ.get('BNT_OUTPUT_DIR', project_dir / 'outputs'))
output_dir.mkdir(parents=True, exist_ok=True)
checkpoint_path = str(output_dir / 'checkpoints') + os.sep
Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
folder = str(output_dir) + os.sep
chainlim = 5e5#5e5

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print(__name__)
print('rank = '+str(rank),flush=True)
print(sys.argv,flush=True)

#try:
if 1 > 0:
    if __name__ == '__main__':
        if len(sys.argv) < 2:
            raise SystemExit('Usage: python -m bnt_nautilus.sampler MODE')
        mode = sys.argv[-1]
        print(mode,flush=True)
    else:
        mode = None
    mode = comm.bcast(mode, root=0)
    if len(mode) != 6 or not mode.isdecimal():
        raise ValueError('MODE must contain exactly six digits, e.g. 000000')
    mode_likelihood = int(mode[0:2])
    mode_scale = int(mode[2:4])
    mode_fiducial = int(mode[4:6])
    print(mode_likelihood,mode_scale,mode_fiducial)
    
name_label = ''

#
# Define the Mode for the Likelihood
#

if mode_likelihood%2 == 1:
    bnt_switch = False
    lklhd_print_1 = '# Conventional C_l likelihood'
    name_label = name_label + 'noBNT_'
elif mode_likelihood%2 == 0:
    bnt_switch = True
    lklhd_print_1 = '# BNT-Transformed C_l likelihood'
    name_label = name_label + 'BNT_'
if mode_likelihood > 0 and mode_likelihood <= 2:
    IA_switch = True;TATT_switch = False;m_switch=False;dz_switch=False;dz_correlation=False
    lklhd_print_2 = '# with NLA IA, without m, without dz'
    name_label = name_label + 'IA_nom_nodz_';KL_dir = str(data_dir / 'KL_new') + os.sep
elif mode_likelihood > 2 and mode_likelihood <= 4:
    IA_switch = True;TATT_switch = False;m_switch=True;dz_switch=False;dz_correlation=False
    lklhd_print_2 = '# with NLA IA, with m, without dz'
    name_label = name_label + 'IA_m_nodz_';KL_dir = str(data_dir / 'KL_new') + os.sep
elif mode_likelihood > 4 and mode_likelihood <= 6:
    IA_switch = True;TATT_switch = False;m_switch=True;dz_switch=True;dz_correlation=False
    lklhd_print_2 = '# with NLA IA, with m, with dz'
    name_label = name_label + 'IA_m_dz_';KL_dir = str(data_dir / 'KL_new') + os.sep
elif mode_likelihood > 6 and mode_likelihood <= 8:
    IA_switch = True;TATT_switch = False;m_switch=True;dz_switch=True;dz_correlation=True
    lklhd_print_2 = '# with NLA IA, with m, with correlated dz'
    name_label = name_label + 'IA_m_cdz_';KL_dir = str(data_dir / 'KL_new') + os.sep
if mode_likelihood > 10 and mode_likelihood <= 12:
    IA_switch = True;TATT_switch = True;m_switch=True;dz_switch=False
    lklhd_print_2 = '# with TATT IA, without m, without dz'
    name_label = name_label + 'TATT_nom_nodz_';KL_dir = str(data_dir / 'KL_new') + os.sep
elif mode_likelihood > 12 and mode_likelihood <= 14:
    IA_switch = True;TATT_switch = True;m_switch=True;dz_switch=False
    lklhd_print_2 = '# with TATT IA, with m, without dz'
    name_label = name_label + 'TATT_m_nodz_';KL_dir = str(data_dir / 'KL_new') + os.sep
elif mode_likelihood > 14 and mode_likelihood <= 16:
    IA_switch = True;TATT_switch = True;m_switch=True;dz_switch=True
    lklhd_print_2 = '# with TATT IA, with m, with dz'
    name_label = name_label + 'TATT_m_dz_';KL_dir = str(data_dir / 'KL_new') + os.sep
elif mode_likelihood > 16 and mode_likelihood <= 18:
    IA_switch = True;TATT_switch = True;m_switch=True;dz_switch=True;dz_correlation=True
    lklhd_print_2 = '# with TATT IA, with m, with correlated dz'
    name_label = name_label + 'TATT_m_cdz_';KL_dir = str(data_dir / 'KL_new') + os.sep
elif mode_likelihood > 26 and mode_likelihood <= 28:
    IA_switch = True;TATT_switch = False;m_switch=True;dz_switch=True;dz_correlation=True
    lklhd_print_2 = '# with NLA IA, with m, with correlated dz, KiDS ell'
    name_label = name_label + 'IA_m_cdz_';KL_dir = str(data_dir / 'KL_KiDS') + os.sep
elif mode_likelihood > 80 and mode_likelihood <= 82:
    IA_switch = True;TATT_switch = False;m_switch=False;dz_switch=False
    lklhd_print_2 = '# with IA, without m, without dz, test';KL_dir = str(data_dir / 'KL_new') + os.sep
    name_label = name_label + 'IA_nom_nodz_short_'; chainlim = 1000
elif mode_likelihood > 90 and mode_likelihood <= 92:
    IA_switch = True;TATT_switch = True;m_switch=False;dz_switch=False
    lklhd_print_2 = '# with TATT IA, without m, without dz, test';KL_dir = str(data_dir / 'KL_new') + os.sep
    name_label = name_label + 'IA_nom_nodz_short_'; chainlim = 1000
elif mode_likelihood > 96 and mode_likelihood <= 98:
    IA_switch = True;TATT_switch = True;m_switch=True;dz_switch=True;dz_correlation=True
    lklhd_print_2 = '# with TATT IA, with m, with correlated dz, test';KL_dir = str(data_dir / 'KL_new') + os.sep
    name_label = name_label + 'IA_m_cdz_short_'; chainlim = 1000
    
# IA_switch = False # if to turn off IA completely
# folder = folder + 'noIA/

nzfile = KL_dir+'KiDS_Legacy3.nz'
cataloguer_nb = np.load(KL_dir+'Cataloguer_noBNT.npy') #needs new
cataloguer_bn = np.load(KL_dir+'Cataloguer_BNT.npy')
    
#
# Define the Mode for the Scale Cut
#
    
if mode_scale in (0,10,20,30,40,50,60,70,80,90):
    k_llm = 0.0025;k_obj = 100;tfd = np.inf
    scale_print = '# no k_max'
    name_label = name_label + 'k100_'
    if mode_scale == 30:
        #tmp_nb = redshift_select(cataloguer_nb,'auto')
        #tmp_bn = redshift_select(cataloguer_bn,'auto')
        tmp_nb = redshift_select_2(cataloguer_bn,keep_diagonal=[1],remove_bin=None,inverse=False)
        tmp_bn = redshift_select_2(cataloguer_bn,keep_diagonal=[1],remove_bin=None,inverse=False)
        name_label = name_label + 'tomo_diag1_'
    elif mode_scale == 40:
        tmp_nb = redshift_select_2(cataloguer_bn,keep_diagonal=[1,2],remove_bin=None,inverse=False)
        tmp_bn = redshift_select_2(cataloguer_bn,keep_diagonal=[1,2],remove_bin=None,inverse=False)
        name_label = name_label + 'tomo_diag12_'
    elif mode_scale == 50:
        tmp_nb = ell_select(cataloguer_nb,ell_min=100,ell_max=3000)
        tmp_bn = ell_select(cataloguer_bn,ell_min=100,ell_max=3000)
        name_label = name_label + 'tomo_ellmin100max3000_'
    elif mode_scale == 60:
        tmp_nb = ell_select(cataloguer_nb,ell_min=100,ell_max=1500)
        tmp_bn = ell_select(cataloguer_bn,ell_min=100,ell_max=1500)
        name_label = name_label + 'tomo_ellmin100max1500_'
    elif mode_scale == 70:
        tmp_nb = redshift_select_2(cataloguer_nb,keep_diagonal=None,remove_bin=[1],inverse=False)
        tmp_bn = redshift_select_2(cataloguer_bn,keep_diagonal=None,remove_bin=[1],inverse=False)
        name_label = name_label + 'tomo_not1'
    elif mode_scale == 80:
        tmp_nb = redshift_select_2(cataloguer_nb,keep_diagonal=None,remove_bin=[3,4,5,6],inverse=False)
        tmp_bn = redshift_select_2(cataloguer_bn,keep_diagonal=None,remove_bin=[3,4,5,6],inverse=False)
        name_label = name_label + 'tomo_t12'
    elif mode_scale == 90:
        tmp_nb = redshift_select_2(cataloguer_nb,keep_diagonal=None,remove_bin=[1,2],inverse=False)
        tmp_bn = redshift_select_2(cataloguer_bn,keep_diagonal=None,remove_bin=[1,2],inverse=False)
        name_label = name_label + 'tomo_t3456'
else:
    if mode_scale in (1,4,7,51,54,57):
        k_llm = 0.0025
        k_obj = 0.1;scale_print = '# k_max = 0.1, '
        name_label = name_label + 'k0.1_'
    elif mode_scale in (2,5,8,52,55,58):
        k_llm = 0.0025
        k_obj = 1./3.;scale_print = '# k_max = 0.33, '
        name_label = name_label + 'k0.33_'
    elif mode_scale in (3,6,9,53,56,59):
        k_llm = 0.0025
        k_obj = 1.0;scale_print = '# k_max = 1.0, '
        name_label = name_label + 'k1.0_'
    elif mode_scale in (11,14,17):
        k_llm = 0.1/3
        k_obj = 0.1;scale_print = '# k_max = 0.1, '
        name_label = name_label + 'k_0.033_0.1_'
    elif mode_scale in (12,15,18):
        k_llm = 0.1
        k_obj = 1./3.;scale_print = '# k_max = 0.33, '
        name_label = name_label + 'k_0.1_0.33_'
    elif mode_scale in (13,16,19):
        k_llm = 1.0/3
        k_obj = 1.0;scale_print = '# k_max = 1.0, '
        name_label = name_label + 'k_0.33_1.0_'
    elif mode_scale in (21,24,27,61,64,67):
        k_llm = 0.1/3.
        k_obj = 1./3.;scale_print = '# k_max = 0.33, '
        name_label = name_label + 'k_0.033_0.3_'
    elif mode_scale in (22,25,28,62,65,68):
        k_llm = 0.1
        k_obj = 1.0;scale_print = '# k_max = 1.0, '
        name_label = name_label + 'k_0.1_1.0_'
    elif mode_scale in (23,26,29,63,66,69):
        k_llm = 1.0/3
        k_obj = 10.0/3;scale_print = '# k_max = 3.33, '
        name_label = name_label + 'k_0.33_3.3_'
    if mode_scale in (1,2,3,11,12,13,21,22,23,51,52,53,61,62,63):
        tfd = 0.1;scale_print += 'TFD = 0.1'
        name_label = name_label + 'T0.1_'
    elif mode_scale in (4,5,6,14,15,16,24,25,26,54,55,56,64,65,66):
        tfd = 0.02;scale_print += 'TFD = 0.02'
        name_label = name_label + 'T0.02_'
    elif mode_scale in (7,8,9,17,18,19,27,28,29,57,58,59,67,68,69):
        tfd = 0.004;scale_print += 'TFD = 0.004'
        name_label = name_label + 'T0.004_'
    if mode_scale > 50 and mode_scale < 70:
        tmp_nb = ell_select(cataloguer_nb,ell_min=100,ell_max=1500)
        tmp_bn = ell_select(cataloguer_bn,ell_min=100,ell_max=1500)
        name_label = name_label + 'tomo_ellmin100_'
    
#
# Define the Mode for the Fiducial Data Vector
#

KL_dir2 = KL_dir

if mode_fiducial == 0:
    fidcl_print = '# HMcode 2016'
    theoretical_fid = True;As_switch=False
    AIAfid = 0.0; A2fid = 0.0; bTAfid = 0.0
    NL_code = 'camb';NL_recipe = 'mead2020_feedback';name_label += 'fid_Mead_'
    MG_switch = False;IA_B1 = 0
elif mode_fiducial == 1:
    fidcl_print = '# HMcode 2016 As'
    theoretical_fid = True;As_switch=True
    AIAfid = 0.0; A2fid = 0.0; bTAfid = 0.0
    NL_code = 'camb';NL_recipe = 'mead2020_feedback';name_label += 'fid_MeadAs_'
    MG_switch = False;IA_B1 = 0
elif mode_fiducial == 2:
    fidcl_print = '# Real Data'
    theoretical_fid = False;As_switch=False
    AIAfid = 0.0; A2fid = 0.0; bTAfid = 0.0
    NL_code = 'camb';NL_recipe = 'mead2020_feedback';name_label += 'fid_KiDS_'
    MG_switch = False;IA_B1 = 0
elif mode_fiducial == 3:
    fidcl_print = '# Real Data As'
    theoretical_fid = False;As_switch=True
    AIAfid = 0.0; A2fid = 0.0; bTAfid = 0.0
    NL_code = 'camb';NL_recipe = 'mead2020_feedback';name_label += 'fid_KiDSAs_'
    MG_switch = False;IA_B1 = 0
elif mode_fiducial == 5:
    fidcl_print = '# HMcode 2016'
    theoretical_fid = True;As_switch=False
    AIAfid = 0.0; A2fid = 0.0; bTAfid = 0.0
    NL_code = 'camb';NL_recipe = 'mead2020_feedback';name_label += 'fid_Meadoldcovmat_'
    MG_switch = False;IA_B1 = 0;KL_dir2 = str(data_dir) + os.sep
    

#
# Define the Basic Setup
#
    
mu_dz = np.array([-2.6,1.4,-0.2,0.8,-1.1,5.4])*1e-2
std0_dz = np.array([0.2,0.1,0.2,0.1,0.2,0.4])*1e-2
std_dz = np.sqrt(std0_dz**2 + (1e-2)**2)

c12 = -0.09;
c13 = 0.05;c23 = 0.19
c14 = 0.02;c24 = -0.19;c34 = -0.35
c15 = 0.00;c25 = -0.11;c35 = -0.20;c45 = 0.14
c16 = -0.00;c26 = 0.07;c36 = -0.22;c46 = 0.28;c56 = -0.02
R_dz = np.array([
    [1.0, c12, c13, c14, c15, c16],
    [c12, 1.0, c23, c24, c25, c26],
    [c13, c23, 1.0, c34, c35, c36],
    [c14, c24, c34, 1.0, c45, c46],
    [c15, c25, c35, c45, 1.0, c56],
    [c16, c26, c36, c46, c56, 1.0]
])
cov_dz = np.diag(std_dz) @ R_dz @ np.diag(std_dz)
L_dz = np.linalg.cholesky(cov_dz)

mu_m = np.array([-2.3,-1.6,-1.1,2.0,3.0,4.5])*1e-2
std_m = np.array([0.6,0.6,0.7,0.7,0.8,0.9])*1e-2
nz_dt = np.loadtxt(nzfile)
lz = nz_dt.shape[1]-1
Cl_cov = np.load(KL_dir2+'Cov_pCl_Additive.npy')
bell_arr = np.load(KL_dir+'ell_pCl.npy')

cell_arr_geom0 = np.geomspace(2,50,10)[:-1]
cell_arr_geom1 = np.geomspace(50,6200,25)
cell_arr_geom = np.append(cell_arr_geom0,cell_arr_geom1)
#ell_arr = np.sort(ell_arr_geom)
cell_arr = np.sort(np.append([0],cell_arr_geom))

if __name__ == '__main__':
    print ('Ready to Compute a BNT Core Module \n',flush=True)

# Paused here, below need more working

#
# Compute the fiducial datavec with and without the scale cut
#

bnt_std = bnt_core(nz_dt,ell_arr=bell_arr,pCl_dir=KL_dir,
                   NL_code=NL_code,NL_recipe=NL_recipe,
                   omega_c = 0.153,S_8=0.797,Omega_h=True,
                   A_IA=AIAfid,A_IA2=A2fid,IA_B1=IA_B1,b_TA=bTAfid,
                   dz1 = mu_dz[0],dz2 = mu_dz[1],
                   dz3 = mu_dz[2],dz4 = mu_dz[3],
                   dz5 = mu_dz[4],dz6 = mu_dz[5],
                   m1 = mu_m[0],m2 = mu_m[1],
                   m3 = mu_m[2],m4 = mu_m[3],
                   m5 = mu_m[4],m6 = mu_m[5],
                   dzmode='additive',MG_switch=False,m_bias=True,
                   kmin=0.0025,kmax=100)
bnt_std.Cl_vec();Cl_std = 1.0*bnt_std.Cl_deorg
bnt_std.aCl_vec_reorg();aCl_std = 1.0*bnt_std.aCl_reorg
bnt_std.ell_arr = cell_arr
bnt_std.pCl_vec();pCl_std = 1.0*bnt_std.pCl_deorg
bpCl_std = bnt_std.Cl_vector_to_bnt(pCl_std)

aCl_cov = bnt_std.covmat_to_bnt_ell(Cl_cov)

bnt_fid = bnt_core(nz_dt,ell_arr=bell_arr,pCl_dir=KL_dir,
                   NL_code=NL_code,NL_recipe=NL_recipe,
                   omega_c = 0.153,S_8=0.797,Omega_h=True,
                   A_IA=AIAfid,A_IA2=A2fid,IA_B1=IA_B1,b_TA=bTAfid,
                   dz1 = mu_dz[0],dz2 = mu_dz[1],
                   dz3 = mu_dz[2],dz4 = mu_dz[3],
                   dz5 = mu_dz[4],dz6 = mu_dz[5],
                   m1 = mu_m[0],m2 = mu_m[1],
                   m3 = mu_m[2],m4 = mu_m[3],
                   m5 = mu_m[4],m6 = mu_m[5],
                   dzmode='additive',MG_switch=False,m_bias=True,
                   kmin=k_llm,kmax=k_obj)
bnt_fid.Cl_vec();Cl_fid = bnt_fid.Cl_deorg
bnt_fid.aCl_vec_reorg();aCl_fid = bnt_fid.aCl_reorg
bnt_fid.ell_arr = cell_arr
bnt_fid.pCl_vec();pCl_fid = 1.0*bnt_std.pCl_deorg
bpCl_fid = bnt_fid.Cl_vector_to_bnt(pCl_fid)


if theoretical_fid == True:
    bnt_tsc = bnt_core(nz_dt,ell_arr=cell_arr,pCl_dir=KL_dir,
                       NL_code=NL_code,NL_recipe=NL_recipe,
                       Omega_m = 0.2,S_8=0.797,Omega_h=False,
                       A_IA=AIAfid,A_IA2=A2fid,IA_B1=IA_B1,b_TA=bTAfid,
                       dz1 = mu_dz[0],dz2 = mu_dz[1],
                       dz3 = mu_dz[2],dz4 = mu_dz[3],
                       dz5 = mu_dz[4],dz6 = mu_dz[5],
                       m1 = mu_m[0],m2 = mu_m[1],
                       m3 = mu_m[2],m4 = mu_m[3],
                       m5 = mu_m[4],m6 = mu_m[5],
                       dzmode='additive',MG_switch=False,m_bias=True,
                       kmin=k_llm,kmax=k_obj)
    bnt_tsc.pCl_vec();pCl_tsc = 1.0*bnt_tsc.pCl_deorg
    bpCl_tsc = bnt_tsc.Cl_vector_to_bnt(pCl_tsc)
else:
    class KiDS_Legacy:
        def __init__(self):
            self.VOID = None

    bnt_kls = KiDS_Legacy()
    bnt_kls.pCls = np.load(KL_dir+'pCl_EE_KiDSLegacy.npy')
    bnt_kls.pCls_BB = np.load(KL_dir+'pCl_BB_KiDSLegacy.npy')
    bnt_kls.bpCls = bnt_std.p_a_i @ (bnt_std.p_a_i @ bnt_kls.pCls).transpose(1,0,2)
    bnt_kls.bpCls_BB = bnt_std.p_a_i @ (bnt_std.p_a_i @ bnt_kls.pCls_BB).transpose(1,0,2)
    pCl_kls = np.zeros_like(pCl_std)
    q = 0
    for i in range(6):
        for j in range(i,6):
            pCl_kls[q*30:(q+1)*30] = bnt_kls.pCls[i,j]
            q = q + 1
    bpCl_kls = bnt_std.Cl_vector_to_bnt(pCl_kls)

S8_print = '# where S_8 fid = '+str(np.sqrt(bnt_fid.Om/0.3)*ccl.power.sigma8(bnt_std.ccl_cosmo))[:8]
s8_print = '# and sigma_8 fid = '+str(ccl.power.sigma8(bnt_std.ccl_cosmo))[:7]


if __name__ == '__main__':
    print ('Fiducial Data Vectors Computed \n')

#
# Computation Related to the Scale Cut
#

Cl_diff = np.abs(Cl_fid/Cl_std - 1) < tfd
aCl_diff = np.abs(aCl_fid/aCl_std - 1) < tfd
print('k-cutted noBNT data vector length:',len(Cl_fid[Cl_diff]))
print('k-cutted BNT data vector length:',len(aCl_fid[aCl_diff]))
if mode_scale >= 30 and mode_scale <= 99:
    Cl_diff = np.logical_and(Cl_diff,tmp_nb)
    aCl_diff = np.logical_and(aCl_diff,tmp_bn)
    print('z-cutted noBNT data vector length:',len(Cl_fid[tmp_nb]))
    print('z-cutted BNT data vector length:',len(aCl_fid[tmp_bn]))
nBNT_l = len(Cl_fid[Cl_diff])
BNT_l = len(aCl_fid[aCl_diff])
len_print = '# non-BNT data vector length '+str(nBNT_l)+'\n'
len_print += '# BNT data vector length '+str(BNT_l)+'\n'

Cl_cov_sc = (Cl_cov[Cl_diff].T)[Cl_diff]
aCl_cov_sc = (aCl_cov[aCl_diff].T)[aCl_diff]
Cl_icov_sc = np.linalg.inv(Cl_cov_sc)
aCl_icov_sc = np.linalg.inv(aCl_cov_sc)

if theoretical_fid == True:
    pCl_tst = pCl_tsc
else:
    pCl_tst = pCl_kls
delta_deorg = (pCl_tst - pCl_std)[Cl_diff]
delta_reorg = bnt_std.Cl_vector_to_bnt(pCl_tst - pCl_std)[aCl_diff]
chi2_deorg = delta_deorg @ Cl_icov_sc @ delta_deorg
chi2_reorg = delta_reorg @ aCl_icov_sc @ delta_reorg

if __name__ == '__main__':
    print ('chi^2 of the test cosmology ($\\Omega_m = 0.35$ or KiDS ODV) is \n',flush=True)
    print ('non-BNT: '+str(chi2_deorg)+'\n',flush=True)
    print ('BNT: '+str(chi2_reorg)+'\n',flush=True)
    #print ('with '+len_print)
    #print ('If using manual integration as backup, then chi^2s are \n')
    #print ('non-BNT: '+str(chi2_deorg2)+'\n')
    #print ('BNT: '+str(chi2_reorg2)+'\n')

#
# Define the Prior and Parameter List
#

prior = Prior()

# Cosmology
#Param_basics = ['Omega_m','Omega_b','h','n_s','A_s']
if As_switch == False:
    Param_basics = ['omega_c','omega_b','h','n_s','S_8']
else:
    Param_basics = ['omega_c','omega_b','h','n_s','A_s']
#prior.add_parameter('Omega_m', dist=(0.1, 0.6)) #0.1 0.6
#prior.add_parameter('Omega_b', dist=(0.1, 0.6)) #0.03 0.07
prior.add_parameter('omega_c', dist=(0.051, 0.355)) #0.051 0.255
prior.add_parameter('omega_b', dist=(0.019, 0.026)) #0.019 0.026
prior.add_parameter('h', dist=(0.63, 0.83)) #0.55 0.85
prior.add_parameter('n_s', dist=(0.83, 1.11)) #0.92 1.02
if As_switch == True:
    prior.add_parameter('A_s', dist=(0.1e-9, 7.0e-9)) #1.5 5.0
elif As_switch == False:
    prior.add_parameter('S_8', dist=(0.5, 1.0)) #1.5 5.0

# Halo Model
#prior.add_parameter('p', dist=(-0.499, +0.499))
#prior.add_parameter('q', dist=(0.001, 1.999))
if NL_recipe == 'mead2020' or NL_recipe == 'mead2020_feedback':
    prior.add_parameter('T_AGN', dist=(7.3, 8.3)) #5.0, 10.0
    Param_basics.append('T_AGN')
elif NL_recipe == 'mead':
    prior.add_parameter('B', dist=(2.0, 3.13)) #1.0, 6.0
    Param_basics.append('B')
    
if MG_switch == True:
    prior.add_parameter('mu_MG', dist = (-3.0,3.0))
    prior.add_parameter('Sigma_MG', dist = (-3.0,3.0))
    Param_basics.append('mu_MG')
    Param_basics.append('Sigma_MG')

# Intrinsic Alignment
if IA_switch == True:
    Param_basics.append('A_IA')
    prior.add_parameter('A_IA', dist=(-0.6, +0.6)) #-6,6
    if mode_likelihood <= -20:
        Param_basics.append('eta_IA')
        prior.add_parameter('eta_IA', dist=(-5, +5)) #-5,5
    elif mode_likelihood > 20:
        Param_basics.append('B_IA')
        prior.add_parameter('B_IA', dist=(-6.0, +6.0))
    if TATT_switch == True:
        Param_basics.append('A_IA2')
        prior.add_parameter('A_IA2', dist=(-6.0, +6.0)) #-6,6
        if mode_likelihood <= 20:
            Param_basics.append('eta_IA2')
            prior.add_parameter('eta_IA2', dist=(-5.0, +5.0)) #-5,5
        else:
            Param_basics.append('B_IA2')
            prior.add_parameter('B_IA2', dist=(-6.0, +6.0))

# Redshift Error
if dz_switch == True:
    for i in range(lz):
        if dz_correlation == False:
            Param_basics.append('dz'+str(i+1))
            prior.add_parameter('dz'+str(i+1), dist=norm(loc=mu_dz[i], scale=std_dz[i]))
        else:
            Param_basics.append('ranz'+str(i+1))
            prior.add_parameter('ranz'+str(i+1), dist=norm(loc=mu_dz[i], scale=std_dz[i]))

# Multiplicative Bias
if m_switch == True:
    for j in range(lz):
        Param_basics.append('m'+str(j+1))
        prior.add_parameter('m'+str(j+1), dist=norm(loc=mu_m[i], scale=std_m[i]))
        
Param_message = '# '
q = 0
Param_columns = [('log_weight',float)]
for item in Param_basics:
    Param_message += item + ' '
    if len(Param_message) - q*70 > 70:
        Param_message += '\n# '
        q += 1
    Param_columns.append((item,float))
    
btype = [('Omega_m_output', float), ('A_s_output', float), ('sigma_4_output', float), ('S_4_output', float), ('sigma_6_output', float), ('S_6_output', float), ('sigma_8_output', float), ('S_8_output', float),  ('sigma_12_output', float), ('S_12_output', float), ('sigma_16_output', float), ('S_16_output', float), ('sigma_20_output', float), ('S_20_output', float), ('chi_2', float),('reduced_chi_2', float)]
#btype = [('Omega_m_output', float), ('A_s_output', float), ('sigma_8_output', float), ('S_8_output', float), ('chi_2', float),('reduced_chi_2', float)]
if dz_correlation == True:
    for i in range(lz):
        btype.append(('dz'+str(i+1), float))
    print(btype,flush=True)
#for btem in btype:
    #Param_columns.append(btype)
        
#
# Define the likelihood function
#

def loglike_cosmo(param_dict):
    ranz = np.array([param_dict['ranz1'], param_dict['ranz2'], param_dict['ranz3'], 
                     param_dict['ranz4'], param_dict['ranz5'], param_dict['ranz6']])
    cdz = mu_dz + L_dz @ ranz
    bnt_the = bnt_core(nz_dt,**param_dict,ell_arr=cell_arr,#dz_mean = mu_dz,
                       dz1 = cdz[0],dz2 = cdz[1],
                       dz3 = cdz[2],dz4 = cdz[3],
                       dz5 = cdz[4],dz6 = cdz[5],
                       pCl_dir=KL_dir,Omega_h=True,
                       NL_code=NL_code,NL_recipe=NL_recipe,
                       Modified_Gravity=MG_switch,
                       dzmode='additive',m_bias=True,
                       kmin=0.0025,kmax=100)
    
    sig8 = ccl.power.sigma8(bnt_the.ccl_cosmo)
    S8_output = np.sqrt(bnt_the.Om/0.3)*sig8
    if 2 > 1:
        sig4 = ccl.power.sigmaR(bnt_the.ccl_cosmo,4./bnt_the.ccl_cosmo["h"])
        sig6 = ccl.power.sigmaR(bnt_the.ccl_cosmo,6./bnt_the.ccl_cosmo["h"])
        si12 = ccl.power.sigmaR(bnt_the.ccl_cosmo,12./bnt_the.ccl_cosmo["h"])
        si16 = ccl.power.sigmaR(bnt_the.ccl_cosmo,16./bnt_the.ccl_cosmo["h"])
        si20 = ccl.power.sigmaR(bnt_the.ccl_cosmo,20./bnt_the.ccl_cosmo["h"])
        S4_output = np.sqrt(bnt_the.Om/0.3)*sig4
        S6_output = np.sqrt(bnt_the.Om/0.3)*sig6
        S12_output = np.sqrt(bnt_the.Om/0.3)*si12
        S16_output = np.sqrt(bnt_the.Om/0.3)*si16
        S20_output = np.sqrt(bnt_the.Om/0.3)*si20
    #try:
    if 1 > 0:
        bnt_the.pCl_vec(klim=False)
        pCl_the = bnt_the.pCl_deorg
        if theoretical_fid == True:
            pCl_std_use = 1.0*pCl_std
        else:
            pCl_std_use = 1.0*pCl_kls
        if bnt_switch == False:
            delta = (pCl_the - pCl_std_use)[Cl_diff]
            icov = Cl_icov_sc
            Ndof = nBNT_l - 5
        elif bnt_switch == True:
            fdelta = pCl_the - pCl_std_use
            delta = bnt_the.Cl_vector_to_bnt(fdelta)[aCl_diff]
            icov = aCl_icov_sc
            Ndof = BNT_l - 5
        chi2 = delta @ icov @ delta
        rchi2 = chi2/Ndof
    #except:
    #else:
        #chi2 = np.inf
        #rchi2 = chi2/Ndof
    return (-0.5*chi2,bnt_the.Om,bnt_the.As,\
            sig4,S4_output,sig6,S6_output,sig8,S8_output,\
            si12,S12_output,si16,S16_output,si20,S20_output,\
            chi2,rchi2,cdz[0],cdz[1],cdz[2],cdz[3],cdz[4],cdz[5])
    #return (-0.5*chi2,bnt_the.Om,bnt_the.As,sig8,S8_output,chi2,rchi2,cdz[0],cdz[1],cdz[2],cdz[3],cdz[4],cdz[5])
    
if __name__ == '__main__':
    #if 1 > 0:
    print('Config Initiated \n')
    Key_Message = name_label+'\n'
    Key_Message += lklhd_print_1+'\n'+lklhd_print_2+'\n'
    Key_Message += scale_print+'\n# Running Against \n'
    Key_Message += fidcl_print+'\n'+S8_print+'\n'+s8_print+'\n'
    Key_Message += '# with \n'+len_print
    Key_Message += '# using Model Parameters: \n'
    Key_Message += Param_message+'\n'
    Key_Message += '# Omega_m_output A_s_output sigma_8_output S_8 chi_2 \n'
    print(Key_Message,flush=True)
    
    sampler = Sampler(prior, loglike_cosmo, 
                      filepath=checkpoint_path+name_label+'checkpoint.hdf5',
                      pass_dict=True, n_live = 5000, 
                      blobs_dtype=btype,
                      #pool=16)
                      pool=MPIPoolExecutor())
                      #)
    sampler.run(verbose=True,n_like_max=chainlim)
    points, log_weight, log_likelihood, blobs = sampler.posterior(return_blobs=True)
    
    #print(points)
    #print(log_weight)
    #print(log_likelihood)
    
    #np.savetxt(folder+subfolder+name_label+'_tmp_points.txt',points)
    #np.savetxt(folder+subfolder+name_label+'_tmp_logweight.txt',log_weight)
    #np.save(folder+subfolder+name_label+'_tmp_blobs.npy',blobs)
    
    plobs = np.zeros(len(log_weight),dtype=Param_columns)
    plobs['log_weight'] = log_weight
    for i in range(1,len(Param_columns)):
        sttr = Param_columns[i][0]
        plobs[sttr] = points[:,i-1]
    flobs = np.lib.recfunctions.merge_arrays((plobs,blobs),
                                     flatten=True,usemask=False)
    np.save(folder+name_label+'.npy',flobs)
    print('Minimum reduced chi2 is:',np.min(flobs['reduced_chi_2']))
    #np.savetxt(folder+name_label+'.dat',flobs,header=Key_Message)
    