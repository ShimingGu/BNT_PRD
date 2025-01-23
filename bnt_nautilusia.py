from scipy.stats import norm,multivariate_normal
from nautilus import Prior,Sampler
import time
import os
import numpy as np
from bnt_mcmc_ccl3 import bnt_core
import pyccl as ccl
from scipy.ndimage import gaussian_filter as gf
from scipy.stats import norm
from datetime import datetime as dt
import sys
import subprocess as sub
from mpi4py.futures import MPIPoolExecutor
os.environ["OMP_NUM_THREADS"] = "1"

folder = '/consus/pm2x/gsm/nulling/test_chains_nautilus/'
subfolder = 'backups/'
checkpoint_path = '/home/gsm/'
chainlim = 5e5
nzfile = '/consus/pm2x/gsm/nulling/Euclid.nz'

print(__name__)

try:
    mode = sys.argv[1]
    print(mode)
    mode_likelihood = int(mode[0:2])
    mode_scale = int(mode[2:4])
    mode_fiducial = int(mode[4:6])
    print(mode_likelihood,mode_scale,mode_fiducial)
except:
    try:
        #liszt = !ls /home/gsm/*_condor.gsm
        liszt = sub.check_output(["ls /home/gsm/*_condor.gsm","-lt"],shell=True)
        mode = liszt[10:16]
        print(mode)
        mode_likelihood = int(mode[0:2])
        mode_scale = int(mode[2:4])
        mode_fiducial = int(mode[4:6])
        print(mode_likelihood,mode_scale,mode_fiducial)
    except:
        mode_fiducial = 0
        mode_scale = 0
        mode_likelihood = 1
        print('no system argument detected')
    
    
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
    IA_switch = True;m_switch=False;dz_switch=False
    lklhd_print_2 = '# with IA, without m, without dz'
    name_label = name_label + 'IA_nom_nodz_'
elif mode_likelihood > 2 and mode_likelihood <= 4:
    IA_switch = True;m_switch=True;dz_switch=False
    lklhd_print_2 = '# with IA, with m, without dz'
    name_label = name_label + 'IA_m_nodz_'
elif mode_likelihood > 4 and mode_likelihood <= 6:
    IA_switch = True;m_switch=False;dz_switch=True
    lklhd_print_2 = '# with IA, without m, with dz'
    name_label = name_label + 'IA_nom_dz_'
elif mode_likelihood > 6 and mode_likelihood <= 8:
    IA_switch = True;m_switch=True;dz_switch=True
    lklhd_print_2 = '# with IA, with m, with dz'
    name_label = name_label + 'IA_m_dz_'
elif mode_likelihood > 90 and mode_likelihood <= 92:
    IA_switch = True;m_switch=False;dz_switch=False
    lklhd_print_2 = '# with IA, without m, without dz, test'
    name_label = name_label + 'IA_nom_nodz_short_'; chainlim = 1000
    
# IA_switch = False # if to turn off IA completely
# folder = folder + 'noIA/
    
#
# Define the Mode for the Scale Cut
#
    
if mode_scale == 0:
    k_obj = 100;tfd = 0.1
    scale_print = '# no k_max'
    name_label = name_label + 'k100_'
else:
    if mode_scale in (1,4,7):
        k_obj = 0.1;scale_print = '# k_max = 0.1, '
        name_label = name_label + 'k0.1_'
    elif mode_scale in (2,5,8):
        k_obj = 1./3.;scale_print = '# k_max = 0.33, '
        name_label = name_label + 'k0.33_'
    elif mode_scale in (3,6,9):
        k_obj = 1.0;scale_print = '# k_max = 1.0, '
        name_label = name_label + 'k1.0_'
    if mode_scale in (1,2,3):
        tfd = 0.1;scale_print += 'TFD = 0.1'
        name_label = name_label + 'T0.1_'
    elif mode_scale in (4,5,6):
        tfd = 0.02;scale_print += 'TFD = 0.02'
        name_label = name_label + 'T0.02_'
    elif mode_scale in (7,8,9):
        tfd = 0.004;scale_print += 'TFD = 0.004'
        name_label = name_label + 'T0.004_'
    
#
# Define the Mode for the Fiducial Data Vector
#
    
if mode_fiducial == 0:
    fidcl_print = '# HMcode 2016';AIAfid = 0.0
    NL_code = 'camb';NL_recipe = 'mead';name_label += 'fid_Mead_'
    MG_switch = False
elif mode_fiducial == 1:
    fidcl_print = '# AxionHMcode, tuned';AIAfid = 0.0
    NL_code = 'camb';NL_recipe = 'vogt';name_label += 'fid_Vogt_'
    MG_switch = False
elif mode_fiducial == 2:
    fidcl_print = '# AxionHMcode, untuned';AIAfid = 0.0
    NL_code = 'camb';NL_recipe = 'vogt_original';name_label += 'fid_Vogt_orig_'
    MG_switch = False
elif mode_fiducial == 3:
    fidcl_print = '# Halofit Takahashi';AIAfid = 0.0
    NL_code = 'camb';NL_recipe = 'takahashi';name_label += 'fid_Takahashi_'
    MG_switch = False
elif mode_fiducial == 4:
    fidcl_print = '# Halofit + Baryon Correction Model';AIAfid = 0.0
    NL_code = 'camb';NL_recipe = 'schneider';name_label += 'fid_Schneider_'
    MG_switch = False
elif mode_fiducial == 5:
    fidcl_print = '# Vanilla Halo Model';AIAfid = 0.0
    NL_code = 'camb';NL_recipe = 'vanilla';name_label += 'fid_Vanilla_'
    MG_switch = False
elif mode_fiducial == 6:
    fidcl_print = '# IA = 2.0 fid; = 0.0 chain';AIAfid = 2.0
    NL_code = 'camb';NL_recipe = 'mead';name_label += 'fid_biasedIA_'
    MG_switch = False;IA_switch = False
elif mode_fiducial == 10:
    fidcl_print = '# builtin halofit';AIAfid = 0.0
    NL_code = 'halofit';NL_recipe = 'mead';name_label += 'fid_Takahashi_ccl_'
    MG_switch = False
elif mode_fiducial == 11:
    fidcl_print = '# builtin halofit with Modified Gravity';AIAfid = 0.0
    NL_code = 'halofit';NL_recipe = 'takahashi';name_label += 'fid_TakahashiMG_ccl_'
    MG_switch = True
    
#
# Define the Basic Setup
#
    
mu_dz = np.array([-0.025749,0.022716,
                  -0.026032,0.012594,
                  0.019285,0.008326,
                  0.038207,0.002732,
                  0.034066,0.049479,
                  0.066490,0.000815,
                  0.049070])
nz_dt = np.loadtxt(nzfile)
lz = nz_dt.shape[1]-1
Cl_cov = np.loadtxt('/consus/pm2x/gsm/nulling/Scale_covmats/covmats_file/covariance_standard_Euclid_standard.mat')
ell_arr = np.geomspace(50,5000,50)

if __name__ == '__main__':
    print ('Ready to Compute a BNT Core Module \n')

#
# Compute the fiducial datavec with and without the scale cut
#

bnt_std = bnt_core(nz_dt,NL_code=NL_code,NL_recipe=NL_recipe,A_IA=AIAfid,
                   dz1 = mu_dz[0],dz2 = mu_dz[1],
                   dz3 = mu_dz[2],dz4 = mu_dz[3],
                   dz5 = mu_dz[4],dz6 = mu_dz[5],
                   dz7 = mu_dz[6],dz8 = mu_dz[7],
                   dz9 = mu_dz[8],dz10 = mu_dz[9],
                   dzmode='additive',
                   ell_arr=ell_arr,kmax=100)
bnt_std.Cl_vec();Cl_std = 1.0*bnt_std.Cl_deorg
bnt_std.aCl_vec_reorg();aCl_std = 1.0*bnt_std.aCl_reorg
bnt_std.Cl_vec(klim=True);Cl_stdc = 1.0*bnt_std.Cl_deorg
bnt_std.aCl_vec_reorg(klim=True);aCl_stdc = 1.0*bnt_std.aCl_reorg

aCl_cov = bnt_std.covmat_to_bnt_ell(Cl_cov)

bnt_fid = bnt_core(nz_dt,NL_code=NL_code,NL_recipe=NL_recipe,A_IA=AIAfid,
                   dz1 = mu_dz[0],dz2 = mu_dz[1],
                   dz3 = mu_dz[2],dz4 = mu_dz[3],
                   dz5 = mu_dz[4],dz6 = mu_dz[5],
                   dz7 = mu_dz[6],dz8 = mu_dz[7],
                   dz9 = mu_dz[8],dz10 = mu_dz[9],
                   dzmode='additive',
                   ell_arr=ell_arr,kmax=k_obj)
bnt_fid.Cl_vec();Cl_fid = bnt_fid.Cl_deorg
bnt_fid.aCl_vec_reorg();aCl_fid = bnt_fid.aCl_reorg

bnt_tsc = bnt_core(nz_dt,NL_code=NL_code,NL_recipe=NL_recipe,
                   Omega_m = 0.2,A_IA=AIAfid,
                   dz1 = mu_dz[0],dz2 = mu_dz[1],
                   dz3 = mu_dz[2],dz4 = mu_dz[3],
                   dz5 = mu_dz[4],dz6 = mu_dz[5],
                   dz7 = mu_dz[6],dz8 = mu_dz[7],
                   dz9 = mu_dz[8],dz10 = mu_dz[9],
                   dzmode='additive',
                   ell_arr=ell_arr,kmax=k_obj)
bnt_tsc.Cl_vec();Cl_tsc = 1.0*bnt_tsc.Cl_deorg
bnt_tsc.aCl_vec_reorg();aCl_tsc = 1.0*bnt_tsc.aCl_reorg
bnt_tsc.Cl_vec(klim=True);Cl_tscc = 1.0*bnt_tsc.Cl_deorg
bnt_tsc.aCl_vec_reorg(klim=True);aCl_tscc = 1.0*bnt_tsc.aCl_reorg

s8_print = '# where s_8 fid = '+str(ccl.power.sigma8(bnt_fid.ccl_cosmo))[:8]

if __name__ == '__main__':
    print ('Fiducial Data Vectors Computed \n')

#
# Computation Related to the Scale Cut
#

Cl_diff = np.abs(Cl_fid/Cl_std - 1) < tfd
aCl_diff = np.abs(aCl_fid/aCl_std - 1) < tfd
nBNT_l = len(Cl_fid[Cl_diff])
BNT_l = len(aCl_fid[aCl_diff])
len_print = '# non-BNT data vector length '+str(nBNT_l)+'\n'
len_print += '# BNT data vector length '+str(BNT_l)+'\n'

Cl_cov_sc = (Cl_cov[Cl_diff].T)[Cl_diff]
aCl_cov_sc = (aCl_cov[aCl_diff].T)[aCl_diff]
Cl_icov_sc = np.linalg.inv(Cl_cov_sc)
aCl_icov_sc = np.linalg.inv(aCl_cov_sc)

delta_deorg = (Cl_tsc - Cl_std)[Cl_diff]
delta_reorg = bnt_std.Cl_vector_to_bnt(Cl_tsc - Cl_std)[aCl_diff]
delta_deorg2 = (Cl_tscc - Cl_stdc)[Cl_diff]
delta_reorg2 = bnt_std.Cl_vector_to_bnt(Cl_tscc - Cl_stdc)[aCl_diff]
chi2_deorg = delta_deorg @ Cl_icov_sc @ delta_deorg
chi2_reorg = delta_reorg @ aCl_icov_sc @ delta_reorg
chi2_deorg2 = delta_deorg2 @ Cl_icov_sc @ delta_deorg2
chi2_reorg2 = delta_reorg2 @ aCl_icov_sc @ delta_reorg2

if __name__ == '__main__':
    print ('chi^2 of the test cosmology $\Omega_m = 0.2 are \n')
    print ('non-BNT: '+str(chi2_deorg)+'\n')
    print ('BNT: '+str(chi2_reorg)+'\n')
    print ('If using manual integration as backup, then chi^2s are \n')
    print ('non-BNT: '+str(chi2_deorg2)+'\n')
    print ('BNT: '+str(chi2_reorg2)+'\n')

#
# Define the Prior and Parameter List
#

prior = Prior()

# Cosmology
Param_basics = ['Omega_m','Omega_b','h','n_s','A_s']
prior.add_parameter('Omega_m', dist=(0.1, 0.6)) #0.1 0.6
prior.add_parameter('Omega_b', dist=(0.03, 0.07)) #0.03 0.07
prior.add_parameter('h', dist=(0.55, 0.85)) #0.55 0.85
prior.add_parameter('n_s', dist=(0.92, 1.02)) #0.92 1.02
prior.add_parameter('A_s', dist=(1.5e-9, 5.0e-9)) #1.5

# Halo Model
#prior.add_parameter('p', dist=(-0.499, +0.499))
#prior.add_parameter('q', dist=(0.001, 1.999))
if NL_recipe == 'mead2020' or NL_recipe == 'mead2020_feedback':
    prior.add_parameter('T_AGN', dist=(5.0, 10.0))
    Param_basics.append('T_AGN')
elif NL_code == 'camb':
    prior.add_parameter('B', dist=(1.0, 6.0))
    Param_basics.append('B')
    
if MG_switch == True:
    prior.add_parameter('mu_MG', dist = (-3.0,3.0))
    prior.add_parameter('Sigma_MG', dist = (-3.0,3.0))
    Param_basics.append('mu_MG')
    Param_basics.append('Sigma_MG')

# Intrinsic Alignment
if IA_switch == True:
    Param_basics.append('A_IA')
    Param_basics.append('eta_IA')
    Param_basics.append('A_IA2')
    Param_basics.append('eta_IA2')
    prior.add_parameter('A_IA', dist=(-6.0, +6.0))
    prior.add_parameter('eta_IA', dist=(-5.0, +5.0))
    prior.add_parameter('A_IA2', dist=(-6.0, +6.0))
    prior.add_parameter('eta_IA2', dist=(-5.0, +5.0))

# Redshift Error
if dz_switch == True:
    for i in range(lz):
        Param_basics.append('dz'+str(i+1))
        prior.add_parameter('dz'+str(i+1), dist=norm(loc=mu_dz[i], scale=np.abs(0.002*(1+mu_dz[i]))))

# Multiplicative Bias
if m_switch == True:
    for j in range(lz):
        Param_basics.append('m'+str(j+1))
        prior.add_parameter('m'+str(j+1), dist=norm(loc=0.0, scale=0.0005))
        
Param_message = '# '
q = 0
Param_columns = [('log_weight',float)]
for item in Param_basics:
    Param_message += item + ' '
    if len(Param_message) - q*70 > 70:
        Param_message += '\n# '
        q += 1
    Param_columns.append((item,float))
    
btype = [('Omega_m_output', float), ('sigma_8_output', float), ('S_8', float), ('chi_2', float)]
#for btem in btype:
    #Param_columns.append(btype)
        
#
# Define the likelihood function
#

def loglike_cosmo(param_dict):
    bnt_the = bnt_core(nz_dt,**param_dict,ell_arr=ell_arr,dz_mean = mu_dz,
                       NL_code=NL_code,NL_recipe='mead',
                       dzmode='additive',m_bias=True)
    sig8 = ccl.power.sigma8(bnt_the.ccl_cosmo)
    S8 = np.sqrt(bnt_the.Om/0.3)*sig8
    try:
        #if 1 > 0:
        try:
            #if 1 > 0:
            bnt_the.Cl_vec(klim=False)
            Cl_the = bnt_the.Cl_deorg
            Cl_std_use = 1.0*Cl_std
        except:
            #else:
            bnt_the.Cl_vec(klim=True)
            Cl_the = bnt_the.Cl_deorg
            Cl_std_use = 1.0*Cl_stdc
            print(param_dict)
            print(sig8,S8)
        if bnt_switch == False:
            delta = (Cl_the - Cl_std_use)[Cl_diff]
            icov = Cl_icov_sc
        elif bnt_switch == True:
            fdelta = Cl_the - Cl_std_use
            delta = bnt_std.Cl_vector_to_bnt(fdelta)[aCl_diff]
            icov = aCl_icov_sc
        chi2 = delta @ icov @ delta
    except:
        #else:
        chi2 = np.inf
    return (-0.5*chi2,bnt_the.Om,sig8,S8,chi2)
    
if __name__ == '__main__':
    #if 1 > 0:
    print('Config Initiated \n')
    Key_Message = name_label+'\n'
    Key_Message += lklhd_print_1+'\n'+lklhd_print_2+'\n'
    Key_Message += scale_print+'\n# Running Against \n'
    Key_Message += fidcl_print+'\n'+s8_print+'\n'
    Key_Message += '# with \n'+len_print
    Key_Message += '# using Model Parameters: \n'
    Key_Message += Param_message+'\n'
    Key_Message += '# Omega_m_output sigma_8_output S_8 chi_2 \n'
    print(Key_Message)
    
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
    #np.savetxt(folder+name_label+'.dat',flobs,header=Key_Message)
    