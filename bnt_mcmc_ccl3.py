import numpy as np
from astropy import constants as cst
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from scipy.interpolate import interp1d,interp2d,RegularGridInterpolator
from scipy.integrate import fixed_quad
from scipy.ndimage import gaussian_filter as gf
import copy
import pyccl as ccl

def cosmo_2_cosmo(ccl_cosmo=None,astropy_cosmo=None,n_s=None,A_s=None,sig_8=None):
    if ccl_cosmo == None and astropy_cosmo == None:
        raise ValueError('One cosmology input is required')
    elif ccl_cosmo == None:
        print('Converting Astropy Cosmology to Pyccl')
        if n_s == None or (A_s == None and sig_8 == None):
            raise ValueError('Need n_s and one of A_s or sigma_8')
        else:
            ns = n_s;h = astropy_cosmo.h
            Om = astropy_cosmo.Om0;Ob = astropy_cosmo.Ob0
            Oc = Om-Ob;Tc = astropy_cosmo.Tcmb0
            if A_s is not None:
                As = A_s;ccl_cosmo = ccl.Cosmology(Omega_c = Oc,
                                          Omega_b = Ob,T_CMB=Tc,
                                          h = h,A_s = As,n_s=ns)
            else:
                s8 = sig_8;ccl_cosmo = ccl.Cosmology(Omega_c = Oc,
                                          Omega_b = Ob,T_CMB=Tc,
                                          h = h,sig_8 = s8,n_s=ns)
        return ccl_cosmo
    elif astropy_cosmo == None:
        #print('Converting Pyccl Cosmology to Astropy')
        Ob = ccl_cosmo['Omega_b'];Om = ccl_cosmo['Omega_c']+Ob
        H0 = ccl_cosmo['h']*100;TC = ccl_cosmo['T_CMB']
        astropy_cosmo = FlatLambdaCDM(Tcmb0=TC,H0=H0,Om0=Om,Ob0=Ob)
        return astropy_cosmo
    else:
        raise ValueError('Input for both, which one do you want to convert to?')
        
def n_0_i_int(z,n_of_z_interp):
    return n_of_z_interp(z)

def n_1_i_int(z,n_of_z_interp,apcosmo):
    chi_of_z = (apcosmo.comoving_distance(z)/(cst.c/apcosmo.H0)).cgs
    n_1_i_int = n_of_z_interp(z)/chi_of_z
    return n_1_i_int

def n_0_n_1(nz_data,apcosmo):
    nzrows = nz_data.shape[0];ntomobin = nz_data.shape[1]-1
    nz_table = np.zeros((nzrows,ntomobin+1))
    n_0 = np.zeros(ntomobin);n_1 = np.zeros(ntomobin)
    z = nz_data[:,0]
    for i in range(ntomobin):
        nz = nz_data[:,i+1]
        zmin=1e-18;zmax=3
        n_of_z_interp = interp1d(z, nz, kind='quadratic',fill_value='extrapolate')
        n_0[i], error = fixed_quad(n_0_i_int, zmin, zmax, args=(n_of_z_interp,),n=10000)
        n_1[i], error = fixed_quad(n_1_i_int, zmin, zmax, args=(n_of_z_interp,apcosmo),n=10000)
    return n_0,n_1,ntomobin

def reorganise_ell(dvec,tomobins,ellbins,verbose=False):
    Nbl = [];Tz = 0
    for h in range(ellbins):
        Tz1 = Tz + h
        for i in range(tomobins):
            Tz2 = Tz1 + i*10000
            for j in range(i,tomobins):   
                Nbl.append(Tz2 + j*100)
    iNbo = np.argsort(Nbl)
    if verbose == True:
        print(iNbo)
    ndvec = np.zeros_like(dvec)
    for i in range(len(dvec)):
        ndvec[iNbo[i]] = dvec[i]
    return ndvec

def deorganise_ell(dvec,tomobins,ellbins):
    Nbl = [];Tz = 0
    for h in range(ellbins):
        Tz1 = Tz + h
        for i in range(tomobins):
            Tz2 = Tz1 + i*1000
            for j in range(i,tomobins):   
                Nbl.append(Tz2 + j*100)
    iNbo = np.argsort(Nbl)
    ndvec = np.zeros_like(dvec)
    for i in range(len(dvec)):
        ndvec[i] = dvec[iNbo[i]]
    return ndvec

def power_func_mc(bntclass,chi_,kcut=1e2):
    cosmo = bntclass.ccl_cosmo;h=cosmo['h']
    lcut=kcut*chi_-1
    a_sf=ccl.background.scale_factor_of_chi(cosmo,chi_)
    ec_grid = np.meshgrid(bntclass.ell_arr,chi_)
    k_grid = (ec_grid[0]+0.5)/ec_grid[1]
    lk = k_grid.shape[0]
    k_inds = np.array([np.searchsorted(k_grid[i]*h,kcut) for i in range(lk)])
    pk = np.array([10**bntclass.paowa((np.log(k_grid[j]),a_sf[j])) for j in range(lk)])
    #for l in range(lk):
        #pk[l,k_inds[l]:] = pk[l,k_inds[l]:]*1e-15
    return pk

def proj_int(chi_,bntclass,kern_1,kern_2=None,kcut=1e2):
    if kern_2 == None:
        kern_2 = kern_1
    return chi_**(-2)*kern_1(chi_)*kern_2(chi_)*power_func_mc(bntclass,chi_,kcut).T
    #print(toreturn.shape)
    #return toreturn
    
def fermi_dirac(x,smooth,cut):
    smooth__ = np.float64(smooth)
    yy = 1/(np.exp((x-cut)*smooth__)+1)
    return yy/yy[0]+1e-19

def fermi_dirac2(x,smooth,cut1,cut2):
    smooth__ = np.float64(smooth)
    y1 = 1/(np.exp((x-cut2)*smooth__)+1)
    y2 = 1/(np.exp((cut1-x)*smooth__)+1)
    return (y1*y2)/(y1[0]*y2[-1])+1e-19

#
# The Class for BNT Computation
#

class bnt_core:
    
    def __init__(self,nz_data,ell_arr=None,**kwargs):
        self.nz_dt = self._dz(nz_data,**kwargs)
        self.zari = np.flip(self.nz_dt[:,0])
        self.aarr = 1./(1.+self.zari)
        aarr1 = self.aarr[:-1]+0.5*np.diff(self.aarr)
        self.aarr2 = np.unique(np.sort(np.append(self.aarr,aarr1)))
        self._mbias(nz_data,**kwargs)
        self.a_sf = 1./(1+self.nz_dt[:,0])
        
        self._inflate(**kwargs)    
        self.IA_on = False
        self.TATT_on = False
        if self.AIA1 != 0.0:
            self.IA_on = True
            self.IA1 = self.AIA1*(self.a0/self.a_sf)**self.eta1
            D = self.ccl_cosmo.growth_factor(self.a_sf)
            rho_m = ccl.physical_constants.RHO_CRITICAL*self.ccl_cosmo['Omega_m']
            self.aia1 = - self.IA1 * 5e-14 * rho_m / D
            if self.AIA2 != 0.0:
                self.TATT_on = True
                self.IA2 = self.AIA2*(self.a0/self.a_sf)**self.eta2
                self.aia2 = 5*self.IA2*5e-14*rho_m/(D**2)
                #self.aia = self.aia1+self.aia2
                if self.bTA == 0.0:
                    self.aia_delta = None
                else:
                    self.aia_delta = self.bTA*self.aia1
                self._inflate_TATT()
            else:
                self.TATT_on = False
        self.ap_cosmo = cosmo_2_cosmo(ccl_cosmo=self.ccl_cosmo)
        self.p_a_i = self.p_a_i_matrix()
        #self.PAI = self.P_a_I()
        if ell_arr is not None:
            self.ell_arr = ell_arr
        else:
            self._ell_max(**kwargs)
        self.nell = len(self.ell_arr)
        self.PAI_ell = self.P_a_I_ell()
            
    def _dz(self,nz_data,**kwargs):
        n_nz = nz_data.shape[1]-1
        nz_data2 = 1.0*nz_data
        dzs = []
        self.dzmode = kwargs.pop('dzmode',None)
        dz_mu = kwargs.pop('dz_mean',None)
        if dz_mu is None:
            dz_mu = np.zeros(n_nz)
        if self.dzmode == 'additive':
            for i in range(n_nz):
                dz = float(kwargs.pop('dz'+str(i+1),dz_mu[i]))
                old_z = nz_data[:,0];new_z = 1.0*old_z-dz
                nz_data2[:,i+1] = np.interp(old_z,new_z,nz_data[:,i+1],left=0,right=0)
                nz_data2[0,i+1] = 0
        elif self.dzmode == 'multiplicative':
            for i in range(n_nz):
                pz = float(kwargs.pop('pz'+str(i+1),0.0))
                old_z = nz_data[:,0];new_z = 1.0*old_z*pz
                nz_data2[:,i+1] = np.interp(old_z,new_z,nz_data[:,i+1],left=0,right=0)
                nz_data2[0,i+1] = 0
        return nz_data2
    
    def _mbias(self,nz_data,**kwargs):
        n_nz = nz_data.shape[1]-1
        self.mbias_ = kwargs.pop('m_bias',None)
        if self.mbias_ == True:
            self.mbias = np.zeros(n_nz)
            for i in range(n_nz):
                self.mbias[i] = float(kwargs.pop('m'+str(i+1),0.0))
                
    def _inflate_cosmology(self,**kwargs):
        Omega_mode = kwargs.pop('Omega_h',False)
        self.MG = kwargs.pop('Modified_Gravity',False)
        self.h = float(kwargs.pop('h',0.6898))
        if Omega_mode == False:
            self.Om = float(kwargs.pop('Omega_m',0.2905))
            self.Ob = float(kwargs.pop('Omega_b',0.0473))
            self.Oc = self.Om-self.Ob
        elif Omega_mode == True:
            self.Och2 = float(kwargs.pop('omega_m',0.11572))
            self.Obh2 = float(kwargs.pop('omega_m',0.0225))
            self.Oc = self.Och2/self.h/self.h
            self.Ob = self.Obh2/self.h/self.h
            self.Om = self.Oc+self.Ob
        if self.MG == False:
            self.mgpara=None
        elif self.MG == True:
            self.mu_mg = float(kwargs.pop('mu_MG',0.0))
            self.Sigma_mg = float(kwargs.pop('Sigma_MG',0.0))
            self.mgpara=ccl.modified_gravity.mu_Sigma.MuSigmaMG(mu_0=self.mu_mg,sigma_0=self.Sigma_mg)
        self.ns = float(kwargs.pop('n_s',0.969))
        self.w_0 = float(kwargs.pop('w_0',-1.0))
        self.w_a = float(kwargs.pop('w_a',0.0))
        self.M_nu = float(kwargs.pop('m_nu',0.0))
        self.As = float(kwargs.pop('A_s',2.18676e-9))
        self.s8 = float(kwargs.pop('sigma_8',0.826))
        self.fDM = False
        #self._inflate_fDM(**kwargs)
        
    def _inflate_fDM(self,**kwargs):
        self.fDM = True
        self.logDMmass = float(kwargs.pop('logM_DM',1))
        #self.logsigmaDM = kwargs.pop('logsigma_DM',None)
        self.fax = float(kwargs.pop('f_ax',0))
        if self.logDMmass > 0:
            self.logDMmass = 1
            self.fax = 0
        
    def _inflate_halo(self,**kwargs):
        self.p = float(kwargs.pop('p',0.3))
        self.q = float(kwargs.pop('q',0.707))
        self.B = float(kwargs.pop('B',3.13))
        self.TAGN = float(kwargs.pop('T_AGN',7.8))
        
    def _inflate_core(self,**kwargs):
        ep = {"camb": {"halofit_version": self.nl_recipe, 
                       "HMCode_A_baryon": self.B, 
                       'HMCode_logT_AGN': self.TAGN,
                       #"HMCode_p_st": self.p, 
                       #"HMCode_q_st": self.q, 
                       "HMCode_A_des": 0.322, 
                       "HMCode_ST_des": 0.2}}
        if self.transfer == 'eisenstein_hu' or self.nl_code == 'emulator':
            self.ccl_cosmo = ccl.Cosmology(Omega_c = self.Oc,
                             Omega_b = self.Ob,h = self.h,
                             n_s = self.ns,sigma8=self.s8,Neff=3.046, 
                             m_nu=self.M_nu,w0 = self.w_0,wa = self.w_a,
                             transfer_function=self.transfer,
                             matter_power_spectrum=self.nl_code,
                             mg_parametrization=self.mgpara)
                             #extra_parameters=ep)
        elif self.nl_code == 'camb':
            self.ccl_cosmo = ccl.Cosmology(Omega_c = self.Oc,
                             Omega_b = self.Ob,h = self.h,
                             n_s = self.ns,A_s=self.As,Neff=3.046, 
                             m_nu=self.M_nu,w0 = self.w_0,wa = self.w_a,
                             transfer_function=self.transfer,
                             matter_power_spectrum=self.nl_code,
                             #mg_parametrization=self.mgpara,
                             extra_parameters=ep)
        else:
            self.ccl_cosmo = ccl.Cosmology(Omega_c = self.Oc,
                             Omega_b = self.Ob,h = self.h,
                             n_s = self.ns,A_s=self.As,Neff=3.046, 
                             m_nu=self.M_nu,w0 = self.w_0,wa = self.w_a,
                             transfer_function=self.transfer,
                             matter_power_spectrum=self.nl_code,
                             mg_parametrization=self.mgpara,
                             extra_parameters=ep)
        
    def _inflate_power(self,**kwargs):
        self.kmin = kwargs.pop('kmin',0.0025)
        self.kminh = self.kmin*self.h
        self.kmax = kwargs.pop('kmax',100)
        self.kmaxh = self.kmax*self.h
        self.kmax1 = kwargs.pop('kmax_1',0)
        self.sharp = kwargs.pop('sharp',1000000)
        self.karr = 10**(np.linspace(np.log10(self.kmin/np.pi),np.log10(self.kmax*np.pi),5000))
        self.ANL = kwargs.pop('A_NL',1)
        
    def _inflate_IA(self,**kwargs):
        self.AIA1 = float(kwargs.pop('A_IA',0.0))
        self.AIA2 = float(kwargs.pop('A_IA2',0.0))
        self.bTA = float(kwargs.pop('b_TA',0.0))
        self.eta1 = float(kwargs.pop('eta_IA',0.0))
        self.eta2 = float(kwargs.pop('eta_IA2',0.0))
        self.z0 = float(kwargs.pop('z0_IA',0.62))
        self.a0 = 1./(1.+self.z0)
    
    def _inflate_TATT(self,**kwargs):
        z_a1 = (self.zari,np.flip(self.aia1))
        z_a2 = (self.zari,np.flip(self.aia2))
        if self.aia_delta is None:
            z_a1d = None
        else:
            z_a1d = (self.zari,np.flip(self.aia_delta))
        PTIA = ccl.nl_pt.tracers.PTIntrinsicAlignmentTracer(z_a1,c2=z_a2,cdelta=z_a1d)
        EPT = ccl.nl_pt.ept.EulerianPTCalculator(with_NC=False, with_IA=True, 
                                             with_matter_1loop=True,
                                             cosmo=self.ccl_cosmo, 
                                             log10k_min=-4, log10k_max=2, 
                                             nk_per_decade=20, a_arr=self.aarr, 
                                             k_cutoff=None, n_exp_cutoff=4, 
                                             b1_pk_kind='nonlinear', 
                                             bk2_pk_kind='nonlinear', 
                                             pad_factor=1.0, 
                                             low_extrap=-5.0, high_extrap=3.0, 
                                             P_window=None, C_window=0.75,
                                             sub_lowk=False)
        k_PT = EPT.k_s
        P_ii_TATT_EE = EPT._get_pii(PTIA,PTIA,False)
        self.pk2d_II = ccl.Pk2D(a_arr=self.aarr,lk_arr=np.log(k_PT),
                pk_arr=P_ii_TATT_EE,is_logp=False,
                extrap_order_lok=1,
                extrap_order_hik=2) 
        #P_ii_TATT_BB = EPT._get_pii(PTIA,PTIA,True)
        #self.pk2d_BB = ccl.Pk2D(a_arr=self.aarr,lk_arr=np.log(k_PT),
                #pk_arr=P_ii_TATT_BB,is_logp=False,
                #extrap_order_lok=1,
                #extrap_order_hik=2) 
        P_gi_TATT = EPT._get_pim(PTIA)
        self.pk2d_GI = ccl.Pk2D(a_arr=self.aarr,lk_arr=np.log(k_PT),
                pk_arr=P_gi_TATT,is_logp=False,
                extrap_order_lok=1,
                extrap_order_hik=2) 
        #P_gg_PT = EPT._get_pmm()
        #self.pk2d_GG = ccl.Pk2D(a_arr=self.aarr,lk_arr=np.log(k_PT),
                #pk_arr=P_gg_PT,is_logp=False,
                #extrap_order_lok=1,
                #extrap_order_hik=2) 
        
    def _inflate_recipe(self,**kwargs):
        self.Ext_input = False
        self.BCM = False
        self.ax_cor = None
        self.Vanilla = False
        self.transfer = kwargs.pop('transfer','boltzmann_camb')
        if self.transfer == 'boltzmann_camb':
            self.nl_code = kwargs.pop('NL_code','camb')
        else:
            self.nl_code = kwargs.pop('NL_code','halofit')
        self.nl_recipe = kwargs.pop('NL_recipe','mead')
        if self.nl_recipe == 'AxionHMcode' or self.nl_recipe == 'vogt':
            self.Ext_input = True;self.nl_recipe = 'mead'
            self.BCM = False;self.ax_cor = True
            self.Vanilla = False
        elif self.nl_recipe == 'vogt_original':
            self.Ext_input = True;self.nl_recipe = 'mead'
            self.BCM = False;self.ax_cor = False
            self.Vanilla = False
        elif self.nl_recipe == 'BCM' or self.nl_recipe == 'schneider':
            self.Ext_input = False;self.nl_code = 'halofit'
            self.BCM = True;self.ax_cor = None
            self.Vanilla = False
        elif self.nl_recipe == 'vanilla' or self.nl_recipe == 'HaloModel':
            self.Ext_input = False;self.BCM = False
            self.ax_cor = None;self.Vanilla = True
            self.nl_recipe = 'mead'
        else:
            self.Ext_input = False;self.BCM = False
            self.ax_cor = None;self.Vanilla = False
            
    def _inflate_halomodel(self,**kwargs): #still working on it
        #halo_def = ccl.halos.MassDefVir() # may need change later
        halo_def = ccl.halos.MassDef200c()
        halo_con = ccl.halos.ConcentrationDiemer15(halo_def)
        halo_hmf = ccl.halos.hmfunc.MassFuncTinker10(self.ccl_cosmo,
                                                     halo_def)
        halo_hmb = ccl.halos.hbias.HaloBiasTinker10(self.ccl_cosmo,halo_def)
        halo_cal = ccl.halos.halo_model.HMCalculator(self.ccl_cosmo,
                                                     halo_hmf,
                                                     halo_hmb,halo_def,
                                                     log10M_min=6.0,
                                                     log10M_max=17.0)
        prof_mas = ccl.halos.profiles.HaloProfileNFW(halo_con)
        self.pka = ccl.halos.halo_model.halomod_power_spectrum(cosmo=self.ccl_cosmo,
                                                hmc=halo_cal,
                                                k=self.karr,
                                                a=self.aarr2,
                                                prof=prof_mas,
                                                normprof1=True)
        self.pk2d = ccl.halos.halo_model.halomod_Pk2D(self.ccl_cosmo,halo_cal,
                                                prof=prof_mas,
                                                normprof1=True,
                                                lk_arr=np.log(self.karr),
                                                a_arr=self.aarr2)
        self.paowa = RegularGridInterpolator((np.log(self.karr),self.aarr2),np.log10(self.pka).T,bounds_error=False, fill_value=-100)
            
    def _inflate_spectra(self,**kwargs):
        pknl_tsf = fermi_dirac2(self.karr,self.sharp,self.kmin,self.kmax)
        if self.ANL >= 0.99:
            if self.BCM == False:
                self.pka = np.array([ccl.power.nonlin_matter_power(self.ccl_cosmo,self.karr,a)*pknl_tsf for a in self.aarr2])
            else:
                self.pka = np.array([ccl.power.nonlin_matter_power(self.ccl_cosmo,self.karr,a)*ccl.bcm.bcm_model_fka(self.ccl_cosmo,self.karr,a)*pknl_tsf for a in self.aarr2])
        elif self.ANL <= 0.01 and self.ANL >= -0.01:
            if self.BCM == False:
                self.pka = np.array([ccl.power.linear_matter_power(self.ccl_cosmo,self.karr,a)*pknl_tsf for a in self.aarr2])
            else:
                self.pka = np.array([ccl.power.linear_matter_power(self.ccl_cosmo,self.karr,a)*ccl.bcm.bcm_model_fka(self.ccl_cosmo,self.karr,a)*pknl_tsf for a in self.aarr2])
        elif self.ANL <= -1:
            if self.BCM == False:
                self.pka = np.array([(ccl.power.nonlin_matter_power(self.ccl_cosmo,self.karr,a)-ccl.power.linear_matter_power(self.ccl_cosmo,self.karr,a))*pknl_tsf for a in self.aarr2])
            else:
                self.pka = np.array([(ccl.power.nonlin_matter_power(self.ccl_cosmo,self.karr,a)-ccl.power.linear_matter_power(self.ccl_cosmo,self.karr,a))*ccl.bcm.bcm_model_fka(self.ccl_cosmo,self.karr,a)*pknl_tsf for a in self.aarr2])
        elif self.ANL > 0.01 and self.ANL < 0.99:
            if self.BCM == False:
                self.pka = np.array([(ccl.power.linear_matter_power(self.ccl_cosmo,self.karr,a)+self.ANL*(ccl.power.nonlin_matter_power(self.ccl_cosmo,self.karr,a)-ccl.power.linear_matter_power(self.ccl_cosmo,self.karr,a)))*pknl_tsf for a in self.aarr2])
            else:
                self.pka = np.array([(ccl.power.linear_matter_power(self.ccl_cosmo,self.karr,a)+self.ANL*(ccl.power.nonlin_matter_power(self.ccl_cosmo,self.karr,a)-ccl.power.linear_matter_power(self.ccl_cosmo,self.karr,a)))*ccl.bcm.bcm_model_fka(self.ccl_cosmo,self.karr,a)*pknl_tsf for a in self.aarr2])
        if self.Ext_input == True:
            kar = np.load('/consus/pm2x/gsm/AxionHMcode_karr.npy')
            aar = np.load('/consus/pm2x/gsm/AxionHMcode_aarr.npy')
            if self.ax_cor == True:
                pkr0 = np.load('/consus/pm2x/gsm/AxionHMcode_pk2d_ratio_2.npy')
                s1,s2 = pkr0.shape
                self.pkr = np.zeros((s1,len(self.karr)))
                for i in range(s1):
                    self.pkr[i] = np.interp(self.karr,kar,pkr0[i])
                p1,p2 = self.pka.shape
                self.pkaa = np.zeros((len(aar),p2))
                for j in range(p2):
                    self.pkaa[:,j] = np.interp(aar,self.aarr2,self.pka[:,j])
                self.paowa = RegularGridInterpolator((np.log(self.karr),aar),np.log10(self.pkaa*self.pkr).T,bounds_error=False, fill_value=-100)
                self.pk2d = ccl.pk2d.Pk2D(a_arr = aar,
                                          lk_arr = np.log(self.karr), 
                                          pk_arr = self.pkaa*self.pkr,
                                          #cosmo=self.ccl_cosmo,
                                          is_logp = False)
            elif self.ax_cor == False:
                self.pkax = np.load('/consus/pm2x/gsm/AxionHMcode_pk2d_2.npy')
                self.paowa = RegularGridInterpolator((np.log(kar),aar),np.log10(self.pkax).T,bounds_error=False,fill_value=-100)
                self.pk2d = ccl.pk2d.Pk2D(a_arr = aar,
                                          lk_arr = np.log(kar), 
                                          pk_arr = self.pkax,
                                          #cosmo=self.ccl_cosmo,
                                          is_logp = False)
        elif self.Vanilla == True:
            self._inflate_halomodel(**kwargs)
        else:
            self.paowa = RegularGridInterpolator((np.log(self.karr),self.aarr2),np.log10(self.pka).T,bounds_error=False, fill_value=-100)
            self.pk2d = ccl.pk2d.Pk2D(a_arr = self.aarr2, 
                                      lk_arr = np.log(self.karr), 
                                      pk_arr = self.pka, 
                                      #cosmo=self.ccl_cosmo, 
                                      is_logp = False)
            
                
    def _inflate(self,**kwargs):
        self._inflate_cosmology(**kwargs)
        self._inflate_halo(**kwargs)
        self._inflate_IA(**kwargs)
        self._inflate_power(**kwargs)
        self._inflate_recipe(**kwargs)
        self._inflate_core(**kwargs)
        self._inflate_spectra(**kwargs)
        
    def _ell_max(self,**kwargs):
        mf = kwargs.pop('l_max_mod_fac',1)
        lmin = kwargs.pop('lmin',2);amin = np.min(self.aarr)
        chi_end = ccl.comoving_radial_distance(self.ccl_cosmo,amin)
        self.l_max = np.int((self.kmax*chi_end)/mf-1)
        self.lmax = np.min([self.l_max,100000]);self.nell = 3000
        self.ell_arr = np.unique(np.int_(np.geomspace(lmin,self.lmax,self.nell)))
        
    def p_a_i_matrix(self):
        n_0,n_1,ntomobin = n_0_n_1(self.nz_dt,self.ap_cosmo)
        self.ntomo = ntomobin
        self.ltomo = int(self.ntomo*(self.ntomo+1)/2)
        p_a_i = np.zeros((ntomobin,ntomobin))
        p_a_i[1,0] = -1
        for i in range(ntomobin):
            p_a_i[i,i] = 1
        for i in range(2,ntomobin):
            p_a_i[i,i-2] = (n_1[i]*n_0[i-1]-n_0[i]*n_1[i-1])/(n_0[i-2]*n_1[i-1]-n_1[i-2]*n_0[i-1])
            p_a_i[i,i-1] = (n_1[i]*n_0[i-2]-n_0[i]*n_1[i-2])/(n_0[i-1]*n_1[i-2]-n_1[i-1]*n_0[i-2])
        return p_a_i

    
    def P_a_I_ell(self):
        p_a_i = self.p_a_i
        nell = self.nell
        ntomo = self.ntomo
        ntomi = int(ntomo*(ntomo+1)/2)
        P_a_I_b = np.zeros((ntomi,ntomi))
        ijind = []
        for i in range(ntomo):
            for j in range(i,ntomo):
                ijind.append((i,j))
        for a in range(ntomi):
            for b in range(ntomi):
                P_a_I_b[a,b] = p_a_i[ijind[a][0],ijind[b][0]]*p_a_i[ijind[a][1],ijind[b][1]]
                if ijind[b][1] > ijind[b][0]:
                    P_a_I_b[a,b] += p_a_i[ijind[a][0],ijind[b][1]]*p_a_i[ijind[a][1],ijind[b][0]]
        P_A_I = np.zeros((ntomi*nell,ntomi*nell))
        for i in range(nell):
            P_A_I[i*ntomi:(i+1)*ntomi,i*ntomi:(i+1)*ntomi] = P_a_I_b
        return P_A_I
        
    def kernel_clean(self,cosmo,z,nz):
        self.chi,kernel = ccl.tracers.get_lensing_kernel(cosmo,dndz=(z,nz),n_chi=len(z))
        return kernel
    
    def kernel_clean_N(self,cosmo,z,nz):
        self.chi_N,kernel = ccl.tracers.get_density_kernel(cosmo,dndz=(z,nz))
        return kernel
    
    def Cl_vector_to_bnt(self,Cl_vector):
        Cl_reorg_vector = reorganise_ell(Cl_vector,self.ntomo,self.nell)
        return self.PAI_ell@Cl_reorg_vector

    def covmat_to_bnt_faster_ell(self,covmat):
        covmat_reorg = reorganise_ell(reorganise_ell(covmat,self.ntomo,self.nell).T,self.ntomo,self.nell)
        return self.PAI_ell @ covmat_reorg @ self.PAI_ell.T
    
    def covmat_to_bnt_ell(self,covmat):
        covmat_reorg = reorganise_ell(reorganise_ell(covmat,self.ntomo,self.nell).T,self.ntomo,self.nell)
        return self.PAI_ell @ np.transpose(self.PAI_ell @ np.transpose(covmat_reorg))
    
    def Cl_tomo(self,extraIA=False):
        if self.IA_on == False:
            tracers = [ccl.tracers.WeakLensingTracer(cosmo=self.ccl_cosmo,dndz=(self.nz_dt[:,0],self.nz_dt[:,it+1])) for it in range(self.ntomo)]
        else:
            tracers = [ccl.tracers.WeakLensingTracer(cosmo=self.ccl_cosmo,dndz=(self.nz_dt[:,0],self.nz_dt[:,it+1]),ia_bias=(self.nz_dt[:,0],self.aia1),use_A_ia=False) for it in range(self.ntomo)]
        Cls = np.array([[ccl.angular_cl(self.ccl_cosmo,tracers[i],tracers[j],self.ell_arr,p_of_k_a = self.pk2d) for i in range(self.ntomo)] for j in range(self.ntomo)])
        self.Cls = Cls
        
    def Cl_tomo_ext(self,pk2d=None,mode='GG'):
        if pk2d is None:
            if mode == 'GG':
                pk2d = self.pk2d
            elif mode == 'GI':
                pk2d = self.pk2d_GI
            elif mode == 'II':
                pk2d = self.pk2d_II
        if mode == 'GG':
            if self.IA_on == False:
                tracer1 = [ccl.tracers.WeakLensingTracer(cosmo=self.ccl_cosmo,dndz=(self.nz_dt[:,0],self.nz_dt[:,it+1])) for it in range(self.ntomo)]
            else:
                tracer1 = [ccl.tracers.WeakLensingTracer(cosmo=self.ccl_cosmo,dndz=(self.nz_dt[:,0],self.nz_dt[:,it+1]),ia_bias=(self.nz_dt[:,0],self.aia1),use_A_ia=False) for it in range(self.ntomo)]
            tracer2 = tracer1
        elif mode == 'GI':
            tracer1 = [ccl.tracers.WeakLensingTracer(cosmo=self.ccl_cosmo,dndz=(self.nz_dt[:,0],self.nz_dt[:,it+1])) for it in range(self.ntomo)]
            tracer2 = [ccl.tracers.NumberCountsTracer(cosmo=self.ccl_cosmo,dndz=(self.nz_dt[:,0],self.nz_dt[:,it+1]), bias=(self.nz_dt[:,0],np.ones_like(self.nz_dt[:,0])), has_rsd=False) for it in range(self.ntomo)]
        elif mode == 'II':
            tracer2 = [ccl.tracers.NumberCountsTracer(cosmo=self.ccl_cosmo,dndz=(self.nz_dt[:,0],self.nz_dt[:,it+1]), bias=(self.nz_dt[:,0],np.ones_like(self.nz_dt[:,0])), has_rsd=False) for it in range(self.ntomo)]
            tracer1 = tracer2
        if self.mbias_ is not None:
            Cls = np.array([[(1+self.mbias[i])*(1+self.mbias[j])*ccl.angular_cl(self.ccl_cosmo,tracer1[i],tracer2[j],self.ell_arr,p_of_k_a = pk2d) for i in range(self.ntomo)] for j in range(self.ntomo)])
        else:
            Cls = np.array([[ccl.angular_cl(self.ccl_cosmo,tracer1[i],tracer2[j],self.ell_arr,p_of_k_a = pk2d) for i in range(self.ntomo)] for j in range(self.ntomo)])
        return Cls
        
    def Cl_tomo2(self):
        chis,kernel0 = ccl.tracers.get_lensing_kernel(self.ccl_cosmo,dndz=(self.nz_dt[:,0],self.nz_dt[:,1]),n_chi=len(self.nz_dt[:,0]))
        self.wkernel = np.array([self.kernel_clean(self.ccl_cosmo,self.nz_dt[:,0],self.nz_dt[:,k+1]) for k in range(self.ntomo)])
        bnt_tracer = []
        for i in range(self.ntomo):
            Tracer_bnt = ccl.tracers.Tracer()
            if self.IA_on == False:
                Tracer_bnt.add_tracer(cosmo=self.ccl_cosmo,kernel=(chis,self.wkernel[i]),der_bessel=-1, der_angles=2)
            else:
                ta = (1./(1+(self.nz_dt[:,0])[::-1]), self.aia1[::-1])
                Tracer_bnt.add_tracer(cosmo=self.ccl_cosmo,kernel=(chis,self.wkernel[i]),transfer_a = self.ta,der_bessel=-1, der_angles=2)
            bnt_tracer.append(Tracer_bnt)
        if self.mbias_ is None:
            Cls = np.array([[ccl.angular_cl(self.ccl_cosmo,bnt_tracer[i],bnt_tracer[j],self.ell_arr,p_of_k_a = self.pk2d) for i in range(self.ntomo)] for j in range(self.ntomo)])
        else:
            Cls = np.array([[(1+self.mbias[i])*(1+self.mbias[j])*ccl.angular_cl(self.ccl_cosmo,bnt_tracer[i],bnt_tracer[j],self.ell_arr,p_of_k_a = self.pk2d) for i in range(self.ntomo)] for j in range(self.ntomo)])
        self.Cls = Cls
        
    def Cl_tomok(self):
        chis,kernel0 = ccl.tracers.get_lensing_kernel(self.ccl_cosmo,dndz=(self.nz_dt[:,0],self.nz_dt[:,1]),n_chi=len(self.nz_dt[:,0]))
        self.wkernel = np.array([self.kernel_clean(self.ccl_cosmo,self.nz_dt[:,0],self.nz_dt[:,k+1]) for k in range(self.ntomo)])
        kern_func = []
        for i in range(self.ntomo):
            kern_func.append( interp1d(chis,self.wkernel[i],kind='quadratic',fill_value='extrapolate'))
        if self.mbias_ == None:
            Cls = np.array([[fixed_quad(proj_int, 0, chis[-1], args=(self,kern_func[i],kern_func[j],self.kmax),n=1000)[0] for i in range(self.ntomo)] for j in range(self.ntomo)])
        else:
            Cls = np.array([[(1+self.mbias[i])*(1+self.mbias[j])*fixed_quad(proj_int, 0, chis[-1], args=(self,kern_func[i],kern_func[j],self.kmax),n=1000)[0] for i in range(self.ntomo)] for j in range(self.ntomo)])
        self.Cls = Cls
        
    def aCl_tomo(self):
        chis,kernel0 = ccl.tracers.get_lensing_kernel(self.ccl_cosmo,dndz=(self.nz_dt[:,0],self.nz_dt[:,1]),n_chi=len(self.nz_dt[:,0]))
        self.wkernels = np.array([self.kernel_clean(self.ccl_cosmo,self.nz_dt[:,0],self.nz_dt[:,k+1]) for k in range(self.ntomo)])
        self.vkernels = self.p_a_i@self.wkernels
        bnt_tracer = []
        for i in range(self.ntomo):
            Tracer_bnt = ccl.tracers.Tracer()
            if self.IA_on == False:
                if self.MG == False:
                    Tracer_bnt.add_tracer(cosmo=self.ccl_cosmo,kernel=(chis,self.vkernels[i]),der_bessel=-1, der_angles=2)
                elif self.MG == True:
                    Tracer_bnt._MG_add_tracer(cosmo=self.ccl_cosmo,kernel=(chis,self.vkernels[i]),z_b=self.nz_dt[:,0],der_bessel=-1, der_angles=2)
            else:
                ta = (1./(1+(self.nz_dt[:,0])[::-1]), self.aia1[::-1])
                if self.MG == False:
                    Tracer_bnt.add_tracer(cosmo=self.ccl_cosmo,kernel=(chis,self.vkernels[i]),transfer_a=ta,der_bessel=-1, der_angles=2)
                elif self.MG == True:
                    Tracer_bnt._MG_add_tracer(cosmo=self.ccl_cosmo,kernel=(chis,self.vkernels[i]),z_b=self.nz_dt[:,0],transfer_a=ta,der_bessel=-1, der_angles=2)
            bnt_tracer.append(Tracer_bnt)
        if self.mbias_ == None:
            self.aCls = np.array([[ccl.angular_cl(self.ccl_cosmo,bnt_tracer[i],bnt_tracer[j],self.ell_arr,p_of_k_a = self.pk2d) for i in range(self.ntomo)] for j in range(self.ntomo)])
        else:
            self.aCls = np.array([[(1+self.mbias[i])*(1+self.mbias[j])*ccl.angular_cl(self.ccl_cosmo,bnt_tracer[i],bnt_tracer[j],self.ell_arr,p_of_k_a = self.pk2d) for i in range(self.ntomo)] for j in range(self.ntomo)])
        #return aCls
        
    def Cl_tomo_ext2(self,pk2d=None,mode='GG'):
        if pk2d is None:
            if mode == 'GG':
                pk2d = self.pk2d
                if self.TATT_on == False:
                    self.tt_on = False
                else:
                    self.tt_on = True
            elif mode == 'GI' or 'IG':
                pk2d = self.pk2d_GI;self.tt_on = True
            elif mode == 'II':
                pk2d = self.pk2d_II;self.tt_on = True
        else:
            self.tt_on = True
        chis,kernel0 = ccl.tracers.get_lensing_kernel(self.ccl_cosmo,dndz=(self.nz_dt[:,0],self.nz_dt[:,1]),n_chi=len(self.nz_dt[:,0]))
        self.wkernel = np.array([self.kernel_clean(self.ccl_cosmo,self.nz_dt[:,0],self.nz_dt[:,k+1]) for k in range(self.ntomo)])
        #self.vkernels = self.p_a_i@self.wkernels
        #self.ntracers = [ccl.tracers.NumberCountsTracer(cosmo=self.ccl_cosmo,dndz=(self.nz_dt[:,0],self.nz_dt[:,it+1]), bias=(self.nz_dt[:,0],np.ones_like(self.nz_dt[:,0])), has_rsd=False) for it in range(self.ntomo)]
        self.nkernel = np.array([self.kernel_clean_N(self.ccl_cosmo,self.nz_dt[:,0],self.nz_dt[:,k+1]) for k in range(self.ntomo)])
        #self.mkernels = self.p_a_i @ self.nkernels
        nobnt_tracer1 = []
        nobnt_tracer2 = []
        for i in range(self.ntomo):
            Tracer_bnt = ccl.tracers.Tracer()
            Tracer_bia = ccl.tracers.Tracer()
            if self.IA_on == False or self.tt_on == True:
                if mode != 'II':
                    if self.MG == False:
                        Tracer_bnt.add_tracer(cosmo=self.ccl_cosmo,kernel=(chis,self.wkernel[i]),der_bessel=-1, der_angles=2)
                    elif self.MG == True:
                        Tracer_bnt._MG_add_tracer(cosmo=self.ccl_cosmo,kernel=(chis,self.wkernel[i]),z_b=self.nz_dt[:,0],der_bessel=-1, der_angles=2)
                    nobnt_tracer1.append(Tracer_bnt)
                if mode != 'GG':
                    Tracer_bia.add_tracer(cosmo=self.ccl_cosmo,kernel=(self.chi_N,self.nkernel[i]))
                    nobnt_tracer2.append(Tracer_bia)
            elif self.IA_on == True and mode != 'II':
                ta = (1./(1+(self.nz_dt[:,0])[::-1]), self.aia1[::-1])
                if self.MG == False:
                    Tracer_bnt.add_tracer(cosmo=self.ccl_cosmo,kernel=(chis,self.wkernel[i]),transfer_a=ta,der_bessel=-1, der_angles=2)
                elif self.MG == True:
                    Tracer_bnt._MG_add_tracer(cosmo=self.ccl_cosmo,kernel=(chis,self.wkernel[i]),z_b=self.nz_dt[:,0],transfer_a=ta,der_bessel=-1, der_angles=2)
                nobnt_tracer1.append(Tracer_bnt)
        if mode == 'GG':
            nobnt_tracer1 = nobnt_tracer1
            nobnt_tracer2 = nobnt_tracer1
        elif mode == 'GI':
            nobnt_tracer1 = nobnt_tracer1
            nobnt_tracer2 = nobnt_tracer2
        elif mode == 'II':
            nobnt_tracer1 = nobnt_tracer2
            nobnt_tracer2 = nobnt_tracer2
        if self.mbias_ == None:
            aCls = np.array([[ccl.angular_cl(self.ccl_cosmo,nobnt_tracer1[i],nobnt_tracer2[j],self.ell_arr,p_of_k_a = pk2d) for i in range(self.ntomo)] for j in range(self.ntomo)])
        else:
            aCls = np.array([[(1+self.mbias[i])*(1+self.mbias[j])*ccl.angular_cl(self.ccl_cosmo,bnt_tracer1[i],bnt_tracer2[j],self.ell_arr,p_of_k_a = pk2d) for i in range(self.ntomo)] for j in range(self.ntomo)])
        return aCls
                         
    def aCl_tomo_ext(self,pk2d=None,mode='GG'):
        if pk2d is None:
            if mode == 'GG':
                pk2d = self.pk2d
                if self.TATT_on == False:
                    self.tt_on = False
                else:
                    self.tt_on = True
            elif mode == 'GI' or 'IG':
                pk2d = self.pk2d_GI;self.tt_on = True
            elif mode == 'II':
                pk2d = self.pk2d_II;self.tt_on = True
        else:
            self.tt_on = True
        chis,kernel0 = ccl.tracers.get_lensing_kernel(self.ccl_cosmo,dndz=(self.nz_dt[:,0],self.nz_dt[:,1]),n_chi=len(self.nz_dt[:,0]))
        self.wkernels = np.array([self.kernel_clean(self.ccl_cosmo,self.nz_dt[:,0],self.nz_dt[:,k+1]) for k in range(self.ntomo)])
        self.vkernels = self.p_a_i@self.wkernels
        #self.ntracers = [ccl.tracers.NumberCountsTracer(cosmo=self.ccl_cosmo,dndz=(self.nz_dt[:,0],self.nz_dt[:,it+1]), bias=(self.nz_dt[:,0],np.ones_like(self.nz_dt[:,0])), has_rsd=False) for it in range(self.ntomo)]
        self.nkernels = np.array([self.kernel_clean_N(self.ccl_cosmo,self.nz_dt[:,0],self.nz_dt[:,k+1]) for k in range(self.ntomo)])
        self.mkernels = self.p_a_i @ self.nkernels
        bnt_tracer1 = []
        bnt_tracer2 = []
        for i in range(self.ntomo):
            Tracer_bnt = ccl.tracers.Tracer()
            Tracer_bia = ccl.tracers.Tracer()
            if self.IA_on == False or self.tt_on == True:
                if mode != 'II':
                    if self.MG == False:
                        Tracer_bnt.add_tracer(cosmo=self.ccl_cosmo,kernel=(chis,self.vkernels[i]),der_bessel=-1, der_angles=2)
                    elif self.MG == True:
                        Tracer_bnt._MG_add_tracer(cosmo=self.ccl_cosmo,kernel=(chis,self.vkernels[i]),z_b=self.nz_dt[:,0],der_bessel=-1, der_angles=2)
                    bnt_tracer1.append(Tracer_bnt)
                if mode != 'GG':
                    Tracer_bia.add_tracer(cosmo=self.ccl_cosmo,kernel=(self.chi_N,self.mkernels[i]))
                    bnt_tracer2.append(Tracer_bia)
            elif self.IA_on == True and mode != 'II':
                ta = (1./(1+(self.nz_dt[:,0])[::-1]), self.aia1[::-1])
                if self.MG == False:
                    Tracer_bnt.add_tracer(cosmo=self.ccl_cosmo,kernel=(chis,self.vkernels[i]),transfer_a=ta,der_bessel=-1, der_angles=2)
                elif self.MG == True:
                    Tracer_bnt._MG_add_tracer(cosmo=self.ccl_cosmo,kernel=(chis,self.vkernels[i]),z_b=self.nz_dt[:,0],transfer_a=ta,der_bessel=-1, der_angles=2)
                bnt_tracer1.append(Tracer_bnt)
        if mode == 'GG':
            bnt_tracer2 = bnt_tracer1
        elif mode == 'GI':
            bnt_tracer1 = bnt_tracer1
            bnt_tracer2 = bnt_tracer2
        elif mode == 'II':
            bnt_tracer1 = bnt_tracer2
        if self.mbias_ == None:
            aCls = np.array([[ccl.angular_cl(self.ccl_cosmo,bnt_tracer1[i],bnt_tracer2[j],self.ell_arr,p_of_k_a = pk2d) for i in range(self.ntomo)] for j in range(self.ntomo)])
        else:
            aCls = np.array([[(1+self.mbias[i])*(1+self.mbias[j])*ccl.angular_cl(self.ccl_cosmo,bnt_tracer1[i],bnt_tracer2[j],self.ell_arr,p_of_k_a = pk2d) for i in range(self.ntomo)] for j in range(self.ntomo)])
        return aCls
        
    def aCl_tomok(self):
        chis,kernel0 = ccl.tracers.get_lensing_kernel(self.ccl_cosmo,dndz=(self.nz_dt[:,0],self.nz_dt[:,1]),n_chi=len(self.nz_dt[:,0]))
        self.wkernels = np.array([self.kernel_clean(self.ccl_cosmo,self.nz_dt[:,0],self.nz_dt[:,k+1]) for k in range(self.ntomo)])
        self.vkernels = self.p_a_i@self.wkernels
        kern_func = []
        for i in range(self.ntomo):
            kern_func.append( interp1d(chis,self.vkernels[i],kind='quadratic',fill_value='extrapolate'))
        if self.mbias_ == None:
            aCls = np.array([[fixed_quad(proj_int, 0, chis[-1], args=(self,kern_func[i],kern_func[j],self.kmax),n=1000)[0] for i in range(self.ntomo)] for j in range(self.ntomo)])
        else:
            aCls = np.array([[(1+self.mbias[i])*(1+self.mbias[j])*fixed_quad(proj_int, 0, chis[-1], args=(self,kern_func[i],kern_func[j],self.kmax),n=1000)[0] for i in range(self.ntomo)] for j in range(self.ntomo)])
        self.aCls = aCls
        
    def Cl_vec(self,klim=False):
        cosmo = self.ccl_cosmo
        ell_arr = self.ell_arr
        ntomo = self.ntomo
        nell = self.nell
        ltomo = int(ntomo*(ntomo+1)/2)
        if klim == False:
            if self.TATT_on == False:
                self.Cl_tomo()
            else:
                self.Cls_gg = self.Cl_tomo_ext2(mode='GG')
                self.Cls_gi = self.Cl_tomo_ext2(mode='GI')
                self.Cls_ii = self.Cl_tomo_ext2(mode='II')
                self.Cls = self.Cls_gg+self.Cls_gi+self.Cls_ii
        else:
            self.Cl_tomok()
        self.Cl_deorg = np.zeros(ltomo*nell)
        if self.TATT_on == True:
            self.Cl_gg_deorg = np.zeros(ltomo*nell)
            self.Cl_gi_deorg = np.zeros(ltomo*nell)
            self.Cl_ii_deorg = np.zeros(ltomo*nell)
        q = 0
        for i in range(ntomo):
            for j in range(i,ntomo):
                self.Cl_deorg[q*nell:(q+1)*nell] = self.Cls[i,j]
                if self.TATT_on == True:
                    self.Cl_gg_deorg[q*nell:(q+1)*nell] = self.Cls_gg[i,j]
                    self.Cl_gi_deorg[q*nell:(q+1)*nell] = self.Cls_gi[i,j]
                    self.Cl_ii_deorg[q*nell:(q+1)*nell] = self.Cls_ii[i,j]
                q += 1
                
    def aCl_vec_reorg(self,klim=False):
        cosmo = self.ccl_cosmo
        ell_arr = self.ell_arr
        ntomo = self.ntomo
        nell = self.nell
        if klim == False:
            if self.TATT_on == False:
                self.aCl_tomo()
            else:
                self.aCls_gg = self.aCl_tomo_ext(mode='GG')
                self.aCls_gi = self.aCl_tomo_ext(mode='GI')
                self.aCls_ii = self.aCl_tomo_ext(mode='II')
                self.aCls = self.aCls_gg+self.aCls_gi+self.aCls_ii
        else:
            self.aCl_tomok()
        ltomo = int(ntomo*(ntomo+1)/2)
        self.aCl_reorg = np.zeros(ltomo*nell)
        if self.TATT_on == True:
            self.aCl_gg_reorg = np.zeros(ltomo*nell)
            self.aCl_gi_reorg = np.zeros(ltomo*nell)
            self.aCl_ii_reorg = np.zeros(ltomo*nell)
        q = 0
        for i in range(ntomo):
            for j in range(i,ntomo):
                self.aCl_reorg[q::ltomo] = self.aCls[i,j]
                if self.TATT_on == True:
                    self.aCl_gg_reorg[q::ltomo] = self.aCls_gg[i,j]
                    self.aCl_gi_reorg[q::ltomo] = self.aCls_gi[i,j]
                    self.aCl_ii_reorg[q::ltomo] = self.aCls_ii[i,j]
                q += 1