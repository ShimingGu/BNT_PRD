"""Optional fast estimates for the primordial-amplitude calibration."""


def matter_only_sigma8(omega_c, omega_b, h, n_s, amplitude, m_nu=0.06):
    """Estimate sigma8 with a z=0, linear, matter-only CAMB calculation.

    This approximation is validated separately against the CCL/CAMB path.
    Its neutrino conventions and sigma8 integration are not identical to CCL's.
    """
    import camb

    params = camb.CAMBparams()
    params.set_cosmology(H0=h * 100, ombh2=omega_b, omch2=omega_c,
                         mnu=m_nu, nnu=3.046)
    params.InitPower.set_params(As=amplitude, ns=n_s)
    params.WantCls = False
    params.DoLensing = False
    params.set_matter_power(redshifts=[0.0], kmax=2.0, nonlinear=False, silent=True)
    params.NonLinear = camb.model.NonLinear_none
    return float(camb.get_results(params).get_sigma8_0())
