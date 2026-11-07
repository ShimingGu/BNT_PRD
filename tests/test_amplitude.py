import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from bnt_nautilus.core import bnt_core


class AmplitudeTest(unittest.TestCase):
    def core(self, s8, amplitude):
        return SimpleNamespace(
            nl_recipe="mead2020_feedback", B=3.13, TAGN=7.8,
            transfer="boltzmann_camb", nl_code="camb", S8=s8,
            Om=0.35, Oc=0.30, Ob=0.05, h=0.7, ns=0.96,
            M_nu=0.06, w_0=-1.0, w_a=0.0, As=amplitude,
        )

    def test_s8_uses_one_amplitude_correction(self):
        core = self.core(s8=0.8, amplitude=2e-9)
        target = 0.8 * np.sqrt(0.3 / 0.35)
        initial_as = 2.18676e-9 * (target / 0.826) ** 2
        initial_cosmo, final_cosmo = object(), object()
        with patch("bnt_nautilus.core.ccl.Cosmology", side_effect=[initial_cosmo, final_cosmo]) as cosmology, patch(
            "bnt_nautilus.core.ccl.power.sigma8", return_value=0.9
        ) as sigma8:
            bnt_core._inflate_core(core)
        self.assertEqual(cosmology.call_count, 2)
        sigma8.assert_called_once_with(initial_cosmo)
        self.assertAlmostEqual(cosmology.call_args_list[0].kwargs["A_s"] / initial_as, 1.0)
        self.assertAlmostEqual(core.As / (initial_as * (target / 0.9) ** 2), 1.0)
        self.assertEqual(cosmology.call_args_list[1].kwargs["A_s"], core.As)
        self.assertIs(core.ccl_cosmo, final_cosmo)

    def test_direct_as_stays_unchanged(self):
        core = self.core(s8=-1.0, amplitude=2.3e-9)
        with patch("bnt_nautilus.core.ccl.Cosmology") as cosmology:
            bnt_core._inflate_core(core)
        cosmology.assert_called_once()
        self.assertEqual(cosmology.call_args.kwargs["A_s"], 2.3e-9)
        self.assertEqual(core.As, 2.3e-9)

    def test_matter_only_calibration_is_opt_in(self):
        core = self.core(s8=0.8, amplitude=2e-9)
        with patch("bnt_nautilus.core.ccl.Cosmology") as cosmology, patch(
            "bnt_nautilus.core.matter_only_sigma8", return_value=0.9
        ) as calibration, patch("bnt_nautilus.core.ccl.power.sigma8") as sigma8:
            bnt_core._inflate_core(core, sigma8_normalization="camb_matter")
        cosmology.assert_called_once()
        calibration.assert_called_once()
        sigma8.assert_not_called()
        self.assertAlmostEqual(core.As / (2.18676e-9 * (core.sig8 / 0.826) ** 2 * (core.sig8 / 0.9) ** 2), 1.0)

    def test_matter_only_rejects_unvalidated_dark_energy(self):
        core = self.core(s8=0.8, amplitude=2e-9)
        core.w_0 = -0.9
        with self.assertRaisesRegex(ValueError, "LCDM"):
            bnt_core._inflate_core(core, sigma8_normalization="camb_matter")

    def test_matter_only_rejects_unvalidated_neutrino_mass(self):
        core = self.core(s8=0.8, amplitude=2e-9)
        core.M_nu = 0.1
        with self.assertRaisesRegex(ValueError, "m_nu"):
            bnt_core._inflate_core(core, sigma8_normalization="camb_matter")


if __name__ == "__main__":
    unittest.main()
