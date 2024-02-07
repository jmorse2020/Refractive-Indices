#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 10:43:25 2023

@author: jackmorse


Refractive indices of known materials.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



class RefractiveIndex:
    def __init__(self):
        pass

    def _nm2um(nm):
        return nm / 1000
    
    def _nm2m(nm):
        return nm / 1e9
    
    def n_air(wavelengths):
        '''
        Refractive index of air is assumed unity, the function returns an array of ones the same size as wavelengths
        '''
        return np.ones_like(wavelengths)
    
    def n_fs(variable, parameter = "wavelength"):
        """
        refractive index of fused silica.
        
        Fused Silica based on Selmeier equation found at https://refractiveindex.info/?shelf=glass&book=fused_silica&page=Malitson
        around 1 micron.
    

        Parameters
        -------
        variable
        parameter = "wavelength" [nm] or "omega" [rad Hz]
         
    
        Returns
        -------
        value for refractive index of fused silica at wavelength.
        """
        if parameter.lower() not in ["wavelength", "omega"]:
            raise NotImplementedError("Error: Input parameter needs to match 'wavelength' or 'omega'")
        if parameter.lower() == "omega":
            variable = 2 * np.pi * 3e17 / variable # Converts from omega to wavelength in nm, c is s.o.l in nm/s.
            
        wavelength = RefractiveIndex._nm2um(variable)
        return np.sqrt(1 + (0.6961663 * wavelength**2) / (wavelength**2  - 0.0684043**2) + (0.4079426 * wavelength**2) / (wavelength**2 - 0.1162414**2) + (0.8974794 * wavelength**2) / (wavelength**2 - 9.896161**2))

    def _find_bessel_zero(m, n):
        from scipy.special import jn_zeros
        if not (isinstance(m, int) and isinstance(n, int)):
            raise ValueError("Arguments to function '_find_bessel_zero' must be of type 'int'")
            return
        # m is the order of the Bessel function
        # n is the index of the zero (0-based)

        # Using jn_zeros to get the zeros of the Bessel function
        zeros = jn_zeros(m, n+1)  # n+1 because the function is 0-indexed

        # Return the nth zero
        return zeros[n]
    
    def HCF(wavelengths, mode = [1, 1], n_gas = None, n_wall = None, R = 20e-6, w=0.7e-6, part = "Real"):
        '''
        Refractive index of hollow-core fibre, model based on Zeisberger paper (doi: 10.1038/s41598-017-12234-5) 'Analytic model for the complex
        effective index of the leaky modes of tube-type anti-resonant hollow-core fibers'

        The function outputs the loss or real part of the refractive index. 

        Parameters
        -------
        wavelengths ([float]): An array of wavelengths in [nm]
        mode ([int, int]): Mode of propagation HE_nm to model (Default [1, 1])
        n_gas (func): Refractive index of gas filling fibre. Must take arguments wavelength [nm]. (Default air = 1.0).
        n_wall (func): Refractive index of the capillary wall. Must take arguments wavelength [nm]. (Default fused silica). 
        R (float): Radius of fibre core [m] 
        w (float): Wall thickness [m]
         
    
        Returns
        -------
        
        '''
        import numpy as np
        if n_gas is None:
            n_gas = RefractiveIndex.n_air
        if n_wall is None:
            n_wall = RefractiveIndex.n_fs
        # *** Convert to SI units *** #
        c = 3e8                                                                 # SOL in ms^-1
        lambdas = RefractiveIndex._nm2m(wavelengths)

        # *** Obtaining parameters *** #
        jz = RefractiveIndex._find_bessel_zero(mode[0] - 1, mode[1] - 1)        # for j_(m-1),n where the nth root here is the "first root of the function", i.e. the zeroth root of Bessel funciton on order m
        j1nz = RefractiveIndex._find_bessel_zero(1, mode[1] - 1) 
        k_0_lambda = 2 * np.pi / lambdas                                        # Vacuum wavenumber 
        n_wall_lambda = n_wall(wavelengths)                                        

        # *** Computing the refractive index from Eq. 25 in Zeisberger paper *** #
        phi_lambda = k_0_lambda * w * np.sqrt(n_wall_lambda**2 - n_gas(wavelengths)**2)
        epsilon_lambda = n_wall_lambda**2/(n_gas(wavelengths)**2)

        HCF_refractive_indices = []
        real_keyword_array = ["r", "re", "real"]
        imaginary_keyword_array = ["i", "im", "imag", "imaginary"]
        # *** Effective index real *** #
        if part.lower() in real_keyword_array:
            n_eff_lambda = n_gas(wavelengths) -(jz**2)/(2 * k_0_lambda**2 * n_gas(wavelengths) * R**2) - (jz**2)/(k_0_lambda**3 * n_gas(wavelengths)**2 * R**3)*(1)/(np.tan(phi_lambda) * np.sqrt(epsilon_lambda - 1))*((epsilon_lambda + 1))/(2)
            HCF_refractive_indices = n_eff_lambda
        # *** Effective index imaginary (loss) *** #
        elif part.lower() in imaginary_keyword_array:
            print("In,,,,,")
            sigma = 1 / (k_0_lambda * n_gas(wavelengths) * R)
            d = j1nz**3 / (epsilon_lambda * (epsilon_lambda - 1)) * (1 + (1 / np.tan(phi_lambda)**2))
            Im_n_eff = n_gas(wavelengths) * d * sigma**4
            HCF_refractive_indices = Im_n_eff
        else:
            print(f"Argument 'part' in HCF should be one of: \nFor the real part of the index: {real_keyword_array}\nFor the imaginary (loss) part of the index: {imaginary_keyword_array}")
            return ValueError("Argument value for 'part' unrecognised.")
        return HCF_refractive_indices

    def _deriv(f, wavelength, h=1e-6):
        return (1/(2  * h)) * (f(wavelength + h) - f(wavelength - h))
    
    def n_group(n, wavelength, h=1e-6):
        return n(wavelength) - wavelength * RefractiveIndex._deriv(n, wavelength, h)
    
