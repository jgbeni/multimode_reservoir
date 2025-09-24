#!/usr/bin/env python
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import datetime

def plot_results(results, lambda_central, window, save):
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, width_ratios=[2, 2])

    ax_signal = fig.add_subplot(gs[0, 0])
    ax_idler = fig.add_subplot(gs[1, 0])
    ax_coefficients = fig.add_subplot(gs[:, 1])

    # Plot the signal temporal modes
    for ii in range(5):
        _smode = results.signal_temporal_modes[:, ii]
        ax_signal.plot(results.signal_range * 1E9, np.abs(_smode), lw=1.0, label=f"Mode nb. {ii}")
    ax_signal.set_xlabel("Signal frequency (nm)")
    ax_signal.set_ylabel("Amplitude (arb. u.)")
    ax_signal.set_title("Signal temporal modes")
    ax_signal.set_xlim([(lambda_central-window)*10**9, (lambda_central+window)*10**9])
    ax_signal.legend()

    # Plot the idler temporal modes
    for ii in range(5):
        _imode = results.idler_temporal_modes[:, ii]
        ax_idler.plot(results.idler_range * 1E9, np.abs(_imode), lw=1.0, label=f"Mode nb. {ii}")
    ax_idler.set_xlabel("Idler wavelength (nm)")
    ax_idler.set_ylabel("Amplitude (arb. u.)")
    ax_idler.set_title("Idler temporal modes")
    ax_idler.set_xlim([(lambda_central-window)*10**9, (lambda_central+window)*10**9])
    ax_idler.legend()

    # Determine Schmidt coefficients to plot
    for ii in range(len(results.schmidt_coeffs)):
        if results.schmidt_coeffs[ii] < (0.08 * results.schmidt_coeffs[0]):
            schdmidt_min = ii
            break
    Schmidt_coeff = results.schmidt_coeffs[:schdmidt_min]

    # Plot the Schmidt coefficients
    posicion = np.arange(len(Schmidt_coeff))
    ax_coefficients.bar(posicion, Schmidt_coeff, color=[0.6, 0.1, 0.3])
    ax_coefficients.set_xlabel("Order")
    ax_coefficients.set_ylabel("Schmidt Coefficient")
    ax_coefficients.set_xlim(0, len(Schmidt_coeff))
    ax_coefficients.set_title("Schmidt Coefficients")

    plt.tight_layout()
    plt.show()

    print(f'Schmidt coefficients = {results.schmidt_coeffs[:5]}')

    # Additional contour plots
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

       # |JSA| plot
    axs[0].contourf(results.signal_rebin * 1E9, results.idler_rebin * 1E9, abs(results.jsa_rebin), 100, extend='neither')
    axs[0].contour(results.signal_rebin * 1E9, results.idler_rebin * 1E9, abs(results.pump_rebin), [0.5 * abs(results.pump_rebin).max()], colors="white", linewidths=0.2)
    axs[0].contour(results.signal_rebin * 1E9, results.idler_rebin * 1E9, abs(results.phasematching_rebin), [0.5 * abs(results.phasematching_rebin).max()], colors="white", linewidths=0.2, linestyles="--")
    axs[0].set_xlabel("Signal wavelength (nm)")
    axs[0].set_ylabel("Idler wavelength (nm)")
    axs[0].axis('square')
    axs[0].set_title("|JSA|, K = %.1f" % (1.0 / sum(results.schmidt_coeffs ** 4)))
    
    # Pump plot
    axs[1].contourf(results.signal_rebin * 1E9, results.idler_rebin * 1E9, np.real(results.pump_rebin), 100, extend='neither')
    axs[1].set_xlabel("Signal wavelength (nm)")
    axs[1].set_ylabel("Idler wavelength (nm)")
    axs[1].axis('square')
#     axs[1].set_xlim([(lambda_central-window/20)*10**9, (lambda_central+window/20)*10**9])
#     axs[1].set_ylim([(lambda_central-window/20)*10**9, (lambda_central+window/20)*10**9])
    fig.colorbar(axs[1].contourf(results.signal_rebin * 1E9, results.idler_rebin * 1E9, np.real(results.pump_rebin), 100, extend='neither'), ax=axs[1])
    axs[1].set_title("Pump")
    
    # Phasematching plot
    axs[2].contourf(results.signal_rebin * 1E9, results.idler_rebin * 1E9, results.phasematching_rebin, 100, extend='neither')
    axs[2].set_xlabel("Signal wavelength (nm)")
    axs[2].set_ylabel("Idler wavelength (nm)")
    axs[2].axis('square')
    fig.colorbar(axs[2].contourf(results.signal_rebin * 1E9, results.idler_rebin * 1E9, results.phasematching_rebin, 100, extend='neither'), ax=axs[2])
    axs[2].set_title("Phasematching")
    
    if save: 
        now = datetime.datetime.now()
        date_str = now.strftime("%Y-%m-%d %H:%M:%S") 

        # Save the full plot as an image
        plt.tight_layout()
        file_name = f'jsa_pump_phasematching_plot_{now.strftime("%Y%m%d_%H%M%S")}.pdf'
        plt.savefig(file_name, bbox_inches='tight')
        plt.show()

        # Save the plot of pump vs. interv as a separate image
        interv=np.arange(770e-9,790e-9,0.01e-9) 
        fig = plt.figure(figsize=(3, 2))
        plt.plot(interv, np.real(results.pump_interv[:len(interv)]))

        # Save the second plot
        file_name_pump = f'pump_vs_interv_plot_{now.strftime("%Y%m%d_%H%M%S")}.pdf'
        plt.savefig(file_name_pump, bbox_inches='tight') 
        plt.show()

