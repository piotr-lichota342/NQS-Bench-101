import matplotlib.pyplot as plt
import numpy as np
from math import e
from datetime import datetime


def infidelities_graph(epochs, save_path_infidelity_regimes, infidelity_f, h1e6_target_path, h1e6_pred_path, h1_0_target_path, h1_0_pred_path, h0_5_target_path, h0_5_pred_path, h2_0_target_path, h2_0_pred_path):

    f3 = plt.figure()
    ax3 = f3.add_subplot(1,1,1)
    regimes_list = [r'$10^{-6}$', r'$0.5$', r'$1.0$', r'$2.0$']
    #errors = [6.9, 6.9, 6.9, 6.9]
    ax3.set_ylabel(r"infidelity ($\mathcal{I}$)")
    ax3.set_xlabel(r"regimes ($h$ value)")


    h1e6_targets = []
    h1e6_preds = []

    h1_0_targets = []
    h1_0_preds = []

    h0_5_targets = []
    h0_5_preds = []

    h2_0_targets = []
    h2_0_preds = []

    file = open(h1e6_pred_path, "r")
    for line in file:
        h1e6_preds.append(float(line)) 
    file.close()

    file = open(h1e6_target_path, "r")
    for line in file:
        h1e6_targets.append(float(line))  
    file.close()

    file = open(h0_5_pred_path, "r")
    for line in file:
        h0_5_preds.append(float(line))  
    file.close()

    file = open(h0_5_target_path, "r")
    for line in file:
        h0_5_targets.append(float(line))  
    file.close()

    file = open(h1_0_pred_path, "r")
    for line in file:
        h1_0_preds.append(float(line))  
    file.close()

    file = open(h1_0_target_path, "r")
    for line in file:
        h1_0_targets.append(float(line))  
    file.close()

    file = open(h2_0_pred_path, "r")
    for line in file:
        h2_0_preds.append(float(line))  
    file.close()

    file = open(h2_0_target_path, "r")
    for line in file:
        h2_0_targets.append(float(line))  
    file.close()

    infid_h1e6 = infidelity_f(h1e6_targets, h1e6_preds)
    print(f"infid_h1e6={infid_h1e6}")
    infid_h0_5 = infidelity_f(h0_5_targets, h0_5_preds)
    print(f"infid_h0_5={infid_h0_5}")
    infid_h1_0 = infidelity_f(h1_0_targets, h1_0_preds)
    infid_h2_0 = infidelity_f(h2_0_targets, h2_0_preds)

    infidelities = [infid_h1e6, infid_h0_5, infid_h1_0, infid_h2_0]
    #infidelities = [1, 2, 3, 4]
    ax3.errorbar(regimes_list, infidelities, capsize=7, fmt='ko-')
    ax3.set_title("(NQS-Bench-101): Infidelity values for different regimes" + f"\nEPOCHS={epochs}")
    ax3.grid(True)
    f3.savefig(save_path_infidelity_regimes, dpi=300, bbox_inches="tight")

    
