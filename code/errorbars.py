import matplotlib.pyplot as plt
import numpy as np
from math import e
from datetime import datetime

def infidelity(true, pred):
    true_np, pred_np = np.array(true), np.array(pred)
    true_np, pred_np = np.power(e, true_np), np.power(e, pred_np) # converting from log amplitudes
    
    
    mult = true_np * pred_np
    #root = np.sqrt(mult)
    sum_r = np.sum(mult)
    sum_r = np.abs(sum_r)
    fid = sum_r ** 2
        
    
    infid = 1 - fid
    infid = float(infid)
    return infid

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path_infidelity_regimes = f'NQS-Bench-101\\curves\\infidelity_regimes_{timestamp}.png'

f3 = plt.figure()
ax3 = f3.add_subplot(1,1,1)
regimes_list = [r'$1.0^{-6}$', r'$0.5$', r'$1.0$', r'$2.0$']
errors = [6.9, 6.9, 6.9, 6.9]
ax3.set_ylabel(r"infidelity ($\mathcal{I}$)")
ax3.set_xlabel(r"regimes ($h$ value)")

h1e6_target_path = 'NQS-Bench-101\\all_architectures\\20260511_142441\\h_1_0e6\\32_epochs\\0_hidden_layers\\gelu\evaluation_metrics\\target_test_h1_0e6.txt'
h1e6_pred_path = 'NQS-Bench-101\\all_architectures\\20260511_142441\\h_1_0e6\\32_epochs\\0_hidden_layers\\gelu\evaluation_metrics\\pred_test_h1_0e6.txt'
h1e6_targets = []
h1e6_preds = []

h1_0_target_path = 'NQS-Bench-101\\all_architectures\\20260513_142305\\h_1_0\\300_epochs\\0_hidden_layers\\gelu\\evaluation_metrics\\target_test_h1_0.txt'
h1_0_pred_path = 'NQS-Bench-101\\all_architectures\\20260513_142305\\h_1_0\\300_epochs\\0_hidden_layers\\gelu\\evaluation_metrics\\pred_test_h1_0.txt'
h1_0_targets = []
h1_0_preds = []

h0_5_target_path = 'NQS-Bench-101\\all_architectures\\20260513_134647\\h_0_5\\300_epochs\\0_hidden_layers\\gelu\\evaluation_metrics\\target_test_h0_5.txt'
h0_5_pred_path = 'NQS-Bench-101\\all_architectures\\20260513_134647\\h_0_5\\300_epochs\\0_hidden_layers\\gelu\\evaluation_metrics\\pred_test_h0_5.txt'
h0_5_targets = []
h0_5_preds = []

h2_0_target_path = 'NQS-Bench-101\\all_architectures\\20260513_145949\\h_2_0\\300_epochs\\0_hidden_layers\\gelu\\evaluation_metrics\\target_test_h2_0.txt'
h2_0_pred_path = 'NQS-Bench-101\\all_architectures\\20260513_145949\\h_2_0\\300_epochs\\0_hidden_layers\\gelu\\evaluation_metrics\\pred_test_h2_0.txt'
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

infid_h1e6 = infidelity(h1e6_targets, h1e6_preds)
infid_h0_5 = infidelity(h0_5_targets, h0_5_preds)
infid_h1_0 = infidelity(h1_0_targets, h1_0_preds)
infid_h2_0 = infidelity(h2_0_targets, h2_0_preds)

infidelities = [infid_h1e6, infid_h0_5, infid_h1_0, infid_h2_0]
#infidelities = [1, 2, 3, 4]
ax3.errorbar(regimes_list, infidelities, capsize=7, fmt='ko-')
ax3.set_title("(NQS-Bench-101): Infidelity values for different regimes")
ax3.grid(True)
f3.savefig(save_path_infidelity_regimes, dpi=300, bbox_inches="tight")