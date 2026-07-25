import config
import os
import subprocess
from datetime import datetime
import itertools
from tqdm import tqdm
import multiprocessing as mp
import socket
from time import time
from config import CUSTOM_ARCH, WIDTH_SEQUENCE
from itertools import permutations, product

epoch_range = [30,60]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

widths_range = [16, 32, 64, 128, 256, 512, 2048, 4096]

width_set = [16, 32, 64, 128, 256, 512, 2048, 4096]
#width_set = [8,16,32,64,128,256,512,2048]

result = []


#print("All combinations: ", result)

regimes_combinations = ["0100", "0010", "0001"]
hidden_layers_range = range(1,4)
act_functions_range = range(0,3)

for k in hidden_layers_range:
    result.extend([",".join(map(str,combo)) for combo in product(width_set,repeat=k)])


width_sequences = result





    
"""
for combination in dict_regimes_combinations:
        for index, (key, value) in enumerate(trained_regimes.items()):
            print(index)
            trained_regimes[key]=combination[index]
"""
    
    
for regim_comb in regimes_combinations:
    print(f"Current regime: {regim_comb}")
    env = os.environ.copy()
    env["trained_regimes"] = regim_comb
    env["TIMESTAMP"] = timestamp
    
    
    for n_epoch in max([epoch_range]):
        print(f"Number of epochs: {n_epoch}")
        env["EPOCHS"] = str(n_epoch)
        
        if not CUSTOM_ARCH:
            for h_layer in hidden_layers_range:
                print(f"Number of hidden layers {h_layer}")
                env["HIDDEN_LAYERS"] = str(h_layer)
                
                for act_fn in act_functions_range:
                    print(f"Activation function: {act_fn}")
                    env["ACT_FUNCTION"] = str(act_fn)

                    for width in widths_range:
                        print(f"Current width (W): {width}")
                        env["W"] = str(width)
                    
                        subprocess.run(
                            ['python', 'code/generating_all_architectures.py'],
                            env=env
                        )
        else:
            for ws in width_sequences:
                env["HIDDEN_LAYERS"] = "custom"
                env["WIDTH_SEQUENCE"] = ws
                print(f"This is a custom architecture with hidden layer sequence: {WIDTH_SEQUENCE}")
                
                for act_fn in act_functions_range:
                    print(f"Activation function: {act_fn}")
                    env["ACT_FUNCTION"] = str(act_fn)

        
                    print(f"This is a custom architecture with width sequence: {WIDTH_SEQUENCE}")
                    env["W"] = ws.replace(",", "_")
                
                    subprocess.run(
                        ['python', 'code/generating_all_architectures.py'],
                        env=env
                    )


'''
for n_epoch in epoch_range:
    env = os.environ.copy()
    env["EPOCHS"] = str(n_epoch)
    env["TIMESTAMP"] = timestamp
    
    for regim_comb in regimes_combinations:

        env["trained_regimes"] = regim_comb


        subprocess.run(
            ['python', 'NQS-Bench-101/code/generating_all_architectures.py'],
            env=env
        )
        #os.system('python NQS-Bench-101/code/generating_all_architectures.py')
'''        

