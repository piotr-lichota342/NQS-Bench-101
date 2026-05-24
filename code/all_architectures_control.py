import config
import os
import subprocess
from datetime import datetime
import itertools
from tqdm import tqdm

epoch_range = [300,500]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")



regimes_combinations = ["0100"]
hidden_layers_range = range(0,1)
act_functions_range = range(0,1)



    
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
    env["timestamp"] = timestamp
    
    for n_epoch in epoch_range:
        print(f"Number of epochs: {n_epoch}")
        env["EPOCHS"] = str(n_epoch)
        
        for h_layer in hidden_layers_range:
            print(f"Number of hidden layers {h_layer}")
            env["HIDDEN_LAYERS"] = str(h_layer)
            
            for act_fn in act_functions_range:
                print(f"Activation function: {act_fn}")
                env["ACT_FUNCTION"] = str(act_fn)
                
                subprocess.run(
                    ['python', 'NQS-Bench-101/code/generating_all_architectures.py'],
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
        #os.system('python NQS-Bench-101\code\generating_all_architectures.py')
'''        

