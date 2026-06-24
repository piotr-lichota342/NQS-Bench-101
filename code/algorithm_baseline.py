import random

# Search space

depth_range = [1,4]
activation_functions = ['relu', 'gelu', 'tanh']
width_ranges = [8,16,32,64,128,256,512,1024,2048]
is_custom = [0,2]

chosen_depth = random.randint(min(depth_range), max(depth_range))  
custom_arch = random.randint(min(is_custom), max(is_custom))
chosen_activation = random.choice(activation_functions)
width_sequence = []

if custom_arch:
    for hl in range(max(depth_range)-2):
        width_sequence.append(random.choice(width_ranges))



print("The randomly chosen architecture from the search space:\n")
print(f"Depth: {chosen_depth}")
print(f"Custom arch: {custom_arch}")
print(f"Activation function: {chosen_activation}")
if custom_arch:
    print(f"Hidden layers sequence: {width_sequence}")
else:
    print(f"Hidden layers sequence: fixed")

