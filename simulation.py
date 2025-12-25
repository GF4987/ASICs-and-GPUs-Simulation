'''Installing Dependencies'''
from litellm import model_list
import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import argparse

#MACS: Multiply-Accumalate Operator
mac_gpu = 5e-9 #GPU Mac; 5nJ of energy (in Joules)
mac_asic = 0.2e-9 #ASIC Mac; 0.2nJ of energy (in Joules)

#Energy per DRAM access (per 32-bit word)
DRAM_energy = 100e-9 #100nJ of energy

#DRAM acess per MAC
DRAM_words_per_mac = 0.5

#C02 per kWh
carb_factor = 0.45 #US Average

#Joules to kwH conversion
def joules_to_kWh(joules):
    conversion = joules / 3.6e6
    return conversion

'''MAC Simulation Functions'''

def count_layer_macs(layer, x, y):
    #Will return the number of MACS for a forward pass of one layer given an input tensor x
    # and output y
    
    if isinstance(layer, nn.Conv2d):
        in_c = layer.in_channels
        out_c = layer.out_channels
        kh, kw = layer.kernel_size
        oh, ow = y.shape[2], y.shape[3]
        batch = y.shape[0]
        
        #MACs per pixel output (in_c = kh * kw)
        macs_per_pixel = in_c * kh * kw
        total_macs = macs_per_pixel * oh * ow * out_c * batch
        return total_macs
    
    #linear/dense
    elif isinstance(layer, nn.Linear):
        batch = y.shape[0]
        in_f = layer.in_features
        out_f = layer.out_features
        total_macs = in_f * out_f * batch
        return total_macs
    
    else:
        return 0

def profile_model(model, input):
    #Returns a dictionary to of MACs per layer
    
    macs_dict = {}
    
    def hook_fn(layer, x, y):
        macs = count_layer_macs(layer, x, y)
        macs_dict[layer] = macs
        
    hooks = []
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            hooks.append(layer.register_forward_hook(hook_fn))
            
    model.eval()
    with torch.no_grad():
        y = model(input)
        
    #remove hookes
    for h in hooks:
        h.remove()
        
    return macs_dict

def compute_energy(macs, mac_per_joule, dram_words, dram_joules, carbon_factor):
    #Compute energy and carbon for a given MAC count and DRAM access count
    
    E_mac = macs * mac_per_joule
    E_dram = dram_words * dram_joules
    E_total = E_mac + E_dram
    kwh = joules_to_kWh(E_total)
    co_2 = kwh * carbon_factor
    
    return E_total, kwh, co_2

def analyze_model(model, input_shape=(1,3,224,224), batch_size = 1):
    #Analyze the model for both GPU and ASIC
    example_input = torch.randn(batch_size, *input_shape[1:])
    macs_dict = profile_model(model, example_input)
    
    total_macs = sum(macs_dict.values())
    total_dram_words = total_macs * DRAM_words_per_mac
    
    #GPU Analysis
    E_gpu, kwh_gpu, co2_gpu = compute_energy(total_macs, mac_gpu, total_dram_words, DRAM_energy, carb_factor)
    
    #ASIC Analysis
    E_asic, kwh_asic, co2_asic = compute_energy(total_macs, mac_asic, total_dram_words, DRAM_energy, carb_factor)
    
    results = {
        'total_macs': total_macs,
        'total_dram_words': total_dram_words,
        'gpu': {
            'energy_joules': E_gpu,
            'energy_kwh': kwh_gpu,
            'co2_kg': co2_gpu
        },
        'asic': {
            'energy_joules': E_asic,
            'energy_kwh': kwh_asic,
            'co2_kg': co2_asic
        }
    }
    
    return results

        
'''Visualization Function'''
def plot_energy_comp(results, model_name):
    labels = ['GPU', 'ASIC']
    energy = [
        results['gpu']['energy_joules'],
        results['asic']['energy_joules']
    ]
    
    co2 = [
        results['gpu']['co2_kg'],
        results['asic']['co2_kg']
    ]
    
    fig, ax = plt.subplots(1, 2, figsize = (12, 5))
    
    ax[0].bar(labels, energy, color=['blue', 'green'])
    ax[0].set_title(f"Energy Consumption for {model_name}")
    ax[0].set_ylabel("Energy (Joules)")
    
    ax[1].bar(labels, co2, color=['blue', 'green'])
    ax[1].set_title(f"CO2 Emissions for {model_name}")
    ax[1].set_ylabel("CO2 (kg)")
    
    plt.tight_layout()
    plt.show()
    
def plot_mac_comparison(model_names, input_shape=(1,3,224,224)):
    mac_counts = []

    for name in model_names:
        model = get_model(name)
        example_input = torch.randn(input_shape)
        macs_dict = profile_model(model, example_input)
        total_macs = sum(macs_dict.values())
        mac_counts.append(total_macs)

    # Plot
    plt.figure(figsize=(8,5))
    plt.bar(model_names, mac_counts)
    plt.ylabel("Total MACs per Inference")
    plt.title("MAC Count Comparison Across Models")
    plt.yscale("log")  # MAC counts span orders of magnitude
    plt.tight_layout()
    plt.show()

    
'''Model Selection'''
def get_model(model_name):
    model_name = model_name.lower()
    
    if model_name == 'resnet18':
        return models.resnet18(weights=None)
    elif model_name == 'mobilenet_v2':
        return models.mobilenet_v2(weights=None)
    elif model_name == 'vgg16':
        return models.vgg16(weights=None)
    elif model_name == 'alexnet':
        return models.alexnet(weights=None)
    elif model_name == 'squeezenet':
        return models.squeezenet1_0(weights=None)
    else:
        raise ValueError("Unsupported model name. Choose from resnet18, mobilenet_v2, vgg16, alexnet, squeezenet.")
    
'''User-Choice for Model Analysis'''
def run_single_model():
    print("\nChoose a model:")
    print("  1 - resnet18")
    print("  2 - mobilenet_v2")
    print("  3 - alexnet")
    print("  4 - squeezenet")

    choice = input("Enter number: ").strip()

    model_map = {
        "1": "resnet18",
        "2": "mobilenet_v2",
        "3": "alexnet",
        "4": "squeezenet"
    }

    if choice not in model_map:
        raise ValueError("Invalid selection.")

    model_name = model_map[choice]
    model = get_model(model_name)

    results = analyze_model(model)

    print("\n=== Tiny ASIC Compiler Results ===")
    print(f"Model: {model_name}")
    print("Total MACs:", f"{results['total_macs']:,}")

    print("\nGPU:")
    print(" Energy (J):", results["gpu"]["energy_joules"])
    print(" CO2 (kg):", results["gpu"]["co2_kg"])

    print("\nASIC:")
    print(" Energy (J):", results["asic"]["energy_joules"])
    print(" CO2 (kg):", results["asic"]["co2_kg"])


'''Main Function'''
def main():
    model_list = ["resnet18", "mobilenet_v2", "alexnet", "squeezenet"]
    plot_mac_comparison(model_list)

main()

    

