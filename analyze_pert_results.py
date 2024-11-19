import os

import argparse
import subprocess

import re
import json
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='evaluate perturbations')
    
    parser.add_argument('--mode', required=True, choices = ['perturbations', 'analyze'])
    
    parser.add_argument('--pass-vis', action='store_true')
    parser.add_argument('--generate-plots', action='store_true', default=True)

    
    parser.add_argument('--batch-size', type=int,
                        default=1,
                        help='')
  
    parser.add_argument('--work-env', type=str,
                        help='')
    
    parser.add_argument('--ablated-component', type=str, 
                        choices=['softmax', 'layerNorm', 'bias'],)
    
    
    parser.add_argument('--variant', choices=['rmsnorm', 'relu', 'batchnorm'], type=str, help="")
    
    parser.add_argument('--output-dir', type=str,
                        help='')
    parser.add_argument('--neg', type=int, choices = [0,1], default = 0)
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    parser.add_argument('--data-set', default='IMNET100', choices=['IMNET100','CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--num-workers', type=int,
                        default= 2,
                        help='')
    parser.add_argument('--method', type=str,
                        default='grad_rollout',
                        choices=['rollout', 'lrp', 'transformer_attribution', 'full_lrp', 'lrp_last_layer',
                                 'attn_last_layer', 'attn_gradcam'],
                        help='')

    parser.add_argument('--both',  action='store_true')
    parser.add_argument('--debug',
                        action='store_true',
                        help='Runs the first 5 samples and visualizes ommited pixels')
    parser.add_argument('--wrong', action='store_true',
                        default=False,
                        help='')

    parser.add_argument('--scale', type=str,
                        default='per',
                        choices=['per', '100'],
                        help='')

    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.', default=True)
    parser.add_argument('--lmd', type=float,
                        default=10,
                        help='')
    parser.add_argument('--vis-class', type=str,
                        default='top',
                        choices=['top', 'target', 'index'],
                        help='')
    parser.add_argument('--class-id', type=int,
                        default=0,
                        help='')
    parser.add_argument('--cls-agn', action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--no-ia', action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--no-fx', action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--no-fgx', action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--no-m', action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--no-reg', action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--is-ablation', type=bool,
                        default=False,
                        help='')
    parser.add_argument('--data-path', type=str,
                        required=True,
                        help='')
    args = parser.parse_args()
    return args



def parse_pert_results(pert_results_path, acc_keys):
    pos_values = {}
    neg_values = {}
    
    for res_dir in os.listdir(pert_results_path):
        res_path = os.path.join(pert_results_path, res_dir)
        if os.path.isdir(res_path):
            # The key corresponds to the number in res_X
            res_key = int(res_dir.split('_')[1])
            
            if res_key not in acc_keys:
                continue
            
            pert_results_file = os.path.join(res_path, 'pert_results.json')
            with open(pert_results_file, 'r') as f:
                pert_data = json.load(f)
                pos_values[res_key] = pert_data.get('transformer_attribution_pos_auc', 0)
                neg_values[res_key] = pert_data.get('transformer_attribution_neg_auc', 0)
    
    return pos_values, neg_values



def parse_acc_results(acc_results_path):
    acc_dict = {}
    with open(acc_results_path, 'r') as f:
        data = json.load(f)
        for key, value in data.items():
            # Extract the numeric part from key like "2_acc"
            if "acc" in key:
               acc_key = int(key.split('_')[0])
               # Extract the accuracy value after "Acc@1"
               acc_value = float(value.split('Acc@1 ')[1].split(' ')[0])
               acc_dict[acc_key] = acc_value
    return acc_dict



def get_sorted_checkpoints(directory):
    # List to hold the relative paths and their associated numeric values
    checkpoints = []

    # Walk through the directory
    for root, _, files in os.walk(directory):
        for file in files:
            # Check if the file matches the pattern 'checkpoint_*.pth'
            match = re.match(r'checkpoint_(\d+)\.pth', file)
            if match:
                # Extract the number from the filename
                number = int(match.group(1))
                # Get the relative path of the file
                relative_path = os.path.relpath(os.path.join(root, file), directory)
                # Append tuple (number, relative_path)
                checkpoints.append((number, relative_path))

    # Sort the checkpoints by the number
    checkpoints.sort(key=lambda x: x[0])

    # Return just the sorted relative paths
    return [f'{directory}/{relative_path}'  for _, relative_path in checkpoints]



'''
TEMPORARY! based on current accuarcy results
'''
def filter_epochs(model, epoch):
   if model == "rmsnorm":
      return epoch in [0,1,2,3,9,13,18,19,23,29]
   elif model == "relu":
      return epoch in [9, 14, 20, 31, 32, 33, 35, 45, 52, 71]
   elif model == "no_bias":
      return epoch in [7,6,1,3] # 22 26 28 29
   else:
      return epoch in [14,12,16,18,29, 28, 26, 24, 22,10,8]
      


def run_perturbations(args):
    eval_pert_cmd        = "python evaluate_perturbations.py"
   
    
    eval_pert_cmd       +=  f' --method {args.method}'
    eval_pert_cmd       +=  f' --both'

   
 
    eval_pert_cmd       +=  f' --data-path {args.data_path}'
    eval_pert_cmd       +=  f' --batch-size {args.batch_size}'
    eval_pert_cmd       +=  f' --num-workers {args.num_workers}'

    
    root_dir = f'finetuned_models'
    
    if args.ablated_component:
      if args.ablated_component != 'none' and args.variant:
        print("does not support both a variant and ablation")
        exit(1)
    

    if args.ablated_component == None and args.variant == None and args.custom_trained_model == None:
       model = 'none'
    
   
    elif args.variant:
       model          = f'{args.variant}'
       eval_pert_cmd += f' --variant {args.variant}'
  

    elif args.ablated_component :
       model          = f'no_{args.ablated_component}'
       eval_pert_cmd += f' --ablated-component {args.ablated_component}'

    model_dir = f'{root_dir}/{model}'

    checkpoints =  get_sorted_checkpoints(model_dir)

    count = 0
    for c in checkpoints:
     
       checkpoint_path  = c.split("/")[-1]
       epoch            = checkpoint_path.split(".")[0].split("_")[-1]
       if filter_epochs(model, int(epoch)) == False:
          continue
       print(f"working on epoch {epoch}")
       eval_pert_epoch_cmd = f"{eval_pert_cmd} --output-dir {model_dir}/pert_results/res_{epoch}"
       eval_pert_epoch_cmd += f" --work-env {model_dir}/work_env/epoch{epoch}" 
       eval_pert_epoch_cmd += f" --custom-trained-model {model_dir}/{checkpoint_path}" 
       print(f'executing: {eval_pert_epoch_cmd}')
       try:
          subprocess.run(eval_pert_epoch_cmd, check=True, shell=True)
          print(f"generated visualizations")
       except subprocess.CalledProcessError as e:
          print(f"Error: {e}")
          exit(1)




def generate_plots(dir_path):
    acc_results_path = os.path.join(dir_path, 'acc_results.json')
    acc_dict = parse_acc_results(acc_results_path)

    pert_results_path = os.path.join(dir_path, 'pert_results')
    pos_dict, neg_dict = parse_pert_results(pert_results_path, acc_dict.keys())

    # Sort the keys (x-axis)
    sorted_keys = sorted(acc_dict.keys())
    
    acc_values = [acc_dict[k] for k in sorted_keys]
    pos_values = [pos_dict.get(k, 0) for k in sorted_keys]
    neg_values = [neg_dict.get(k, 0) for k in sorted_keys]

    # Plotting
    plt.figure(figsize=(10, 6))
    #plt.plot(sorted_keys, acc_values, label='Accuracy', marker='o')
    plt.plot(sorted_keys, pos_values, label='Positive', marker='s')
    plt.plot(sorted_keys, neg_values, label='Negative', marker='^')

    plt.xlabel('Iteration')  # x-axis label
    plt.ylabel('Values')  # y-axis label
    plt.title(f'Performance Metrics for {os.path.basename(dir_path)}')  # Title
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f"{dir_path}/plot.png")
    plt.show()



def analyze(args):
   choices = [ "relu", "none", "rmsnorm", "no_bias" ] 
   root_dir = f'finetuned_models'
   
   if args.generate_plots:
      for c in choices:
        subdir = f'{root_dir}/{c}'
        generate_plots(subdir)
   
   
   neg_list        = []
   max_neg         = -float('inf')
   max_neg_subdir  = None
   max_neg_key     = None
  
   for c in choices:
    subdir = f'{root_dir}/{c}'
    acc_results_path = os.path.join(subdir, 'acc_results.json')
    acc_dict = parse_acc_results(acc_results_path)
    pert_results_path = os.path.join(subdir, 'pert_results')
    pos_dict, neg_dict = parse_pert_results(pert_results_path, acc_dict.keys())
    for key, neg_value in neg_dict.items():
       neg_list.append((neg_value, pos_dict[key], subdir, key))
       if neg_value > max_neg:
        pos_val = pos_dict[key]
        max_neg = neg_value
        max_neg_subdir = subdir
        max_neg_key = key


    neg_list.sort(reverse=True, key=lambda x: x[0])
    print(f"The subdir with the highest neg value is {max_neg_subdir}")
    print(f"Iter: {max_neg_key}, Neg Value: {max_neg}, Pos Value: {pos_val}")
    
    print("best pert score by negative perutrbations")
    for i in range(min(10, len(neg_list))):  # Make sure to not go beyond the available number of values
        neg_value, pos_value, subdir, key = neg_list[i]
        print(f"{i+1}. Subdir: {subdir} | Iter: {key} | Neg AUC: {neg_value} | POS AUC: {pos_value}")



if __name__ == "__main__":
    args                   = parse_args()
    if args.mode == "perturbations":
       run_perturbations(args)
    else:
       analyze(args)
    