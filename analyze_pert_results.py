import os
import config
import argparse
import subprocess
#import pandas as pd
import re
import json
import matplotlib.pyplot as plt
import numpy as np
def parse_args():
    parser = argparse.ArgumentParser(description='evaluate perturbations')
    
    parser.add_argument('--mode', required=True, choices = ['perturbations', 'analyze'])
    parser.add_argument('--normalized-pert', type=int, default=1, choices = [0,1])

    parser.add_argument('--fract', type=float,
                        default=0.1,
                        help='')

    
    parser.add_argument('--pass-vis', action='store_true')
    parser.add_argument('--gen-latex', action='store_true')
    parser.add_argument('--check-all', action='store_true')
    parser.add_argument('--default-norm', action='store_true')



    parser.add_argument('--generate-plots', action='store_true', default=True)
   

    
    parser.add_argument('--batch-size', type=int,
                        default=1,
                        help='')
  
    parser.add_argument('--work-env', type=str,
                        help='')
    
    parser.add_argument('--ablated-component', type=str, 
                        choices=['softmax', 'layerNorm', 'bias', 'softplus'],)
    
    
    parser.add_argument('--variant', default = 'basic',  type=str, help="")
    
   
    parser.add_argument('--neg', type=int, choices = [0,1], default = 0)
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    parser.add_argument('--data-set', default='IMNET100', choices=['IMNET100','CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--num-workers', type=int,
                        default= 1,
                        help='')
    parser.add_argument('--method', type=str,
                        default='transformer_attribution',
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
                  
                        help='')
    args = parser.parse_args()
    return args


def gen_latex_table(global_top_mapper,args):

   ops      = ["top", "target"] 
   if args.normalized_pert == 0:
      ops   =   ["top_blur", "target_blur" ] + ops

   # Start the LaTeX table
 
   latex_code = r'\begin{table}[h!]\centering' + '\n' + r'\begin{tabular}{c c c c c| c c c c}' + '\n'
   latex_code += r'\hline' + '\n'
   
   header_row = ''


   a_values = ['']
   if args.normalized_pert == 0:
      a_values += ['_blur']
   
   b_values = ['neg_', 'pos_']
   c_values = ['top', 'target']

   x_values = ['Black']
   if args.normalized_pert == 0:
      x_values += ['Blur']
   
   y_values = ['Negative', 'Positive']
   z_values = ['Predicted', 'Target']

   for x in x_values:
      header_row += f' & \\multicolumn{{4}}{{c|}}{{{x}}}'
   header_row += r'\\ \hline' + '\n'


   # Header for y and z values
   subheader_row = ''
   for x in x_values:
      for y in y_values:
         subheader_row += f' & \\multicolumn{{2}}{{c}}{{{y}}}'
   subheader_row += r'\\ ' + '\n'


       # Header for z values
   subsubheader_row = ''
   for x in x_values:
      for y in y_values:
         for z in z_values:
            subsubheader_row += f' & \\multicolumn{{1}}{{c}}{{{z}}}'
   subsubheader_row += r'\\ ' + '\n'


   latex_code = latex_code + header_row + subheader_row + subsubheader_row
   for experiment in global_top_mapper:
      row = experiment
      for a in a_values:
         for b in b_values:
            for c in c_values:
              
               row += f' & {100*global_top_mapper[experiment][b+c+a]:.3f}'
               print(row)
      row += r'\\ ' + '\n'
      latex_code += row
    
   latex_code += "\\hline\n\\end{tabular}\n\\caption{Positive AUC}\n\\end{table}"
   print(latex_code)
   
   
  

def parse_pert_results(pert_results_path, acc_keys, args, op):
    pos_values = {}
    neg_values = {}
    pos_lists = {}    # New dictionary for transformer_attribution_pos lists
    neg_lists = {} 
    
    for res_dir in os.listdir(pert_results_path):
        res_path = os.path.join(pert_results_path, res_dir)
        if os.path.isdir(res_path):
            # The key corresponds to the number in res_X
            res_key = int(res_dir.split('_')[1])
            
            if res_key not in acc_keys:
                continue
            
            if ((args.normalized_pert == 0 and "base" not in res_path) or (args.normalized_pert and "base" in res_path)):
               continue
            
            pert_results_file = os.path.join(res_path, 'pert_results.json')
            with open(pert_results_file, 'r') as f:
                pert_data = json.load(f)
                pos_values[res_key] = pert_data.get(f'transformer_attribution_pos_auc_{op}', 0)
                neg_values[res_key] = pert_data.get(f'transformer_attribution_neg_auc_{op}', 0)

                pos_lists[res_key] = pert_data.get(f'transformer_attribution_pos_{op}', [])
                neg_lists[res_key] = pert_data.get(f'transformer_attribution_neg_{op}', [])
    
    return pos_values, neg_values, pos_lists, neg_lists



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
def filter_epochs(args, epoch, variant):
   return epoch in args.epochs_to_perturbate[variant]



def run_perturbations_env(args):
   choices = args.epochs_to_perturbate.keys()
   for c in choices:
      args.variant = c
      run_perturbations(args)
  


def run_perturbations(args):
    eval_pert_cmd        = "python evaluate_perturbations.py"
   
    
    eval_pert_cmd       +=  f' --method {args.method}'
    eval_pert_cmd       +=  f' --both'

   
 
    eval_pert_cmd       +=  f' --data-path {args.data_path}'
    eval_pert_cmd       +=  f' --data-set {args.data_set}'

    eval_pert_cmd       +=  f' --batch-size {args.batch_size}'
    eval_pert_cmd       +=  f' --num-workers {args.num_workers}'

    eval_pert_cmd       +=  f' --normalized-pert {args.normalized_pert}'
    eval_pert_cmd       +=  f' --fract {args.fract}'



    
    root_dir = f"{args.dirs['finetuned_models_dir']}{args.data_set}"
    
    variant          = f'{args.variant}'
    eval_pert_cmd += f' --variant {args.variant}'
  

    model_dir = f'{root_dir}/{variant}'

    checkpoints =  get_sorted_checkpoints(model_dir)


    for c in checkpoints:
     
       checkpoint_path  = c.split("/")[-1]
       epoch            = checkpoint_path.split(".")[0].split("_")[-1]
       if filter_epochs(args, int(epoch), variant ) == False:
          continue
       print(f"working on epoch {epoch}")
       pert_results_dir = 'pert_results/op_norm' if args.default_norm else 'pert_results'
       eval_pert_epoch_cmd = f"{eval_pert_cmd} --output-dir {model_dir}/{pert_results_dir}/res_{epoch}"
       if args.normalized_pert == 0:
          eval_pert_epoch_cmd+="_base"
      
       eval_pert_epoch_cmd += f" --work-env {model_dir}/work_env/epoch{epoch}" 
       eval_pert_epoch_cmd += f" --custom-trained-model {model_dir}/{checkpoint_path}" 
       print(f'executing: {eval_pert_epoch_cmd}')
       try:
          subprocess.run(eval_pert_epoch_cmd, check=True, shell=True)
          print(f"generated visualizations")
       except subprocess.CalledProcessError as e:
          print(f"Error: {e}")
          exit(1)




def generate_plots(dir_path,args):
    acc_results_path = os.path.join(dir_path, 'acc_results.json')
    acc_dict = parse_acc_results(acc_results_path)
    pert_results_dir = 'pert_results/op_norm' if args.default_norm else 'pert_results'
    
    pert_results_path = os.path.join(dir_path, pert_results_dir)
    pos_dict, neg_dict, pos_lists, neg_lists = parse_pert_results(pert_results_path, acc_dict.keys(),args)

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


def parse_subdir(subdir):
   exp_name = subdir.split("/")[-1]
   exp_name = exp_name.replace("_"," ")
   exp_name = exp_name if exp_name != "none" else "basic"
   return exp_name


def analyze(args):
   choices  =  args.epochs_to_perturbate.keys() 
   root_dir = f"{args.dirs['finetuned_models_dir']}{args.data_set}"
   ops      = ["target", "top"] 
   if args.normalized_pert == 0:
      ops+=   ["target_blur", "top_blur"] 

   print(f"variants to consider: {choices}")
   print(f"operating on : {root_dir}")

   global_top_mapper = {}

   #if args.generate_plots:
   #   for c in choices:
   #     subdir = f'{root_dir}/{c}'
   #     generate_plots(subdir,args)
   
   for op in ops:
   
      neg_list        = []
      max_neg         = -float('inf')
      max_neg_subdir  = None
      max_neg_key     = None
   
      pos_list        = []
      min_pos         = float('inf')
      min_pos_subdir  = None
      min_pos_key     = None

      best_exp        = {}

      for c in choices:


       subdir = f'{root_dir}/{c}'
       acc_results_path = os.path.join(subdir, 'acc_results.json')
       acc_dict = parse_acc_results(acc_results_path)
       pert_results_dir = 'pert_results/op_norm' if args.default_norm else 'pert_results'
       pert_results_path = os.path.join(subdir, pert_results_dir)
       pos_dict, neg_dict, pos_lists, neg_lists = parse_pert_results(pert_results_path, acc_dict.keys(),args, op)
       tmp_max_neg         = -float('inf')
       exp = parse_subdir(subdir)


       if exp not in global_top_mapper:
          global_top_mapper[exp] = {}

       
       for key, neg_value in neg_dict.items():
          neg_list.append((neg_value, pos_dict[key], subdir, key, acc_dict[key]))
          
          if exp not in best_exp:
             best_exp[exp] = neg_lists[key]
             tmp_max_neg   = neg_value
             global_top_mapper[exp][f"neg_{op}"] = neg_value
             global_top_mapper[exp][f"pos_{op}"] = pos_dict[key]
          else:
             if neg_value > tmp_max_neg:
                best_exp[exp] = neg_lists[key]
                tmp_max_neg   = neg_value

                global_top_mapper[exp][f"neg_{op}"] = neg_value
                global_top_mapper[exp][f"pos_{op}"] = pos_dict[key]





          if neg_value > max_neg:
           pos_val = pos_dict[key]
           max_neg = neg_value
           max_neg_subdir = subdir
           max_neg_key = key

       for key, pos_value in pos_dict.items():
          pos_list.append((pos_value, neg_dict[key], subdir, key, acc_dict[key] ))
          if pos_value < min_pos:
           neg_val = neg_dict[key]
           min_pos = pos_value
           min_pos_subdir = subdir
           min_pos_key = key
   

      neg_list.sort(reverse=True, key=lambda x: x[0])
      pos_list.sort(reverse=False, key=lambda x: x[0])


      print(f"The subdir with the highest neg value for {op} critertion is : {max_neg_subdir}")
      print(f"Iter: {max_neg_key}, Neg Value: {max_neg}, Pos Value: {pos_val}")
      print("best pert score by negative perutrbations")
      for i in range(min(60, len(neg_list))):  # Make sure to not go beyond the available number of values
       neg_value, pos_value, subdir, key, acc = neg_list[i]
       print(f"{i+1}. experiment: {parse_subdir(subdir)} | Iter: {key} | Neg AUC: {neg_value} | POS AUC: {pos_value} | ACC1: {acc}")
      print("\n\n")
      for i in range(min(50, len(pos_list))):  # Make sure to not go beyond the available number of values
       pos_value, neg_value, subdir, key, acc = pos_list[i]
       print(f"{i+1}. experiment: {parse_subdir(subdir)} | Iter: {key} | POS AUC: {pos_value} | Neg AUC: {neg_value} | ACC1: {acc}")
   #

      if args.generate_plots:
         x_values = np.arange(0.1, 1.0, 0.1)
         plt.figure(figsize=(8, 6))
         for key, value in best_exp.items():
            plt.plot(x_values, value, label=key) 
         plt.legend()
         plt.savefig(f'{root_dir}plot_{op}.png')
   
   if args.gen_latex:
      gen_latex_table(global_top_mapper,args)




if __name__ == "__main__":
    args                   = parse_args()
    config.get_config(args, skip_further_testing = True, get_epochs_to_perturbate = True)

    if args.mode == "perturbations":
       if args.check_all:
          run_perturbations_env(args)
       else: 
         run_perturbations(args)
    else:
       analyze(args)
    