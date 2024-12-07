

#finetune deit from scratch

###########################################################################################
###########################################################################################

#BASIC DEIT

###########################################################################################
###########################################################################################

# Finetune basic DEIT

python main.py --auto-save --results-dir finetuned_models  --finetune https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth  --model deit_tiny_patch16_224  --seed 0 --lr 5e-6 --min-lr 1e-5 --warmup-lr 1e-5 --drop-path 0.0 --weight-decay 1e-8   --epochs 30  --data-path ./ --num_workers 4 --batch-size 128  --warmup-epochs 1

#continue finetuning basic DEIT

python main.py --auto-save --auto-resume --results-dir finetuned_models  --model deit_tiny_patch16_224  --seed 0 --lr 5e-6 --min-lr 1e-5 --warmup-lr 1e-5 --drop-path 0.0 --weight-decay 1e-8   --epochs 30  --data-path ./ --num_workers 4 --batch-size 128  --warmup-epochs 1





###########################################################################################
###########################################################################################

# ABLATED COMPONENT

###########################################################################################
###########################################################################################


#train ablated component fron finetuned
python main.py --is-ablation --ablated-component bias --auto-save --results-dir finetuned_models   --finetune finetuned_models/none_IMNET100/best_checkpoint.pth  --model deit_tiny_patch16_224  --seed 0 --lr 5e-6 --min-lr 1e-5 --warmup-lr 1e-5 --drop-path 0.0 --weight-decay 1e-8   --epochs 30  --data-path ./ --num_workers 4 --batch-size 128  --warmup-epochs 1



#continue train ablated component (similar for variant)
python main.py --is-ablation --ablated-component bias --auto-save --auto-resume --results-dir finetuned_models   --model deit_tiny_patch16_224  --seed 0 --lr 5e-6 --min-lr 1e-5 --warmup-lr 1e-5 --drop-path 0.0 --weight-decay 1e-8   --epochs 30  --data-path ./ --num_workers 4 --batch-size 128  --warmup-epochs 1

#eval ablated component
python main.py --ablated-component bias --eval --auto-resume --results-dir finetuned_models --model deit_tiny_patch16_224 --seed 0 --lr 5e-6 --min-lr 1e-5 --warmup-lr 1e-5 --drop-path 0.0 --weight-decay 1e-8 --epochs 30 --data-path ./ --num_workers 4 --batch-size 128 --warmup-epochs 1



###########################################################################################
###########################d################################################################

# VARIANT

#train variant
python main.py --variant act_softplus_norm_rms --auto-save --results-dir finetuned_models   --finetune finetuned_models/IMNET/basic/best_checkpoint.pth  --model deit_tiny_patch16_224  --seed 0 --lr 5e-6 --min-lr 1e-5 --warmup-lr 1e-5 --drop-path 0.0 --weight-decay 1e-8   --epochs 30  --data-path ./ --num_workers 4 --batch-size 128  --warmup-epochs 1


#eval variant(same)
python main.py --variant relu --eval --resume finetuned_models/relu/best_checkpoint.pth --results-dir finetuned_models --model deit_tiny_patch16_224 --seed 0 --lr 5e-6 --min-lr 1e-5 --warmup-lr 1e-5 --drop-path 0.0 --weight-decay 1e-8 --epochs 30 --data-path ./ --num_workers 4 --batch-size 128 --warmup-epochs 1

#continue finetuning relu
python main.py --variant rmsnorm_softplus --auto-save --auto-resume --results-dir finetuned_models  --model deit_tiny_patch16_224  --seed 0 --lr 5e-6 --min-lr 1e-5 --warmup-lr 1e-5 --drop-path 0.0 --weight-decay 1e-8   --epochs 80  --data-path ./ --num_workers 4 --batch-size 128  --warmup-epochs 1



###########################################################################################
###########################################################################################



###########################################################################################
###########################################################################################

# PERTURBATIONS

###########################################################################################
###########################################################################################


#evaluate perturbations for basic automatically using best_checkpoint

python evaluate_perturbations.py --output-dir finetuned_models  --method transformer_attribution --data-path ./ --batch-size 1 --custom-trained-model finetuned_models/none/best_checkpoint.pth --num-workers 1 --both


#evaluate perturbations for basic with a specific model - RECOMMENDED


python evaluate_perturbations.py --custom-trained-model finetuned_models/none/checkpoint_14.pth --output-dir finetuned_models  --method transformer_attribution --data-path ./ --batch-size 1  --num-workers 1 --both

#evalutae for perturbation or ablated

python evaluate_perturbations.py --ablated_component bias   --method transformer_attribution --data-path ./ --batch-size 1  --num-workers 1 --both
python evaluate_perturbations.py --variant rmsnorm --custom-trained-model finetuned_models/rmsnorm_IMNET100/checkpoint_29.pth  --method transformer_attribution --data-path ./ --batch-size 1  --num-workers 1 --both


#analyze pertubations
python analyze_pert_results.py --variant norm_bias_ablation --mode pertubations --method transformer_attribution --data-path ./ --batch-size 1  --num-workers 1 --both

###########################################################################################
###########################################################################################

# HEATMAP


#visualize
python visualize_heatmap.py --custom-trained-model finetuned_models/rmsnorm/checkpoint_13.pth --variant rmsnorm --method transformer_attribution --sample-path val/n01877812/ILSVRC2012_val_00014040.JPEG

###########################################################################################
###########################################################################################


#from here



#BATCH -START
python main.py --variant softplus --auto-save --finetune finetuned_models/none/best_checkpoint.pth  --results-dir finetuned_models   --model deit_tiny_patch16_224  --seed 0 --lr 5e-6 --min-lr 1e-5 --warmup-lr 1e-5 --drop-path 0.0 --weight-decay 1e-8   --epochs 30  --data-path ./ --num_workers 4 --batch-size 128  --warmup-epochs 1


#BATCH - CONTINUE
python main.py --variant batchnorm --auto-save --auto-resume --results-dir finetuned_models  --model deit_tiny_patch16_224  --seed 0 --lr 5e-6 --min-lr 1e-5 --warmup-lr 1e-5 --drop-path 0.0 --weight-decay 1e-8   --epochs 50  --data-path ./ --num_workers 4 --batch-size 128  --warmup-epochs 1





val/n01614925/ILSVRC2012_val_00006571.JPEG
val/n01877812/ILSVRC2012_val_00014040.JPEG
val/n02006656/ILSVRC2012_val_00028586.JPEG
val/n01514859/ILSVRC2012_val_00032162.JPEG
val/n01440764/ILSVRC2012_val_00046252.JPEG
val/n01985128/ILSVRC2012_val_00032174.JPEG
finetuned_models/relu/checkpoint_29.pth

python analyze_pert_results.py --variant relu --mode runPerts --method transformer_attribution --data-path ./ --batch-size 1  --num-workers 1 --both







python visualize_heatmap.py --custom-trained-model finetuned_models/batchnorm_IMNET1000/deit_tiny_prepbn.pth --variant batchnorm_param --method transformer_attribution --sample-path val/n01877812/ILSVRC2012_val_00014040.JPEG


###############

python generate_visualizations.py --method transformer_attribution --data-path /home/ai_center/ai_users/zimerman1/datasets/Imagenet/data/ --data-set IMNET --batch-size 1 --num-workers 1 --variant batchnorm_param --custom-trained-model finetuned_models/batchnorm_IMNET1000/deit_tiny_prepbn.pth --work-env finetuned_models/tmp2/work_env



python generate_perturbations.py --debug --method transformer_attribution --data-set IMNET --variant batchnorm_param --custom-trained-model finetuned_models/batchnorm_IMNET1000/deit_tiny_prepbn.pth  --work-env finetuned_models/tmp2/work_env --output-dir finetuned_models/batchnorm_param_IMNET --neg 1





python generate_visualizations.py --method transformer_attribution --data-path /home/ai_center/ai_users/zimerman1/datasets/Imagenet/data/ --data-set IMNET --batch-size 1 --num-workers 1 --custom-trained-model finetuned_models/basic_IMNET100/deit_tiny_patch16_224-a1311bcf.pth --work-env finetuned_models/tmp/work_env


python generate_perturbations.py  --method transformer_attribution --data-set IMNET --custom-trained-model finetuned_models/basic_IMNET100/deit_tiny_patch16_224-a1311bcf.pth  --work-env finetuned_models/tmp/work_env --neg 1






python evaluate_perturbations.py --work-env finetuned_models/batchnorm_IMNET1000/work_env --variant batchnorm_param  --custom-trained-model finetuned_models/batchnorm_IMNET1000/deit_tiny_prepbn.pth --data-set IMNET  --method transformer_attribution --data-path /home/ai_center/ai_users/zimerman1/datasets/Imagenet/data/ --batch-size 1  --num-workers 1 --both



python evaluate_perturbations.py --work-env finetuned_models/basic_IMNET100/work_env  --custom-trained-model finetuned_models/basic_IMNET100/deit_tiny_patch16_224-a1311bcf.pth --data-set IMNET  --method transformer_attribution --data-path /home/ai_center/ai_users/zimerman1/datasets/Imagenet/data/ --batch-size 1  --num-workers 1 --both --output-dir  finetuned_models/basic_IMNET100/pert_results/












python main.py --variant attn_variant_light --auto-start-train --auto-save --model deit_tiny_patch16_224 --seed 0 --lr 5e-6 --min-lr 1e-5 --warmup-lr 1e-5 --drop-path 0.0 --weight-decay 1e-8 --epochs 30  --num_workers 4 --batch-size 128 --warmup-epochs 1
