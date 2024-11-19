

#finetune deit from scratch
python main.py --auto-save --results-dir finetuned_models  --finetune https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth  --model deit_tiny_patch16_224  --seed 0 --lr 5e-6 --min-lr 1e-5 --warmup-lr 1e-5 --drop-path 0.0 --weight-decay 1e-8   --epochs 30  --data-path ./ --num_workers 4 --batch-size 128  --warmup-epochs 1

#continue finetuning basic
python main.py --auto-save --auto-resume --results-dir finetuned_models  --model deit_tiny_patch16_224  --seed 0 --lr 5e-6 --min-lr 1e-5 --warmup-lr 1e-5 --drop-path 0.0 --weight-decay 1e-8   --epochs 30  --data-path ./ --num_workers 4 --batch-size 128  --warmup-epochs 1


#evaluate basic
python main.py --eval --resume finetuned_models/none/best_checkpoint.pth --model deit_tiny_patch16_224 --seed 0 --lr 5e-6 --min-lr 1e-5 --warmup-lr 1e-5 --drop-path 0.0 --weight-decay 1e-8 --epochs 30 --data-path ./ --num_workers 4 --batch-size 128 --warmup-epochs 1


#train ablated component fron finetuned
python main.py --is-ablation --ablated-component bias --auto-save --results-dir finetuned_models   --finetune finetuned_models/none/best_checkpoint.pth  --model deit_tiny_patch16_224  --seed 0 --lr 5e-6 --min-lr 1e-5 --warmup-lr 1e-5 --drop-path 0.0 --weight-decay 1e-8   --epochs 30  --data-path ./ --num_workers 4 --batch-size 128  --warmup-epochs 1



#continue train ablated component (similar for variant)
python main.py --is-ablation --ablated-component bias --auto-save --auto-resume --results-dir finetuned_models   --model deit_tiny_patch16_224  --seed 0 --lr 5e-6 --min-lr 1e-5 --warmup-lr 1e-5 --drop-path 0.0 --weight-decay 1e-8   --epochs 30  --data-path ./ --num_workers 4 --batch-size 128  --warmup-epochs 1


#evaluate perturbations
python evaluate_perturbations.py --output-dir finetuned_models  --method transformer_attribution --data-path ./ --batch-size 1 --custom-trained-model finetuned_models/none/best_checkpoint.pth --num-workers 1 --both



python evaluate_perturbations.py --custom-trained-model finetuned_models/none/checkpoint_14.pth --output-dir finetuned_models  --method transformer_attribution --data-path ./ --batch-size 1  --num-workers 1 --both


python evaluate_perturbations.py --ablated_component bias --output-dir finetuned_models  --method transformer_attribution --data-path ./ --batch-size 1  --num-workers 1 --both
python evaluate_perturbations.py --variant relu --output-dir finetuned_models  --method transformer_attribution --data-path ./ --batch-size 1  --num-workers 1 --both

# full_lrp


#evaluate ablation
python main.py --ablated-component bias --eval --auto-resume --results-dir finetuned_models --model deit_tiny_patch16_224 --seed 0 --lr 5e-6 --min-lr 1e-5 --warmup-lr 1e-5 --drop-path 0.0 --weight-decay 1e-8 --epochs 30 --data-path ./ --num_workers 4 --batch-size 128 --warmup-epochs 1

python main.py --variant relu --eval --resume finetuned_models/relu/best_checkpoint.pth --results-dir finetuned_models --model deit_tiny_patch16_224 --seed 0 --lr 5e-6 --min-lr 1e-5 --warmup-lr 1e-5 --drop-path 0.0 --weight-decay 1e-8 --epochs 30 --data-path ./ --num_workers 4 --batch-size 128 --warmup-epochs 1

#visualize
python visualize_heatmap.py --custom-trained-model finetuned_models/rmsnorm/checkpoint_2.pth --variant rmsnorm --method transformer_attribution --sample-path val/n01614925/ILSVRC2012_val_00006571.JPEG



python main.py --variant relu --auto-save --auto-resume --results-dir finetuned_models  --model deit_tiny_patch16_224  --seed 0 --lr 5e-6 --min-lr 1e-5 --warmup-lr 1e-5 --drop-path 0.0 --weight-decay 1e-8   --epochs 60  --data-path ./ --num_workers 4 --batch-size 128  --warmup-epochs 1




#BATCH -START
python main.py --variant batchnorm --auto-save --finetune finetuned_models/none/best_checkpoint.pth  --results-dir finetuned_models   --model deit_tiny_patch16_224  --seed 0 --lr 5e-6 --min-lr 1e-5 --warmup-lr 1e-5 --drop-path 0.0 --weight-decay 1e-8   --epochs 50  --data-path ./ --num_workers 4 --batch-size 128  --warmup-epochs 1


#BATCH - CONTINUE
python main.py --variant batchnorm --auto-save --auto-resume --results-dir finetuned_models  --model deit_tiny_patch16_224  --seed 0 --lr 5e-6 --min-lr 1e-5 --warmup-lr 1e-5 --drop-path 0.0 --weight-decay 1e-8   --epochs 50  --data-path ./ --num_workers 4 --batch-size 128  --warmup-epochs 1


python analyze_pert_results.py --variant relu --mode runPerts --method transformer_attribution --data-path ./ --batch-size 1  --num-workers 1 --both



val/n01614925/ILSVRC2012_val_00006571.JPEG
val/n01877812/ILSVRC2012_val_00014040.JPEG
val/n02006656/ILSVRC2012_val_00028586.JPEG
val/n01514859/ILSVRC2012_val_00032162.JPEG
val/n01440764/ILSVRC2012_val_00046252.JPEG
val/n01985128/ILSVRC2012_val_00032174.JPEG
finetuned_models/relu/checkpoint_29.pth

python analyze_pert_results.py --variant relu --mode runPerts --method transformer_attribution --data-path ./ --batch-size 1  --num-workers 1 --both