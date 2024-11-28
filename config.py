from modules.layers_ours import *
from functools import partial

DEFAULT_MODEL = {
    'norm'                 : partial(LayerNorm, eps=1e-6) ,
    'last_norm'            : LayerNorm ,
    'activation'           : GELU(),
    'isWithBias'           : True,
    'attn_activation'      : Softmax(dim=-1) 
}

DEFAULT_PATHS = {
        
    'imagenet_1k_Dir'        : '/home/ai_center/ai_users/zimerman1/datasets/Imagenet/data/',
    'imagenet_100_Dir'       : './',
    'finetuned_models_dir'   : 'finetuned_models/', 
    'results_dir'            : 'finetuned_models/', 

}


MODEL_VARIANTS = {
            'basic':                  DEFAULT_MODEL.copy(),
            'attn_act_relu':          {**DEFAULT_MODEL, 'attn_activation': ReluAttention()},
            'attn_act_sigmoid':       {**DEFAULT_MODEL, 'attn_activation': SigmoidAttention()},
            'act_softplus':           {**DEFAULT_MODEL, 'activation': Softplus()},

            'act_softplus_norm_rms':  {**DEFAULT_MODEL, 'activation': Softplus(), 'norm': partial(RMSNorm, eps=1e-6), 'last_norm': RMSNorm },
            'norm_rms':               {**DEFAULT_MODEL, 'norm': partial(RMSNorm, eps=1e-6), 'last_norm': RMSNorm },
            'norm_bias_ablation':     {**DEFAULT_MODEL, 
                                       'norm': partial(UncenteredLayerNorm, eps=1e-6, has_bias=False), 
                                       'last_norm': partial(UncenteredLayerNorm,has_bias=False)},
            'norm_center_ablation':   {**DEFAULT_MODEL, 
                                       'norm': partial(UncenteredLayerNorm, eps=1e-6, center=False),
                                       'last_norm': partial(UncenteredLayerNorm,center=False)},
            'norm_batch':             {**DEFAULT_MODEL, 'norm': RepBN,'last_norm' : RepBN},
            'bias_ablation':          {**DEFAULT_MODEL, 'isWithBias': False,
                                       
                                       },


             'attn_act_sparsemax':     {**DEFAULT_MODEL, 'attn_activation': Sparsemax(dim=-1)},


}



EPOCHS_TO_PERTURBATE = {
            'basic':     [29, 28,26, 24, 22,10, 14,12,16, 18]  ,          #, 
            'attn_act_relu':       [ 14, 20, 31,  33, 35, 45, 52, 71],    
            'act_softplus':       [33,34,40,45,46,49]   ,
            'act_softplus_norm_rms': [78,79,73,60,59,58,50,48,46,44,40],
            'norm_rms':              [13,0,1,2,3,9,13,18,19,23,29,] ,  #
            'norm_bias_ablation':    [2, 13,19,23,18,19,29] ,
            'bias_ablation':        [59,58,56,54,44,47,40,37,33,32]   #
}








DEFAULT_PARAMETERS = {
    'model'                  : 'deit_tiny_patch16_224',
    'seed'                   : 0,
    'lr'                     : 5e-6, 
    'min_lr'                 : 1e-5,
    'warmup-lr'              : 1e-5,
    'drop_path'              : 0.0,
    'weight_decay'           : 1e-8,
    'num_workers'            : 4,
    'batch_size'             : 128,
    'warmup_epochs'          : 1
}

PRETRAINED_MODELS_URL = {
    'deit_tiny_patch16_224': 'finetuned_models/IMNET100/basic/best_checkpoint.pth'

}




def SET_VARIANTS_CONFIG(args):
    if args.variant not in MODEL_VARIANTS:
        print(f"only allowed to use the following variants: {MODEL_VARIANTS.keys()}")
        exit(1)
    
    
    args.model_components = MODEL_VARIANTS[args.variant]



def SET_PATH_CONFIG(args):

    
    if args.data_set == 'IMNET100':
        args.data_path = args.dirs['imagenet_100_Dir']
    else:
        args.data_path = args.dirs['imagenet_1k_Dir']


   


def get_config(args, skip_further_testing = False, get_epochs_to_perturbate = False):

    SET_VARIANTS_CONFIG(args)
    args.dirs = DEFAULT_PATHS

    
    if get_epochs_to_perturbate:
        args.epochs_to_perturbate = EPOCHS_TO_PERTURBATE
        
    
   # if skip_further_testing == False:
   #     vars(args).update(DEFAULT_PARAMETERS)


    

    if args.data_path == None:
        SET_PATH_CONFIG(args)

    if skip_further_testing:
        return
    
    if args.auto_start_train:
        args.finetune =  PRETRAINED_MODELS_URL[args.model]
      

    if args.eval and args.resume =='' and args.auto_resume == False:
        print("for evaluation please add --resume  with your model path, or add --auto-resume to automatically find it ")
        exit(1)
    if args.verbose:
        print(f"working with model {args.model} | dataset: {args.data_set} | variant: {args.variant}")


    
