from models.model import deit_tiny_patch16_224 as vit_LRP
from models.model_train import deit_tiny_patch16_224 as vit_LRP_train


def model_env(pretrained=False,args  = None , hooks = False,  **kwargs):
    
    
    print(args.model_components["isWithBias"])
    print(args.model_components["norm"])
    print(args.model_components["attn_activation"])
    print(args.model_components["activation"])

    if hooks:
        return vit_LRP(
            isWithBias      = args.model_components["isWithBias"],
            layer_norm      = args.model_components["norm"],
            last_norm       = args.model_components["last_norm"],

            activation      = args.model_components["activation"],
            attn_activation = args.model_components["attn_activation"],
            num_classes     = args.nb_classes,
        )
    else:
        return vit_LRP_train(
            isWithBias      = args.model_components["isWithBias"],
            layer_norm      = args.model_components["norm"],
            last_norm       = args.model_components["last_norm"],

            activation      = args.model_components["activation"],
            attn_activation = args.model_components["attn_activation"],
            num_classes     = args.nb_classes,
        )


   