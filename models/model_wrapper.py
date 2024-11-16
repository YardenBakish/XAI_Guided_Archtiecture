from models.model_no_hooks_ablation import deit_tiny_patch16_224 as vit_LRP_no_hooks
from models.model_ablation import deit_tiny_patch16_224 as vit_LRP




def model_env(pretrained=False, hooks = False, nb_classes =100, ablated_component ="none", variant = "", **kwargs):

    if ablated_component != "none" and variant!= "":
        print("can't have both a variant and ablation")
        exit(1)
    
    #variants
    if variant == "rmsnorm":
        if hooks:
            pass
        else:
            pass
        pass
    if variant == "relu":
        if hooks:
            pass
        else:
            pass
    if variant == "batchnorm":
        if hooks:
            pass
        else:
            pass
    
    
    #ablated or normal
    if hooks:
        return vit_LRP(
            pretrained=pretrained,
            num_classes=nb_classes,
            ablated_component= ablated_component
            )
    else:
        return vit_LRP_no_hooks(
            pretrained=pretrained,
            num_classes=nb_classes,
            ablated_component= ablated_component
            )
 