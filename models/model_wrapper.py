from model_no_hooks_ablation import deit_tiny_patch16_224 as vit_LRP_no_hooks
from model_ablation import deit_tiny_patch16_224 as vit_LRP
from models.variant_relu.model_variant_relu_no_hooks import deit_tiny_patch16_224 as model_variant_relu_no_hooks
from models.variant_relu.model_varient_relu import deit_tiny_patch16_224 as model_variant_relu






def model_env(pretrained=False, hooks = False, nb_classes = 100, ablated_component ="none", variant = "", **kwargs):

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
            return model_variant_relu(
            pretrained=pretrained,
            num_classes=nb_classes,
           # ablated_component= ablated_component
            )
        else:
            return model_variant_relu_no_hooks(
            pretrained=pretrained,
            num_classes=nb_classes,
           # ablated_component= ablated_component
            )
            
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
 