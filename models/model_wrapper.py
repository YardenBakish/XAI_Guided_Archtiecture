from model_no_hooks_ablation import deit_tiny_patch16_224 as vit_LRP_no_hooks
from model_ablation import deit_tiny_patch16_224 as vit_LRP
from models.variant_relu.model_variant_relu_no_hooks import deit_tiny_patch16_224 as model_variant_relu_no_hooks
from models.variant_relu.model_variant_relu import deit_tiny_patch16_224 as model_variant_relu
from models.variant_norm.model_variant_norm_no_hooks import deit_tiny_patch16_224 as model_variant_norm_no_hooks
from models.variant_norm.model_variant_norm import deit_tiny_patch16_224 as model_variant_norm





def model_env(pretrained=False, hooks = False, nb_classes = 100, ablated_component ="none", variant = None, **kwargs):

    if ablated_component:
        if ablated_component != "none" and variant!= None:
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
            print(f"calling model RELU with hooks: {hooks}")

            return model_variant_relu(
            pretrained=pretrained,
            num_classes=nb_classes,
           # ablated_component= ablated_component
            )
        else:
            print(f"calling model RELU with hooks: {hooks}")

            return model_variant_relu_no_hooks(
            pretrained=pretrained,
            num_classes=nb_classes,
           # ablated_component= ablated_component
            )
            
    if variant == "batchnorm":
        if hooks:
            print(f"calling model BatchNorm with hooks: {hooks}")

            return model_variant_norm(
            pretrained=pretrained,
            num_classes=nb_classes,
           # ablated_component= ablated_component
            )
        else:
            print(f"calling model BatchNorm with hooks: {hooks}")

            return model_variant_norm_no_hooks(
            pretrained=pretrained,
            num_classes=nb_classes,
           # ablated_component= ablated_component
            )
    
    
    #ablated or normal
    if hooks:
        print(f"calling model with ablation {ablated_component} with hooks: {hooks}")

        return vit_LRP(
            pretrained=pretrained,
            num_classes=nb_classes,
            ablated_component= ablated_component
            )
    else:
        print(f"calling model with ablation {ablated_component} with hooks: {hooks}")

        return vit_LRP_no_hooks(
            pretrained=pretrained,
            num_classes=nb_classes,
            ablated_component= ablated_component
            )
 