from models.model import deit_tiny_patch16_224 as vit_LRP


def model_env(pretrained=False,args  = None , hooks = False,  **kwargs):
    
    
    print(args.model_components["isWithBias"])
    print(args.model_components["norm"])
    print(args.model_components["attn_activation"])
    print(args.model_components["activation"])


    return vit_LRP(
        isWithBias      = args.model_components["isWithBias"],
        layer_norm      = args.model_components["norm"],
        last_norm       = args.model_components["last_norm"],

        activation      = args.model_components["activation"],
        attn_activation = args.model_components["attn_activation"],
        num_classes     = args.nb_classes,
    )

    if ablated_component:
        if ablated_component != "none" and variant!= None:
            print("can't have both a variant and ablation")
            exit(1)
    
    #variants
    if variant == "rmsnorm":
        if hooks:
            print(f"calling model RMSNorm with hooks: {hooks}")

            return model_variant_rmsnorm(
            pretrained=pretrained,
            num_classes=nb_classes,
           # ablated_component= ablated_component
            )
        else:
            print(f"calling model RMSNorm with hooks: {hooks}")

            return model_variant_rmsnorm_no_hooks(
            pretrained=pretrained,
            num_classes=nb_classes,
           # ablated_component= ablated_component
            )
        
    if variant == "rmsnorm_softplus":
        if hooks:
            print(f"calling model RMSNorm-softplus with hooks: {hooks}")

            return model_variant_rms_softplus(
            pretrained=pretrained,
            num_classes=nb_classes,
           # ablated_component= ablated_component
            )
        else:
            print(f"calling model RMSNorm-softplus with hooks: {hooks}")
            return model_variant_rms_softplus_no_hooks(
            pretrained=pretrained,
            num_classes=nb_classes,
           # ablated_component= ablated_component
            )
    
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
        
    if variant == "sigmoid":
        if hooks:
            print(f"calling model SIGMOID with hooks: {hooks}")

            return model_variant_sigmoid(
            pretrained=pretrained,
            num_classes=nb_classes,
           # ablated_component= ablated_component
            )
        else:
            print(f"calling model SIGMOID with hooks: {hooks}")

            return model_variant_sigmoid_no_hooks(
            pretrained=pretrained,
            num_classes=nb_classes,
           # ablated_component= ablated_component
            )
    if variant == "softplus":
        if hooks:
            print(f"calling model Softplus with hooks: {hooks}")

            return model_variant_softplus(
            pretrained=pretrained,
            num_classes=nb_classes,
           # ablated_component= ablated_component
            )
        else:
            print(f"calling model RELU with hooks: {hooks}")

            return model_variant_softplus_no_hooks(
            pretrained=pretrained,
            num_classes=nb_classes,
           # ablated_component= ablated_component
            )
        

    if variant == "batchnorm_param":
        if hooks:
            print(f"calling model bacthnorm_parm with hooks: {hooks}")

            return model_variant_batch_param(
            pretrained=pretrained,
            num_classes=nb_classes,
           # ablated_component= ablated_component
            )
        else:
            print(f"calling model bacthnorm_parm with hooks: {hooks}")


            return model_variant_batch_param(
            pretrained=pretrained,
            num_classes=nb_classes,
           # ablated_component= ablated_component
            )
        
    if variant == "batchnorm":
        if hooks:
            print(f"calling model BatchNorm with hooks: {hooks}")

            return model_variant_batchnorm(
            pretrained=pretrained,
            num_classes=nb_classes,
           # ablated_component= ablated_component
            )
        else:
            print(f"calling model BatchNorm with hooks: {hooks}")

            return model_variant_batchnorm_no_hooks(
            pretrained=pretrained,
            num_classes=nb_classes,
           # ablated_component= ablated_component
            )
    
            
    if variant in ['norm_bias_ablation' , 'norm_center_ablation' , 'norm_ablation']:
        if hooks:
            print(f"calling model norm ablation with hooks: {hooks} with {variant}")

            return model_variant_norm_ablation(
            pretrained=pretrained,
            num_classes=nb_classes,
            ablated_norm = variant
           # ablated_component= ablated_component
            )
        else:
            print(f"calling model norm ablation with hooks: {hooks} with {variant}")

            return model_variant_norm_ablation_no_hooks(
            pretrained=pretrained,
            num_classes=nb_classes,
            ablated_norm = variant

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
 