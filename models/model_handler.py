from models.model import deit_tiny_patch16_224 as vit_LRP
from models.model_train import deit_tiny_patch16_224 as vit_LRP_train


from models.variant_light_attention.variant_model_light_attn_train import deit_tiny_patch16_224 as model_variant_light_attention_train
from models.variant_light_attention.variant_model_light_attn import deit_tiny_patch16_224 as model_variant_light_attention

from models.variant_layer_scale.variant_model_layer_scale_train import deit_tiny_patch16_224 as model_variant_layer_scale_train
from models.variant_layer_scale.variant_model_layer_scale import deit_tiny_patch16_224 as model_variant_layer_scale

from models.variant_diff_attention.variant_diff_attention import deit_tiny_patch16_224 as model_variant_diff_attention

from models.variant_weight_normalization.model_variant_weight_normalization import deit_tiny_patch16_224 as model_variant_weight_normalization
from models.variant_weight_normalization.model_variant_weight_normalization_train import deit_tiny_patch16_224 as model_variant_weight_normalization_train




#TODO: add support to remaining variants instead of keeping multiple documents with redundant code

def model_env(pretrained=False,args  = None , hooks = False,  **kwargs):



    if args.variant == 'variant_weight_normalization':
        if hooks:
            return model_variant_weight_normalization(
            isWithBias      = args.model_components["isWithBias"],
            layer_norm      = args.model_components["norm"],
            last_norm       = args.model_components["last_norm"],

            activation      = args.model_components["activation"],
            attn_activation = args.model_components["attn_activation"],
            num_classes     = args.nb_classes,
        )

        else:
            return model_variant_weight_normalization_train(
            isWithBias      = args.model_components["isWithBias"],
            layer_norm      = args.model_components["norm"],
            last_norm       = args.model_components["last_norm"],

            activation      = args.model_components["activation"],
            attn_activation = args.model_components["attn_activation"],
            num_classes     = args.nb_classes,
        )


    if args.variant == 'variant_diff_attn':
        if hooks:
            return model_variant_diff_attention(
            isWithBias      = args.model_components["isWithBias"],
            layer_norm      = args.model_components["norm"],
            last_norm       = args.model_components["last_norm"],

            activation      = args.model_components["activation"],
            attn_activation = args.model_components["attn_activation"],
            num_classes     = args.nb_classes,
        )

   
   
   
    if args.variant == 'variant_layer_scale':
        if hooks:
            return model_variant_layer_scale(
            isWithBias      = args.model_components["isWithBias"],
            layer_norm      = args.model_components["norm"],
            last_norm       = args.model_components["last_norm"],

            activation      = args.model_components["activation"],
            attn_activation = args.model_components["attn_activation"],
            num_classes     = args.nb_classes,
        )

        else:
            return model_variant_layer_scale_train(
            isWithBias      = args.model_components["isWithBias"],
            layer_norm      = args.model_components["norm"],
            last_norm       = args.model_components["last_norm"],

            activation      = args.model_components["activation"],
            attn_activation = args.model_components["attn_activation"],
            num_classes     = args.nb_classes,
        )
    
    
    if args.variant == 'attn_variant_light':
        if hooks:
            return model_variant_light_attention(
            isWithBias      = args.model_components["isWithBias"],
            layer_norm      = args.model_components["norm"],
            last_norm       = args.model_components["last_norm"],

            activation      = args.model_components["activation"],
            attn_activation = args.model_components["attn_activation"],
            num_classes     = args.nb_classes,
        )

        else:
            return model_variant_light_attention_train(
            isWithBias      = args.model_components["isWithBias"],
            layer_norm      = args.model_components["norm"],
            last_norm       = args.model_components["last_norm"],

            activation      = args.model_components["activation"],
            attn_activation = args.model_components["attn_activation"],
            num_classes     = args.nb_classes,
        )





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


   