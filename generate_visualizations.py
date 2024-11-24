import os
from tqdm import tqdm
import h5py

import argparse

# Import saliency methods and models
from baselines.ViT.misc_functions import *
#from dataset.label_index_corrector  import *
from ViT_explanation_generator import Baselines, LRP
from model_ablation import deit_tiny_patch16_224 as vit_LRP
from models.model_wrapper import model_env 

from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
from deit.datasets import build_dataset
import torch

from torchvision.datasets import ImageNet
from torchvision import datasets, transforms


def normalize(tensor,
              mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    tensor.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
    return tensor


def compute_saliency_and_save(args):

    first = True
    #correct_label_dic = convertor_dict()

    with h5py.File(os.path.join(args.method_dir, 'results.hdf5'), 'a') as f:
        data_cam = f.create_dataset('vis',
                                    (1, 1, 224, 224),
                                    maxshape=(None, 1, 224, 224),
                                    dtype=np.float32,
                                    compression="gzip")
        data_image = f.create_dataset('image',
                                      (1, 3, 224, 224),
                                      maxshape=(None, 3, 224, 224),
                                      dtype=np.float32,
                                      compression="gzip")
        data_target = f.create_dataset('target',
                                       (1,),
                                       maxshape=(None,),
                                       dtype=np.int32,
                                       compression="gzip")
        for batch_idx, (data, target) in enumerate(tqdm(sample_loader)):
            
            
            
            
            if first:
                 first = False
                 data_cam.resize(data_cam.shape[0] + data.shape[0] - 1, axis=0)
                 data_image.resize(data_image.shape[0] + data.shape[0] - 1, axis=0)
                 data_target.resize(data_target.shape[0] + data.shape[0] - 1, axis=0)
            else:
                 data_cam.resize(data_cam.shape[0] + data.shape[0], axis=0)
                 data_image.resize(data_image.shape[0] + data.shape[0], axis=0)
                 data_target.resize(data_target.shape[0] + data.shape[0], axis=0)

            # Add data
            data_image[-data.shape[0]:] = data.data.cpu().numpy()
            data_target[-data.shape[0]:] = target.data.cpu().numpy()

            target = target.to(device)

            data = normalize(data)
            data = data.to(device)
            data.requires_grad_()

            index = None
            if args.vis_class == 'target':
                index = target

            if args.method == 'rollout':
                Res = baselines.generate_rollout(data, start_layer=1).reshape(data.shape[0], 1, 14, 14)
                # Res = Res - Res.mean()

            elif args.method == 'lrp':
                Res = lrp.generate_LRP(data, start_layer=1, index=index).reshape(data.shape[0], 1, 14, 14)
                # Res = Res - Res.mean()

            elif args.method == 'transformer_attribution':
                #print(model_LRP(data).shape)
          

                print("attribution")
                Res = lrp.generate_LRP(data, start_layer=1, method="grad", index=index).reshape(data.shape[0], 1, 14, 14)
                # Res = Res - Res.mean()

            elif args.method == 'full_lrp':
              
                Res = lrp.generate_LRP(data, method="full", index=index).reshape(data.shape[0], 1, 224, 224)
                # Res = Res - Res.mean()

            elif args.method == 'lrp_last_layer':
              pass
              #  Res = orig_lrp.generate_LRP(data, method="last_layer", is_ablation=args.is_ablation, index=index) \
              #      .reshape(data.shape[0], 1, 14, 14)
                # Res = Res - Res.mean()

            elif args.method == 'attn_last_layer':
                Res = lrp.generate_LRP(data, method="last_layer_attn", is_ablation=args.is_ablation) \
                    .reshape(data.shape[0], 1, 14, 14)

            elif args.method == 'attn_gradcam':
                Res = baselines.generate_cam_attn(data, index=index).reshape(data.shape[0], 1, 14, 14)

            if args.method != 'full_lrp' and args.method != 'input_grads':
                Res = torch.nn.functional.interpolate(Res, scale_factor=16, mode='bilinear').cuda()
            Res = (Res - Res.min()) / (Res.max() - Res.min())

            data_cam[-data.shape[0]:] = Res.data.cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a segmentation')
    parser.add_argument('--batch-size', type=int,
                        default=1,
                        help='')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    parser.add_argument('--ablated-component', type=str,
                        choices=['softmax', 'layerNorm', 'bias'],)
    
    parser.add_argument('--work-env', type=str,
                    
                        help='')
    parser.add_argument('--variant', choices=['rmsnorm', 'relu', 'batchnorm', 'softplus', 'rmsnorm_softplus', 'norm_bias_ablation', 'norm_center_ablation', 'norm_ablation', 'sigmoid'], type=str, help="")

    parser.add_argument('--custom-trained-model', type=str,
                   
                        help='')
    parser.add_argument('--data-set', default='IMNET100', choices=['IMNET100','CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--num-workers', type=int,
                        default= 2,
                        help='')
    parser.add_argument('--method', type=str,
                        default='grad_rollout',
                        choices=['rollout', 'lrp', 'transformer_attribution', 'full_lrp', 'lrp_last_layer',
                                 'attn_last_layer', 'attn_gradcam'],
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
                        required=True,
                        help='')
    args = parser.parse_args()

    # PATH variables
    PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
    if args.work_env:
        PATH = args.work_env
    os.makedirs(os.path.join(PATH, 'visualizations'), exist_ok=True)

    try:
        os.remove(os.path.join(PATH, 'visualizations/{}/{}/results.hdf5'.format(args.method,
                                                                                args.vis_class)))
    except OSError:
        pass


    os.makedirs(os.path.join(PATH, 'visualizations/{}'.format(args.method)), exist_ok=True)
    if args.vis_class == 'index':
        os.makedirs(os.path.join(PATH, 'visualizations/{}/{}_{}'.format(args.method,
                                                                        args.vis_class,
                                                                        args.class_id)), exist_ok=True)
        args.method_dir = os.path.join(PATH, 'visualizations/{}/{}_{}'.format(args.method,
                                                                              args.vis_class,
                                                                              args.class_id))
    else:
        ablation_fold = 'ablation' if args.is_ablation else 'not_ablation'
        os.makedirs(os.path.join(PATH, 'visualizations/{}/{}/{}'.format(args.method,
                                                                     args.vis_class, ablation_fold)), exist_ok=True)
        args.method_dir = os.path.join(PATH, 'visualizations/{}/{}/{}'.format(args.method,
                                                                           args.vis_class, ablation_fold))

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    # Model
    model = vit_LRP(pretrained=True).cuda()
    baselines = Baselines(model)

    # LRP
    if args.custom_trained_model != None:
        if args.data_set == 'IMNET100':
            args.nb_classes = 100
        else:
            args.nb_classes = 1000
    
       
        model_LRP = model_env(pretrained=False, 
                      nb_classes=args.nb_classes,  
                      ablated_component= args.ablated_component,
                      variant = args.variant,
                      hooks = True,
                    )
        #model_LRP.head = torch.nn.Linear(model_LRP.head.weight.shape[1],100)
        checkpoint = torch.load(args.custom_trained_model, map_location='cpu')

        model_LRP.load_state_dict(checkpoint['model'], strict=False)
        model_LRP.to(device)
    else:
        model_LRP = vit_LRP(pretrained=True).cuda()
    model_LRP.eval()
    lrp = LRP(model_LRP)

    # orig LRP
    ''' model_orig_LRP = vit_orig_LRP(pretrained=True).cuda()
    model_orig_LRP.eval()
    orig_lrp = LRP(model_orig_LRP)'''

    # Dataset loader for sample images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset_val, _ = build_dataset(is_train=False, args=args)
    
    #random 0.4
    np.random.seed(42)
    total_size  = len(dataset_val)
    indices = list(range(total_size))
    subset_size = int(total_size * 0.04)
    random_indices = np.random.choice(indices, size=subset_size, replace=False)
    sampler = SubsetRandomSampler(random_indices)

    #first 0.1
    '''total_size  = len(dataset_val)
    subset_size = int(total_size * 0.1)
    indices     = list(range(subset_size))
    dataset_val = Subset(dataset_val, indices)'''

    #print(subset.indices)
    #sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    #imagenet_ds = ImageNet(args.imagenet_validation_path, split='val', download=False, transform=transform)
    sample_loader = torch.utils.data.DataLoader(    
        dataset_val, sampler=sampler,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=args.pin_mem,
        num_workers=args.num_workers,
        drop_last = True

    )

    compute_saliency_and_save(args)
