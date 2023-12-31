import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'using device: {device}')


ALBUM_IMG_KEY = 'album_image'
TIMM_MODELS = [ 'vit_base_patch16_224', 'maxvit_xlarge_tf_224.in21k', 'vit_huge_patch14_clip_224.laion2b', 'vit_base_patch16_clip_224.laion2b_ft_in12k_in1k', 'vit_large_patch14_clip_224.laion2b_ft_in12k_in1k']
BASE_DIR = 'data/tracks2/'