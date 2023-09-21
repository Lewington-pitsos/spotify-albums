import timm 

# model = timm.create_model('vit_base_patch16_clip_224.laion2b_ft_in12k_in1k', pretrained=True) 

pretrained_model_names = timm.list_models(pretrained=True)
for name in pretrained_model_names:
    if 'vit' in name.lower() and '384' in name.lower():
        print(name)