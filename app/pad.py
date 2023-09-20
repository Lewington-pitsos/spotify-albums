import timm 


model = timm.create_model('maxvit_xlarge_tf_224.in21k', pretrained=True)