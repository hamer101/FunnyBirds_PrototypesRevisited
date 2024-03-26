import tomllib 

with open("../model_selection.toml", "rb") as f:
    TOML = tomllib.load(f)

base_architecture = TOML['model_paramas']['base_architecture'] # or 'vgg19' or 'densenet169'

# There are 500 test images (10 images per one of 50 classes) of 256x256 resolution
img_size = 256
prototype_shape = (500, TOML['model_paramas']['prototype_size'], 1, 1) #the 128 is a prototype size. We tested 256 and 512 as well. 
num_classes = 50
prototype_activation_function = 'log'
add_on_layers_type = 'regular'

experiment_run = TOML['experiment_run']

# While using Funnybirds Loader, we just have to provide the Funnybirds Dataset's root dir
data_path = TOML['paths']['dataset_dir']

# The funnybirds dataloader implementation automatically finds test/train/push paths in root path
train_dir = data_path
test_dir = data_path 
train_push_dir = data_path 

train_batch_size = 80
test_batch_size = 100
train_push_batch_size = 75

joint_optimizer_lrs = {'features': 1e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3}

# Interval between decreases of learning rate extended to every 10th epoch 
joint_lr_step_size = TOML['model_paramas']['joint_lr_step_size']

warm_optimizer_lrs = {'add_on_layers': 3e-3,
                      'prototype_vectors': 3e-3}

last_layer_optimizer_lr = 1e-4

coefs = {
    'crs_ent': 1,
    'clst': 0.8,
    'sep': -0.08,
    'l1': 1e-4,
}

num_train_epochs = 100
num_warm_epochs = 5

# Push start postponed to 25th epoch
push_start = TOML['model_paramas']['push_start']

push_epochs = [i for i in range(num_train_epochs) if i % 10 == 0]
