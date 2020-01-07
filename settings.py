import os

class Settings:
    root = os.path.dirname(os.path.realpath(__file__))

    # Train dataset settings
    dataset_info = {
        '291': {
            'path': './',
            'link': '1Rt3asDLuMgLuJvPA1YrhyjWhb97Ly742',
            'is_gray': False,
            'random_scale': False,
            'crop_size': 64,
            'rotate': True,
            'fliplr': True,
            'fliptb': True,
            'scale_factor': 4
        },
        'SR_testing_datasets': {
            'id': 'Set5',
            'path': './',
            'link': 'http://vllab.ucmerced.edu/wlai24/LapSRN/results/SR_testing_datasets.zip',
            'is_gray': False,
            'scale_factor': 4
        }
    }
    dataset_root = os.path.join(root, 'datasets')
    
    # Model Settings
    num_channels = 3
    num_threads = 1
    batch_size = 128
    test_batch_size = 2
    crop_size = 48
    scale_factor = 2
    clip = 0.4
    lr = 0.1
  