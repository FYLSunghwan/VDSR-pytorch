import os

class Settings:
    root = os.path.dirname(os.path.realpath(__file__))

    # Train dataset settings
    dataset_info = {
        '291': {
            'path': './',
            'link': '1Rt3asDLuMgLuJvPA1YrhyjWhb97Ly742',
            'is_gray': False,
            'random_scale': True,
            'crop_size': 48,
            'rotate': True,
            'fliplr': True,
            'fliptb': True,
            'scale_factor': 2,
            'random_scale_factor': True
        },
        'SR_testing_datasets': {
            'id': 'Set5',
            'path': './',
            'link': 'http://vllab.ucmerced.edu/wlai24/LapSRN/results/SR_testing_datasets.zip',
            'is_gray': False,
            'scale_factor': 3
        }
    }
    dataset_root = os.path.join(root, 'datasets')
    
    # Model Settings
    num_channels = 3
    num_threads = 4
    batch_size = 64
    test_batch_size = 1
    clip = 0.4
    lr = 0.1
  