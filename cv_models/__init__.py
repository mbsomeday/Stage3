import torch


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

LOCAL = {
    'dataset_base_dir': r'D:\my_phd\dataset\Stage3',
    'weights_save_path': r'D:\my_phd\Model_Weights\Stage3',
    'D1_ECPDaytime': {
        'base_dir': r'D:\my_phd\dataset\Stage3\D1_ECPDaytime',
        'txt_dir': 'dataset_txt'
    },
    'D2_CityPersons': {
        'base_dir': r'D:\my_phd\dataset\Stage3\D2_CityPersons',
        'txt_dir': 'dataset_txt'
    },
    'D3_ECPNight': {
        'base_dir': r'D:\my_phd\dataset\Stage3\D3_ECPNight',
        'txt_dir': 'dataset_txt'
    },
    'D4_BDD100K': {
        'base_dir': r'D:\my_phd\dataset\Stage3\D4_BDD100K',
        'txt_dir': 'dataset_txt'
    }
}














