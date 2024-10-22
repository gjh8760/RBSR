class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = './'
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        self.pretrained_nets_dir = '/data/gjh8760/Codes/RBSR/pretrained_networks/'    # Directory for pre-trained networks.
        self.save_data_path = self.workspace_dir + '/evaluation'    # Directory for saving network predictions for evaluation.
        self.zurichraw2rgb_dir = '/data/gjh8760/datasets/Zurich-RAW-to-DSLR-Dataset'    # Zurich RAW 2 RGB path
        self.burstsr_dir = ''    # BurstSR dataset path
        self.synburstval_dir = '/data/gjh8760/datasets/SyntheticBurstVal'    # SyntheticBurst validation set path
