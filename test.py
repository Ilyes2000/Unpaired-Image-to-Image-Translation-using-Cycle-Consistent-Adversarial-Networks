import torch
from options.test_options import TestOptions
from data.unaligned_dataset import get_dataloader
from models.networks import define_G
from utils import save_sample_v2
import os

def main():
    opt = TestOptions().parse()
    device = torch.device('cuda' if opt.gpu_ids else 'cpu')

    # Charger générateurs
    netG_A = define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout,
                    opt.init_gain, opt.gpu_ids)
    netG_B = define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout,
                    opt.init_gain, opt.gpu_ids)
    # load weights
    epoch = opt.epoch
    suffix = opt.model_suffix

    checkpoint_dir = "model_checkpoints"  # Remplace ce chemin par celui qui est correct pour toi
    path_G_A = os.path.join(checkpoint_dir, f"G_AB_epoch_{epoch}.pth")
    path_G_B = os.path.join(checkpoint_dir, f"G_BA_epoch_{epoch}.pth")

    print(path_G_A)
    netG_A.load_state_dict(torch.load(path_G_A, map_location=device))
    netG_B.load_state_dict(torch.load(path_G_B, map_location=device))

    netG_A.eval()
    netG_B.eval()

    loader = get_dataloader(
        opt.dataroot, 'test', opt.image_size,
        opt.batch_size, opt.num_threads
    )

    for i, data in enumerate(loader):
        real_A, real_B = data['A'].to(device), data['B'].to(device)
        fake_B = netG_A(real_A)
        fake_A = netG_B(real_B)
        # enregistrer
        save_sample_v2(fake_A, fake_B, real_A, real_B, f"{opt.epoch}_test_{i}", 'results')

if __name__ == '__main__':
    main()