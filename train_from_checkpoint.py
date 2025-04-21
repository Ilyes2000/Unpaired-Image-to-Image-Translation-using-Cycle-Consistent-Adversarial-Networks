import torch, itertools, time, csv, os, re
from options.train_options import TrainOptions
from data.unaligned_dataset import get_dataloader
from models.networks import define_G, define_D, GANLoss
from utils import save_sample, save_model, ImagePool
from tqdm import tqdm

def extract_epoch(path):
	if path is None:
		return None
	match = re.search(r'_epoch_(\d+)\.pth', path)
	return int(match.group(1)) if match else None

def train():
	print("Je récupère les options ..")
	opt = TrainOptions().parse()
	device = torch.device('cuda' if opt.gpu_ids else 'cpu')
	torch.backends.cudnn.benchmark = True
	print(f"device : {device}")

	print("Je récupère les données ..")
	loader = get_dataloader(
		opt.dataroot, 'train', opt.image_size,
		opt.batch_size, opt.num_threads,
		pin_memory=True
	)

	print("Je définie le réseau ..")
	netG_A = define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_gain, opt.gpu_ids)
	netG_B = define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_gain, opt.gpu_ids)
	netD_A = define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_gain, opt.gpu_ids)
	netD_B = define_D(opt.input_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_gain, opt.gpu_ids)

	pretrained_weights = {
        "netG_A": os.path.join("model_checkpoints", "G_AB_epoch_55.pth"),
        "netG_B": os.path.join("model_checkpoints", "G_BA_epoch_55.pth"),
        "netD_A": os.path.join("model_checkpoints", "D_A_epoch_55.pth"),
        "netD_B": os.path.join("model_checkpoints", "D_B_epoch_55.pth")
    }

	# Charger les poids si disponibles
	for net_name, net in zip(pretrained_weights.keys(), [netG_A, netG_B, netD_A, netD_B]):
		weight_path = pretrained_weights[net_name]
		if weight_path and os.path.exists(weight_path):
			print(f"Chargement de {weight_path} pour {net_name}")
			state_dict = torch.load(weight_path, map_location=device)
			net.load_state_dict(state_dict)
		else:
			print(f"Pas de checkpoint trouvé pour {net_name}, initialisation normale.")

	# Détermination de l'époque de reprise
	epochs_loaded = [extract_epoch(path) for path in pretrained_weights.values()]
	start_epoch = max(filter(None, epochs_loaded), default=opt.epoch_count)
	print(f"Reprise à l'époque {start_epoch}")

	criterionGAN   = GANLoss(opt.gan_mode).to(device)
	criterionCycle = torch.nn.L1Loss().to(device)
	criterionIdt   = torch.nn.L1Loss().to(device)

	optG = torch.optim.Adam(itertools.chain(netG_A.parameters(), netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
	optD = torch.optim.Adam(itertools.chain(netD_A.parameters(), netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))

	print("Je génère les fake pool initiaux")
	fake_A_pool, fake_B_pool = ImagePool(opt.pool_size), ImagePool(opt.pool_size)

	csv_path = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.csv')
	os.makedirs(os.path.dirname(csv_path), exist_ok=True)
	if not os.path.exists(csv_path):
		with open(csv_path, mode='w', newline='') as file:
			writer = csv.writer(file)
			writer.writerow([
				"epoch", "loss_G", "loss_G_A", "loss_G_B",
				"loss_cycle_A", "loss_cycle_B", "loss_idt_A", "loss_idt_B",
				"loss_D_A", "loss_D_B"
			])

	print("Je commence l'entraînement")
	for epoch in tqdm(range(start_epoch, opt.n_epochs + opt.n_epochs_decay + 1), desc="Epochs"):
		start_time = time.time()
		total_loss_G = total_loss_G_A = total_loss_G_B = 0
		total_loss_cycle_A = total_loss_cycle_B = 0
		total_loss_idt_A = total_loss_idt_B = 0
		total_loss_D_A = total_loss_D_B = 0
		n_batches = 0

		for i, data in enumerate(loader):
			n_batches += 1
			real_A, real_B = data['A'].to(device), data['B'].to(device)

			# --- G ---
			optG.zero_grad()
			fake_B = netG_A(real_A);    rec_A = netG_B(fake_B)
			fake_A = netG_B(real_B);    rec_B = netG_A(fake_A)

			idt_A = netG_A(real_B); loss_idt_A = criterionIdt(idt_A, real_B) * opt.lambda_B * opt.lambda_identity
			idt_B = netG_B(real_A); loss_idt_B = criterionIdt(idt_B, real_A) * opt.lambda_A * opt.lambda_identity

			loss_G_A = criterionGAN(netD_A(fake_B), True)
			loss_G_B = criterionGAN(netD_B(fake_A), True)
			loss_cycle_A = criterionCycle(rec_A, real_A) * opt.lambda_A
			loss_cycle_B = criterionCycle(rec_B, real_B) * opt.lambda_B

			loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B
			loss_G.backward(); optG.step()

			# --- D ---
			optD.zero_grad()
			pred_real = netD_A(real_B); loss_D_real = criterionGAN(pred_real, True)
			pred_fake = netD_A(fake_B_pool.query(fake_B.detach())); loss_D_fake = criterionGAN(pred_fake, False)
			loss_D_A = (loss_D_real + loss_D_fake) * 0.5; loss_D_A.backward()

			pred_real = netD_B(real_A); loss_D_real = criterionGAN(pred_real, True)
			pred_fake = netD_B(fake_A_pool.query(fake_A.detach())); loss_D_fake = criterionGAN(pred_fake, False)
			loss_D_B = (loss_D_real + loss_D_fake) * 0.5; loss_D_B.backward()
			optD.step()

			total_loss_G += loss_G.item()
			total_loss_G_A += loss_G_A.item()
			total_loss_G_B += loss_G_B.item()
			total_loss_cycle_A += loss_cycle_A.item()
			total_loss_cycle_B += loss_cycle_B.item()
			total_loss_idt_A += loss_idt_A.item()
			total_loss_idt_B += loss_idt_B.item()
			total_loss_D_A += loss_D_A.item()
			total_loss_D_B += loss_D_B.item()

			if i % 100 == 0:
				print(f"[Epoch {epoch}] [Batch {i}] "
					f"[D_A loss: {loss_D_A.item():.4f}] [D_B loss: {loss_D_B.item():.4f}] "
					f"[G loss: {loss_G.item():.4f}]")

		with open(csv_path, mode='a', newline='') as file:
			writer = csv.writer(file)
			writer.writerow([
				epoch,
				total_loss_G / n_batches,
				total_loss_G_A / n_batches,
				total_loss_G_B / n_batches,
				total_loss_cycle_A / n_batches,
				total_loss_cycle_B / n_batches,
				total_loss_idt_A / n_batches,
				total_loss_idt_B / n_batches,
				total_loss_D_A / n_batches,
				total_loss_D_B / n_batches
			])

		save_sample(netG_A, netG_B, real_A, real_B, epoch)
		save_model(netG_A, netG_B, netD_A, netD_B, epoch)
		print(f"[Epoch {epoch}] terminé en {time.time() - start_time:.2f} sec.")

if __name__ == "__main__":
	train()
