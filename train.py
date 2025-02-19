import cv2

from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

model = Unet(
    dim = 32,
    dim_mults = (1, 2, 2, 2),
    dropout = 0.1,
    flash_attn = False
)

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000,           # number of steps
    sampling_timesteps = 250,  # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
)

trainer = Trainer(
    diffusion,
    '/data/ddpm_data/',
    train_batch_size = 32,
    train_lr = 8e-5,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    save_and_sample_every = 70000,
    amp = True,                       # turn on mixed precision
    calculate_fid = True,             # whether to calculate fid during training
    num_fid_samples = 200,
)

trainer.train()

sampled_images = diffusion.sample(batch_size = 2)
for index in range(sampled_images.shape[0]):
    image = sampled_images[index].permute(1, 2, 0).detach().cpu().numpy()
    image = (image * 255.0).astype('uint8')
    print(image)
    cv2.imwrite(f"{index}.jpg", image)
