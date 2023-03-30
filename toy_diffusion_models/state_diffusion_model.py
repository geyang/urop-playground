import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_swiss_roll

from params_proto import ParamsProto

from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Lambda, CenterCrop, Resize
from tqdm import tqdm

import torch
import torch.nn.functional as F

from toy_diffusion_models.models.state_model import CondMlpUnet


class Schedules:
    @staticmethod
    def cosine_beta(T, s=0.008):
        """
        cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = T + 1
        x = torch.linspace(0, T, steps)
        alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    @staticmethod
    def linear_beta(T):
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, T)

    @staticmethod
    def quadratic_beta(T):
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, T) ** 2

    @staticmethod
    def sigmoid_beta(T):
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(-6, 6, T)
        return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


class Args(ParamsProto):
    seed = 100

    T = 200
    t_dim = 100
    betas = "Schedules.sigmoid_beta(Args.T)"
    n_epochs = 200
    lr = 2e-4
    # gradient_accumulate_every = 2
    loss_type = "huber"

    batch_size = 64
    checkpoint = None
    save_and_sample_every = 1000
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_interval = None  # 200
    render_interval = 20


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def main(**deps):
    from ml_logger import logger

    Args._update(deps)
    print(logger)
    torch.manual_seed(Args.seed)
    logger.log_params(Args=vars(Args))

    # define alphas
    betas = eval(Args.betas)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    # calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    # forward diffusion
    def q_sample(x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(denoise_model, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = denoise_model(x_noisy, t)

        if Args.loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif Args.loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif Args.loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss

    xyz, label = make_swiss_roll(n_samples=10000, noise=0.01, random_state=None)

    # normalize
    def normalize(t):
        scale = t.max(axis=0) - t.min(axis=0)
        offset = (t.max(axis=0) + t.min(axis=0)) * 0.5
        return (t - offset) / scale

    dataloader = DataLoader(normalize(xyz[:, [0, 2]]).astype(np.float32), batch_size=Args.batch_size, shuffle=True)

    @torch.no_grad()
    def p_sample(model, x, t, t_index):
        betas_t = extract(betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

        # Note: Equation 11 in the paper Use our model (noise predictor)
        #  to predict the mean
        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Note: Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(model, shape):
        b = shape[0]
        # start from pure noise (for each example in the batch)
        x_start = torch.randn(shape, device=Args.device)
        history = []

        for t in tqdm(reversed(range(0, Args.T)), desc='sampling loop time step', total=Args.T):
            cond = torch.full((b,), t, device=Args.device, dtype=torch.long)
            x_start = p_sample(model, x_start, cond, t)
            history.append(x_start.cpu().numpy())
        return history

    def plot_history(history, prefix):
        files = []
        for t, frame in enumerate(history):
            plt.scatter(*frame.T)
            plt.xlim(-1.2, 1.2)
            plt.ylim(-1.2, 1.2)
            plt.gca().set_aspect('equal')
            path = logger.savefig(f"{prefix}/frame_{t:04d}.png")
            files.append(path)
        # logger.make_video(files, f"{prefix}.gif")

    @torch.no_grad()
    def sample(model, batch_size=16, channels=2):
        return p_sample_loop(model, shape=(batch_size, channels))

    def num_to_groups(num, divisor):
        groups = num // divisor
        remainder = num % divisor
        arr = [divisor] * groups
        if remainder > 0:
            arr.append(remainder)
        return arr

    model = CondMlpUnet(in_dim=2, lat_dim=200, mid_dim=200, cond_dim=Args.t_dim, mults=[1, 1, 1])
    model.to(Args.device)

    from torchinfo import summary
    summary(model)

    optimizer = Adam(model.parameters(), lr=Args.lr)

    for epoch in range(Args.n_epochs + 1):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()

            batch = batch.to(Args.device)

            # Note: Algorithm 1 line 3: sample t uniformly for every example in the batch
            t = torch.randint(0, Args.T, batch.shape[:1], device=Args.device).long()
            loss = p_losses(model, batch, t)

            loss.backward()
            optimizer.step()

            logger.store_metrics(loss=loss.cpu().item())

        logger.log_metrics_summary(key_values={"epoch": epoch})

        if Args.checkpoint_interval and epoch % Args.checkpoint_interval == 0:
            logger.print(f"Saving the checkpoint at epoch {epoch}", color="yellow")
            logger.torch_save(model, f"checkpoints/unet_{epoch}.pkl")

        if epoch % Args.render_interval == 0:
            # save generated images
            logger.print('rendering...', color="yellow")
            # batches = num_to_groups(4, Args.batch_size)

            history = sample(model, batch_size=400, channels=2)
            logger.print('saving the samples...', color="green")
            plot_history(history, f"samples/ep_{epoch}")
            # logger.save_images(all_images, f'samples/sample_{epoch}.png', n_rows=4)

    # # sample 64 images
    # samples = sample(model, image_size=image_size, batch_size=64, channels=channels)
    #
    # # show a random one
    # random_index = 5
    # plt.imshow(samples[-1][random_index].reshape(image_size, image_size, channels), cmap="gray")
    # plt.show()
    #
    # for random_index in trange(53, desc="sampling images"):
    #
    #     ims = []
    #     for i in range(Args.T):
    #         im = samples[i][random_index, 0]
    #         im -= im.min()
    #         im /= im.max()
    #         ims.append(im)
    #
    #     logger.save_video(ims, f"figures/diffusion_{random_index}.gif")


if __name__ == '__main__':
    from ml_logger import logger, instr

    thunk = instr(main)
    logger.log_text("""
    charts:
    - xKey: epoch
      yKey: loss/mean
      yDomain: [0, 0.2]
    """, ".charts.yml", dedent=True)
    thunk()
