import torch
import math
import torch.nn as nn
import numpy as np


# class FF(nn.Linear):
#     def __init__(self, in_dim, out_dim, scale):
#         self.in_dim, self.out_dim = in_dim, out_dim
#         super().__init__(in_features=in_dim, out_features=out_dim)
#         self.weight = torch.normal(0, scale)
#         self.bias = torch.uniform(-0.5, 0.5)
#
#     def forward(self, input):
#         z = self(input)
#         return torch.sin(2 * np.pi * z)


class PosEmb(nn.Module):
    def __init__(self, dim, b=10000):
        super().__init__()

        assert dim % 2 == 0, "dimension needs to be multiples of 2"

        half_dim = dim // 2
        scale = math.log(b) / half_dim
        self.register_buffer(
            'ladder', torch.exp(torch.arange(half_dim) * -scale))

    def forward(self, t):
        # this is position encoding -- it is low-rank product
        mixure = t[:, None] * self.ladder[None, :]
        return torch.cat([mixure.sin(), mixure.cos()], dim=-1)


if __name__ == '__main__':
    # 40 is my batch size
    ts = torch.arange(40)

    pe = PosEmb(20)
    embedded = pe(ts)
    import matplotlib.pyplot as plt

    plt.imshow(embedded)
    plt.show()


class CondResBlock(nn.Module):
    """Conditioned ResNet Block"""

    def __init__(self, in_dim, out_dim, cond_dim):
        super().__init__()

        self.in_features, self.out_features = in_dim, out_dim

        # Note: Not sure why the Mish/SiLU is needed
        self.cond_mlp = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, out_dim))

        # from ml_logger import logger
        # logger.print('I am using non-linearity', color="green")
        self.block_1 = nn.Sequential(nn.Linear(in_dim, out_dim), nn.Mish())
        self.block_2 = nn.Sequential(nn.Linear(out_dim, out_dim), nn.Mish())

        if in_dim != out_dim:
            self.res_linear = nn.Linear(in_dim, out_dim)
        else:
            self.res_linear = nn.Identity()

    def forward(self, x, cond):
        """
        inputs:
            x : [ batch_size x in_dim ]
            cond : [ batch_size x embed_dim ]
        returns:
            out : [ batch_size x out_channels x horizon ]
        """
        h, cond = self.block_1(x), self.cond_mlp(cond)
        # Note: standard practice in transformer. Add the conditioning. Can change to multiplication
        out = self.block_2(cond + h)
        # einops.rearrange(cond, 'b -> b 1') + h
        # 我加，我加，我加加加
        return out + self.res_linear(x)


class CondMlpUnet(nn.Module):
    def __init__(self, in_dim, lat_dim, mid_dim, cond_dim, mults=(1, 4, 8)):
        super().__init__()

        in_dims = [in_dim] + [lat_dim * mul for mul in mults]
        out_dims = in_dims[::-1]

        self.pe = PosEmb(cond_dim, 10000)
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        for i, o in zip(in_dims[:-1], in_dims[1:]):
            self.downs.append(nn.ModuleList([
                CondResBlock(i, o, cond_dim),
                CondResBlock(o, o, cond_dim)
            ]))

        self.mid_block_1 = CondResBlock(o, mid_dim, cond_dim)
        self.mid_block_2 = CondResBlock(mid_dim, o, cond_dim)

        for i, o in zip(out_dims[:-1], out_dims[1:]):
            self.ups.append(nn.ModuleList([
                CondResBlock(i * 2, o, cond_dim),
                CondResBlock(o, o, cond_dim),
            ]))

    def forward(self, x, cond):
        """
            x : [ batch x horizon x transition ]
        """
        cond = self.pe(cond)

        # the latent, res buffer
        h = []
        for resnet_1, resnet_2 in self.downs:
            x = resnet_1(x, cond)
            x = resnet_2(x, cond)
            h.append(x)

        x = self.mid_block_1(x, cond)
        x = self.mid_block_2(x, cond)

        for resnet_1, resnet_2 in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet_1(x, cond)
            x = resnet_2(x, cond)

        return x


if __name__ == '__main__':
    # The non-Conv UNet does not have an additional dimension.
    T = 100
    net = CondMlpUnet(5, 32, 8, 10, mults=[1, 4, 8])
    xs = torch.rand([30, 5])
    ts = torch.randint(T, [30])
    # Note: can also diffusion on the time axis.
    output = net(xs, ts)
    assert xs.shape == output.shape, "the output shape should be identical"
