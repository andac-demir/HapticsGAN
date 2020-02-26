import torch
from torch import nn
from torch.autograd import Variable


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )

    def forward(self, x):
        return self.encoder(x)


class Generator(nn.Module):
    def __init__(self):
		super(Generator, self).__init__()
		self.image_size = 64
		self.num_channels = 3
		self.noise_dim = 100
		self.embed_dim = 1024
		self.projected_embed_dim = 128
		self.latent_dim = self.noise_dim + self.projected_embed_dim
		self.ngf = 64
		self.projection = nn.Sequential(
			nn.Linear(in_features=self.embed_dim,
                      out_features=self.projected_embed_dim),
			nn.BatchNorm1d(num_features=self.projected_embed_dim),
			nn.LeakyReLU(negative_slope=0.2, inplace=True)
			)
		self.genNet = nn.Sequential(
			nn.ConvTranspose2d(in_channels=self.latent_dim,
                               out_channels=self.ngf * 8,
                               kernel_size=4, stride=1, padding=0, bias=False),
			nn.BatchNorm2d(num_features=self.ngf * 8),
			nn.ReLU(inplace=True),
            # 4 x 4 x (ngf x 8)
			nn.ConvTranspose2d(in_channels=self.ngf * 8,
                               out_channels=self.ngf * 4,
                               kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(num_features=self.ngf * 4),
			nn.ReLU(inplace=True),
            # 8 x 8 x (ngf x 4)
			nn.ConvTranspose2d(in_channels=self.ngf * 4,
                               out_channels=self.ngf * 2,
                               kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(num_features=self.ngf * 2),
			nn.ReLU(inplace=True),
            # 16 x 16 x (ngf x 2)
			nn.ConvTranspose2d(in_channels=self.ngf * 2,
                               out_channels=self.ngf,
                               kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(num_features=self.ngf),
			nn.ReLU(inplace=True),
            # 32 x 32 x (ngf)
			nn.ConvTranspose2d(in_channels=self.ngf,
                               out_channels=self.num_channels,
                               kernel_size=4, stride=2, padding=1, bias=False),
			nn.Tanh()
            # 64 x 64 x num_channels
			)

	def forward(self, embed_vector, z):
		projected_embed = self.projection(embed_vector).unsqueeze(2).\
                                                        unsqueeze(3)
		latent_vector = torch.cat([projected_embed, z], 1)
		output = self.genNet(latent_vector)
		return output


class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()
		self.image_size = 64
		self.num_channels = 3
		self.embed_dim = 1024
		self.projected_embed_dim = 128
		self.ndf = 64
		self.B_dim = 128
		self.C_dim = 16
		self.discNet1 = nn.Sequential(
			# input is 64 x 64 x (num_channels)
			nn.Conv2d(in_channels=self.num_channels, out_channels=self.ndf,
                      kernel_size=4, stride=2, padding=1, bias=False),
			nn.LeakyReLU(negative_slope=0.2, inplace=True),
			# 32 x 32 x (ndf)
			nn.Conv2d(in_channels=self.ndf,
                      out_channels=self.ndf * 2,
                      kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(num_features=self.ndf * 2),
			nn.LeakyReLU(negative_slope=0.2, inplace=True),
			# 16 x 16 x (ndf*2)
			nn.Conv2d(in_channels=self.ndf * 2, out_channels=self.ndf * 4,
                      kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(num_features=self.ndf * 4),
			nn.LeakyReLU(negative_slope=0.2, inplace=True),
			# 8 x 8 x (ndf*4)
			nn.Conv2d(in_channels=self.ndf * 4, out_channels=self.ndf * 8,
                      kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(num_features=self.ndf * 8),
			nn.LeakyReLU(negative_slope=0.2, inplace=True))
		self.projector = concatEmbed(self.embed_dim, self.projected_embed_dim)
		self.discNet2 = nn.Sequential(
			# 4 x 4 x (ndf*8)
			nn.Conv2d(in_channels=self.ndf * 8 + self.projected_embed_dim,
                      out_channels=1, kernel_size=4, stride=1, padding=0,
                      bias=False),
			nn.Sigmoid())

	def forward(self, x, embed):
		x_intermediate = self.discNet1(x)
		x = self.projector(x_intermediate, embed)
		x = self.discNet2(x)
		return x.view(-1, 1).squeeze(1), x_intermediate


