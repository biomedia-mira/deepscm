from torch import nn


class Encoder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.h1_nchan = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, self.h1_nchan, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(.1, inplace=True)
        )
        self.h2_nchan = 128
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.h1_nchan, self.h2_nchan, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.h2_nchan),
            nn.LeakyReLU(.1, inplace=True)
        )
        self.h3_dim = hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * self.h2_nchan, self.h3_dim),
            nn.BatchNorm1d(self.h3_dim),
            nn.LeakyReLU(.1, inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x).view(-1, 7 * 7 * self.h2_nchan)
        return self.fc(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, upconv=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.h1_dim = 1024
        self.fc1 = nn.Sequential(
            nn.Linear(self.latent_dim, self.h1_dim),
            nn.BatchNorm1d(self.h1_dim),
            nn.ReLU(inplace=True)
        )
        self.h2_nchan = 128
        h2_dim = 7 * 7 * self.h2_nchan
        self.fc2 = nn.Sequential(
            nn.Linear(self.h1_dim, h2_dim),
            nn.BatchNorm1d(h2_dim),
            nn.ReLU(inplace=True)
        )
        self.h3_nchan = 64

        if upconv:
            conv1_ops = [
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(self.h2_nchan, self.h3_nchan, kernel_size=5, stride=1, padding=2)
            ]
            conv2_ops = [
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(self.h3_nchan, 1, kernel_size=5, stride=1, padding=2)
            ]
        else:
            conv1_ops = [nn.ConvTranspose2d(self.h2_nchan, self.h3_nchan,
                                            kernel_size=4, stride=2, padding=1)]
            conv2_ops = [nn.ConvTranspose2d(self.h3_nchan, 1, kernel_size=4, stride=2, padding=1)]

        self.conv1 = nn.Sequential(
            *conv1_ops,
            nn.BatchNorm2d(self.h3_nchan),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(*conv2_ops)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x).view(-1, self.h2_nchan, 7, 7)
        x = self.conv1(x)
        return self.conv2(x)


def weights_init(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.)
    elif 'BatchNorm' in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.)
