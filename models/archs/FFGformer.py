import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.archs.Modules.FcaNet import MultiSpectralAttentionLayer




class MDTA(nn.Module):
    def __init__(self, channels):
        super(MDTA, self).__init__()
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_conv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3, bias=False)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.patch_size = 8

    def low_frequency_fliter(self, x, alpha: float = 0.5, cutoff: float = 0.1,):
        """
        Suppress low frequencies (center region) in the frequency domain to reduce haze
        and grayish veiling in underwater images.

        Args:
            x: [B, C, H, W]
            alpha: Retention factor for the lowest frequencies in the center (0~1).
                   Smaller values apply stronger suppression (e.g., 0.5 halves the center energy).
            cutoff: Controls the spatial extent of low-frequency suppression (0~1),
                    interpreted as a normalized radius. Larger values suppress a wider area
                    around the center.
        """

        device = x.device
        dtype = x.dtype
        _,_, H, W = x.size()
        # Use normalized coordinates with the center as origin,
        # radius ranging from 0 to ~sqrt(2)
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype),
            indexing="ij",
        )
        r = torch.sqrt(xx ** 2 + yy ** 2)  # [H, W], 0 at the center, ~sqrt(2) at the corners

        # Gaussian decay: when r = 0, mask ≈ alpha; when r >> cutoff, mask ≈ 1
        # exp(-(r^2)/(2*cutoff^2)) is 1 at r = 0 and approaches 0 as r increases
        cutoff = max(cutoff, 1e-6)  # Avoid division by zero
        low_weight = torch.exp(-(r ** 2) / (2 * (cutoff ** 2)))  # [H,W] in (0,1]
        mask = 1.0 - (1.0 - alpha) * low_weight  # [H,W]

        mask = mask.unsqueeze(0).unsqueeze(0) #[B,C,H,W]

        return mask

    def forward1(self, x):

        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1) #[B,C,H,W]
        q_fft = torch.fft.fft2(q.float())
        k_fft = torch.fft.fft2(k.float())
        out = q_fft * k_fft

        out = torch.fft.fftshift(out, dim=(-2, -1))
        scores_fft = out * self.low_frequency_fliter(out)

        out = torch.fft.ifftshift(scores_fft, dim=(-2, -1))
        out = torch.fft.ifft2(out).real

        out = torch.softmax(out, dim=-1)

        output = self.project_out(torch.matmul(out, v))

        return output

    def forward(self, x):
        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)  # [B,C,H,W]
        d_k = q.size(-1)


        q = torch.fft.fft2(q, dim=(-2, -1), norm='ortho')  #spatial->frequency
        k = torch.fft.fft2(k, dim=(-2, -1), norm='ortho')  #spatial->frequency

        scores = torch.matmul(q, k.conj().transpose(-2, -1).contiguous()) / math.sqrt(d_k)

        scores_fft = torch.fft.fftshift(scores, dim=(-2, -1)) #permutation, for filtering

        scores_fft = scores_fft * self.low_frequency_fliter(scores_fft)

        scores_fft = torch.fft.ifftshift(scores_fft, dim=(-2, -1)) #repermutation
        scores_fft = torch.fft.ifft2(scores_fft, dim=(-2, -1), norm='ortho') #frequency->spatial
        scores = torch.real(scores_fft)   #Spatial domain data (real images are taken as .real)

        att_map = torch.softmax(scores, dim=-1)

        out = self.project_out(torch.matmul(att_map,v))

        return out




class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor):
        super(GDFN, self).__init__()

        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1,
                              groups=hidden_channels * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, channels, expansion_factor):
        super(TransformerBlock, self).__init__()

        self.norm1 = nn.LayerNorm(channels)
        self.attn = MDTA(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = GDFN(channels, expansion_factor)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x + self.attn(self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                          .contiguous().reshape(b, c, h, w))
        x = x + self.ffn(self.norm2(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                         .contiguous().reshape(b, c, h, w))
        return x


class DownSample(nn.Module):
    def __init__(self, channels):
        super(DownSample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class UpSample(nn.Module):
    def __init__(self, channels):
        super(UpSample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)



class FFGformer(nn.Module):
    def __init__(self, num_blocks=[2, 3, 3, 4], channels=[16, 32, 64, 128], num_refinement=4,
                 expansion_factor=2.66, ch=[64, 32, 16], size=[64, 128, 256]):
        super(FFGformer, self).__init__()
        self.attention = nn.ModuleList([MultiSpectralAttentionLayer(num_ch, dct_size, dct_size) for num_ch, dct_size in zip(ch,size)])

        self.embed_conv_rgb = nn.Conv2d(3, channels[0], kernel_size=3, padding=1, bias=False)

        self.encoders = nn.ModuleList(
            [nn.Sequential(*[TransformerBlock(num_ch, expansion_factor) for _ in range(num_tb)]) for
             num_tb, num_ch in
             zip(num_blocks, channels)])
        # the number of down sample or up sample == the number of encoder - 1
        self.downs = nn.ModuleList([DownSample(num_ch) for num_ch in channels[:-1]])
        self.ups = nn.ModuleList([UpSample(num_ch) for num_ch in list(reversed(channels))[:-1]])
        # the number of reduce block == the number of decoder - 1
        self.reduces = nn.ModuleList([nn.Conv2d(channels[i], channels[i - 1], kernel_size=1, bias=False)
                                      for i in reversed(range(2, len(channels)))])
        # the number of decoder == the number of encoder - 1
        self.decoders = nn.ModuleList([nn.Sequential(*[TransformerBlock(channels[2], expansion_factor)
                                                       for _ in range(num_blocks[2])])])
        self.decoders.append(nn.Sequential(*[TransformerBlock(channels[1], expansion_factor)
                                             for _ in range(num_blocks[1])]))
        # the channel of last one is not change
        self.decoders.append(nn.Sequential(
            *[TransformerBlock(channels[1], expansion_factor) for _ in range(num_blocks[0])]))

        self.refinement = nn.Sequential(*[TransformerBlock(channels[1], expansion_factor)
                                          for _ in range(num_refinement)])
        self.output = nn.Conv2d(8, 3, kernel_size=3, padding=1, bias=False)

        self.outputl = nn.Conv2d(32, 8, kernel_size=3, padding=1, bias=False)


    def forward(self, RGB_input):
        fo_rgb = self.embed_conv_rgb(RGB_input) #(16,256,256)
        out_enc_rgb1 = self.encoders[0](fo_rgb) #(16,256,256)
        out_enc_rgb2 = self.encoders[1](self.downs[0](out_enc_rgb1))#(32,128,128)
        out_enc_rgb3 = self.encoders[2](self.downs[1](out_enc_rgb2))#(64,64,64)
        fea_light = self.encoders[3](self.downs[2](out_enc_rgb3))#(128,32,32)



        out_dec3 = self.decoders[0](
            self.reduces[0](torch.cat([self.ups[0](fea_light), self.attention[0](out_enc_rgb3)], dim=1)))

        out_dec2 = self.decoders[1](
            self.reduces[1](torch.cat([self.ups[1](out_dec3), self.attention[1](out_enc_rgb2)], dim=1)))

        fd = self.decoders[2](torch.cat([self.ups[2](out_dec2), self.attention[2](out_enc_rgb1)], dim=1))
        fr = self.refinement(fd)

        return self.output(self.outputl(fr))




