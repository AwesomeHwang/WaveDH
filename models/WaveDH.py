import torch
import torch.nn as nn
from .BSRN_arch import BSConvU, CCALayer, ESDB
from .shufflemixer_arch import FMBlock
from .torch_wavelets import DWT_2D, IDWT_2D
from .deconv import FastDeconv
    
class WaveDownampler(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.dwt = DWT_2D(wave='haar')
        self.conv_lh = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels)
        self.conv_hl = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels)
        self.to_att = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels * 2, 1, 1, 0),
                    nn.Sigmoid()
        )
        self.pw = nn.Conv2d(in_channels * 4, in_channels * 2, 1, 1, 0)

    def forward(self, x):
        x = self.dwt(x)
        x_ll, x_lh, x_hl, x_hh = x.chunk(4, dim=1)
        # get attention
        lh =  self.conv_lh(x_ll + x_lh)
        hl =  self.conv_hl(x_ll + x_hl)
        att_map = self.to_att(lh + hl)
        # squeeze
        x_s = self.pw(x)
        o = torch.mul(x_s, att_map) + x_s
        hi_bands = torch.cat([x_lh, x_hl, x_hh], dim=1)
        return o, hi_bands

class FeatureInteract(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        self.cca = CCALayer(in_ch * 2)
        
    def forward(self, x_pix, x_idwt):
        x = torch.cat([x_pix, x_idwt], dim=1)
        x_o = self.cca(x)
        pix_up, o_idwt = x_o.chunk(2, dim=1)
        return pix_up + o_idwt

class FMBConv(nn.Module):
    def __init__(self, dim, conv_ratio=1.5, cg = 16):
        super().__init__()
        hidden_dim = int(dim * conv_ratio)
        group = int(hidden_dim / cg)
        
        self.conv = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, 1, 0),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=group),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0)
        )

    def forward(self, x):
        x = self.conv(x) + x
        return x

class WaveUpsampler(nn.Module):
    def __init__(self, pix_ch):
        super().__init__()

        self.idwt = IDWT_2D(wave='haar')      
        self.upsapling = nn.Sequential(
            nn.Conv2d(pix_ch, pix_ch * 4, 1, 1, 0),
            nn.PixelShuffle(2)
        )
        self.interact = FeatureInteract(pix_ch)
        self.fuse_conv = FMBConv(dim=pix_ch)

    def forward(self, x, hi_bands):
        x_1, x_2 = x.chunk(2, dim=1)
        pix_up = self.upsapling(x_1)
        o_idwt = self.idwt(torch.cat([x_2, hi_bands], dim=1))
        o = self.interact(pix_up, o_idwt)
        return self.fuse_conv(o)

class WaveBottleNeck(nn.Module):
    def __init__(self, in_ch=64, n_lo_block=2):
        super().__init__()
        
        self.dwt = DWT_2D(wave='haar')
        self.idwt = IDWT_2D(wave='haar')
        
        self.lo_blocks = nn.Sequential(
            *[FMBlock(dim=in_ch, kernel_size=7, mlp_ratio=1.25, conv_ratio=1.5) for _ in range(n_lo_block)]
        )
        self.esdb = ESDB(in_channels=in_ch, out_channels=in_ch, conv=BSConvU)

    def forward(self, x):
        x_w = self.dwt(x)
        x_ll, x_lh, x_hl, x_hh = x_w.chunk(4, dim=1)
        x_ll = self.lo_blocks(x_ll)
        
        x_out = self.idwt(torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1))
        out = self.esdb(x_out) + x
        return out
    
class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out

class WaveDH(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=32, use_dropout=False, n_lo_b=2, n_bottles=3):
        super(WaveDH, self).__init__()

        self.deconv = FastDeconv(3, 3, kernel_size=3, stride=1, padding=1)
        
        self.down1 = nn.Sequential(nn.ReflectionPad2d(3),
                                   nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                                   nn.SiLU(inplace=True))
        
        self.down2 = nn.Sequential(WaveBottleNeck(in_ch=ngf, n_lo_block=n_lo_b),
                                   WaveDownampler(ngf)
        )
        
        self.down3 = nn.Sequential(WaveBottleNeck(in_ch=ngf*2, n_lo_block=n_lo_b),
                                   WaveBottleNeck(in_ch=ngf*2, n_lo_block=n_lo_b),
                                   WaveDownampler(ngf*2)
        )

        self.bottleneck = nn.Sequential(
            *[WaveBottleNeck(in_ch=ngf*4, n_lo_block=n_lo_b) for _ in range(n_bottles)]
        )
        
        self.up1 = WaveUpsampler(pix_ch=ngf*2)
        self.up2 = WaveUpsampler(pix_ch=ngf)
        self.up3 = nn.Sequential(nn.ReflectionPad2d(3),
                                 nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)
                                )
        
        self.tanh = nn.Tanh()
        
        self.mix1 = Mix(m=-1)
        self.mix2 = Mix(m=-0.6)

    def forward(self, input):
        x_deconv = self.deconv(input) 

        x_down1 = self.down1(x_deconv) 
        x_down2, hi_bands_hr = self.down2(x_down1) 
        x_down3, hi_bands_lr = self.down3(x_down2) 
        
        x = self.bottleneck(x_down3)

        x_out_mix = self.mix1(x_down3, x)
        x_up1 = self.up1(x_out_mix, hi_bands_lr) 
        x_up1_mix = self.mix2(x_down2, x_up1)
        x_up2 = self.up2(x_up1_mix, hi_bands_hr) 
        out = self.up3(x_up2) 

        return self.tanh(out + x_deconv)