import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super(ConvBlock, self).__init__()
        self.main = nn.Sequential(nn.ReflectionPad2d(padding),
                                  nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0, bias=bias))
        self.norm = nn.InstanceNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        out = self.main(x)
        out = self.norm(out)
        out = self.activation(out)
        return out

class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, n_contents=8):
        super(SPADEResnetBlock, self).__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        self.norm_0 = SPADE(fin, n_contents)
        self.norm_1 = SPADE(fmiddle, n_contents)
        if self.learned_shortcut:
            self.norm_s = SPADE(fin, n_contents)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)
        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))
        out = x_s + dx
        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)

class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super(SPADE, self).__init__()

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 64
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, segmap):
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)
        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out

class EncoderContent(nn.Module):
    def __init__(self, in_ch, n_classes, n_contents, encoder=[64, 64, 128, 256], decoder=[256, 128, 64, 64]):
        super(EncoderContent, self).__init__()
        self.enc_nf, self.dec_nf = encoder, decoder
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.init = ConvBlock(in_ch, 32, kernel_size=5, stride=1, padding=2)
        # configure encoder
        prev_nf = 32 # initial number of channels
        self.downarm = nn.ModuleList()

        for nf in self.enc_nf:
            self.downarm.append(ConvBlock(prev_nf, nf, kernel_size=5, stride=2, padding=2))
            prev_nf = nf

        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()

        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(ConvBlock(channels, nf, kernel_size=3, stride=1, padding=1))
            prev_nf = nf

        prev_nf += 32
        # final layers
        self.final = ConvBlock(prev_nf, 32, kernel_size=3, stride=1, padding=1)
        self.atten_layer = nn.Conv2d(32, n_contents, kernel_size=3, stride=1, padding=1)
        self.logit_layer = nn.Conv2d(n_contents, n_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # get encoder activations
        x_enc = [self.init(x)]
        for layer in self.downarm:
            x_enc.append(layer(x_enc[-1]))

         # conv, upsample, concatenate series
        x = x_enc.pop()
        for layer in self.uparm:
            x = layer(x)
            x = self.upsample(x)
            x = torch.cat([x, x_enc.pop()], dim=1)
        
        x = self.final(x)
        contents = self.atten_layer(x)
        logits = self.logit_layer(contents)
        return contents, logits

class EncoderStyle(nn.Module):
    def __init__(self, in_ch, out_ch=256):
        super(EncoderStyle, self).__init__()
        ndf = 64
        n_blocks=4
        max_ndf = 4
        conv_layers = [ConvBlock(in_ch, ndf, kernel_size=4, stride=2, padding=1)]

        for n in range(1, n_blocks):
            input_ndf = ndf * min(max_ndf, n)  # 2**(n-1)
            output_ndf = ndf * min(max_ndf, n+1)  # 2**n
            conv_layers += [ConvBlock(input_ndf, output_ndf, kernel_size=3, stride=1, padding=1)]
        conv_layers += [nn.AdaptiveAvgPool2d(1)] 
        self.fc_mu = nn.Sequential(*[nn.Linear(output_ndf, out_ch)])
        self.fc_var = nn.Sequential(*[nn.Linear(output_ndf, out_ch)])
        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

class SPADEGenerator(nn.Module):
    def __init__(self, z_dim=256, n_contents=8, nf=32, n_spade_layers=3):
        super(SPADEGenerator, self).__init__()
        self.nf = nf
        self.n_spade_layers = n_spade_layers
        self.size = z_dim // (2**n_spade_layers)
        self.fc = nn.Linear(z_dim, 8 * nf * self.size * self.size)
        self.spade_layers = nn.ModuleList()
        for i in range(n_spade_layers):
            ch = 2 ** (n_spade_layers - i)
            self.spade_layers.append(SPADEResnetBlock(nf * ch, int(nf*ch//2), n_contents))
        final_nc = nf
        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, z, inputs):
        content = inputs
        x = self.fc(z)
        x = x.view(-1, (2**self.n_spade_layers) * self.nf, self.size, self.size)
        for i in range(self.n_spade_layers):
            x = self.spade_layers[i](x, content)
            x = self.up(x)
            
        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = torch.tanh(x)
        return x

class Discriminator(nn.Module):
    """
    70 x 70 PatchGAN
    """
    def __init__(self, nch_input, nfilters=64, nlayers=3):
        super(Discriminator, self).__init__()
        model = [nn.Conv2d(nch_input, nfilters, kernel_size=4, stride=2, padding=2),
                 nn.LeakyReLU(0.2, True)]
        fl = nfilters
        for n in range(1, nlayers):
            fl = 2 * fl
            stride = 2 if n < 2 else 1
            model += [ConvBlock(fl//2, fl, 4, stride, 2)]
        model += [nn.Conv2d(fl, 1, kernel_size=1, stride=1, padding=2)]
        self.model = nn.Sequential(*model)
    
    def forward(self, img):    
        x = self.model(img)
        return x

class EncoderContentPaired(nn.Module):
    def __init__(self, in_ch, n_contents, encoder=[64, 64, 128, 256], decoder=[256, 128, 64, 64]):
        super(EncoderContentPaired, self).__init__()
        self.enc_nf, self.dec_nf = encoder, decoder
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.init_a = ConvBlock(in_ch, 32, kernel_size=5, stride=1, padding=2)
        # configure encoder
        prev_nf = 32 # initial number of channels
        self.downarm_a = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm_a.append(ConvBlock(prev_nf, nf, kernel_size=5, stride=2, padding=2))
            prev_nf = nf
        enc_history = list(reversed(self.enc_nf))
        self.uparm_a = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm_a.append(ConvBlock(channels, nf, kernel_size=3, stride=1, padding=1))
            prev_nf = nf

        self.init_b = ConvBlock(in_ch, 32, kernel_size=5, stride=1, padding=2)
        # configure encoder
        prev_nf = 32 # initial number of channels
        self.downarm_b = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm_b.append(ConvBlock(prev_nf, nf, kernel_size=5, stride=2, padding=2))
            prev_nf = nf
        enc_history = list(reversed(self.enc_nf))
        self.uparm_b = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm_b.append(ConvBlock(channels, nf, kernel_size=3, stride=1, padding=1))
            prev_nf = nf

        prev_nf += 32
        # share layers
        enc_share = []
        enc_share.append(ConvBlock(prev_nf, 32, kernel_size=3, stride=1, padding=1))
        enc_share.append(nn.Conv2d(32, n_contents, kernel_size=3, stride=1, padding=1))
        self.enc_share = nn.Sequential(*enc_share)

    def forward(self, a, b):
        # get encoder activations
        a_enc = [self.init_a(a)]
        for layer in self.downarm_a:
            a_enc.append(layer(a_enc[-1]))
         # conv, upsample, concatenate series
        a = a_enc.pop()
        for layer in self.uparm_a:
            a = layer(a)
            a = self.upsample(a)
            a = torch.cat([a, a_enc.pop()], dim=1)

        b_enc = [self.init_b(b)]
        for layer in self.downarm_b:
            b_enc.append(layer(b_enc[-1]))
         # conv, upsample, concatenate series
        b = b_enc.pop()
        for layer in self.uparm_b:
            b = layer(b)
            b = self.upsample(b)
            b = torch.cat([b, b_enc.pop()], dim=1)

        a = self.enc_share(a)
        b = self.enc_share(b)

        return a, b

class EncoderStylePaired(nn.Module):
    def __init__(self, in_ch, out_ch=256):
        super(EncoderStylePaired, self).__init__()
        ndf = 64
        n_blocks=4
        max_ndf = 4
        conv_layers_a = [ConvBlock(in_ch, ndf, kernel_size=4, stride=2, padding=1)]
        for n in range(1, n_blocks):
            input_ndf = ndf * min(max_ndf, n)  # 2**(n-1)
            output_ndf = ndf * min(max_ndf, n+1)  # 2**n
            conv_layers_a += [ConvBlock(input_ndf, output_ndf, kernel_size=3, stride=1, padding=1)]
        conv_layers_a += [nn.AdaptiveAvgPool2d(1)] 
        self.fc_mu_a = nn.Sequential(*[nn.Linear(output_ndf, out_ch)])
        self.fc_var_a = nn.Sequential(*[nn.Linear(output_ndf, out_ch)])
        self.conv_a = nn.Sequential(*conv_layers_a)

        conv_layers_b = [ConvBlock(in_ch, ndf, kernel_size=4, stride=2, padding=1)]
        for n in range(1, n_blocks):
            input_ndf = ndf * min(max_ndf, n)  # 2**(n-1)
            output_ndf = ndf * min(max_ndf, n+1)  # 2**n
            conv_layers_b += [ConvBlock(input_ndf, output_ndf, kernel_size=3, stride=1, padding=1)]
        conv_layers_b += [nn.AdaptiveAvgPool2d(1)] 
        self.fc_mu_b = nn.Sequential(*[nn.Linear(output_ndf, out_ch)])
        self.fc_var_b = nn.Sequential(*[nn.Linear(output_ndf, out_ch)])
        self.conv_b = nn.Sequential(*conv_layers_b)

    def forward(self, a, b):
        a = self.conv_a(a)
        a = a.view(a.size(0), -1)
        mu_a = self.fc_mu_a(a)
        logvar_a = self.fc_var_a(a)
        z_a = self.reparameterize(mu_a, logvar_a)

        b = self.conv_b(b)
        b = b.view(b.size(0), -1)
        mu_b = self.fc_mu_b(b)
        logvar_b = self.fc_var_b(b)
        z_b = self.reparameterize(mu_b, logvar_b)
        return z_a, mu_a, logvar_a, z_b, mu_b, logvar_b
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

def calc_vector_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    feat_var = feat.var(dim=1) + eps
    feat_std = feat_var.sqrt()
    feat_mean = feat.mean(dim=1)
    return feat_mean, feat_std

def calc_tensor_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat, style_feat):
    size = content_feat.size()
    style_mean, style_std = calc_vector_mean_std(style_feat)
    content_mean, content_std = calc_tensor_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.view(style_std.shape[0],1,1,1).expand(size) + style_mean.view(style_mean.shape[0],1,1,1).expand(size)

class AdaINDecoder(nn.Module):
    def __init__(self, anatomy_out_channels):
        super(AdaINDecoder, self).__init__()
        self.anatomy_out_channels = anatomy_out_channels
        self.conv1 = ConvBlock(self.anatomy_out_channels, 128, 3, 1, 1)
        self.conv2 = ConvBlock(128, 64, 3, 1, 1)
        self.conv3 = ConvBlock(64, 32, 3, 1, 1)
        self.conv4 = nn.Conv2d(32, 3, 3, 1, 1)

        nn.init.xavier_normal_(self.conv4.weight.data)
        self.conv4.bias.data.zero_()

    def forward(self, a, z):
        out = adaptive_instance_normalization(a, z)
        out = self.conv1(out)
        out = adaptive_instance_normalization(out, z)
        out = self.conv2(out)
        out = adaptive_instance_normalization(out, z)
        out = self.conv3(out)
        out = adaptive_instance_normalization(out, z)
        out = torch.tanh(self.conv4(out))
        return out

class Round(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

class ResnetBlock(nn.Module):
    """Resnet block"""
    def __init__(self, in_ch, out_ch):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(in_ch, out_ch)
        self.learned_shortcut = (in_ch != out_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        
    def build_conv_block(self, in_ch, dim):
        # padd input
        conv_block = [nn.ReflectionPad2d(1)]
        # add convolutional layer followed by normalization and ReLU
        conv_block += [nn.Conv2d(in_ch, dim, kernel_size=3, padding=0), 
                       nn.InstanceNorm2d(dim), 
                       nn.LeakyReLU(0.2, True),
                       nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, padding=0),
                       nn.InstanceNorm2d(dim)]
        return nn.Sequential(*conv_block)

    def shortcut(self, x):
        x_s = self.conv(x) if self.learned_shortcut else x
        return x_s
    
    def forward(self, x):
        """Forward pass (with skip connections)"""
        x_s = self.shortcut(x)
        out = x_s + self.conv_block(x)  # add skip connections
        return F.leaky_relu(out)

class Bottleneck(nn.Module):
    def __init__(self, filters=32, n_block=4, depth=4, kernel_size=(3,3)):
        super(Bottleneck, self).__init__()
        out_ch = filters * 2 ** n_block
        in_ch = filters * 2 ** (n_block - 1)
        self.model = nn.ModuleList()
        for i in range(depth):
            dilate = 2 ** i
            self.model.append(nn.Sequential(*[nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, padding=dilate,
                          dilation=dilate), nn.InstanceNorm2d(out_ch), nn.LeakyReLU(inplace=True)]))
            if i == 0:
                in_ch = out_ch

    def forward(self, x):
        bottleneck_output = 0
        output = x
        for layer in self.model:
            output = layer(output)
            bottleneck_output += output
        return bottleneck_output

class Up(nn.Module):
    def __init__(self, filters=32, n_block=4, kernel_size=(3, 3), batch_norm=True, padding='same', drop=False):
        super(Up, self).__init__()
        self.n_block = n_block
        self.up = nn.UpsamplingNearest2d(scale_factor=2)
        self.model = nn.ModuleList()
        for i in reversed(range(n_block)):
            out_ch = filters * 2 ** i
            in_ch = 3 * out_ch
            model = [ConvBlock(in_ch, out_ch)] + [ConvBlock(out_ch, out_ch)]
            self.model.append(nn.Sequential(*model))

    def forward(self, x, skip):
        output = x
        for layer in self.model:
            output = self.up(output)
            output = torch.cat([skip.pop(), output], 1)
            output = layer(output)
        return output

class Down(nn.Module):
    def __init__(self, in_ch, ndf=32):
        super(Down, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.down_num = 4
        self.pool = nn.MaxPool2d(2)
        self.down = nn.ModuleList()
        for i in range(self.down_num):
            in_ch = in_ch if i == 0 else ndf * 2 ** (i - 1)
            out_ch = ndf * 2 ** i
            self.down.append(ResnetBlock(in_ch, out_ch))
    
    def forward(self, x):
        skip = []
        for i in range(self.down_num):
            x = self.down[i](x)
            skip.append(x)
            x = self.pool(x)
        return x, skip

class EncoderShared(nn.Module):
    def __init__(self, in_ch):
        super(EncoderShared, self).__init__()
        self.down = Down(in_ch)
        self.bottleneck = Bottleneck()
    
    def forward(self, x):
        x, skip = self.down(x)
        x = self.bottleneck(x)
        return x, skip

class EncoderAnatomy(nn.Module):
    def __init__(self, n_contents):
        super(EncoderAnatomy, self).__init__()
        self.up = Up()
        self.content_layer = nn.Conv2d(32, n_contents, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x, skip):
        x = self.up(x, skip)
        x = self.content_layer(x)
        return F.softmax(x, dim=1)

class Segmenter(nn.Module):
    def __init__(self, n_contents, n_classes):
        super(Segmenter, self).__init__()
        self.seg = nn.Sequential(*[
            ConvBlock(n_contents, 64), ConvBlock(64, 64), ConvBlock(64, n_classes)
        ])
    
    def forward(self, x):
        x = self.seg(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_ch, n_contents, n_classes):
        super(UNet, self).__init__()
        self.down = Down(in_ch)
        self.bottleneck = Bottleneck()
        self.up = Up()
        self.content_layer = nn.Conv2d(32, n_contents, kernel_size=3, stride=1, padding=1)
        self.seg = nn.Sequential(*[
            ConvBlock(n_contents, 64), ConvBlock(64, 64), ConvBlock(64, n_classes)
        ])
    
    def forward(self, x):
        x, skip = self.down(x)
        x = self.bottleneck(x)
        x = self.up(x, skip)
        contents = self.content_layer(x)
        seg = self.seg(contents)
        return F.softmax(contents, dim=1), seg


if __name__ == '__main__':
    x = torch.randn(4, 3, 256, 256)
    E_content = EncoderContent(3, 5, 8)
    E_style = EncoderStyle(3, 256)

    contents, logits = E_content(x)
    print(contents.shape)
    print(logits.shape)

    z, mu, logvar = E_style(x)
    print(z.shape)
    print(mu.shape)
    print(logvar.shape)

    decoder = SPADEGenerator()
    rec_x = decoder(z, contents)
    print(rec_x.shape)


