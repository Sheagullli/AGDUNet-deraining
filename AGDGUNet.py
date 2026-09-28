## 主代码

import torch
import torch.nn as nn
import torch.nn.functional as F


################# 一、 Designing a simple net to replacing gradient operator#########################################################
# Basic modules
def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer


def default_conv(in_channels, out_channels, kernel_size, stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), stride=stride, bias=bias)


class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.PReLU(), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            if i == 0:
                m.append(conv(n_feats, 64, kernel_size, bias=bias))
            else:
                m.append(conv(64, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)

        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class Gradient_Net(nn.Module):
    def __init__(self):
        super(Gradient_Net, self).__init__()
        # Sobel算子
        self.ratio = 15
        kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0)
        kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0)
        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

    def forward(self, x):
        # 水平梯度
        grad_x_r = F.conv2d(x[:, 0, :, :].unsqueeze(1), self.weight_x / self.ratio, stride=1, padding=1)
        grad_x_g = F.conv2d(x[:, 1, :, :].unsqueeze(1), self.weight_x / self.ratio, stride=1, padding=1)
        grad_x_b = F.conv2d(x[:, 2, :, :].unsqueeze(1), self.weight_x / self.ratio, stride=1, padding=1)
        grad_x = torch.cat([grad_x_r, grad_x_g, grad_x_b], 1)
        # 垂直梯度
        grad_y_r = F.conv2d(x[:, 0, :, :].unsqueeze(1), self.weight_y / self.ratio, stride=1, padding=1)
        grad_y_g = F.conv2d(x[:, 1, :, :].unsqueeze(1), self.weight_y / self.ratio, stride=1, padding=1)
        grad_y_b = F.conv2d(x[:, 2, :, :].unsqueeze(1), self.weight_y / self.ratio, stride=1, padding=1)
        grad_y = torch.cat([grad_y_r, grad_y_g, grad_y_b], 1)

        return grad_x, grad_y  # , Px, Py


class Gradient_Net_transpose(nn.Module):
    def __init__(self):
        super(Gradient_Net_transpose, self).__init__()
        # Sobel算子
        self.ratio = 15
        kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0)
        kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0)
        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

    def forward(self, x, xORy):

        if xORy == 'x':
            # 水平梯度
            grad_x_r_T = F.conv_transpose2d(x[:, 0, :, :].unsqueeze(1), self.weight_x / self.ratio, stride=1, padding=1)
            grad_x_g_T = F.conv_transpose2d(x[:, 1, :, :].unsqueeze(1), self.weight_x / self.ratio, stride=1, padding=1)
            grad_x_b_T = F.conv_transpose2d(x[:, 2, :, :].unsqueeze(1), self.weight_x / self.ratio, stride=1, padding=1)
            # grad_x_r = F.conv2d(x[:, 0, :, :].unsqueeze(1), self.weight_x, stride=1, padding=1)
            # grad_x_g = F.conv2d(x[:, 1, :, :].unsqueeze(1), self.weight_x, stride=1, padding=1)
            # grad_x_b = F.conv2d(x[:, 2, :, :].unsqueeze(1), self.weight_x, stride=1, padding=1)
            grad_T = torch.cat([grad_x_r_T, grad_x_g_T, grad_x_b_T], 1)
        else:
            # 垂直梯度
            grad_y_r_T = F.conv_transpose2d(x[:, 0, :, :].unsqueeze(1), self.weight_y / self.ratio, stride=1, padding=1)
            grad_y_g_T = F.conv_transpose2d(x[:, 1, :, :].unsqueeze(1), self.weight_y / self.ratio, stride=1, padding=1)
            grad_y_b_T = F.conv_transpose2d(x[:, 2, :, :].unsqueeze(1), self.weight_y / self.ratio, stride=1, padding=1)
            grad_T = torch.cat([grad_y_r_T, grad_y_g_T, grad_y_b_T], 1)

        return grad_T


##########################################################################
class Basic_block(nn.Module):
    def __init__(self):
        super(Basic_block, self).__init__()
        self.gra = Gradient_Net().cuda()
        self.gra_T = Gradient_Net_transpose().cuda()
        self.unet = Unet(in_channels=35, num_features=35)
        self.rnet = Rnet(in_channels=3, num_features=32)

    def forward(self, img, U, gra_O_x, gra_O_y, Z_U, rR_1, rU_1, rR_2, rU_2):
        # img:rainy image, U: previous stage U, R: previous stage R , Z: auxiliary variable
        # Update Rain streaks R
        ER = img - U
        # gra_R_x, gra_R_y, gra_R_Px, gra_R_Py = self.gra(ER)
        # gra_U_x, gra_U_y, gra_U_Px, gra_U_Py = self.gra(U)
        gra_R_x, gra_R_y = self.gra(ER)
        gra_U_x, gra_U_y = self.gra(U)
        delat_r_x = self.gra_T(gra_R_x + gra_U_x - gra_O_x, 'x')
        delat_r_y = self.gra_T(gra_R_y + gra_U_y - gra_O_y, 'y')
        # x2_imgR = ER - self.rR_1 * (gra_R_Px * (gra_R_x + gra_U_x - gra_O_x)) - self.rR_2 * (gra_R_Py * (gra_R_y + gra_U_y - gra_O_y))
        x2_imgR = ER - rR_1 * delat_r_x - rR_2 * delat_r_y
        # x2_imgR = ER - self.rR_1 * (gra_R_Px * (gra_R_x + gra_U_x - gra_O_x)) - self.rR_2 * (gra_R_Py * (gra_R_y + gra_U_y - gra_O_y))
        # x2_imgR = torch.cat((x2_imgR, Z_R), dim=1)
        R = self.rnet(x2_imgR)

        # Update background U
        # gra_R_x, gra_R_y, gra_R_Px, gra_R_Py = self.gra(R)
        gra_R_x, gra_R_y = self.gra(R)
        delat_u_x = self.gra_T(gra_R_x + gra_U_x - gra_O_x, 'x')
        delat_u_y = self.gra_T(gra_R_y + gra_U_y - gra_O_y, 'y')
        x2_imgU = U - rU_1 * delat_u_x - rU_2 * delat_u_y
        # x2_imgU = U - self.rU_1 * (gra_U_Px * (gra_R_x + gra_U_x - gra_O_x)) - self.rU_2 * (gra_U_Py * (gra_R_y + gra_U_y - gra_O_y))
        input_dual = torch.cat((x2_imgU, Z_U), dim=1)
        out_dual = self.unet(input_dual)
        U = out_dual[:, :3, :, :]
        Z_U = out_dual[:, 3:, :, :]

        return R, U, Z_U  # , Z_R


##########################################################################
# proxNet_R
class Rnet(nn.Module):
    def __init__(self, in_channels, num_features):
        super(Rnet, self).__init__()
        self.in_channels = in_channels
        self.num_features = num_features
        self.out_channels = in_channels
        self.f = nn.ReLU()
        self.conv2d_t = nn.Conv2d(self.in_channels, self.num_features, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv2d = nn.Conv2d(self.num_features, self.out_channels, kernel_size=3, stride=1, padding=1, dilation=1)
        self.relu_mask = ResBlock(default_conv, 3, 3)
        self.tau = nn.Parameter(torch.Tensor([1]), requires_grad=True)  # for sparse input map
        # self.tau0 = torch.Tensor([0.5])
        # self.taum = self.tau0.unsqueeze(dim=0).unsqueeze(dim=0).unsqueeze(dim=0).expand(-1, num_features, -1, -1)
        # self.tau = nn.Parameter(self.taum, requires_grad=True)  # for sparse input map
        self.sigmoid = nn.Sigmoid()
        self.resm1 = nn.Sequential(
            nn.Conv2d(self.num_features, self.num_features, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.num_features),
            nn.ReLU(),
            nn.Conv2d(self.num_features, self.num_features, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.num_features),
        )
        self.resm2 = nn.Sequential(
            nn.Conv2d(self.num_features, self.num_features, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.num_features),
            nn.ReLU(),
            nn.Conv2d(self.num_features, self.num_features, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.num_features),
        )
        self.resm3 = nn.Sequential(
            nn.Conv2d(self.num_features, self.num_features, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.num_features),
            nn.ReLU(),
            nn.Conv2d(self.num_features, self.num_features, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.num_features),
        )
        self.resm4 = nn.Sequential(
            nn.Conv2d(self.num_features, self.num_features, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.num_features),
            nn.ReLU(),
            nn.Conv2d(self.num_features, self.num_features, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.num_features),
        )
        self.resm5 = nn.Sequential(
            nn.Conv2d(self.num_features, self.num_features, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.num_features),
            nn.ReLU(),
            nn.Conv2d(self.num_features, self.num_features, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.num_features),
        )

    def forward(self, input):
        input = self.conv2d_t(input)
        m1 = F.relu(input + self.resm1(input))
        m2 = F.relu(m1 + self.resm2(m1))
        m3 = F.relu(m2 + self.resm3(m2))
        m4 = F.relu(m3 + self.resm4(m3))
        m5 = self.conv2d(m4 + self.resm5(m4))
        # R = m5[:, :3, :, :]
        # Z_R = m5[:, 3:, :, :]
        # relu_mask = self.sigmoid(self.relu_mask(m5))
        # R = self.f(R - self.tau)

        # relu_mask = self.sigmoid(self.relu_mask(m5))
        m5 = self.f(m5 - self.tau)

        return m5  # , Z_R


##########################################################################
# proxNet_U
class Unet(nn.Module):
    def __init__(self, in_channels, num_features):
        super(Unet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_features = num_features

        self.resx1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.num_features, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.num_features),
            nn.ReLU(),
            nn.Conv2d(self.num_features, self.num_features, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.num_features),
        )
        self.resx2 = nn.Sequential(
            nn.Conv2d(self.num_features, self.num_features, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.num_features),
            nn.ReLU(),
            nn.Conv2d(self.num_features, self.num_features, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.num_features),
        )
        self.resx3 = nn.Sequential(
            nn.Conv2d(self.num_features, self.num_features, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.num_features),
            nn.ReLU(),
            nn.Conv2d(self.num_features, self.num_features, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.num_features),
        )
        self.resx4 = nn.Sequential(
            nn.Conv2d(self.num_features, self.num_features, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.num_features),
            nn.ReLU(),
            nn.Conv2d(self.num_features, self.num_features, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.num_features),

        )

    def forward(self, input):
        x1 = F.relu(input + self.resx1(input))
        x2 = F.relu(x1 + self.resx2(x1))
        x3 = F.relu(x2 + self.resx3(x2))
        x4 = F.relu(x3 + self.resx4(x3))
        return x4


##########################################################################
# AGDUNet
class AGDUNet(nn.Module):
    def __init__(self, depth=8):
        super(AGDUNet, self).__init__()
        self.depth = depth
        self.step_R = 1
        self.step_U = 1
        # 每次迭代不共享参数
        basics = []
        for i in range(self.depth):
            basics.append(Basic_block().cuda())
        self.basics = nn.Sequential(*basics)
        # 每次迭代共享参数
        # self.basic = Basic_block()

        self.gra = Gradient_Net().cuda()
        self.gra_T = Gradient_Net_transpose().cuda()
        self.conv2d_U = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2d_R = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.unet0 = Unet(in_channels=35, num_features=35)
        self.unet1 = Unet(in_channels=35, num_features=35)
        self.rnet1 = Rnet(in_channels=3, num_features=32)
        self.unet_adjust = Unet(in_channels=35, num_features=35)
        self.rnet_adjust = Rnet(in_channels=3, num_features=32)

        self.rR_1 = nn.Parameter(torch.Tensor([self.step_R]), requires_grad=True)
        self.rR_1_S = self.make_eta(self.depth, torch.Tensor([self.step_R]))
        self.rR_2 = nn.Parameter(torch.Tensor([self.step_R]), requires_grad=True)
        self.rR_2_S = self.make_eta(self.depth, torch.Tensor([self.step_R]))

        self.rU_1 = nn.Parameter(torch.Tensor([self.step_U]), requires_grad=True)
        self.rU_1_S = self.make_eta(self.depth, torch.Tensor([self.step_U]))
        self.rU_2 = nn.Parameter(torch.Tensor([self.step_U]), requires_grad=True)
        self.rU_2_S = self.make_eta(self.depth, torch.Tensor([self.step_U]))

    def make_eta(self, iters, const):
        const_dimadd = const.unsqueeze(dim=0)
        const_f = const_dimadd.expand(iters, -1)
        eta = nn.Parameter(data=const_f, requires_grad=True)
        return eta

    def forward(self, img):
        ListU = []
        ListR = []

        # initialize U0 and Z0
        Z0_U = self.conv2d_U(img)
        input_ini0 = torch.cat((img, Z0_U), dim=1)
        out_dual0 = self.unet0(input_ini0)
        U0 = out_dual0[:, :3, :, :]
        Z1 = out_dual0[:, 3:, :, :]

        ##-------------------------------------------
        ##-------------- Stage 1---------------------
        ##-------------------------------------------
        # Updating U0-->R1
        ER = img - U0
        Z0_R = self.conv2d_R(ER)
        # gra_R_x, gra_R_y, gra_R_Px, gra_R_Py = self.gra(ER)  # R梯度
        # gra_U_x, gra_U_y, gra_U_Px, gra_U_Py = self.gra(U0)  # U梯度
        # gra_O_x, gra_O_y, gra_O_Px, gra_O_Py = self.gra(img)  # O梯度
        gra_R_x, gra_R_y = self.gra(ER)  # R梯度
        gra_U_x, gra_U_y = self.gra(U0)  # U梯度
        gra_O_x, gra_O_y = self.gra(img)  # O梯度
        delat_r_x = self.gra_T(gra_R_x + gra_U_x - gra_O_x, 'x')
        delat_r_y = self.gra_T(gra_R_y + gra_U_y - gra_O_y, 'y')
        # x2_imgR = ER - self.rR_1 * (gra_R_Px * (gra_R_x + gra_U_x - gra_O_x)) - self.rR_2 * (gra_R_Py * (gra_R_y + gra_U_y - gra_O_y))
        x2_imgR = ER - self.rR_1 * delat_r_x - self.rR_2 * delat_r_y
        # x2_imgR = torch.cat((x2_imgR, Z0_R), dim=1)
        # R1, Z_R = self.rnet1(x2_imgR)
        R1 = self.rnet1(x2_imgR)
        ListR.append(R1)
        # Updating R1-->B1
        # gra_R_x, gra_R_y, gra_R_Px, gra_R_Py = self.gra(R1)
        gra_R_x, gra_R_y = self.gra(R1)
        delat_u_x = self.gra_T(gra_R_x + gra_U_x - gra_O_x, 'x')
        delat_u_y = self.gra_T(gra_R_y + gra_U_y - gra_O_y, 'y')
        x2_imgU = U0 - self.rU_1 * delat_u_x - self.rU_2 * delat_u_y
        input_dual1 = torch.cat((x2_imgU, Z1), dim=1)
        out_dual11 = self.unet1(input_dual1)
        U1 = out_dual11[:, :3, :, :]
        Z_U = out_dual11[:, 3:, :, :]
        ListU.append(U1)

        ##-------------------------------------------
        ##-------------- Stage 2-6 ---------------------
        ##-------------------------------------------
        # 每次迭代共享参数
        # for i in range(self.depth):
        #     R1, U1, Z_U, Z_R = self.basic(img, U1, R1, gra_O_x, gra_O_y, gra_O_Px, gra_O_Py, Z_U, Z_R)
        #     ListR.append(R1)
        #     ListU.append(U1)
        # for i in range(self.depth):
        #     R1, U1, Z_U = self.basic(img, U1, R1, gra_O_x, gra_O_y, gra_O_Px, gra_O_Py, Z_U)
        #     ListR.append(R1)
        #     ListU.append(U1)

        # 每次迭代不共享参数
        # for i in range(self.depth):
        #     R1, U1, Z_U, Z_R = self.basics[i](img, U1, R1, gra_O_x, gra_O_y, gra_O_Px, gra_O_Py, Z_U, Z_R)
        #     ListR.append(R1)
        #     ListU.append(U1)
        for i in range(self.depth):
            R1, U1, Z_U = self.basics[i](img, U1, gra_O_x, gra_O_y, Z_U, self.rR_1_S[i], self.rU_1_S[i], self.rR_2_S[i], self.rU_2_S[i])
            ListR.append(R1)
            ListU.append(U1)

        # 最后调整
        # R1 = torch.cat((R1, Z_R), dim=1)
        # R_adjust, Z_R = self.rnet_adjust(R1)
        # R_adjust = self.rnet_adjust(R1)
        # ListU.append(img - R_adjust)
        # ListR.append(R_adjust)

        output_dual = torch.cat((U1, Z_U), dim=1)
        output = self.unet_adjust(output_dual)
        ListU.append(output[:, :3, :, :])
        ListR.append(img - output[:, :3, :, :])

        return U0, ListU, ListR
