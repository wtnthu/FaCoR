import torch, pdb
from inference import load_pretrained_model, to_input


# from FAC.kernelconv2d import KernelConv2D


def l2_norm(input,axis=1):
    # pdb.set_trace()
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.encoder=KitModel("./kit_resnet101.pkl")

        self.projection=nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight -0.05, 0.05)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, imgs, aug=False):
        img1,img2=imgs
        embeding1, _= self.encoder(img1)
        embeding2, _= self.encoder(img2)

        embeding11 = l2_norm(embeding1)
        embeding22 = l2_norm(embeding2)

        pro1 ,pro2=self.projection(embeding1),self.projection(embeding2)

        return embeding11,embeding22,pro1,pro2, pro2


class Net_ada3(torch.nn.Module):
    def __init__(self):
        super(Net_ada3, self).__init__()
        # self.encoder=KitModel("./kit_resnet101.pkl")
        self.Ada_model = load_pretrained_model('ir_101')
        #'''
        self.projection=nn.Sequential(
            torch.nn.Linear(512*6, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
        )
        #'''
        # self.projection = torch.nn.Linear(1024, 1)
        self.channel=64
        CA = True
        # self.enhance_block = FSFNet(self.channel*8//4, CA)
        self.enhance_block1 = FSFNet2(self.channel*8, CA)
        self.enhance_block2 = CAM_Module(self.channel*8)
        self.CCA=CALayer2(1024)
        self.CCA2=CALayer(1024)
        self.out = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.CA1 = CAM_Module(1536)



        self._initialize_weights()
        # self._initialize_weights2()

    def _initialize_weights(self):
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight -0.05, 0.05)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _initialize_weights2(self):
        for m in self.enhance_block1.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight -0.05, 0.05)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.enhance_block2.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight -0.05, 0.05)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, imgs, aug=False):
        img1,img2=imgs
        # embeding1 ,embeding2= self.encoder(img1),self.encoder(img2)
        # f1_0, x1_feat = self.encoder(img1)
        # f2_0, x2_feat = self.encoder(img2)
        img1 = img1#((img1/255.0) - 0.5)/0.5
        img2 = img2#(img2/255.0 - 0.5)/0.5
        idx=[2,1,0]
        #with torch.no_grad():
        f1_0, x1_feat = self.Ada_model(img1[:,idx])
        f2_0, x2_feat = self.Ada_model(img2[:,idx])

        _, _ ,att_map0 = self.enhance_block1(x1_feat, x2_feat)

        f1_0 = l2_norm(f1_0)
        f2_0 = l2_norm(f2_0)

        x1_feat = l2_norm(x1_feat)
        x2_feat = l2_norm(x2_feat)


        f1_1, f2_1 ,att_map1 = self.enhance_block2(f1_0, f2_0)
        f1_2, f2_2 ,att_map2 = self.enhance_block1(x1_feat, x2_feat)


        
        f1_2 = torch.flatten(self.avg_pool(f1_2),1)
        f2_2 = torch.flatten(self.avg_pool(f2_2),1)


        wC = self.CCA(torch.cat([f1_1, f1_2],1).unsqueeze(2).unsqueeze(3))
        wC = wC.view(-1,2,512)[:,:,:,None,None]
        f1s = f1_1.unsqueeze(2).unsqueeze(3) * wC[:,0] + f1_2.unsqueeze(2).unsqueeze(3)* wC[:,1]

        wC2 = self.CCA(torch.cat([f2_1, f2_2],1).unsqueeze(2).unsqueeze(3))
        wC2 = wC2.view(-1,2,512)[:,:,:,None,None]
        f2s = f2_1.unsqueeze(2).unsqueeze(3) * wC2[:,0] + f2_2.unsqueeze(2).unsqueeze(3)* wC2[:,1]

        f1s = torch.flatten(f1s,1)
        f2s = torch.flatten(f2s,1)


        return f1s,f2s,f1s,f2s, att_map0


class FSFNet2(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim, CA=False):
        super(FSFNet2,self).__init__()
        self.chanel_in = in_dim
        reduction_ratio=16
        self.query_conv = nn.Conv2d(in_channels = in_dim*2 , out_channels = in_dim//reduction_ratio , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim*2 , out_channels = in_dim//reduction_ratio , kernel_size= 1)
        # self.value_conv = nn.Conv2d(in_channels = in_dim*2 , out_channels = in_dim*2 , kernel_size= 1)
        self.value_conv1 = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.value_conv2 = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.CA = CALayer(in_dim*2)
        self.gamma = nn.Parameter(torch.zeros(1))
        #pdb.set_trace()
        self.UseCA = CA

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self, x1, x2):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        # pdb.set_trace()
        m_batchsize,C,width ,height = x1.size()
        # pdb.set_trace()
        x = torch.cat([x1, x2],1)
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        # proj_value = self.value_conv(x)
        proj_value1 = self.value_conv1(x1)
        proj_value2 = self.value_conv2(x2)
        # proj_value = torch.cat([proj_value1, proj_value2],1)
        #pdb.set_trace()
        # if self.UseCA==True:
        #     #pdb.set_trace()
        #     proj_value = self.CA(proj_value)
            #Ch_w=Ch_w.view(-1,2,self.chanel_in)[:,:,:,None,None]
            #Ca_feat = proj_value[:, 0:self.chanel_in] * Ch_w[:,0] + proj_value[:, self.chanel_in:2*self.chanel_in] * Ch_w[:,1]
            #proj_value = proj_value.view(m_batchsize,-1,width*height) # B X C X N

        proj_value1 = proj_value1.view(m_batchsize,-1,width*height) # B X C X N
        proj_value2 = proj_value2.view(m_batchsize,-1,width*height) # B X C X N

        out1 = torch.bmm(proj_value1,attention.permute(0,2,1) )
        out2 = torch.bmm(proj_value2,attention.permute(0,2,1) )
        # out = out.view(m_batchsize,2*C,width,height)
        #pdb.set_trace()
        #out = self.gamma*out.view(m_batchsize,-1,width,height) + x.view(m_batchsize,-1,width,height)
        # pdb.set_trace()
        out1 = out1.view(m_batchsize,-1,width,height) + x1.view(m_batchsize,-1,width,height)
        out2 = out2.view(m_batchsize,-1,width,height) + x2.view(m_batchsize,-1,width,height)
        # if self.UseCA==True:
        #     #pdb.set_trace()
        #     out = self.CA(out)
        #out = self.gamma*out + x.view(m_batchsize,2*C,width,height)
        return out1, out2, attention

class FSFNet_fac(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim, CA=False):
        super(FSFNet_fac,self).__init__()
        self.chanel_in = in_dim
        reduction_ratio=16
        # self.query_conv = nn.Conv2d(in_channels = in_dim*2 , out_channels = in_dim//reduction_ratio , kernel_size= 1)
        # self.key_conv = nn.Conv2d(in_channels = in_dim*2 , out_channels = in_dim//reduction_ratio , kernel_size= 1)
        # self.value_conv = nn.Conv2d(in_channels = in_dim*2 , out_channels = in_dim*2 , kernel_size= 1)
        self.value_conv1 = nn.Conv2d(in_channels = in_dim , out_channels = in_dim*4 , kernel_size= 1)
        self.value_conv2 = nn.Conv2d(in_channels = in_dim , out_channels = in_dim*4 , kernel_size= 1)
        self.CA = CALayer(in_dim*2)
        self.gamma = nn.Parameter(torch.zeros(1))
        #pdb.set_trace()
        self.UseCA = CA

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self, x1, x2):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        # pdb.set_trace()
        x = torch.cat([x1, x2],1)
        m_batchsize,C,width ,height = x.size()
        
        # pdb.set_trace()
        proj_query  = x.view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  x.view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        # proj_value = self.value_conv(x)
        proj_value1 = self.value_conv1(x1)
        proj_value2 = self.value_conv2(x2)
        # proj_value = torch.cat([proj_value1, proj_value2],1)
        #pdb.set_trace()
        # if self.UseCA==True:
        #     #pdb.set_trace()
        #     proj_value = self.CA(proj_value)
            #Ch_w=Ch_w.view(-1,2,self.chanel_in)[:,:,:,None,None]
            #Ca_feat = proj_value[:, 0:self.chanel_in] * Ch_w[:,0] + proj_value[:, self.chanel_in:2*self.chanel_in] * Ch_w[:,1]
            #proj_value = proj_value.view(m_batchsize,-1,width*height) # B X C X N

        proj_value1 = proj_value1.view(m_batchsize,-1,width*height) # B X C X N
        proj_value2 = proj_value2.view(m_batchsize,-1,width*height) # B X C X N

        out1 = torch.bmm(proj_value1,attention.permute(0,2,1) )
        out2 = torch.bmm(proj_value2,attention.permute(0,2,1) )
        # out = out.view(m_batchsize,2*C,width,height)
        #pdb.set_trace()
        #out = self.gamma*out.view(m_batchsize,-1,width,height) + x.view(m_batchsize,-1,width,height)
        # pdb.set_trace()
        out1 = out1.view(m_batchsize,-1,width,height)# + x1.view(m_batchsize,-1,width,height)
        out2 = out2.view(m_batchsize,-1,width,height)# + x2.view(m_batchsize,-1,width,height)
        # if self.UseCA==True:
        #     #pdb.set_trace()
        #     out = self.CA(out)
        #out = self.gamma*out + x.view(m_batchsize,2*C,width,height)
        return out1, out2, attention



class FSFNet(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim, CA=False):
        super(FSFNet,self).__init__()
        self.chanel_in = in_dim
        reduction_ratio=8
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//reduction_ratio , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//reduction_ratio , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim*2 , out_channels = in_dim*2 , kernel_size= 1)
        self.CA = CALayer(in_dim*2)
        self.gamma = nn.Parameter(torch.zeros(1))
        #pdb.set_trace()
        self.UseCA = CA

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self, x1, x2):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        # pdb.set_trace()
        m_batchsize,C,width ,height = x1.size()
        # pdb.set_trace()
        proj_query  = self.query_conv(x1).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x2).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        x = torch.cat([x1, x2],1)
        proj_value = self.value_conv(x)
        #pdb.set_trace()
        if self.UseCA==True:
            #pdb.set_trace()
            proj_value = self.CA(proj_value)
            #Ch_w=Ch_w.view(-1,2,self.chanel_in)[:,:,:,None,None]
            #Ca_feat = proj_value[:, 0:self.chanel_in] * Ch_w[:,0] + proj_value[:, self.chanel_in:2*self.chanel_in] * Ch_w[:,1]
            #proj_value = proj_value.view(m_batchsize,-1,width*height) # B X C X N

        proj_value = proj_value.view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        # out = out.view(m_batchsize,2*C,width,height)
        #pdb.set_trace()
        out = self.gamma*out.view(m_batchsize,-1,width,height) + x.view(m_batchsize,-1,width,height)
        # pdb.set_trace()
        #out = self.gamma*out + x.view(m_batchsize,2*C,width,height)
        return out,attention

class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.conv = nn.Conv2d(in_channels = in_dim*2 , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self, x1, x2):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        x1 = x1.unsqueeze(2).unsqueeze(3)
        x2 = x2.unsqueeze(2).unsqueeze(3)
        x = torch.cat([x1, x2], 1)
        x = self.conv(x)
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)

        proj_value1 = x1.view(m_batchsize, C, -1)
        proj_value2 = x2.view(m_batchsize, C, -1)

        out1 = torch.bmm(attention, proj_value1)
        out1 = out1.view(m_batchsize, C, height, width)

        out2 = torch.bmm(attention, proj_value2)
        out2 = out2.view(m_batchsize, C, height, width)

        out1 = out1 + x1
        out2 = out2 + x2
        # out = self.gamma*out + x
        return out1.reshape(m_batchsize, -1), out2.reshape(m_batchsize, -1), attention



class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x*y


class Local_CNN(nn.Module):
    def __init__(self):
        super(Local_CNN, self).__init__()

        channel_num=32
        self.layer = nn.Sequential(
            nn.Conv2d(3, channel_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel_num),
            nn.ReLU(),
            # nn.MaxPool2d(2),
            nn.Conv2d(channel_num, channel_num*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel_num*2),
            nn.ReLU(),
            # nn.MaxPool2d(2),
            nn.Conv2d(channel_num*2, channel_num*4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel_num*4),
            nn.ReLU(),
            # nn.MaxPool2d(2),
            nn.Conv2d(channel_num*4, channel_num*8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel_num*8),
            nn.ReLU(),
        )
        # self.fc1 = nn.Linear(16384, 2)
        # # self.dropout1 = nn.Dropout(0.3)
        # self.out = nn.Sigmoid()


    def forward(self, x):
        out = self.layer(x)
        return out


class CALayer2(nn.Module):
    def __init__(self, channel):
        super(CALayer2, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        # y = self.avg_pool(x)
        y=x
        # pdb.set_trace()
        y = self.ca(y)
        return y