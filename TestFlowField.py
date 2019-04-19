import os
import torch
import argparse
from libs.FlowLoader import FlowDataset
from libs.Matrix import MulLayer
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
from libs.utils import print_options
from libs.models import encoder3,encoder4, encoder5
from libs.models import decoder3,decoder4, decoder5

parser = argparse.ArgumentParser()
parser.add_argument("--vgg_dir", default='models/vgg_r41.pth',
                    help='pre-trained encoder path')
parser.add_argument("--decoder_dir", default='models/dec_r41.pth',
                    help='pre-trained decoder path')
parser.add_argument("--matrixPath", default='models/r41.pth',
                    help='pre-trained model path')
parser.add_argument("--stylePath", default="data/style/",
                    help='path to style image')
parser.add_argument("--contentPath", default="data/content/",
                    help='path to frames')
parser.add_argument("--outf", default="ArtisticFlow/",
                    help='path to transferred images')
parser.add_argument("--batchSize", type=int,default=1,
                    help='batch size')
parser.add_argument('--loadSize', type=int, default=130,
                    help='scale image size')
parser.add_argument('--fineSize', type=int, default=130,
                    help='crop image size')
parser.add_argument("--layer", default="r41",
                    help='which features to transfer, either r31 or r41')

################# PREPARATIONS #################
opt = parser.parse_args()
opt.cuda = torch.cuda.is_available()
print_options(opt)

os.makedirs(opt.outf,exist_ok=True)
cudnn.benchmark = True

################# DATA #################
flow_dataset = FlowDataset(opt.contentPath,opt.loadSize,opt.fineSize,test=True)
flow_loader = torch.utils.data.DataLoader(dataset=flow_dataset,
                                             batch_size = opt.batchSize,
                                             shuffle = False,
                                             num_workers = 1)

################# MODEL #################
if(opt.layer == 'r31'):
    vgg = encoder3()
    dec = decoder3()
elif(opt.layer == 'r41'):
    vgg = encoder4()
    dec = decoder4()
matrix = MulLayer(opt.layer)
vgg.load_state_dict(torch.load(opt.vgg_dir))
dec.load_state_dict(torch.load(opt.decoder_dir))
matrix.load_state_dict(torch.load(opt.matrixPath))

################# GLOBAL VARIABLE #################
contentV = torch.Tensor(opt.batchSize,3,opt.fineSize,opt.fineSize)
styleV = torch.Tensor(opt.batchSize,3,opt.fineSize,opt.fineSize)

################# GPU  #################
if(opt.cuda):
    vgg.cuda()
    dec.cuda()
    matrix.cuda()
    contentV = contentV.cuda()
    styleV = styleV.cuda()

for vel, blur_vel, name in enumerate(flow_loader):
    name = name[0]
    contentV.resize_(blur_vel.size()).copy_(blur_vel)
    styleV.resize_(vel.size()).copy_(vel)

    # forward
    with torch.no_grad():
        sF = vgg(styleV)
        cF = vgg(contentV)

        if(opt.layer == 'r41'):
            feature,transmatrix = matrix(cF[opt.layer],sF[opt.layer])
        else:
            feature,transmatrix = matrix(cF,sF)
        transfer = dec(feature)

    transfer = transfer.clamp(0,1)
    vutils.save_image(transfer,'%s/%s.png'%(opt.outf,name),normalize=True,scale_each=True,nrow=opt.batchSize)
    print('Transferred image saved at %s%s.png'%(opt.outf,name))

'''
for ci,(content,contentName) in enumerate(content_loader):
    contentName = contentName[0]
    contentV.resize_(content.size()).copy_(content)
    for sj,(style,styleName) in enumerate(style_loader):
        styleName = styleName[0]
        styleV.resize_(style.size()).copy_(style)

        # forward
        with torch.no_grad():
            sF = vgg(styleV)
            cF = vgg(contentV)

            if(opt.layer == 'r41'):
                feature,transmatrix = matrix(cF[opt.layer],sF[opt.layer])
            else:
                feature,transmatrix = matrix(cF,sF)
            transfer = dec(feature)

        transfer = transfer.clamp(0,1)
        vutils.save_image(transfer,'%s/%s_%s.png'%(opt.outf,contentName,styleName),normalize=True,scale_each=True,nrow=opt.batchSize)
        print('Transferred image saved at %s%s_%s.png'%(opt.outf,contentName,styleName))
'''