
import logging
import os
import json
import torch
import torch.nn.functional as F
import torchvision

from utils.models import create_layer_map, load_model,load_models_dirpath
import util.smartparse as smartparse


def root():
    return '../trojai-datasets/cyber-network-c2-feb2024-train/models'

#The user provided engine that our algorithm interacts with
class engine:
    def __init__(self,folder=None,params=None):
        default_params=smartparse.obj();
        default_params.model_filepath='';
        default_params.examples_dirpath='';
        default_params.scratch_dirpath=''
        params=smartparse.merge(params,default_params)
        
        if params.model_filepath=='':
            params.model_filepath=os.path.join(folder,'model.pt');
        else:
            folder=os.path.dirname(params.model_filepath)
        
        self.folder=folder
        try:
            config_path=os.path.join(folder,'reduced-config.json')
            self.config=json.load(open(config_path,'r'))
        except:
            config_path=os.path.join(folder,'config.json')
            self.config=json.load(open(config_path,'r'))
        
        
        if params.examples_dirpath=='':
            params.examples_dirpath=os.path.join(folder,'clean-example-data');
        if params.scratch_dirpath=='':
            params.scratch_dirpath='scratch';
        
        self.params=params
        
        #Load model
        print('entering load model')
        model, model_repr, model_class = load_model(params.model_filepath)
        print('done loading model')
        self.model=model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.model.to(self.device)
        self.model.model.eval()
    
    def load_examples(self,examples_dirpath=None,scratch_dirpath=None):
        if examples_dirpath is None:
            examples_dirpath=self.params.examples_dirpath
        if scratch_dirpath is None:
            scratch_dirpath=self.params.scratch_dirpath;
        
        #Load examples
        fnames=[fname for fname in os.listdir(examples_dirpath) if fname.endswith('.png') or fname.endswith('.PNG')]
        fnames=sorted(fnames)
        im=[]
        gt=[]
        for fname in fnames:
            im.append(torchvision.io.read_image(os.path.join(examples_dirpath,fname)).float())
            fname_base = os.path.splitext(fname)[0]
            gt.append(json.load(open(os.path.join(examples_dirpath, '{}.json'.format(fname_base)),'r')))
        
        im=torch.stack(im,dim=0)
        
        return {'im':im,'gt':torch.LongTensor(gt)}
    
    def load_poisoned_examples(self,examples_dirpath=None,scratch_dirpath=None):
        examples_dirpath=os.path.join(self.folder,'poisoned-example-data');
        if not os.path.exists(examples_dirpath):
            return None
        
        #Load examples
        fnames=[fname for fname in os.listdir(examples_dirpath) if fname.endswith('.png') or fname.endswith('.PNG')]
        fnames=sorted(fnames)
        im=[]
        gt=[]
        for fname in fnames:
            im.append(torchvision.io.read_image(os.path.join(examples_dirpath,fname)).float())
            fname_base = os.path.splitext(fname)[0]
            gt.append(json.load(open(os.path.join(examples_dirpath, '{}.json'.format(fname_base)),'r')))
        
        im=torch.stack(im,dim=0)
        
        return {'im':im,'gt':torch.LongTensor(gt)}
    
    def eval_actv0(self,data):
        with torch.no_grad():
            im=data['im'].cuda()
            gt=data['gt'].cuda()
            pred=self.model.model(im)
        
        return pred
    
    def eval_grad(self,data):
        im=data['im'].cuda()
        gt=data['gt'].cuda()
        
        im0=im.clone().requires_grad_()
        pred=self.model.model(im0)
        loss=F.cross_entropy(pred,1-gt)
        loss.backward()
        g0=im0.grad.data.clone()
        
        im1=im.clone().requires_grad_()
        pred=self.model.model(im1)
        loss=F.cross_entropy(pred,gt)
        loss.backward()
        g1=im1.grad.data.clone()
        
        return torch.cat((g0,g1),dim=-3)

'''
interface=engine(folder=os.path.join(root(),'id-00000020'))

for n,p in interface.model.model.named_parameters():
    print(n,p.shape)


im,gt=interface.load_examples()
print(im.shape,gt)

pred=interface.model.model(im.cuda())
loss=F.cross_entropy(pred,torch.LongTensor(gt).cuda())
print(pred,loss)
'''
