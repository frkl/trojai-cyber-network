import torch
import util.video
import os
import helper


fnames=os.listdir(helper.root())
fnames=sorted(fnames)

'''
ims=[]
gts=[]
for fname in fnames:
    interface=helper.engine(folder=os.path.join(helper.root(),fname))
    data=interface.load_examples()
    if not data is None:
        ims.append(data['im'])
        gts.append(data['gt'])

ims=torch.cat(ims,dim=0)
gts=torch.cat(gts,dim=0)

torch.save({'im':ims,'gt':gts},'all_clean.pt')

v=ims/255
v=v.repeat(1,3,1,1)

util.video.write_video(v,'tmp2.avi',fps=10)
'''

interface=helper.engine(folder=os.path.join(helper.root(),'id-00000048'))

data=torch.load('all_clean.pt')

data2=torch.load('all_poisoned.pt')

s=interface.eval_actv0(data)