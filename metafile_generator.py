import torch

#Github link
url='https://github.com/frkl/trojai-cyber-network'
#commit id
'''
import subprocess
import re
process = subprocess.Popen(["git", "ls-remote", url], stdout=subprocess.PIPE)
stdout, stderr = process.communicate()
sha = re.split(r'\t+', stdout.decode('ascii'))[0]
print(sha)
'''


sha='ea6d7866e18f1aa8cb0753023e67babae848fdfb' #override sha
'''
from parse import parse
import sys, zlib

raw_commit = sys.stdin.buffer.read()

commit = zlib.decompress(raw_commit).decode('utf-8').split('\x00')[1]
(headers, body) = commit.split('\n\n')

for line in headers.splitlines():
    # `{:S}` is a type identifier meaning 'non-whitespace', so that
    # the fields will be captured successfully.
    p = parse('author {name} <{email:S}> {time:S} {tz:S}', line)
    if (p):
        print("Author: {} <{}>\n".format(p['name'], p['email']))
        print(body)
        break
'''

#Load current hyperparameters
checkpoint='learned_parameters/model.pt'
tmp=torch.load(checkpoint)

#Generate metaparam template
hp_config=tmp[0]['config'];
template={};
for cfg in hp_config:
    template_k={'description':cfg[0]}
    if cfg[1]==str:
        template_k['type']='string'
        template_k['enum']=cfg[2]
    elif cfg[1]==int:
        template_k['type']='integer'
        template_k['minimum']=cfg[2][0]
        template_k['maximum']=cfg[2][1]
        template_k['suggested_minimum']=cfg[2][0]
        template_k['suggested_maximum']=cfg[2][1]
    elif cfg[1]==float:
        template_k['type']='number'
        template_k['suggested_minimum']=cfg[2][0]
        template_k['suggested_maximum']=cfg[2][1]
    
    template[cfg[0]]=template_k


meta_schema={"$schema": "http://json-schema.org/draft-07/schema#","title": "SRI Trinity Framework",
"technique": "SRI trigger search + attribution analysis pipeline","technique_description": "Using model's activations on 288 clean examples as features for Trojan detection. Further trigger search integration is intended.","technique_changes": "Applying the universal Trojan detection pipelines developed R11 image classification and R12 cyber-pdf, but using model activations as features instead of weight gradients.","technique_type": ["Attribution Analysis"],"commit_id": sha,"repo_name": url,"required": [],"additionalProperties": False,"type": "object"}
meta_schema['properties']=template;

#Generate metaparam file
params=vars(tmp[0]['params'])
params2={};
for cfg in hp_config:
    k=cfg[0]
    params2[k]=params[k]

import json
with open('metaparameters_schema.json','w') as f:
    json.dump(meta_schema,f)

with open('metaparameters.json','w') as f:
    json.dump(params2,f)

