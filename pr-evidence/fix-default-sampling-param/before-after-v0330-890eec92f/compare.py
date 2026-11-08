import json, math
from pathlib import Path
import numpy as np
from PIL import Image
root=Path('/comparison')
summary={'before':'2b610b3e9','after':'890eec92f','models':{}}
comparisons={}
for model in ['moss','qwen','h3']:
 versions={v:json.loads((root/'results'/model/v/'results.json').read_text()) for v in ['before','after']}
 launches={v:json.loads((root/'results'/model/v/'launch.json').read_text()) for v in versions}
 assert launches['before']['args']==launches['after']['args']
 for case in versions['before']:
  a,b=(versions[v][case] for v in ['before','after'])
  assert a['http_status']==b['http_status']==200
  requests=[json.loads((root/'results'/model/v/(case+'-request.json')).read_text()) for v in versions]
  assert requests[0]==requests[1]
  assert not {'seed','generator_device','num_inference_steps','guidance_scale','true_cfg_scale','flow_shift'} & requests[0].keys()
  if model!='moss':assert a['rgb_sha256']==b['rgb_sha256']
  if model=='h3':assert a['audio_pcm_sha256']==b['audio_pcm_sha256']
  else:
   arrays=[np.asarray(Image.open(root/'results'/model/v/(case+'.png')).convert('RGB'),dtype=np.float64) for v in versions]
   mse=float(np.mean((arrays[0]-arrays[1])**2))
   comparisons[model+'/'+case]={'mse':mse,'psnr_db':10*math.log10(255**2/mse) if mse else 'infinity'}
 summary['models'][model]=versions
assert summary['models']['moss']['before']['generations']['rgb_sha256']!=summary['models']['moss']['after']['generations']['rgb_sha256']
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(root/'image-comparisons.json').write_text(json.dumps(comparisons,indent=2)+'\n')
print(json.dumps(summary));print(json.dumps(comparisons))
