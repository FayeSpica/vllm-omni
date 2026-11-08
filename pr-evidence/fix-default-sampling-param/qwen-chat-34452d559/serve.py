import functools,json,os,pathlib,types
model=os.environ['BA_MODEL'];version=os.environ['BA_VERSION'];root=pathlib.Path('/comparison/results')/model/version;root.mkdir(parents=True,exist_ok=True)
fd=os.open(root/'server.log',os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o644);os.dup2(fd,1);os.dup2(fd,2)
# Test-only control of omitted-seed generation. No sampling settings are injected.
import vllm_omni.entrypoints.openai.api_server as api
import vllm_omni.diffusion.request as dr
from vllm_omni.entrypoints.async_omni import AsyncOmni
fixed_rng=types.SimpleNamespace(randint=lambda low,high:42)
api.random=fixed_rng;dr.random=fixed_rng
keys=['num_inference_steps','guidance_scale','guidance_scale_provided','true_cfg_scale','seed','generator_device','height','width','num_frames','frame_rate','num_outputs_per_prompt','extra_args']
def record(stage,request_id,params):
 values=[]
 for sp in params or []:values.append({k:getattr(sp,k,None) for k in keys})
 with (root/'sampling.jsonl').open('a') as f:f.write(json.dumps({'stage':stage,'request_id':request_id,'params':values},default=lambda x:type(x).__name__)+'\n')
original=AsyncOmni.generate
@functools.wraps(original)
def generate(self,*a,**kw):
 record('api_to_engine',kw.get('request_id',''),kw.get('sampling_params_list'))
 return original(self,*a,**kw)
AsyncOmni.generate=generate
post=dr.OmniDiffusionRequest.__post_init__
@functools.wraps(post)
def post_init(self):
 post(self);record('backend_request',self.request_id,[self.sampling_params])
dr.OmniDiffusionRequest.__post_init__=post_init
args=['serve'];extra=[]
if model=='moss':identifier='OpenMOSS-Team/MOSS-Image-Base'
elif model=='qwen':
 identifier='Qwen/Qwen-Image-2.1-Turbo';extra=['--default-sampling-params',json.dumps({'0':{'num_inference_steps':8,'true_cfg_scale':1.0}})]
else:
 identifier='/root/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-H3/snapshots/42ed227ee7df40d41602854ae760620d6eb651fe/FL2VA'
 extra=['--served-model-name','MiniMaxAI/MiniMax-H3','--trust-remote-code','--task-type','fl2va','--num-gpus','1','--diffusion-offload-config',json.dumps({'mode':'layer','components':['dit','text_encoder']}),'--diffusion-attention-backend','FLASH_ATTN']
args+=[identifier,'--omni','--host','0.0.0.0','--port','8000','--enforce-eager','--stage-init-timeout','1800',*extra]
(root/'launch.json').write_text(json.dumps({'source':version,'args':args,'rng_control':'Only omitted-seed randint is fixed at 42 in the test harness; request and sampling defaults are unchanged.'},indent=2)+'\n')
import sys
sys.argv=['vllm',*args]
from vllm_omni.entrypoints.cli.main import main
main()
