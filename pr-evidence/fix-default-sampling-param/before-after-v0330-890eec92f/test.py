import base64,hashlib,io,json,os,pathlib,time
import httpx,numpy as np,av
from PIL import Image
kind=os.environ['BA_MODEL'];version=os.environ['BA_VERSION'];p=pathlib.Path('/comparison/results')/kind/version;res={}
ref=pathlib.Path('/comparison/reference.png').read_bytes();moss_prompt=json.loads(pathlib.Path('/comparison/moss-request.json').read_text())['prompt']
with httpx.Client(base_url='http://127.0.0.1:8000',timeout=1800,trust_env=False) as c:
 endpoints=['generations'] if kind=='moss' else ['edits','chat'] if kind=='qwen' else ['async','sync']
 for name in endpoints:
  if kind=='moss':
   body={'model':'OpenMOSS-Team/MOSS-Image-Base','prompt':moss_prompt,'size':'1024x1024','n':1};endpoint='/v1/images/generations'
  elif name=='edits':
   body={'model':'Qwen/Qwen-Image-2.1-Turbo','prompt':'把人物手中的木桶改成蓝色，保持人物和背景不变。','size':'512x512'};endpoint='/v1/images/edits'
  elif name=='chat':
   body={'model':'Qwen/Qwen-Image-2.1-Turbo','messages':[{'role':'user','content':'A blue ceramic teapot on a wooden table beside a window, soft daylight, detailed illustration.'}],'modalities':['image'],'height':512,'width':512};endpoint='/v1/chat/completions'
  else:
   body={'model':'MiniMaxAI/MiniMax-H3','prompt':'A small blue bird hops along a wooden garden fence in morning sunlight. Gentle camera movement, birds chirping and soft wind.','size':'512x288','num_frames':100,'fps':24,'extra_params':json.dumps({'task':'t2va','aspect_ratio':'16:9'})};endpoint='/v1/videos' if name=='async' else '/v1/videos/sync'
  assert not {'seed','generator_device','num_inference_steps','guidance_scale','true_cfg_scale','flow_shift'}&body.keys()
  (p/(name+'-request.json')).write_text(json.dumps(body,ensure_ascii=False,indent=2)+'\n');start=time.perf_counter()
  if name=='edits':r=c.post(endpoint,data={k:str(v) for k,v in body.items()},files={'image':('reference.png',ref,'image/png')})
  elif kind=='h3':r=c.post(endpoint,files=[(k,(None,str(v))) for k,v in body.items()])
  else:r=c.post(endpoint,json=body)
  if r.status_code!=200:(p/(name+'-error.txt')).write_text(r.text);raise RuntimeError(f'{endpoint} {r.status_code}: {r.text[:1000]}')
  if kind=='h3' and name=='async':
   job=r.json();ident=job['id']
   while True:
    r=c.get('/v1/videos/'+ident)
    if r.status_code!=200:(p/(name+'-error.txt')).write_text(r.text);r.raise_for_status()
    job=r.json()
    if job['status']=='completed':break
    if job['status'] in ['failed','cancelled']:raise RuntimeError(json.dumps(job))
    if time.perf_counter()-start>1800:raise TimeoutError(ident)
    time.sleep(.5)
   (p/(name+'-job.json')).write_text(json.dumps(job,indent=2)+'\n');r=c.get('/v1/videos/'+ident+'/content');r.raise_for_status()
  elapsed=time.perf_counter()-start
  if kind!='h3':
   d=r.json()
   if name=='chat':
    content=d['choices'][0]['message']['content'];items=json.loads(content) if isinstance(content,str) else content;raw=base64.b64decode(next(x['image_url']['url'] for x in items if x['type']=='image_url').split(',',1)[1])
   else:raw=base64.b64decode(d['data'][0]['b64_json'])
   (p/(name+'.png')).write_bytes(raw);a=np.array(Image.open(io.BytesIO(raw)).convert('RGB'));assert a.std()>1
   meta={'shape':list(a.shape),'rgb_sha256':hashlib.sha256(a.tobytes()).hexdigest()}
  else:
   raw=r.content;(p/(name+'.mp4')).write_bytes(raw)
   with av.open(io.BytesIO(raw)) as v:a=np.stack([f.to_ndarray(format='rgb24') for f in v.decode(video=0)])
   with av.open(io.BytesIO(raw)) as v:
    sr=v.streams.audio[0].rate;audio=np.concatenate([f.to_ndarray().astype(np.float32).reshape(-1) for f in v.decode(audio=0)])
   delta=float(np.abs(a[1:].astype(float)-a[:-1]).mean());rms=float(np.sqrt(np.mean(audio.astype(float)**2)));assert delta>.01 and rms>1e-6
   meta={'shape':list(a.shape),'rgb_sha256':hashlib.sha256(a.tobytes()).hexdigest(),'audio_pcm_sha256':hashlib.sha256(audio.tobytes()).hexdigest(),'audio_sample_rate':sr,'audio_rms':rms,'frame_delta':delta}
  res[name]={'http_status':r.status_code,'e2e_seconds':round(elapsed,3),'endpoint':endpoint,**meta};(p/'results.json').write_text(json.dumps(res,indent=2)+'\n');print(kind,version,name,json.dumps(res[name]),flush=True)
