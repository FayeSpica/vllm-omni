import json,os,pathlib,subprocess,sys,time,urllib.request
root=pathlib.Path('/tmp/pr9700-qwen-chat-34452d559');group=sys.argv[1]
models=['qwen'];gpu='6' if group=='images' else '7'
for model in models:
 for version in ['after']:
  name=f'lwm_pr9700_chat344_{model}_{version}';out=root/'results'/model/version;out.mkdir(parents=True,exist_ok=True)
  cache='/opt/docker_cache/.cache' if model=='h3' else '/mnt/sfs_turbo'
  args=['docker','run','-d','--name',name,'--gpus','device='+gpu,'--shm-size=8g','-e','CUDA_VISIBLE_DEVICES=0','-e','HF_HUB_OFFLINE=1','-e','BA_MODEL='+model,'-e','BA_VERSION='+version,'-e','PYTHONPATH=/comparison/'+version,'-e','VLLM_OMNI_VIDEO_SYNC_TIMEOUT=3600','-v',str(root)+':/comparison','-v',cache+':/root/.cache','-w','/comparison/'+version,'--entrypoint','python3','lwm/pr9700-cuda0330:44b53bc4f','/comparison/serve.py']
  subprocess.run(args,check=True);print('started',name,flush=True)
  try:
   # Probe inside the container: no published host port is required.
   for i in range(180):
    r=subprocess.run(['docker','exec',name,'python3','-c','import urllib.request; assert urllib.request.urlopen("http://127.0.0.1:8000/health",timeout=2).status==200'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    if r.returncode==0:break
    state=subprocess.check_output(['docker','inspect',name,'--format','{{.State.Running}}'],text=True).strip()
    if state!='true':raise RuntimeError(name+' exited; see '+str(out/'server.log'))
    time.sleep(5)
   else:raise TimeoutError(name+' readiness')
   r=subprocess.run(['docker','exec',name,'python3','/comparison/test.py'],stdout=(out/'test.log').open('w'),stderr=subprocess.STDOUT)
   print((out/'test.log').read_text()[-2500:],flush=True)
   if r.returncode:raise RuntimeError(name+' test failed')
  finally:subprocess.run(['docker','stop','-t','10',name],check=False)
