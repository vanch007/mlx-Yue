import json,pathlib,sys,numpy as np,soundfile as sf
from lyra.artifacts import load_artifacts
from lyra.music_tools.abc_tools import parse_abc,report
root=pathlib.Path.cwd(); results=[]
for path in sorted((root/'outputs').rglob('result.json')):
    if not (path.parent/'latent.npy').exists() or 'listen' in str(path.parent): continue
    entry={'path':str(path.parent.relative_to(root))}
    try:
        s=load_artifacts(path.parent)
        peak=rms_sum=samples=zeros=0; finite=True
        with sf.SoundFile(path.parent/'audio.flac') as f:
            for block in f.blocks(blocksize=48000,dtype='float32',always_2d=True):
                finite &= bool(np.isfinite(block).all()); peak=max(peak,float(np.max(np.abs(block)))); rms_sum+=float(np.sum(block.astype(np.float64)**2));samples+=block.size; zeros+=int(np.count_nonzero(block==0))
        entry.update(status='pass',frames=len(s.semantic.tokens),audio_seconds=s.result['audio_seconds'],truncated=s.result['truncated'],finite=finite,peak=peak,rms=(rms_sum/samples)**.5,zero_fraction=zeros/samples,ode_steps=s.config['generation']['ode_steps'],attention_inputs=s.config.get('attention_inputs','original'),precision=s.config.get('ar_precision'),identity=s.result['identity'])
        if s.semantic.plan.abc:
            try: score=parse_abc(s.semantic.plan.abc);entry['score_parse']='pass'
            except ValueError as e: entry['score_parse']='fail';entry['score_error']=str(e)
    except Exception as e:entry.update(status='fail',error=f'{type(e).__name__}: {e}')
    results.append(entry)
resources=[]
for p in sorted((root/'outputs').rglob('*.resources.json')):
    r=json.loads(p.read_text());resources.append({'path':str(p.relative_to(root)),'monitor_error':r.get('monitor_error'),'exception':r.get('exception'),'peak_gib':r.get('sampled_maxima',{}).get('physical_footprint_bytes',0)/2**30})
report={'git_head':__import__('subprocess').check_output(['git','rev-parse','HEAD'],text=True).strip(),'songs':results,'resources':resources,'summary':{'verified':sum(r['status']=='pass' for r in results),'failed':sum(r['status']=='fail' for r in results),'naturally_ended':sum(not any(r.get('truncated',{'unknown':True}).values()) for r in results)}}
(root/'reports/acceptance-audit/artifacts.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print(report['summary']);print('score failures',[r['path'] for r in results if r.get('score_parse')=='fail'])
for r in results:
 if r['status']=='fail':print(r)
