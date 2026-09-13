import json,tempfile,pathlib
from contextlib import nullcontext
from unittest.mock import patch
from lyra.cli import main,parser,_pipeline
from lyra.pipeline import YuE2Pipeline
from yue2.protocol import GenerationConfig
records=[]
with tempfile.TemporaryDirectory() as tmp:
 p=pathlib.Path(tmp); model=p/'saved';model.mkdir(); (model/'conversion.json').write_text('{}')
 cfg=GenerationConfig.from_dict({'semantic':{'max_tokens':555},'ode_steps':17})
 (model/'pipeline.json').write_text(json.dumps({'model':'.','vae':'vae','generation_config':cfg.to_dict()}))
 args=parser().parse_args(['generate','--style','pop','--lyrics','hello','--model',str(model),'--output',str(p/'out')])
 # Capture the effective argument passed to constructor after real saved metadata loading.
 with patch.object(YuE2Pipeline,'load_timing',{},create=True),patch.object(YuE2Pipeline,'__init__',return_value=None) as init,patch('lyra.pipeline.resolve_model',side_effect=lambda x,**kw:x):
  _pipeline(args)
  actual=init.call_args.kwargs['generation_config']
  records.append({'case':'serialized_generation_config_via_cli','saved_steps':17,'saved_max_tokens':555,'constructor_config':None if actual is None else actual.to_dict(),'expected':'saved config','status':'fail' if actual is None else 'pass'})
class FakePipe:
 def __enter__(self):return self
 def __exit__(self,*a):pass
 def __call__(self,**data):
  class FakeSong:
   def save_artifacts(self,path):
    path.mkdir(parents=True)
    return {'identity':'fake','truncated':{'abc':False,'semantic':False}}
  return FakeSong()
with tempfile.TemporaryDirectory() as tmp:
 p=pathlib.Path(tmp);req=p/'requests.jsonl';req.write_text(json.dumps({'id':'batch.json','style':'pop','lyrics':'hello'})+'\n')
 with patch('lyra.cli._pipeline',return_value=FakePipe()):
  try:main(['batch','--input',str(req),'--output',str(p/'out')]);err=None
  except Exception as e:err=f'{type(e).__name__}: {e}'
 records.append({'case':'valid_request_id_collides_with_batch_manifest','status':'fail' if err else 'pass','error':err,'expected':'preflight rejection before generation or noncolliding output layout'})
pathlib.Path('reports/acceptance-audit/cli-reproductions.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(records,indent=2))
