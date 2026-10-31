import base64,hashlib,json,sys
p=json.load(sys.stdin);t=p['ticket'];lib={'__name__':'readonly_failed_metadata'};exec(p['rpc_source'],lib);r=lib['check_ticket'](t,False);lib['existing'](r,t)
out=r/'output'/t['request']['arms'][0]/'measured';response={'ok':True,'run_id':t['request']['run_id'],'remote_file_writes':False,'GPU_or_model_calls':False,'files':{}}
for rel in ('artifacts/profile_export.jsonl','artifacts/profile_export_aiperf.json','aiperf.log'):
 path=out/rel;data=lib['regular'](path);assert len(data)<=2*2**20
 response['files'][rel]={'sha256':hashlib.sha256(data).hexdigest(),'size':len(data),'base64':base64.b64encode(data).decode()}
print(json.dumps(response))
