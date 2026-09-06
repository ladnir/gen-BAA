"""Generate a covering tree of entropy upper certificates, with checkpoints."""
import argparse,json,time
from pathlib import Path
from fractions import Fraction as F
from math import log
from bulk_bound import Model,entropy
from verify_bulk import setup,impossible,verify_leaf,threshold


def box_at(path,nodes,root):
    box=root.copy();prefix=''
    for char in path:
        ix=nodes[prefix]*2;mid=(box[ix]+box[ix+1])/2
        box[ix+1 if char=='0' else ix]=mid
        prefix+=char
    return box


def run(sigma,path,limit,delta=None):
    path=Path(path);md=setup(sigma,delta=delta)
    root=[md['amin'],F(1),md['amin']/2,F(1)]
    model=Model(sigma=sigma,delta=float(md['delta']),n=md['n'],qbits=127)
    if path.exists():data=json.loads(path.read_text())
    else:data={'sigma':sigma,'delta':str(md['delta']),'nodes':{},'leaves':{},'pending':[''],'solves':0}
    if data['sigma']!=sigma or setup(sigma,delta=data.get('delta'))['delta']!=md['delta']:
        raise ValueError('Checkpoint parameters do not match requested parameters')
    if not data['nodes'] and not data['leaves'] and not data['pending']:
        data['pending']=['']
    # Recover unfinished leaves after any interrupted proposal. Verification
    # independently rejects incomplete covers; checkpoints never certify them.
    frontier=[]
    def unfinished(node):
        if node in data['leaves']:return
        if node in data['nodes']:
            unfinished(node+'0');unfinished(node+'1')
        else:frontier.append(node)
    unfinished('')
    data['pending']=frontier
    target=threshold(sigma)
    started=time.monotonic();solves=0
    def save():
        temp=path.with_suffix('.writing.json');temp.write_text(json.dumps(data,separators=(',',':')))
        temp.replace(path)
    node=None
    try:
        while data['pending'] and time.monotonic()-started<limit:
            node=data['pending'].pop()
            box=box_at(node,data['nodes'],root)
            if impossible(md,box):
                data['leaves'][node]={'status':'infeasible'}
                continue
            result=model.solve([float(v) for v in box])
            solves+=1;data['solves']+=1
            if result['status']=='candidate':
                upper=verify_leaf(sigma,box,result,delta=md['delta'])
                if upper<target:
                    data['leaves'][node]=result
                    if solves%20==0:
                        print(json.dumps({'solves':data['solves'],'leaves':len(data['leaves']),
                                          'pending':len(data['pending']),'last_upper':str(upper)}),flush=True)
                    continue
            else:
                data.setdefault('failures',[]).append({'path':node,'status':result['status']})
            errors=[]
            for ix in (0,2):
                lo,hi=float(box[ix]),float(box[ix+1]);mid=(lo+hi)/2
                errors.append(float(entropy(mid)-(entropy(lo)+entropy(hi))/2))
            axis=0 if errors[0]>=errors[1] else 1
            if len(node)>=35:
                data['pending'].append(node)
                raise RuntimeError('Unresolved deep box: '+str(box)+' '+str(result))
            data['nodes'][node]=axis
            data['pending'].extend([node+'1',node+'0'])
            if solves%20==0:save()
    finally:
        if node is not None and node not in data['nodes'] and node not in data['leaves'] and node not in data['pending']:
            data['pending'].append(node)
        save()
    print(json.dumps({'sigma':sigma,'solves':data['solves'],'leaves':len(data['leaves']),
                      'pending':len(data['pending']),'seconds':time.monotonic()-started}),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--sigma',type=int,default=4)
    ap.add_argument('--output',required=True);ap.add_argument('--seconds',type=float,default=45)
    ap.add_argument('--delta',help='Exact rational or decimal target distance')
    args=ap.parse_args();run(args.sigma,args.output,args.seconds,delta=args.delta)
