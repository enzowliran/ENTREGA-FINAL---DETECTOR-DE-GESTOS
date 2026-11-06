import json,re,sys,collections as C,statistics as S
R=sys.argv[1]
for lado in('left','right'):
    d=json.load(open(f'{R}/Annotation/all_{lado}.json'))
    print(lado,list(d.keys()), d.get('labels'))
    db=d['database']; k=list(db)[0]; print(k,db[k])
    c=C.Counter();sub=C.Counter();sp=C.Counter();dur=C.defaultdict(list);rec=C.Counter()
    for n,v in db.items():
        l=v['annotations']['label']; c[l]+=1; sp[v['subset']]+=1
        m=re.search(r'Subject(\d+)',n); sub[int(m.group(1))]+=1
        s=v['annotations']['segment']; dur[l].append(s[1]-s[0])
    print(' total',len(db),'subset',dict(sp)); print(' classes',dict(c)); print(' sujeitos',len(sub),min(sub.values()),max(sub.values()))
    print(' duracao media frames',{k:(round(S.mean(v),1),min(v),max(v)) for k,v in dur.items()})
