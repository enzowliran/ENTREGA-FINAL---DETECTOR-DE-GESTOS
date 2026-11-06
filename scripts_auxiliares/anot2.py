import re,sys,collections as C,statistics as S,struct
R=sys.argv[1]; P=sys.argv[2]
nomes={}
for lado in ('left','right'):
    for sub in ('train','val'):
        n=0;cl=C.Counter();pes=C.Counter();dur=C.defaultdict(list)
        for l in open(f'{R}/Annotation/{sub}list_all_{lado}.txt'):
            q=l.split()
            if len(q)!=4: continue
            n+=1; cl[int(q[1])]+=1; pes[q[0].split('/')[0]]+=1; dur[int(q[1])].append(int(q[3])-int(q[2]))
        print(lado,sub,'n',n,'classes',dict(sorted(cl.items())),'pessoas',len(pes),'dur media',{k:round(S.mean(v),1) for k,v in sorted(dur.items())})
b=open(P,'rb').read()
print('assinatura',b[:7]); typ=b[9]; fl=struct.unpack('<H',b[10:12])[0]; print('MAIN_HEAD tipo',hex(typ),'flags',hex(fl),'volume',bool(fl&1),'solid',bool(fl&8),'novo-numero',bool(fl&0x10))
o=7+struct.unpack('<H',b[12:14])[0]
contagem=C.Counter();ex=0;metodos=C.Counter();ultimo=None
while o+32<len(b) and b[o+2]==0x74:
    f=struct.unpack('<H',b[o+3:o+5])[0];hs=struct.unpack('<H',b[o+5:o+7])[0];ps=struct.unpack('<I',b[o+7:o+11])[0];us=struct.unpack('<I',b[o+11:o+15])[0];m=b[o+25];ns=struct.unpack('<H',b[o+26:o+28])[0]
    nome=b[o+32:o+32+ns].decode('latin1'); metodos[hex(m)]+=1
    if ex<2 and nome.endswith('jpg'): print(nome[-40:],'empacotado',ps,'original',us,'flag solid-file',bool(f&0x10));ex+=1
    ultimo=(nome[-40:],o); o+=hs+ps
print('metodos',dict(metodos),'ultimo',ultimo)
