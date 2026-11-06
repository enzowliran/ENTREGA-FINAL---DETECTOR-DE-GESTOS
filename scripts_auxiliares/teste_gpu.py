import sys,time; sys.path.insert(0,'src')
import torch, modelos as M
torch.backends.cudnn.benchmark=True
for nome in ('cnn_gru','cnn_3d'):
    torch.cuda.reset_peak_memory_stats(); m=M.criar_modelo(nome,27).cuda(); o=torch.optim.AdamW(m.parameters())
    x=torch.randn(48,16,3,96,96,device='cuda'); y=torch.randint(0,27,(48,),device='cuda')
    t=None
    for i in range(12):
        if i==2: t=time.time()
        with torch.autocast('cuda',dtype=torch.bfloat16): l=torch.nn.functional.cross_entropy(m(x).float(),y)
        o.zero_grad(set_to_none=True); l.backward(); o.step()
    torch.cuda.synchronize(); d=time.time()-t
    print(nome,f'{M.contar_parametros(m):,} params',round(48*10/d),'videos/s','pico',torch.cuda.max_memory_allocated()>>20,'MB')
    a=M.adaptar_entrada(m,1); print(' adaptar 1 canal', a(torch.randn(2,16,1,96,96,device='cuda')).shape); del m,o,x,a
