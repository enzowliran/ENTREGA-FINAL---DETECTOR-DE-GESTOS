p='demo/ao_vivo.py'; s=open(p,encoding='utf-8').read()
rep=[('''# O treino amostra 16 frames de vídeos de cerca de 3 s, o que dá cerca de 5,3 frames por segundo.
QUADROS_POR_SEGUNDO = 5.3
''','''# O buffer guarda 12 frames por segundo, como o Jester. Cada janela sorteia 16 deles, então uma
# janela curta enxerga gestos rápidos e uma longa enxerga gestos lentos.
QUADROS_POR_SEGUNDO = 12
'''),
('''def desenhar(quadro, legenda, confianca, destaque, top3, disparos, minutos, ms_inferencia, aquecendo, fontes):
''','''def desenhar(quadro, legenda, confianca, destaque, top3, disparos, minutos, ms_inferencia, aquecendo, fontes,
             entrada=None, fps=0.0, janela_ativa=None):
'''),
('''    return np.array(Image.alpha_composite(imagem, camada).convert("RGB"))[..., ::-1].copy()
''','''    janela_texto = f" · janela {janela_ativa:.1f} s" if janela_ativa else ""
    d.text((largura - 10, 8), f"Câmera: {fps:.0f} fps{janela_texto}", font=fontes[1], fill=(235, 235, 230, 255), anchor="ra")
    final = Image.alpha_composite(imagem, camada).convert("RGB")
    if entrada is not None:
        # Miniatura com o que o modelo recebe, para conferir o tamanho da mão na imagem.
        mini = Image.fromarray(entrada.transpose(1, 2, 0).astype(np.uint8)).resize((192, 192), Image.NEAREST)
        final.paste(mini, (largura - 202, 34))
    return np.array(final)[..., ::-1].copy()
'''),
('''    janela = collections.deque(maxlen=N_FRAMES)
''','''    maximo = int(round(max(args.janelas) * QUADROS_POR_SEGUNDO))
    janela = collections.deque(maxlen=maximo)
    minimo = int(round(min(args.janelas) * QUADROS_POR_SEGUNDO))
    tempos = collections.deque(maxlen=30)
    janela_ativa = None
'''),
('''        agora = time.time()
        # A entrada''','''        agora = time.time()
        tempos.append(agora)
        fps = (len(tempos) - 1) / (tempos[-1] - tempos[0]) if len(tempos) > 1 and tempos[-1] > tempos[0] else 0.0
        # A entrada'''),
('''        if len(janela) == N_FRAMES and agora - ultima_inferencia >= args.intervalo:
            t0 = time.perf_counter()
            with torch.no_grad(), torch.autocast(dispositivo, dtype=torch.bfloat16):
                x = (torch.from_numpy(np.stack(janela))[None].to(dispositivo) - 127.5) / 127.5
                p = modelo(x).float().softmax(1)[0].cpu().numpy()
            ms = (time.perf_counter() - t0) * 1000
''','''        if len(janela) >= minimo and agora - ultima_inferencia >= args.intervalo:
            t0 = time.perf_counter()
            buffer = list(janela)
            lotes_janela, usadas = [], []
            for duracao in args.janelas:
                n = int(round(duracao * QUADROS_POR_SEGUNDO))
                if len(buffer) >= n:
                    indices = np.round(np.linspace(len(buffer) - n, len(buffer) - 1, N_FRAMES)).astype(int)
                    lotes_janela.append(np.stack([buffer[i] for i in indices]))
                    usadas.append(duracao)
            with torch.no_grad(), torch.autocast(dispositivo, dtype=torch.bfloat16):
                x = (torch.from_numpy(np.stack(lotes_janela)).to(dispositivo) - 127.5) / 127.5
                todas = modelo(x).float().softmax(1).cpu().numpy()
            # Fica a janela com o gesto mais confiante. Isso favorece detectar, então pode aumentar os falsos disparos.
            indices_gesto = [i for i, nome in enumerate(nomes) if nome not in SEM_GESTO]
            escolhida = int(np.argmax([q[indices_gesto].max() for q in todas]))
            p, janela_ativa = todas[escolhida], usadas[escolhida]
            ms = (time.perf_counter() - t0) * 1000
'''),
('''        elif len(janela) < N_FRAMES:
            legenda = f"Aquecendo... {len(janela)}/{N_FRAMES}"
''','''        elif len(janela) < minimo:
            legenda = f"Aquecendo... {len(janela)}/{minimo}"
'''),
('''(agora - inicio) / 60, ms, len(janela) < N_FRAMES, fontes)
''','''(agora - inicio) / 60, ms, len(janela) < minimo, fontes,
                        entrada=janela[-1] if janela else None, fps=fps, janela_ativa=janela_ativa)
'''),
('''    a.add_argument("--limiar", type=float, default=0.6, help="Confiança mínima para mostrar um gesto")
    a.add_argument("--suavizar", type=int, default=3, help="Número de predições na média móvel")
    a.add_argument("--intervalo", type=float, default=0.2, help="Segundos entre inferências")
''','''    a.add_argument("--limiar", type=float, default=0.5, help="Confiança mínima para mostrar um gesto")
    a.add_argument("--suavizar", type=int, default=2, help="Número de predições na média móvel")
    a.add_argument("--intervalo", type=float, default=0.15, help="Segundos entre inferências")
    a.add_argument("--janelas", type=float, nargs="+", default=[1.0, 1.5, 3.0], help="Durações das janelas, em segundos")
'''),
('''MODELOS_PADRAO = [RAIZ / "resultados/jester/cnn_3d/melhor.pt", RAIZ / "resultados/jester/cnn_3d_pequeno/melhor.pt"]''',
 '''MODELOS_PADRAO = [RAIZ / "resultados/jester/cnn_3d/melhor.pt", RAIZ / "resultados/jester/cnn_3d_curto/melhor.pt",
                  RAIZ / "resultados/jester/cnn_3d_pequeno/melhor.pt"]'''),
]
for a,b in rep:
    assert a in s, a[:60]
    s=s.replace(a,b)
open(p,'w',encoding='utf-8').write(s)

p='src/jester.py'; s=open(p,encoding='utf-8').read()
a='''                 classe_sem_gesto=None, classes_estaticas=()):
'''
b='''                 classe_sem_gesto=None, classes_estaticas=(), aug_tempo=False):
        # Sorteia um trecho e uma velocidade, para a rede reconhecer gestos curtos na janela.
        self.aug_tempo = aug_tempo
'''
assert a in s; s=s.replace(a,b)
a='''        pontos = np.linspace(0, contagem, self.n_frames + 1)
        if self.treino:
'''
b='''        base, extensao = 0.0, float(contagem)
        if self.treino and self.aug_tempo:
            extensao = contagem * gerador.uniform(0.5, 1.0)
            base = gerador.uniform(0, contagem - extensao)
        pontos = base + np.linspace(0, extensao, self.n_frames + 1)
        if self.treino:
'''
assert a in s; s=s.replace(a,b)
open(p,'w',encoding='utf-8').write(s)

p='src/treinar_jester.py'; s=open(p,encoding='utf-8').read()
a='''    analisador.add_argument("--frac-sinteticos", '''
b='''    analisador.add_argument("--aug-tempo", action="store_true", help="Treina com trechos e velocidades variados")
    analisador.add_argument("--suavizacao", type=float, default=0.1, help="Suavização de rótulos; valores altos limitam a confiança")
    analisador.add_argument("--frac-sinteticos", '''
assert a in s; s=s.replace(a,b)
a='''classes_estaticas=[nomes.index(n) for n in ESTATICAS + SEM_GESTO])
'''
b='''classes_estaticas=[nomes.index(n) for n in ESTATICAS + SEM_GESTO],
                                aug_tempo=args.aug_tempo)
'''
assert a in s; s=s.replace(a,b)
a='''label_smoothing=0.1)'''
b='''label_smoothing=args.suavizacao)'''
assert a in s; s=s.replace(a,b)
open(p,'w',encoding='utf-8').write(s)
print('ok')
