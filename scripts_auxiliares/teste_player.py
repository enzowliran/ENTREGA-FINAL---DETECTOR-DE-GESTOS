import json, re, sys, threading, time, urllib.request
sys.path.insert(0, 'demo')
import player as P

# Gatilho: exige N janelas seguidas, dispara uma vez e pede para soltar o gesto
g = P.Gatilho(0.6, 2, 1.2)
seq = [('Swiping Right', .9, 0.0), ('Swiping Right', .9, 0.15), ('Swiping Right', .9, 0.3), ('Swiping Right', .9, 2.0),
       ('No gesture', .9, 2.1), ('Swiping Right', .9, 2.2), ('Swiping Right', .9, 2.35), ('Stop Sign', .5, 3.0), ('Stop Sign', .9, 4.0), ('Thumb Up', .99, 5.0)]
print('gatilho:', [g.atualizar(*s) for s in seq])

base = 'http://127.0.0.1:8766'
eventos = []
def ouvir():
    with urllib.request.urlopen(base + '/eventos', timeout=20) as r:
        fim = time.time() + 6
        for linha in r:
            if linha.startswith(b'data: '):
                eventos.append(json.loads(linha[6:]))
            if time.time() > fim:
                break
t = threading.Thread(target=ouvir, daemon=True); t.start()
time.sleep(1)
for c in ('proxima', 'anterior', 'alternar'):
    urllib.request.urlopen(f'{base}/testar?c={c}', timeout=5).read()
t.join(8)
print('comandos recebidos:', [e['comando'] for e in eventos if e['tipo'] == 'comando'])
est = [e for e in eventos if e['tipo'] == 'estado']
print('estados:', len(est), est[-1] if est else None)
with urllib.request.urlopen(base + '/video', timeout=5) as r:
    d = r.read(60000)
print('video bytes', len(d), 'tem jpeg', b'\xff\xd8' in d, r.headers.get('Content-Type'))
html = open('demo/player.html', encoding='utf-8').read()
open(sys.argv[1], 'w', encoding='utf-8').write(re.search(r'<script>(.*)</script>', html, re.S).group(1))
