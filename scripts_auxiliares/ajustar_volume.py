import re, esprima
p = 'demo/player.html'
s = open(p, encoding='utf-8').read()
trocas = [
    ('mestre = ctx.createGain(); mestre.gain.value = 0.85;', 'mestre = ctx.createGain(); mestre.gain.value = 0.7;'),
    ('faixaBus = ctx.createGain(); faixaBus.gain.value = 1; faixaBus.connect(ctx.saida);',
     'faixaBus = ctx.createGain(); faixaBus.gain.value = MUSICAS[faixa].ganho; faixaBus.connect(ctx.saida);'),
    ('{ titulo: "Estrada Azul", estilo: "Lo-fi",', '{ titulo: "Estrada Azul", ganho: 1.15, estilo: "Lo-fi",'),
    ('{ titulo: "Noite na Via", estilo: "Synthwave",', '{ titulo: "Noite na Via", ganho: 0.7, estilo: "Synthwave",'),
    ('{ titulo: "Sinal Verde", estilo: "House",', '{ titulo: "Sinal Verde", ganho: 0.95, estilo: "House",'),
    ('{ titulo: "Chuva Fina", estilo: "Piano ambiente",', '{ titulo: "Chuva Fina", ganho: 1.3, estilo: "Piano ambiente",'),
    ('{ titulo: "Quilômetro Zero", estilo: "Funk retrô",', '{ titulo: "Quilômetro Zero", ganho: 0.85, estilo: "Funk retrô",'),
]
for a, b in trocas:
    assert a in s, a
    s = s.replace(a, b)
open(p, 'w', encoding='utf-8').write(s)
esprima.parseScript(re.search(r'<script>(.*)</script>', s, re.S).group(1))
print('ok')
