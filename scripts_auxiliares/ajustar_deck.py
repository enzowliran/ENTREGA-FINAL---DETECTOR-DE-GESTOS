p = 'demo/apresentacao_modelo.html'
s = open(p, encoding='utf-8').read()
trocas = [
    ('.grade2 { display: grid; grid-template-columns: 1fr 1fr; gap: 28px; flex: 1; min-height: 0; }',
     '.grade2 { display: grid; grid-template-columns: 1fr 1fr; gap: 28px; align-items: start; }'),
    ('  svg text { font-family: inherit; }', '  svg { display: block; width: 100%; height: auto; }\n  svg text { font-family: inherit; }'),
    ('<div class="cartao" style="padding:16px"><svg id="grafico" viewBox="0 0 680 370"',
     '<div class="cartao" style="padding:12px 16px"><svg id="grafico" viewBox="0 0 680 370"'),
    ('o.n.replace("Treino ", "") + " " + String(o.ult).replace(".", ",") + "%"',
     'o.n.replace("Treino ", "") + " " + o.ult.toFixed(1).replace(".", ",") + "%"'),
    ('Com cerca de 80% na validação, a webcam só reconhecia o joinha parado. Arrastar a mão rápido não funcionava.',
     'Com cerca de 80% na validação, o joinha só era reconhecido depois de segurar cerca de 10 s, e arrastar a mão rápido não funcionava.'),
]
for a, b in trocas:
    assert a in s, a[:50]
    s = s.replace(a, b)
open(p, 'w', encoding='utf-8').write(s)
print('ok')
