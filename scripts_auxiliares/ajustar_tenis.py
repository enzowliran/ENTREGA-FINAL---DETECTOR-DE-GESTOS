import re, esprima
p = 'demo/tenis.html'
s = open(p, encoding='utf-8').read()
trocas = [
    ('alturaGolpe: 1.05, ladoBola: 1.55 }', 'alturaGolpe: 1.05, ladoBola: 1.25 }'),
    ('jogador = criarPessoa(0xe6663c, 0xe7b48f); jogador.position.set(0, 0, QUADRA.zJogador); jogador.rotation.y = Math.PI;',
     'jogador = criarPessoa(0xe6663c, 0xe7b48f); jogador.position.set(0, 0, QUADRA.zJogador);'),
    ('rival = criarPessoa(0x3a4fa8, 0xb98563); rival.position.set(0, 0, QUADRA.zRival);',
     'rival = criarPessoa(0x3a4fa8, 0xb98563); rival.position.set(0, 0, QUADRA.zRival); rival.rotation.y = Math.PI;'),
    ('''  const f = Math.sin((1 - estado.balanco) * Math.PI);
  const ombro = jogador.userData.ombro;
  ombro.rotation.y = estado.balanco > 0 ? estado.ladoBalanco * (f * 2.2 - 0.3) : 0.35;
  jogador.rotation.y = Math.PI + (estado.balanco > 0 ? -estado.ladoBalanco * f * 0.5 : 0);
''', '''  const avanco = 1 - estado.balanco, f = Math.sin(avanco * Math.PI);
  const ombro = jogador.userData.ombro;
  // Ângulo do ombro: 0 aponta para a direita da tela e valores positivos giram para a frente.
  if (estado.balanco > 0) ombro.rotation.y = estado.ladoBalanco > 0 ? -0.7 + avanco * 3.1 : 2.7 - avanco * 3.2;
  else ombro.rotation.y = 0.35;
  jogador.rotation.y = estado.balanco > 0 ? estado.ladoBalanco * f * 0.5 : 0;
'''),
    ('''    const alvo = estado.para.x;
    rival.position.x''', '''    rival.position.x'''),
    ('''    return void alvo;
''', '''    return;
'''),
]
for a, b in trocas:
    assert a in s, a[:60]
    s = s.replace(a, b)
open(p, 'w', encoding='utf-8').write(s)
esprima.parseScript(re.findall(r'<script>(.*?)</script>', s, re.S)[0])
print('ok')
