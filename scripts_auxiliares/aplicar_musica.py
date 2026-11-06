import re, sys
sp = sys.argv[1]
p = 'demo/player.html'
s = open(p, encoding='utf-8').read()
novo = open(sp + '/musica.js', encoding='utf-8').read()
i = s.index('const notas = n =>')
j = s.index('const capa = document.getElementById("capa");')
s = s[:i] + novo + s[j:]
open(p, 'w', encoding='utf-8').write(s)
import esprima
js = re.search(r'<script>(.*)</script>', s, re.S).group(1)
tree = esprima.parseScript(js)
nomes = [n.id.name for n in tree.body if n.type == 'FunctionDeclaration']
print('JS válido;', len(tree.body), 'declarações; funções:', len(nomes))
for f in ('iniciarAudio', 'agendar', 'tocar', 'pausar', 'alternar', 'trocar', 'atualizarInterface', 'criarMotivo', 'aleatorio'):
    assert f in nomes, f
print('faixas:', re.findall(r'titulo: "([^"]+)"', novo))
for m in re.finditer(r'(ritmoMelodia|kick|caixa|chimbal|baixo): "([^"]*)"', novo):
    pass
tam = [len(x) for x in re.findall(r'ritmoMelodia: "([^"]+)"', novo)]
print('motivos', tam)
