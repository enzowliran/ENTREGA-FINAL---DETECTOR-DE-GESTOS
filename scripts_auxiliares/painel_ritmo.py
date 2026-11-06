from pathlib import Path
import importlib.util, sys

p = Path('demo/player.py')
s = p.read_bytes().decode('utf-8')
crlf = '\r\n' in s
s = s.replace('\r\n', '\n')
pares = [
    ('''    Os gestos de `repetiveis` são a exceção: mantidos, voltam a disparar a cada `repetir` segundos.
    """
''', '''    Os gestos de `repetiveis` são a exceção: mantidos, voltam a disparar a cada `repetir` segundos.
    A repetição exige o gesto na janela mais curta (`presente`): as janelas longas ainda enxergam um
    gesto rápido alguns segundos depois de a mão sair, o que contaria como se ele continuasse.
    """
'''),
    ('repetiveis=(), repetir=0.8):', 'repetiveis=(), repetir=2.0):'),
    ('    def atualizar(self, nome, confianca, agora):\n', '    def atualizar(self, nome, confianca, agora, presente=True):\n'),
    ('if not self.armado and nome in self.repetiveis and agora >= self.proxima_repeticao:',
     'if not self.armado and presente and nome in self.repetiveis and agora >= self.proxima_repeticao:'),
    ('''            disparo = gatilho.atualizar(nome_atual, confianca, agora)
''', '''            curta = todas[0]  # lotes segue a ordem de --janelas, da mais curta para a mais longa
            presente = nomes[int(curta.argmax())] == nome_atual and float(curta.max()) >= args.limiar
            disparo = gatilho.atualizar(nome_atual, confianca, agora, presente)
'''),
    ('a.add_argument("--repetir", type=float, default=0.8, help="Segundos entre repetições de um polegar mantido, no painel")',
     'a.add_argument("--repetir", type=float, default=2.0, help="Segundos entre repetições de um polegar mantido, no painel")'),
]
for a, b in pares:
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)
p.write_bytes((s.replace('\n', '\r\n') if crlf else s).encode('utf-8'))

sys.path.insert(0, 'demo')
e = importlib.util.spec_from_file_location('player_teste', 'demo/player.py')
m = importlib.util.module_from_spec(e); e.loader.exec_module(m)
m.COMANDOS.update(m.COMANDOS_PAINEL)

def simular(presenca, duracao):
    g = m.Gatilho(0.7, 3, 1.2, m.COMANDOS_VOLUME, 2.0)
    t, saida = 0.0, []
    while t < duracao:
        if g.atualizar('Thumb Down', 0.95, t, presenca(t)):
            saida.append((round(t, 2), 'repete' if g.foi_repeticao else 'primeiro'))
        t += 0.15
    return saida

print('mantido 7 s:', simular(lambda t: True, 7))
print('polegar rapido, janela longa ainda ve por 4 s:', simular(lambda t: t < 1.0, 4))
