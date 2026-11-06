from pathlib import Path

def trocar(caminho, pares):
    p = Path(caminho)
    s = p.read_bytes().decode('utf-8')
    crlf = '\r\n' in s
    s = s.replace('\r\n', '\n')
    for a, b in pares:
        assert s.count(a) == 1, a[:70]
        s = s.replace(a, b)
    p.write_bytes((s.replace('\n', '\r\n') if crlf else s).encode('utf-8'))

trocar('demo/player.py', [
    ('''class Gatilho:
    """Só dispara com a mesma classe em N janelas seguidas e exige soltar o gesto para repetir."""

    def __init__(self, limiar, seguidas, pausa):
        self.limiar, self.seguidas, self.pausa = limiar, seguidas, pausa
        self.atual, self.contagem, self.fim_pausa, self.armado = None, 0, 0.0, True
''', '''class Gatilho:
    """Só dispara com a mesma classe em N janelas seguidas e exige soltar o gesto para repetir.

    Os gestos de `repetiveis` são a exceção: mantidos, voltam a disparar a cada `repetir` segundos.
    """

    def __init__(self, limiar, seguidas, pausa, repetiveis=(), repetir=0.8):
        self.limiar, self.seguidas, self.pausa = limiar, seguidas, pausa
        self.repetiveis, self.repetir = set(repetiveis), repetir
        self.atual, self.contagem, self.fim_pausa, self.armado = None, 0, 0.0, True
        self.proxima_repeticao, self.foi_repeticao = 0.0, False
'''),
    ('''        self.contagem += 1
        if self.armado and self.contagem >= self.seguidas and agora >= self.fim_pausa:
            self.armado, self.fim_pausa = False, agora + self.pausa
            return nome
        return None
''', '''        self.contagem += 1
        self.foi_repeticao = False
        if self.armado and self.contagem >= self.seguidas and agora >= self.fim_pausa:
            self.armado, self.fim_pausa = False, agora + self.pausa
            self.proxima_repeticao = agora + self.repetir
            return nome
        if not self.armado and nome in self.repetiveis and agora >= self.proxima_repeticao:
            self.proxima_repeticao, self.fim_pausa = agora + self.repetir, agora + self.pausa
            self.foi_repeticao = True
            return nome
        return None
'''),
    ('''    gatilho = Gatilho(args.limiar, args.seguidas, args.pausa)
''', '''    gatilho = Gatilho(args.limiar, args.seguidas, args.pausa, COMANDOS_VOLUME if args.painel else (), args.repetir)
'''),
    ('''                                 "confianca": confianca})
''', '''                                 "confianca": confianca, "repeticao": gatilho.foi_repeticao})
'''),
    ('''COMANDOS_PAINEL = {"Thumb Up": "volume_mais", "Thumb Down": "volume_menos", "Shaking Hand": "acenar"}
''', '''COMANDOS_VOLUME = {"Thumb Up": "volume_mais", "Thumb Down": "volume_menos"}
COMANDOS_PAINEL = {**COMANDOS_VOLUME, "Shaking Hand": "acenar"}
'''),
    ('''    a.add_argument("--painel", action''', '''    a.add_argument("--repetir", type=float, default=0.8, help="Segundos entre repetições de um polegar mantido, no painel")
    a.add_argument("--painel", action'''),
])
trocar('demo/painel_carro.html', [
    ('''  if (e.tipo === "comando") executar(e.comando);
''', '''  // Manter o polegar só repete o comando dentro do menu; o volume muda um passo por gesto.
  if (e.tipo === "comando" && (!e.repeticao || menuAberto)) executar(e.comando);
'''),
])

# Teste da regra do gatilho, sem câmera.
import importlib.util, sys
sys.path.insert(0, 'demo')
spec = importlib.util.spec_from_file_location('player_teste', 'demo/player.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.COMANDOS.update(m.COMANDOS_PAINEL)
g = m.Gatilho(0.7, 3, 1.2, m.COMANDOS_VOLUME, 0.8)
disparos = []
t = 0.0
def passo(nome, segundos):
    global t
    for _ in range(int(segundos / 0.15)):
        d = g.atualizar(nome, 0.95, t)
        if d: disparos.append((round(t, 2), d, g.foi_repeticao))
        t += 0.15
passo('Thumb Down', 3.0); print('polegar mantido 3 s:', disparos); disparos.clear()
passo('No gesture', 1.5); passo('Stop Sign', 3.0); print('pare mantido 3 s:', disparos); disparos.clear()
passo('No gesture', 1.5); passo('Thumb Up', 0.6); print('polegar rapido:', disparos); disparos.clear()
