"""Motor da parte nova: CNN 3D para os gestos de movimento e pontos da mão (MediaPipe) para os números.

Usado por ao_vivo_pontos.py, player_pontos.py e testar_webcam_pontos.py. Reaproveita as funções de
ao_vivo.py (carregar o modelo, preparar o quadro) sem alterá-las.

A cada quadro da câmera, `SistemaPontos.processar`:
  1. guarda o quadro no buffer da CNN (12 por segundo, como sempre);
  2. roda o Hand Landmarker e passa a mão para o DecisorNumeros, que pode disparar um número;
  3. pega o resultado mais novo da CNN, que roda numa thread própria, e passa pelo gatilho de
     sempre. Se o gatilho disparar, o disparo só vale quando o decisor deixa (a mão não está
     parada numa pose de número, e nenhum número disparou há menos de `bloqueio` segundos).
     Os disparos barrados ficam contados em `suprimidos`.
"""
import collections
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

PASTA_DEMO = Path(__file__).resolve().parent
RAIZ = PASTA_DEMO.parent
sys.path.insert(0, str(PASTA_DEMO))
sys.path.insert(0, str(RAIZ / "src"))
import ao_vivo as V  # noqa: E402
import pontos_mao as P  # noqa: E402

MODELO_PADRAO = RAIZ / "resultados/jester/cnn_3d/melhor.pt"


class AnalisadorCNN:
    """A CNN 3D numa thread própria, sempre sobre os quadros mais recentes (como no modo sem NVIDIA das demos)."""

    def __init__(self, caminho, janelas=None, intervalo=0.15, suavizar=2, cpu=False):
        self.dispositivo = "cuda" if torch.cuda.is_available() and not cpu else "cpu"
        self.modelo, self.nomes, self.info = V.carregar_modelo(Path(caminho), self.dispositivo)
        self.janelas = janelas or ([1.0, 1.5, 3.0] if self.dispositivo == "cuda" else [1.0, 1.5])
        self.minimo = int(round(min(self.janelas) * V.QUADROS_POR_SEGUNDO))
        self.buffer = collections.deque(maxlen=int(round(max(self.janelas) * V.QUADROS_POR_SEGUNDO)))
        self.historico = collections.deque(maxlen=suavizar)
        self.indices_gesto = [i for i, n in enumerate(self.nomes) if n not in V.SEM_GESTO]
        self.intervalo, self.ultimo_quadro = intervalo, 0.0
        self.trava, self.parar, self.novo = threading.Lock(), threading.Event(), None
        self.thread = threading.Thread(target=self._rodar, daemon=True)
        self.thread.start()

    def adicionar(self, quadro, agora):
        if agora - self.ultimo_quadro >= 1 / V.QUADROS_POR_SEGUNDO:
            preparado = V.preparar_quadro(quadro)
            with self.trava:
                self.buffer.append(preparado)
            self.ultimo_quadro = agora

    @property
    def aquecendo(self):
        return len(self.buffer) < self.minimo

    def _rodar(self):
        ultima = 0.0
        while not self.parar.is_set():
            with self.trava:
                lista = list(self.buffer)
            if len(lista) < self.minimo or time.time() - ultima < self.intervalo:
                time.sleep(0.01)
                continue
            ultima = time.time()
            t0 = time.perf_counter()
            lotes = []
            for duracao in self.janelas:
                n = int(round(duracao * V.QUADROS_POR_SEGUNDO))
                if len(lista) >= n:
                    indices = np.round(np.linspace(len(lista) - n, len(lista) - 1, V.N_FRAMES)).astype(int)
                    lotes.append(np.stack([lista[i] for i in indices]))
            # Sem NVIDIA, sem autocast: em processadores sem bfloat16, uma análise passou de 4 minutos (ver ao_vivo.py).
            with torch.no_grad(), torch.autocast(self.dispositivo, dtype=torch.bfloat16, enabled=self.dispositivo == "cuda"):
                x = (torch.from_numpy(np.stack(lotes)).to(self.dispositivo) - 127.5) / 127.5
                todas = self.modelo(x).float().softmax(1).cpu().numpy()
            with self.trava:
                self.novo = (todas, (time.perf_counter() - t0) * 1000)

    def resultado(self):
        """Devolve (nome, confiança, média, presente, ms) quando há análise nova; senão None."""
        with self.trava:
            novo, self.novo = self.novo, None
        if novo is None:
            return None
        todas, ms = novo
        escolhida = int(np.argmax([q[self.indices_gesto].max() for q in todas]))
        self.historico.append(todas[escolhida])
        media = np.mean(self.historico, axis=0)
        melhor = int(media.argmax())
        nome = self.nomes[melhor]
        curta = todas[0]  # a janela mais curta, para saber se o gesto ainda está acontecendo (ver player.py)
        presente = (self.nomes[int(curta.argmax())] == nome, float(curta.max()))
        return nome, float(media[melhor]), media, presente, ms

    def fechar(self):
        self.parar.set()
        self.thread.join(timeout=2)


class SistemaPontos:
    """Junta a CNN e os pontos. `disparar_cnn(nome, confiança, agora, presente)` é o gatilho de quem usa."""

    def __init__(self, disparar_cnn, modelo=MODELO_PADRAO, janelas=None, intervalo=0.15, suavizar=2, cpu=False,
                 tempo_estavel=0.5, movimento=0.5, escala=0.25, bloqueio=1.0, limiar=0.7):
        self.limiar = limiar
        self.cnn = AnalisadorCNN(modelo, janelas, intervalo, suavizar, cpu)
        self.detector = P.DetectorMao(video=True)
        self.decisor = P.DecisorNumeros(tempo_estavel, movimento, escala, bloqueio)
        self.disparar_cnn = disparar_cnn
        self.cnn_nome, self.cnn_conf, self.cnn_media, self.ms = "Aquecendo", 0.0, None, 0.0
        self.ms_mao, self.suprimidos = 0.0, collections.Counter()

    def processar(self, quadro, agora):
        """Devolve {"disparo", "origem"} (origem "pontos" ou "cnn") e "suprimido" quando a CNN foi barrada."""
        self.cnn.adicionar(quadro, agora)
        t0 = time.perf_counter()
        mao = self.detector.detectar(quadro, agora)
        self.ms_mao = (time.perf_counter() - t0) * 1000
        altura, largura = quadro.shape[:2]
        evento = {"disparo": None, "origem": None, "suprimido": None}
        numero = self.decisor.atualizar(mao, agora, largura, altura)
        if numero:
            evento.update(disparo=numero, origem="pontos")

        r = self.cnn.resultado()
        if r is not None:
            self.cnn_nome, self.cnn_conf, self.cnn_media, (mesma, conf_curta), self.ms = r
            presente = mesma and conf_curta >= self.limiar
            # Um modelo com os números (cnn_3d_mais_gestos) também funciona: para o gatilho, número da CNN é "sem gesto".
            nome = "No gesture" if self.cnn_nome in P.NUMEROS else self.cnn_nome
            disparo = self.disparar_cnn(nome, self.cnn_conf, agora, presente)
            if disparo and evento["disparo"] is None:
                if self.decisor.cnn_pode_disparar(agora):
                    self.decisor.cnn_disparou(agora)
                    evento.update(disparo=disparo, origem="cnn")
                else:
                    self.suprimidos[disparo] += 1
                    evento["suprimido"] = disparo
        return evento

    def gesto_atual(self):
        """O que mostrar como "gesto atual": a pose de número quando a mão está parada nela, senão a CNN."""
        d = self.decisor
        if d.pose in P.NUMEROS and d.parado:
            return d.pose, max(d.progresso, 0.01)
        return self.cnn_nome, self.cnn_conf

    def fechar(self):
        self.cnn.fechar()
        self.detector.fechar()
