"""Redes de vídeo treinadas do zero, com inicialização aleatória e sem pesos de terceiros.

Entrada: tensor (lote, tempo, canais, altura, largura).
"""
import torch
from torch import nn


def _bloco_2d(entrada, saida):
    return nn.Sequential(nn.Conv2d(entrada, saida, 3, padding=1, bias=False), nn.BatchNorm2d(saida), nn.ReLU(inplace=True),
                         nn.Conv2d(saida, saida, 3, padding=1, bias=False), nn.BatchNorm2d(saida), nn.ReLU(inplace=True),
                         nn.MaxPool2d(2))


class CNNGRU(nn.Module):
    """CNN 2D aplicada a cada frame, seguida de uma GRU sobre a sequência."""

    def __init__(self, n_classes, canais_entrada=3, canais=(32, 64, 128, 256), oculta=256, dropout=0.3):
        super().__init__()
        camadas, entrada = [], canais_entrada
        for saida in canais:
            camadas.append(_bloco_2d(entrada, saida))
            entrada = saida
        self.corpo = nn.Sequential(*camadas, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.gru = nn.GRU(entrada, oculta, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.cabeca = nn.Linear(oculta, n_classes)

    def forward(self, x):
        lote, tempo = x.shape[:2]
        h = self.corpo(x.flatten(0, 1)).view(lote, tempo, -1)
        saida, _ = self.gru(h)
        return self.cabeca(self.dropout(saida.mean(dim=1)))


class CNNTransformer(nn.Module):
    """A mesma CNN 2D por frame da CNNGRU, com atenção temporal no lugar da GRU."""

    def __init__(self, n_classes, canais_entrada=3, canais=(32, 64, 128, 256), camadas_atencao=2,
                 cabecas=4, n_frames=16, dropout=0.3):
        super().__init__()
        blocos, entrada = [], canais_entrada
        for saida in canais:
            blocos.append(_bloco_2d(entrada, saida))
            entrada = saida
        self.corpo = nn.Sequential(*blocos, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.posicao = nn.Parameter(torch.zeros(1, n_frames, entrada))
        nn.init.trunc_normal_(self.posicao, std=0.02)
        camada = nn.TransformerEncoderLayer(entrada, cabecas, dim_feedforward=2 * entrada, dropout=0.1,
                                            batch_first=True, norm_first=True)
        self.atencao = nn.TransformerEncoder(camada, camadas_atencao, enable_nested_tensor=False)
        self.dropout = nn.Dropout(dropout)
        self.cabeca = nn.Linear(entrada, n_classes)

    def forward(self, x):
        lote, tempo = x.shape[:2]
        h = self.corpo(x.flatten(0, 1)).view(lote, tempo, -1)
        h = self.atencao(h + self.posicao[:, :tempo])
        return self.cabeca(self.dropout(h.mean(dim=1)))


class CNN3D(nn.Module):
    """Convolução no espaço e no tempo. O tempo só é reduzido nas camadas finais."""

    def __init__(self, n_classes, canais_entrada=3, canais=(32, 64, 128, 256), dropout=0.3):
        super().__init__()
        camadas, entrada = [], canais_entrada
        for i, saida in enumerate(canais):
            passo_tempo = 2 if i >= 1 else 1
            camadas += [nn.Conv3d(entrada, saida, 3, padding=1, bias=False), nn.BatchNorm3d(saida), nn.ReLU(inplace=True),
                        nn.Conv3d(saida, saida, 3, padding=1, bias=False), nn.BatchNorm3d(saida), nn.ReLU(inplace=True),
                        nn.MaxPool3d((passo_tempo, 2, 2))]
            entrada = saida
        self.corpo = nn.Sequential(*camadas, nn.AdaptiveAvgPool3d(1), nn.Flatten())
        self.dropout = nn.Dropout(dropout)
        self.cabeca = nn.Linear(entrada, n_classes)

    def forward(self, x):
        return self.cabeca(self.dropout(self.corpo(x.transpose(1, 2))))


def _conv3d(entrada, saida):
    return nn.Sequential(nn.Conv3d(entrada, saida, 3, padding=1, bias=False), nn.BatchNorm3d(saida))


class Conv2mais1D(nn.Module):
    """Convolução (2+1)D: primeiro no espaço, depois no tempo, com cerca dos parâmetros de uma 3x3x3.

    Feita com Conv2d e Conv1d comuns: as Conv3d "achatadas" (1x3x3 e 3x1x1) ficaram dezenas de vezes mais lentas nesta GPU.
    """

    def __init__(self, entrada, saida):
        super().__init__()
        meio = (27 * entrada * saida) // (9 * entrada + 3 * saida)
        meio = max(16, round(meio / 16) * 16)
        self.espaco = nn.Conv2d(entrada, meio, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(meio)
        self.tempo = nn.Conv1d(meio, saida, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(saida)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        n, c, t, h, w = x.shape
        x = self.espaco(x.transpose(1, 2).reshape(n * t, c, h, w))
        x = self.relu(self.bn1(x.view(n, t, -1, h, w).transpose(1, 2).contiguous()))
        m = x.shape[1]
        x = self.tempo(x.permute(0, 3, 4, 1, 2).reshape(n * h * w, m, t))
        return self.bn2(x.view(n, h, w, -1, t).permute(0, 3, 4, 1, 2).contiguous())


def _conv2mais1d(entrada, saida):
    return Conv2mais1D(entrada, saida)


class BlocoResidual(nn.Module):
    """Duas convoluções com um atalho: a entrada é somada à saída, o que facilita o treino de redes mais fundas."""

    def __init__(self, entrada, saida, conv):
        super().__init__()
        self.a, self.b, self.relu = conv(entrada, saida), conv(saida, saida), nn.ReLU(inplace=True)
        self.atalho = nn.Identity() if entrada == saida else nn.Sequential(nn.Conv3d(entrada, saida, 1, bias=False), nn.BatchNorm3d(saida))

    def forward(self, x):
        return self.relu(self.b(self.relu(self.a(x))) + self.atalho(x))


class Res3D(nn.Module):
    """Mesma estrutura da CNN3D, com blocos residuais; a convolução pode ser 3D ou (2+1)D."""

    def __init__(self, n_classes, canais_entrada=3, canais=(32, 64, 128, 256), fatorada=False, dropout=0.3):
        super().__init__()
        conv, camadas, entrada = (_conv2mais1d if fatorada else _conv3d), [], canais_entrada
        for i, saida in enumerate(canais):
            camadas += [BlocoResidual(entrada, saida, conv), nn.MaxPool3d((2 if i >= 1 else 1, 2, 2))]
            entrada = saida
        self.corpo = nn.Sequential(*camadas, nn.AdaptiveAvgPool3d(1), nn.Flatten())
        self.dropout = nn.Dropout(dropout)
        self.cabeca = nn.Linear(entrada, n_classes)

    def forward(self, x):
        return self.cabeca(self.dropout(self.corpo(x.transpose(1, 2))))


ARQUITETURAS = {
    "cnn_3d": lambda n, c: CNN3D(n, c),
    "cnn_3d_leve": lambda n, c: CNN3D(n, c, canais=(16, 32, 64, 128)),
    "cnn_3d_larga": lambda n, c: CNN3D(n, c, canais=(48, 96, 192, 384)),
    "res_3d": lambda n, c: Res3D(n, c),
    # Com lote de 48, a (2+1)D passa dos 8 GB da GPU e fica dezenas de vezes mais lenta: ficou fora da comparação.
    "r2plus1d": lambda n, c: Res3D(n, c, fatorada=True),
    "cnn_gru": lambda n, c: CNNGRU(n, c),
    "cnn_transformer": lambda n, c: CNNTransformer(n, c),
}


def criar_modelo(nome, n_classes, canais_entrada=3):
    if nome not in ARQUITETURAS:
        raise ValueError(nome)
    return ARQUITETURAS[nome](n_classes, canais_entrada)


def contar_parametros(modelo):
    return sum(p.numel() for p in modelo.parameters())


def adaptar_entrada(modelo, canais_novos):
    """Troca os canais da primeira convolução, repetindo a média dos pesos aprendidos."""
    primeira = next(m for m in modelo.modules() if isinstance(m, (nn.Conv2d, nn.Conv3d)))
    with torch.no_grad():
        media = primeira.weight.mean(dim=1, keepdim=True)
        forma = [1] * media.dim()
        forma[1] = canais_novos
        primeira.weight = nn.Parameter(media.repeat(*forma) * primeira.in_channels / canais_novos)
    primeira.in_channels = canais_novos
    return modelo
