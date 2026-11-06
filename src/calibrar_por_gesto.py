"""Compara um limiar único (0,7 e 3 janelas) com um limiar para cada gesto, só na validação do Jester.

Usa o mesmo fluxo simulado de calibrar_fluxo.py. Os limiares são escolhidos em validação cruzada:
ajuste em 4/5 das janelas e avaliação no 1/5 restante, para o ganho medido não vir de sobreajuste.
Falsos disparos são raros, então o custo de cada limiar também conta janelas isoladas sem gesto
acima dele, um sinal mais denso do risco. O conjunto de teste não é usado.

Uso:
    python src/calibrar_por_gesto.py --modelo resultados/jester/cnn_3d/melhor.pt
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import calibrar_fluxo as C
import jester as J
import modelos as M
import treinar_jester as T

GRADE = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98])
LIMIAR_BASE, N_BASE = 0.7, 3
PESO_ISOLADAS = 0.1  # peso das janelas isoladas no custo de risco
N_DOBRAS = 5
COMANDOS_DEMO = ["Swiping Left", "Swiping Right", "Stop Sign", "Thumb Up", "Thumb Down"]


def resumir(p, negativos):
    """Reduz as probabilidades à melhor classe de gesto e sua confiança, por janela."""
    gesto = p.copy()
    gesto[:, negativos] = 0
    return gesto.argmax(1).astype(np.int16), gesto.max(1).astype(np.float32)


def extrair_janelas(args):
    """Passa o modelo no fluxo sem gesto e nos vídeos com gesto, em blocos, e guarda só o resumo."""
    cache = Path(args.cache)
    if cache.exists():
        return dict(np.load(cache, allow_pickle=False))
    estado = torch.load(args.modelo, map_location="cuda", weights_only=False)
    nomes = estado["nomes"]
    modelo = M.criar_modelo(estado["arquitetura"], len(nomes)).cuda().eval()
    modelo.load_state_dict(estado["modelo"])
    negativos = [nomes.index(n) for n in T.SEM_GESTO]
    _, _, validacao, _ = J.ler_rotulos(Path(args.dados) / "rotulos/jester_labels")
    conjunto = J.ConjuntoJester(args.dados, validacao, treino=False)

    sem_gesto = [i for i, s in enumerate(conjunto.amostras) if s[3] in negativos]
    com_gesto = [i for i, s in enumerate(conjunto.amostras) if s[3] not in negativos]
    frames, _ = C.decodificar(conjunto, sem_gesto)
    minutos = len(frames) / C.FPS_ORIGEM / 60
    p, _ = C.prever_fluxo(modelo, frames)
    classe_neg, conf_neg = resumir(p, negativos)
    del frames, p
    print(f"Fluxo sem gesto: {len(sem_gesto)} vídeos, {minutos:.1f} min, {len(conf_neg)} janelas", flush=True)

    classes, confs, videos = [], [], []
    for b in range(0, len(com_gesto), args.bloco):
        parte = com_gesto[b:b + args.bloco]
        frames, inicios = C.decodificar(conjunto, parte)
        p, pontos = C.prever_fluxo(modelo, frames)
        classe, conf = resumir(p, negativos)
        classes.append(classe); confs.append(conf)
        videos.append(b + np.searchsorted(inicios, pontos + 18, side="right") - 1)
        print(f"Vídeos com gesto: {b + len(parte)}/{len(com_gesto)}", flush=True)
        del frames, p
    dados = {"classe_neg": classe_neg, "conf_neg": conf_neg, "classe_pos": np.concatenate(classes),
             "conf_pos": np.concatenate(confs), "video_pos": np.concatenate(videos),
             "alvo": np.array([conjunto.amostras[i][3] for i in com_gesto]), "minutos": np.array(minutos),
             "nomes": np.array(nomes), "negativos": np.array(negativos)}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **dados)
    return dados


def disparos(classe, conf, limiares, n_seguidas, pausa=C.PAUSA_JANELAS):
    """Mesma regra de calibrar_fluxo.contar_disparos, com limiar por classe. Devolve (janelas, classes)."""
    janelas = np.nonzero(conf >= limiares[classe])[0]
    saida, seguidas, atual, anterior, bloqueio = [], 0, -1, -2, -1
    for i in janelas.tolist():
        c = int(classe[i])
        seguidas = seguidas + 1 if i == anterior + 1 and c == atual else 1
        atual, anterior = c, i
        if seguidas >= n_seguidas and i > bloqueio:
            saida.append(i)
            bloqueio = i + pausa
            seguidas = 0
    saida = np.array(saida, dtype=np.int64)
    return saida, classe[saida] if len(saida) else np.array([], dtype=np.int16)


def avaliar_positivos(d, limiares):
    """Por vídeo: se houve disparo da classe certa, e quantos disparos errados houve em cada vídeo."""
    janelas, classes = disparos(d["classe_pos"], d["conf_pos"], limiares, N_BASE)
    videos = d["video_pos"][janelas]
    certos = classes == d["alvo"][videos]
    acerto = np.zeros(len(d["alvo"]), dtype=bool)
    acerto[videos[certos]] = True
    errados = np.bincount(videos[~certos], minlength=len(d["alvo"]))
    return acerto, errados


def curvas(d, classes_gesto):
    """Eventos de cada gesto em cada limiar da grade, mantendo os outros gestos desligados."""
    total = len(d["nomes"])
    eventos = {}
    for c in classes_gesto:
        for l in GRADE:
            limiares = np.full(total, np.inf)
            limiares[c] = l
            neg3, _ = disparos(d["classe_neg"], d["conf_neg"], limiares, N_BASE)
            neg1, _ = disparos(d["classe_neg"], d["conf_neg"], limiares, 1, pausa=0)
            pos, _ = disparos(d["classe_pos"], d["conf_pos"], limiares, N_BASE)
            videos = d["video_pos"][pos]
            eventos[c, float(l)] = (neg3, neg1, videos)
    return eventos


def escolher(d, eventos, classes_gesto, treino_neg, treino_video):
    """Escolhe os limiares que mais ganham sem passar do risco do limiar único, nos dados de ajuste."""
    n = len(GRADE)
    ganho, risco = np.zeros((len(classes_gesto), n)), np.zeros((len(classes_gesto), n))
    for a, c in enumerate(classes_gesto):
        for b, l in enumerate(GRADE):
            neg3, neg1, videos = eventos[c, float(l)]
            risco[a, b] = treino_neg[neg3].sum() + PESO_ISOLADAS * treino_neg[neg1].sum()
            usados = videos[treino_video[videos]]
            certos = np.unique(usados[d["alvo"][usados] == c]).size
            ganho[a, b] = certos - (d["alvo"][usados] != c).sum()
    base = int(np.argmin(np.abs(GRADE - LIMIAR_BASE)))
    orcamento = risco[:, base].sum()
    melhor, melhor_ganho = np.full(len(classes_gesto), base), ganho[:, base].sum()
    for peso in np.geomspace(0.05, 2000, 80):
        escolha = (ganho - peso * risco).argmax(1)
        linhas = np.arange(len(classes_gesto))
        if risco[linhas, escolha].sum() <= orcamento and ganho[linhas, escolha].sum() > melhor_ganho:
            melhor, melhor_ganho = escolha, ganho[linhas, escolha].sum()
    limiares = np.full(len(d["nomes"]), np.inf)
    limiares[classes_gesto] = GRADE[melhor]
    return limiares


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--modelo", required=True)
    a.add_argument("--dados", default="data/Jester")
    a.add_argument("--bloco", type=int, default=1500, help="Vídeos com gesto decodificados por vez")
    a.add_argument("--cache", default="data/interno/janelas_validacao_cnn_3d.npz")
    a.add_argument("--semente", type=int, default=42)
    args = a.parse_args()
    d = extrair_janelas(args)
    nomes = [str(n) for n in d["nomes"]]
    negativos = d["negativos"].tolist()
    classes_gesto = [i for i in range(len(nomes)) if i not in negativos]
    minutos = float(d["minutos"])
    gerador = np.random.default_rng(args.semente)

    unico = np.full(len(nomes), np.inf)
    unico[classes_gesto] = LIMIAR_BASE

    # Conferência: nos primeiros 1500 vídeos e no fluxo inteiro, deve repetir calibracao_fluxo.csv.
    falsos_base, _ = disparos(d["classe_neg"], d["conf_neg"], unico, N_BASE)
    acerto_base, errados_base = avaliar_positivos(d, unico)
    print(f"Conferência com o CSV: {len(falsos_base)} falsos; {acerto_base[:args.bloco].mean():.3f} nos primeiros {args.bloco} vídeos", flush=True)

    eventos = curvas(d, classes_gesto)
    print("Curvas por gesto prontas", flush=True)

    n_neg, n_vid = len(d["conf_neg"]), len(d["alvo"])
    dobra_neg = np.minimum((np.arange(n_neg) * N_DOBRAS) // n_neg, N_DOBRAS - 1)  # blocos contíguos do fluxo
    dobra_vid = gerador.permutation(n_vid) % N_DOBRAS

    acerto_novo, errados_novo = np.zeros(n_vid, dtype=bool), np.zeros(n_vid, dtype=np.int64)
    falsos_novo, por_dobra = 0, []
    for k in range(N_DOBRAS):
        limiares = escolher(d, eventos, classes_gesto, dobra_neg != k, dobra_vid != k)
        janelas, _ = disparos(d["classe_neg"], d["conf_neg"], limiares, N_BASE)
        falsos_k = int((dobra_neg[janelas] == k).sum())
        acerto, errados = avaliar_positivos(d, limiares)
        fora = dobra_vid == k
        acerto_novo[fora], errados_novo[fora] = acerto[fora], errados[fora]
        falsos_novo += falsos_k
        base_k = int((dobra_neg[falsos_base] == k).sum())
        por_dobra.append({"dobra": k, "falsos_unico": base_k, "falsos_por_gesto": falsos_k,
                          "det_unico": float(acerto_base[fora].mean()), "det_por_gesto": float(acerto[fora].mean())})
        print(f"Dobra {k}: falsos {base_k} -> {falsos_k}; detecção {acerto_base[fora].mean():.3f} -> {acerto[fora].mean():.3f}", flush=True)

    amostras = gerador.integers(0, n_vid, (2000, n_vid))
    diferencas = acerto_novo[amostras].mean(1) - acerto_base[amostras].mean(1)
    ic = [float(x) for x in np.percentile(diferencas, [2.5, 97.5])]

    final = escolher(d, eventos, classes_gesto, np.ones(n_neg, dtype=bool), np.ones(n_vid, dtype=bool))
    resumo = {
        "minutos_sem_gesto": minutos, "n_videos_com_gesto": n_vid, "dobras": por_dobra,
        "unico": {"falsos": len(falsos_base), "falsos_por_min": len(falsos_base) / minutos,
                  "deteccao": float(acerto_base.mean()), "classe_errada": float(errados_base.sum() / n_vid)},
        "por_gesto_fora_da_amostra": {"falsos": falsos_novo, "falsos_por_min": falsos_novo / minutos,
                                      "deteccao": float(acerto_novo.mean()), "classe_errada": float(errados_novo.sum() / n_vid)},
        "diferenca_deteccao_ic95": ic,
        "limiares_finais": {nomes[c]: float(final[c]) for c in classes_gesto},
    }
    pasta = Path(args.modelo).parent
    (pasta / "limiar_por_gesto.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
    with (pasta / "limiar_por_gesto.csv").open("w", newline="", encoding="utf-8") as f:
        e = csv.writer(f)
        e.writerow(["gesto", "n", "det_unico", "det_por_gesto_fora_da_amostra", "limiar_final", "comando_da_demo"])
        for c in classes_gesto:
            m = d["alvo"] == c
            e.writerow([nomes[c], int(m.sum()), round(float(acerto_base[m].mean()), 4), round(float(acerto_novo[m].mean()), 4),
                        float(final[c]), nomes[c] in COMANDOS_DEMO])

    u, g = resumo["unico"], resumo["por_gesto_fora_da_amostra"]
    print(f"\nFluxo de {minutos:.1f} min; {n_vid} vídeos com gesto")
    print(f"Limiar único:    {u['falsos']} falsos ({u['falsos_por_min']:.3f}/min); detecção {u['deteccao']:.1%}; classe errada {u['classe_errada']:.1%}")
    print(f"Limiar por gesto: {g['falsos']} falsos ({g['falsos_por_min']:.3f}/min); detecção {g['deteccao']:.1%}; classe errada {g['classe_errada']:.1%}")
    print(f"Diferença de detecção: {g['deteccao'] - u['deteccao']:+.1%} (IC 95% {ic[0]:+.1%} a {ic[1]:+.1%})")
    print("\nComandos da demo:")
    for n in COMANDOS_DEMO:
        if n in nomes:
            c = nomes.index(n)
            m = d["alvo"] == c
            print(f"  {n:15s} limiar {final[c]:.2f}; detecção {acerto_base[m].mean():.1%} -> {acerto_novo[m].mean():.1%}")


if __name__ == "__main__":
    main()
