# Passagem de sessão: 16/11 à noite (entrega 17–18/11)

Projeto: controle por gestos no carro (Insper, Deep Learning; Enzo, Isabela, Maria Luiza). Tudo em português.

## Como rodar
- Python só via `demo\python.bat` ou `demo\python_pontos.bat` (o da `.venv` é bloqueado pelo Windows).
- Apresentação final: `demo\abrir_apresentacao_final.bat` (porta 8767). O slide 8 é a demo do carro.
- Carro sozinho: `demo\abrir_carro3d.bat` (porta 8772).
- Mão 3D: `demo\abrir_mao3d.bat` (porta 8774), com o modelo novo de mão (`demo/mao_modelo.js` e `demo/mao/*.glb`, licença MIT).
- Teste guiado na webcam: `demo\abrir_teste_ao_vivo.bat` (porta 8776). Grava em `resultados/webcam_ao_vivo/`. O relatório sai com `src/relatorio_teste_ao_vivo.py`. Nenhuma sessão foi gravada ainda.

**Segurança:** o notebook deu choque pela saída de ar com o ventilador no máximo, durante os treinos. Não treinar nem usar a GPU pesada sem o usuário confirmar bateria ou tomada aterrada. Inferência de cerca de 10 minutos foi autorizada na bateria.

## Modelo ao vivo hoje (`demo/player.py`, `laco_pontos` e `Pontos.combinar`)
- **Movimentos:** CNN 3D (`resultados/pontos/modelo_final/cnn.pt`, 36 classes). Dispara com ≥ 0,7 em 3 análises seguidas, a cada 0,15 s, com janelas de 1, 1,5 e 3 s.
- **Números (1, 2, 4):** MediaPipe (21 pontos) e uma regra geométrica (`src/pontos_mao.py`).
  - Mão parada: o pulso andou ≤ 0,5 palma em 6 quadros.
  - Mesma contagem em 5 de 6 quadros: o número passa direto, sem porcentagem (`Gatilho._pela_regra`).
  - O 0,9 fixo foi removido.
- **Veto da CNN: DESLIGADO na noite de 16/11** (`Pontos.VETO_CNN = False`). O que ele fazia: as classes que o carro não usa são somadas em "Outros gestos". Se, na janela mais longa, a CNN aponta "Outros gestos" ou um movimento do carro, o número é ignorado. "Sem gesto" e números não vetam.
- **GRU:** não roda ao vivo. `--pontos gru.pt` continua obrigatório, só para ligar esse caminho.
- **Janela do OK** (no navegador, `carro3d.html`): abre por 7 s.

## RESOLVIDO na noite de 16/11: o veto foi desligado
Teste de 2 minutos na webcam, sem veto: os números voltaram (um dedo 6, dois dedos 5, quatro dedos 11 disparos), e o usuário confirmou. Em todas as 116 leituras de número, o veto teria derrubado o número. O modelo final é: CNN 3D + contagem de dedos, sem veto. Os slides já mostram essa versão (70,9% e 1,62 falsos/min).

Ativação: a janela abriu 8 vezes, todas pelo detector do OK no navegador; três vieram com 7,0 s entre si (reabriu ao fechar). Não foi alterada. Cada abertura é gravada em `resultados/webcam_ao_vivo/ativacao_*.json` (diagnóstico em `carro3d.html`, `registrarAtivacao`).

## Problema original (histórico)
O usuário testou o carro na webcam com o veto: **os números (1, 2, 4) não estão sendo reconhecidos**, e às vezes **a ativação abre sozinha**. Na manhã de 16/11, sem veto, os números funcionavam bem.

- **Suspeita para os números:** o veto compara a soma de cerca de 17 classes ("Outros gestos") com uma única classe. Com a CNN em dúvida na webcam, a soma ganha e veta o número.
- **Correção mais simples:** desligar o veto. Em `Pontos.combinar`, a linha `veto = outros_soma > ...`; dá para pôr atrás de uma opção. Depois, testar na webcam.
- **Alternativa:** vetar só quando a classe mais alta das 36, sem somar, é um gesto não usado ou um movimento.
- **Ativação sozinha:** ainda não diagnosticada. O OK é detectado no navegador (`carro3d.html`: pinça < 0,5 palma, três dedos retos > 0,78, mantido 0,4 s). Ligação recebida também abre a janela.
- **Para diagnosticar:** gravar `curl -N http://127.0.0.1:8767/eventos` enquanto o usuário faz os gestos. O evento "estado" traz `pontos.regra` (contagem) e `pontos.numero` (o número depois do veto). Se a regra dá número e o número sai null, foi o veto. A primeira tentativa falhou porque a apresentação estava fechada.

## Resultados medidos (todos sem a janela do OK)

| Versão | Acerto, teste misto (36 classes) | Falsos/min (IPN) | 1 dedo (trechos) | 2 dedos (trechos) |
|---|---|---|---|---|
| CNN sozinha | 84,0% | 0,19 | 5/265 | 0/264 |
| Números pela GRU (só nos dados; ao vivo não disparava) | 92,0% | 0,25 | 251/265 | 240/264 |
| **Contagem sem veto (MODELO FINAL, confirmado ao vivo)** | 70,9% | 1,62 | 228/265 | 242/264 |
| Contagem + veto (testada, desligada) | 87,9% | 1,24 | 182/265 | 180/264 |
| Contagem + veto com "sem gesto" (removido) | 88,3% | 0,35 | 170/265 | 163/264 |

- **Confiança da contagem** (quando diz k, é k; 1 / 2 / 4 dedos): 74 / 23 / 53% sem veto; 98 / 84 / 94% com o veto atual.
- **Arquivos:**
  - teste por clipe: `src/medir_veto_outros.py` → `resultados/pontos/veto_outros.json`;
  - vídeo contínuo: `src/fluxo_ao_vivo.py` → `fluxo_ao_vivo*.json`;
  - acerto sem veto: `src/medir_modelo_ao_vivo.py`;
  - confiança da regra: `src/medir_confianca_numeros.py`.

**Importante:** o acerto nos dados e o funcionamento na webcam divergem. A GRU foi a melhor nos dados e não funcionou ao vivo; o veto melhorou os dados e parece ter quebrado os números ao vivo. Decidir pela demo ao vivo.

## Apresentação final
- **Arquivos:** `demo/apresentacao_final_modelo.html` mais o gerador `demo/gerar_apresentacao_final.py`, que lê os números de `resultados/` e gera `demo/apresentacao_final.html`. São 33 slides.
- **Estado:** os slides mostram o modelo final, sem veto (70,9% e 1,62/min); o veto aparece como versão testada. Histórico: para trocar, mudaram-se no gerador as chaves `vo.get("veto B (ao vivo)...")` e `fvar.get("...veto da CNN (janela mais longa)")` pela versão sem veto. Também revisar os textos de "veto" em "Como funciona" (slide 5), A7, A9, A10 e Limitações.
- **Slide 13 "Se tivéssemos mais tempo":** pronto (GRU treinada com vídeos da webcam; placa FPGA Kria KV260 com câmera infravermelha, em ilustração SVG).
- **Estilo exigido:** formal, só tons de azul, sem emojis, sem tabelas arredondadas. Nunca inventar números.

## Apresentação refeita em 16/11 à noite (38 slides)
- **Destaque do corpo:** só o modelo final. Acerto de 87,7% nos 11 gestos do carro (`src/medir_gestos_do_carro.py` → `resultados/pontos/gestos_do_carro.json`); 1,05 falsos/min só com os comandos do carro.
- **Slides refeitos:** Como funciona (5), Os gestos (7), Resultados (9 e 10). As comparações entre modelos foram para o apêndice.
- **Apêndice novo:** A4 arquitetura da CNN 3D; A7 como contamos os dedos; A8 por que não o de maior acerto, arquiteturas do zero e linha do tempo.
- **Gráficos e tabelas novos:** `demo/graficos_finais.py`, chamado pelo gerador.
- **Roteiro das falas:** `ROTEIRO_APRESENTACAO.md` (3 pessoas, tempos e perguntas prováveis).
- **Falta:** ensaio completo pela `demo\abrir_apresentacao_final.bat`; os slides novos só foram vistos por um servidor de arquivos, sem a câmera.

## Retorno do grupo aplicado em 16/11 à noite (34 slides)
- **Corpo (13 slides):** 8 é novo ("A ativação pelo OK"); saíram Segurança e Limitações; 12 é "Próximos passos" (4 itens curtos).
- **Apêndice:** saíram A4 arquitetura, A7 combinações / modelo ao vivo / confiança e A13 riscos. Entraram A1 "Jester e preparo" e A7 "MediaPipe e mão parada". A14 é Referências.
- **Mesma régua (11 gestos do carro):** GRU + CNN 95,6%; contagem + veto 86,2%; modelo final 87,7%; CNN sozinha 81,1% (`gestos_do_carro.json`, `gestos_do_carro_gru.json`). Nos dados a GRU + CNN é melhor; na webcam só o modelo final reconheceu os números.
- **Janela do OK (simulação, `src/fluxo_janela_ok.py`):** 1,05 → 0,73 comando falso por minuto; o OK abre sozinho 3,08 vezes por minuto no IPN. Proposta não aplicada: pinça < 0,3 e 0,6 s (0,19 falso por minuto). Precisa de teste ao vivo.
- **Pendente:** foto da câmera para o A12 (o grupo ficou de mandar o arquivo); decidir se aperta o OK.

## Regras do usuário
- Decidir, avançar e reportar; respostas curtas e didáticas.
- Registrar achados no `notebook.ipynb`.
- Não baixar nada sem dizer o arquivo, a fonte e o tamanho.
- Não mexer em configurações do sistema.
- EgoGesture descartada.
- ASL Citizen e DriverMHG: sem redistribuição.
- Sem imagens de pessoas das bases nos slides.

O último commit é `200a21b`.
