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
    ('"Pulling Hand In": "funcao"}', '"Pulling Hand In": "funcao", "Pushing Hand Away": "ajuda"}'),
    ('polegares, tamborilar e puxar a mão")', 'polegares, tamborilar, puxar e empurrar a mão")'),
])
trocar('demo/painel_carro.html', [
    ('// ---- Voz: ao trocar de tela, diz onde a pessoa está e o que cada gesto faz, para não precisar olhar ----',
     '// ---- Voz: ao trocar de tela, diz onde a pessoa está; os comandos só são lidos quando ela pede ----'),
    ('''function narrarTela(inicio) {
  const titulo = chamada ? (chamada.estado === "tocando" ? "Chamada de " + chamada.nome : "Em chamada") : menuAberto ? "Menu de músicas" : NOMES_PAGINA[pagina];
  falar([...(inicio ? [inicio] : []), titulo, ...comandosAtuais().map(([gesto, acao, falado]) => (falado || gesto) + ": " + acao)]);
}
''', '''function tituloTela() {
  return chamada ? (chamada.estado === "tocando" ? "Chamada de " + chamada.nome : "Em chamada") : menuAberto ? "Menu de músicas" : NOMES_PAGINA[pagina];
}
function narrarTela(inicio) { falar([...(inicio ? [inicio] : []), tituloTela()]); }
function explicarComandos() {
  falar([tituloTela(), ...comandosAtuais().map(([gesto, acao, falado]) => (falado || gesto) + ": " + acao)]);
}
'''),
    ('comandosAtuais().map(([g, a]) => `<b>${g}:</b> ${a}`).join("<br>");',
     '[...comandosAtuais(), ["Empurrar a mão", "explica os comandos"]].map(([g, a]) => `<b>${g}:</b> ${a}`).join("<br>");'),
    ('  let feito = true;\n',
     '  if (c === "ajuda") {\n    if (!repeticao) { explicarComandos(); avisar("Comandos de " + tituloTela()); $("nDisparos").textContent = ++comandos; }\n    return;\n  }\n  let feito = true;\n'),
    ('tocar(); narrarTela(); };', 'tocar(); falar([tituloTela(), "Empurre a mão para ouvir os comandos"]); };'),
    ('''  if (e.key.toLowerCase() === "m") executar("tamborilar");\n''',
     '''  if (e.key.toLowerCase() === "m") executar("tamborilar");\n  if (e.key.toLowerCase() === "e") executar("ajuda");\n'''),
])
print('ok')
