from pathlib import Path

p = Path('demo/painel_carro.html')
s = p.read_bytes().decode('utf-8')
crlf = '\r\n' in s
s = s.replace('\r\n', '\n')
VOZ = '''// ---- Voz: ao trocar de tela, diz onde a pessoa está e o que cada gesto faz, para não precisar olhar ----
const temVoz = "speechSynthesis" in window;
let vozPt = null, vozesLidas = false, idFala = 0, tempoFala, tempoGuarda, falaAtual = null, avisouSemVoz = false;
function escolherVoz() {
  const todas = speechSynthesis.getVoices();
  if (!todas.length) return;
  vozesLidas = true;
  const br = todas.filter(v => /^pt[-_]BR/i.test(v.lang)), pt = br.length ? br : todas.filter(v => /^pt/i.test(v.lang));
  vozPt = pt.find(v => /natural|online|google/i.test(v.name)) || pt[0] || null;
}
if (temVoz) { escolherVoz(); speechSynthesis.addEventListener("voiceschanged", escolherVoz); }
function semVoz() {
  terminarFala();
  if (!avisouSemVoz) { avisouSemVoz = true; mostrarErro("Sem voz em português disponível neste navegador."); }
}
function terminarFala() { clearTimeout(tempoGuarda); falaAtual = null; fatorVoz = 1; aplicarGanho(); }
function calarVoz() {
  idFala++; clearTimeout(tempoFala);
  if (temVoz) speechSynthesis.cancel();
  terminarFala();
}
// Uma frase por vez: o Chrome corta falas longas, e cada comando vira uma pausa natural.
function falarFrase(frases, i, id) {
  if (id !== idFala) return;
  if (i >= frases.length) { terminarFala(); return; }
  const u = new SpeechSynthesisUtterance(frases[i]);
  u.lang = "pt-BR"; u.rate = 1.05;
  if (vozPt) u.voice = vozPt;
  falaAtual = u; // guarda a referência para o navegador não descartar a fala no meio
  const seguir = () => { clearTimeout(tempoGuarda); falarFrase(frases, i + 1, id); };
  u.onend = seguir;
  u.onerror = e => { if (id === idFala && e.error !== "canceled" && e.error !== "interrupted") semVoz(); };
  clearTimeout(tempoGuarda);
  tempoGuarda = setTimeout(() => { if (id === idFala) { speechSynthesis.cancel(); semVoz(); } }, 10000); // voz que não responde não pode deixar a música baixa
  speechSynthesis.speak(u);
}
function falar(frases) {
  calarVoz();
  if (!temVoz) { semVoz(); return; }
  if (vozesLidas && !vozPt) { semVoz(); return; } // uma voz em inglês lendo português não seria entendida
  const id = idFala;
  fatorVoz = 0.3; aplicarGanho();
  tempoFala = setTimeout(() => falarFrase(frases, 0, id), 150);
}
function narrarTela(inicio) {
  const titulo = chamada ? (chamada.estado === "tocando" ? "Chamada de " + chamada.nome : "Em chamada") : menuAberto ? "Menu de músicas" : NOMES_PAGINA[pagina];
  falar([...(inicio ? [inicio] : []), titulo, ...comandosAtuais().map(([gesto, acao, falado]) => (falado || gesto) + ": " + acao)]);
}
'''
pares = [
    ('fatorAudio = 1, volumeToque = 0.1;', 'fatorAudio = 1, fatorVoz = 1, volumeToque = 0.1;'),
    ('mestre.gain.value = volume * fatorAudio;', 'mestre.gain.value = volume * fatorAudio * fatorVoz;'),
    ('mestre.gain.setTargetAtTime(volume * fatorAudio, ctx.currentTime, 0.05)', 'mestre.gain.setTargetAtTime(volume * fatorAudio * fatorVoz, ctx.currentTime, 0.05)'),
    ('function atualizarLegenda() {\n  const FUNCAO', 'function comandosAtuais() {\n  const FUNCAO'),
    ('linhas = [["Para a direita", "próxima"], ["Para a esquerda", "última tocada"], ["Sinal de pare", tocando ? "pausa" : "toca"], ["Polegar", "volume"]];',
     'linhas = [["Para a direita", "próxima", "Deslizar para a direita"], ["Para a esquerda", "última tocada", "Deslizar para a esquerda"], ["Sinal de pare", tocando ? "pausa" : "toca"], ["Polegar", "volume", "Polegar para cima ou baixo"]];'),
    ('  $("legenda").innerHTML = linhas.map(([g, a]) => `<b>${g}:</b> ${a}`).join("<br>");\n}\n',
     '  return linhas;\n}\nfunction atualizarLegenda() {\n  $("legenda").innerHTML = comandosAtuais().map(([g, a]) => `<b>${g}:</b> ${a}`).join("<br>");\n}\n'),
    ('  $("menu").classList.add("aberto");\n  atualizarLegenda();\n', '  $("menu").classList.add("aberto");\n  atualizarLegenda();\n  narrarTela();\n'),
    ('$("menu").classList.remove("aberto"); atualizarLegenda(); }', '$("menu").classList.remove("aberto"); atualizarLegenda(); narrarTela(); }'),
    ('function mostrarPagina(nova) {\n', VOZ + 'function mostrarPagina(nova) {\n'),
    ('b.dataset.pagina === nova));\n  atualizarLegenda();\n}', 'b.dataset.pagina === nova));\n  atualizarLegenda();\n  if (ligado) narrarTela();\n}'),
    ('mostrarChamada(); avisar("Chamada de " + nome);', 'mostrarChamada(); avisar("Chamada de " + nome); narrarTela();'),
    ('mostrarChamada(); avisar("Chamada atendida");', 'mostrarChamada(); avisar("Chamada atendida"); narrarTela();'),
    ('  mostrarChamada(); avisar(texto);\n', '  mostrarChamada(); avisar(texto);\n  narrarTela(texto.startsWith("Última") ? "Chamada encerrada" : texto.replace(":", ","));\n'),
    ('  let feito = true;\n', '  let feito = true;\n  const falaAntes = idFala;\n'),
    ('  if (feito) $("nDisparos").textContent = ++comandos;\n',
     '  if (feito) $("nDisparos").textContent = ++comandos;\n  // Quem já sabe os comandos não precisa ouvir tudo: um comando interrompe a explicação.\n  if (feito && idFala === falaAntes) calarVoz();\n'),
    ('$("inicio").style.display = "none"; tocar(); };', '$("inicio").style.display = "none"; tocar(); narrarTela(); };'),
]
for a, b in pares:
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)
p.write_bytes((s.replace('\n', '\r\n') if crlf else s).encode('utf-8'))
print('ok', len(pares))
