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
    ('COMANDOS_VOLUME = {"Thumb Up": "volume_mais", "Thumb Down": "volume_menos"}\n',
     'COMANDOS_PAINEL = {"Thumb Up": "volume_mais", "Thumb Down": "volume_menos", "Shaking Hand": "acenar",\n'
     '                   "Swiping Up": "menu_cima", "Swiping Down": "menu_baixo"}\n'),
    ('    a.add_argument("--volume", action="store_true", help="Polegar para cima e para baixo controlam o volume")\n',
     '    a.add_argument("--painel", action="store_true", help="Liga os gestos extras do painel: polegares, aceno e arrastar na vertical")\n'),
    ('    if args.volume:\n        COMANDOS.update(COMANDOS_VOLUME)\n',
     '    if args.painel:\n        COMANDOS.update(COMANDOS_PAINEL)\n'),
])
trocar('demo/abrir_painel.bat', [('--porta 8770 --volume %*', '--porta 8770 --painel %*')])

trocar('demo/painel_carro.html', [
    # Estilo do menu
    ('  #legenda { left: 1282px;',
     '''  #menu { position: absolute; inset: 0; z-index: 1; display: none; flex-direction: column; padding: 12px 20px 12px; background: #0e1012; }
  #menu.aberto { display: flex; }
  #menu h2 { margin: 0 0 8px; font-size: 21px; font-weight: 700; }
  #lista { list-style: none; margin: 0; padding: 0; flex: 1; min-height: 0; overflow: hidden; position: relative; }
  #lista li { display: flex; align-items: center; gap: 12px; height: 50px; padding: 0 12px; border-radius: 8px; cursor: pointer; }
  #lista li.selecionada { background: var(--texto); color: #101214; }
  #lista .numero { flex: none; width: 22px; color: var(--apoio); font-variant-numeric: tabular-nums; }
  #lista li.selecionada .numero, #lista li.selecionada small { color: #4a5055; }
  #lista .nome { flex: 1; min-width: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  #lista .nome b { font-weight: 600; font-size: 17px; }
  #lista small { display: block; color: var(--apoio); font-size: 13px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  #lista .atual { flex: none; font-size: 12px; font-weight: 700; color: var(--destaque); }
  #lista li.selecionada .atual { color: #16704a; }

  #legenda { left: 1282px;'''),
    # Menu e legenda
    ('''    <div id="volumeCaixa" aria-hidden="true"><div id="volumeBarra"></div></div>
''', '''    <div id="volumeCaixa" aria-hidden="true"><div id="volumeBarra"></div></div>
    <div id="menu" role="dialog" aria-label="Lista de músicas">
      <h2>Músicas</h2>
      <ol id="lista"></ol>
    </div>
'''),
    ('''  <div id="legenda">
    <b>Para a esquerda:</b> volta<br>
    <b>Para a direita:</b> avança<br>
    <b>Sinal de pare:</b> pausa<br>
    <b>Polegar:</b> volume
  </div>
''', '''  <div id="legenda"></div>
'''),
    # Histórico das faixas tocadas
    ('let faixas = [], atual = 0, ligado = false, audioEl = null, volume = 0.7, ultimoPasso = 0;',
     'let faixas = [], atual = 0, ligado = false, audioEl = null, volume = 0.7, historico = [], menuAberto = false, selecionada = 0, comandos = 0;'),
    ('''function trocar(d) {
  atual = (atual + d + faixas.length) % faixas.length;
  if (usaArquivos()) {''', '''function trocar(d) { irPara((atual + d + faixas.length) % faixas.length); }
// Volta para a última faixa tocada; sem histórico, para a anterior da lista.
function voltarUltima() {
  if (historico.length) irPara(historico.pop(), false);
  else irPara((atual - 1 + faixas.length) % faixas.length, false);
}
function irPara(indice, registrar = true) {
  if (registrar && indice !== atual) historico.push(atual);
  atual = indice;
  if (usaArquivos()) {'''),
    # Botão e legenda conforme o estado
    ('''function atualizarBotao() { $("iconeAlternar").setAttribute("d", tocando ? "M6 5h4.5v14H6zM13.5 5H18v14h-4.5z" : "M7 5l13 7-13 7z"); }
''', '''function atualizarBotao() {
  $("iconeAlternar").setAttribute("d", tocando ? "M6 5h4.5v14H6zM13.5 5H18v14h-4.5z" : "M7 5l13 7-13 7z");
  atualizarLegenda();
}
function atualizarLegenda() {
  const linhas = menuAberto
    ? [["Arrastar para cima ou baixo", "navega"], ["Polegar para cima", "toca a faixa"], ["Acenar", "fecha o menu"]]
    : [["Para a direita", "próxima"], ["Para a esquerda", "última tocada"], ["Sinal de pare", tocando ? "pausa" : "toca"], ["Polegar", "volume"]];
  if (!menuAberto && !tocando) linhas.push(["Acenar", "abre o menu"]);
  $("legenda").innerHTML = linhas.map(([g, a]) => `<b>${g}:</b> ${a}`).join("<br>");
}
function abrirMenu() {
  menuAberto = true; selecionada = atual;
  const lista = $("lista"); lista.replaceChildren();
  faixas.forEach((f, i) => {
    const li = document.createElement("li");
    const numero = document.createElement("span"); numero.className = "numero"; numero.textContent = i + 1;
    const nome = document.createElement("div"); nome.className = "nome";
    const titulo = document.createElement("b"); titulo.textContent = f.titulo;
    const artista = document.createElement("small"); artista.textContent = f.artista;
    nome.append(titulo, artista); li.append(numero, nome);
    if (i === atual) { const m = document.createElement("span"); m.className = "atual"; m.textContent = "ATUAL"; li.append(m); }
    li.onclick = () => { selecionada = i; confirmarMenu(); };
    lista.append(li);
  });
  $("menu").classList.add("aberto");
  destacarMenu(); atualizarLegenda();
}
function fecharMenu() { menuAberto = false; $("menu").classList.remove("aberto"); atualizarLegenda(); }
function destacarMenu() {
  const lista = $("lista");
  [...lista.children].forEach((li, i) => li.classList.toggle("selecionada", i === selecionada));
  const li = lista.children[selecionada];
  if (li.offsetTop < lista.scrollTop) lista.scrollTop = li.offsetTop;
  else if (li.offsetTop + li.offsetHeight > lista.scrollTop + lista.clientHeight) lista.scrollTop = li.offsetTop + li.offsetHeight - lista.clientHeight;
}
function moverMenu(d) {
  selecionada = (selecionada + d + faixas.length) % faixas.length;
  destacarMenu(); avisar(faixas[selecionada].titulo);
}
function confirmarMenu() {
  const i = selecionada;
  fecharMenu();
  if (i !== atual) irPara(i);
  tocar(); avisar("▶ " + faixas[atual].titulo);
}
'''),
    # Comandos
    ('''  if (!faixas.length || !ligado) return;
  if (c === "proxima") { trocar(1); if (!tocando) tocar(); avisar("▶ " + faixas[atual].titulo); }
  else if (c === "anterior") { trocar(-1); if (!tocando) tocar(); avisar("◀ " + faixas[atual].titulo); }
  else if (c === "alternar") { tocando ? pausar() : tocar(); avisar(tocando ? "Tocando" : "Pausado"); }
  else if (c === "volume_mais") { mudarVolume(0.1); avisar("Volume " + Math.round(volume * 100) + "%"); }
  else if (c === "volume_menos") { mudarVolume(-0.1); avisar("Volume " + Math.round(volume * 100) + "%"); }
}
''', '''  if (!faixas.length || !ligado) return;
  let feito = true;
  if (menuAberto) {
    // Com o menu aberto, só valem navegar, confirmar e fechar.
    if (c === "menu_cima") moverMenu(-1);
    else if (c === "menu_baixo") moverMenu(1);
    else if (c === "volume_mais") confirmarMenu();
    else if (c === "acenar") { fecharMenu(); avisar("Menu fechado"); }
    else feito = false;
  }
  else if (c === "proxima") { trocar(1); if (!tocando) tocar(); avisar("▶ " + faixas[atual].titulo); }
  else if (c === "anterior") { voltarUltima(); if (!tocando) tocar(); avisar("◀ " + faixas[atual].titulo); }
  else if (c === "alternar") { tocando ? pausar() : tocar(); avisar(tocando ? "Tocando" : "Pausado"); }
  else if (c === "volume_mais") { mudarVolume(0.1); avisar("Volume " + Math.round(volume * 100) + "%"); }
  else if (c === "volume_menos") { mudarVolume(-0.1); avisar("Volume " + Math.round(volume * 100) + "%"); }
  else if (c === "acenar" && !tocando) { abrirMenu(); avisar("Menu de músicas"); }
  else feito = false;
  // O contador mostra só os comandos que tiveram efeito neste estado.
  if (feito) $("nDisparos").textContent = ++comandos;
}
'''),
    ('''  if (e.key === "ArrowUp") { e.preventDefault(); executar("volume_mais"); }
  if (e.key === "ArrowDown") { e.preventDefault(); executar("volume_menos"); }
''', '''  if (e.key === "ArrowUp") { e.preventDefault(); executar(menuAberto ? "menu_cima" : "volume_mais"); }
  if (e.key === "ArrowDown") { e.preventDefault(); executar(menuAberto ? "menu_baixo" : "volume_menos"); }
  if (e.key === "Enter" && menuAberto) { e.preventDefault(); executar("volume_mais"); }
  if (e.key.toLowerCase() === "m") executar("acenar");
'''),
    ('''    $("nDisparos").textContent = e.disparos; $("nMs").textContent = Math.round(e.ms);
''', '''    $("nMs").textContent = Math.round(e.ms);
'''),
    ('''$("volumeBarra").style.height = volume * 100 + "%";
carregarFaixas();
''', '''$("volumeBarra").style.height = volume * 100 + "%";
atualizarLegenda();
carregarFaixas();
'''),
])
print('ok')
