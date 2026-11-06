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
    ('''"Shaking Hand": "acenar",
                   "Swiping Up": "menu_cima", "Swiping Down": "menu_baixo"}
''', '''"Shaking Hand": "acenar"}
'''),
    ('help="Liga os gestos extras do painel: polegares, aceno e arrastar na vertical"', 'help="Liga os gestos extras do painel: polegares e aceno"'),
])
trocar('demo/painel_carro.html', [
    ('''    ? [["Arrastar para cima ou baixo", "navega"], ["Polegar para cima", "toca a faixa"], ["Acenar", "fecha o menu"]]
''', '''    ? [["Polegar para cima ou baixo", "navega"], ["Sinal de pare", "seleciona"], ["Acenar", "volta"]]
'''),
    ('''function abrirMenu() {
  menuAberto = true; selecionada = atual;
  const lista = $("lista"); lista.replaceChildren();
''', '''function abrirMenu() {
  menuAberto = true; selecionada = atual;
  montarLista();
  $("menu").classList.add("aberto");
  atualizarLegenda();
}
function montarLista() {
  const lista = $("lista"), rolagem = lista.scrollTop; lista.replaceChildren();
'''),
    ('''    li.onclick = () => { selecionada = i; confirmarMenu(); };
    lista.append(li);
  });
  $("menu").classList.add("aberto");
  destacarMenu(); atualizarLegenda();
}
''', '''    li.onclick = () => { selecionada = i; confirmarMenu(); };
    lista.append(li);
  });
  lista.scrollTop = rolagem;
  destacarMenu();
}
'''),
    ('''function confirmarMenu() {
  const i = selecionada;
  fecharMenu();
  if (i !== atual) irPara(i);
  tocar(); avisar("▶ " + faixas[atual].titulo);
}
''', '''// Só escolhe a faixa: ela começa quando a pessoa voltar à tela principal e der play.
function confirmarMenu() {
  if (selecionada !== atual) irPara(selecionada);
  montarLista();
  avisar("Selecionada: " + faixas[atual].titulo);
}
'''),
    ('''    if (c === "menu_cima") moverMenu(-1);
    else if (c === "menu_baixo") moverMenu(1);
    else if (c === "volume_mais") confirmarMenu();
    else if (c === "acenar")''', '''    if (c === "volume_mais") moverMenu(-1);
    else if (c === "volume_menos") moverMenu(1);
    else if (c === "alternar") confirmarMenu();
    else if (c === "acenar")'''),
    ('''  if (e.key === "ArrowUp") { e.preventDefault(); executar(menuAberto ? "menu_cima" : "volume_mais"); }
  if (e.key === "ArrowDown") { e.preventDefault(); executar(menuAberto ? "menu_baixo" : "volume_menos"); }
  if (e.key === "Enter" && menuAberto) { e.preventDefault(); executar("volume_mais"); }
''', '''  if (e.key === "ArrowUp") { e.preventDefault(); executar("volume_mais"); }
  if (e.key === "ArrowDown") { e.preventDefault(); executar("volume_menos"); }
'''),
])
print('ok')
