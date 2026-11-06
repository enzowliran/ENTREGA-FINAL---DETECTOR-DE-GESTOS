from pathlib import Path

def trocar(caminho, pares):
    p = Path(caminho)
    s = p.read_bytes().decode('utf-8')
    crlf = '\r\n' in s
    s = s.replace('\r\n', '\n')
    for a, b in pares:
        assert s.count(a) == 1, (s.count(a), a[:80])
        s = s.replace(a, b)
    p.write_bytes((s.replace('\n', '\r\n') if crlf else s).encode('utf-8'))

trocar('demo/player.py', [
    ('COMANDOS_PAINEL = {**COMANDOS_VOLUME, "Drumming Fingers": "tamborilar"}',
     'COMANDOS_PAINEL = {**COMANDOS_VOLUME, "Drumming Fingers": "tamborilar", "Rolling Hand Backward": "funcao"}'),
    ('help="Liga os gestos extras do painel: polegares e tamborilar"', 'help="Liga os gestos extras do painel: polegares, tamborilar e rolar a mão"'),
])

trocar('demo/painel_carro.html', [
    # ---- Estilos ----
    ('  #capa { flex: none; width: 216px; height: 216px;', '  #capa { flex: none; width: 190px; height: 190px;'),
    ('  .faixa { display: flex; gap: 18px; margin-top: 12px; }', '''  #abas { display: flex; gap: 6px; margin-top: 8px; }
  #abas button { flex: 1; width: auto; height: 30px; border-radius: 15px; font-size: 14px; font-weight: 600; color: var(--apoio); background: #24282c; }
  #abas button.ativa { background: var(--texto); color: #101214; }
  .pagina { display: none; flex: 1; min-height: 0; flex-direction: column; }
  .pagina.ativa { display: flex; }
  .faixa { display: flex; gap: 18px; margin-top: 10px; }
  .campos { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 14px; flex: 1; }
  .campo { background: #1d2125; border: 2px solid transparent; border-radius: 12px; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; padding: 8px; }
  .campo.ativo { border-color: var(--texto); background: #282d32; }
  .campo small { color: var(--apoio); font-size: 14px; }
  .campo b { font-size: 36px; line-height: 1.25; font-variant-numeric: tabular-nums; }
  .campo b.texto { font-size: 24px; }
  .situacao { display: flex; justify-content: space-between; align-items: center; margin-top: 10px; font-size: 16px; color: var(--apoio); }
  .situacao b { color: var(--texto); }
  .destaque { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; gap: 6px; }
  .destaque svg { width: 84px; height: 84px; }
  .destaque h3 { margin: 0; font-size: 32px; }
  .destaque p { margin: 0; color: var(--apoio); font-size: 16px; }
  #velocidades { display: flex; gap: 8px; width: 100%; margin-top: 8px; }
  #velocidades span { flex: 1; padding: 10px 2px; border-radius: 10px; background: #1d2125; font-size: 15px; font-weight: 600; color: var(--apoio); }
  #velocidades span.ativo { background: var(--texto); color: #101214; }
  #chamada { position: absolute; inset: 0; z-index: 2; display: none; flex-direction: column; align-items: center; justify-content: center; gap: 8px; text-align: center; background: #0d1a14; padding: 16px; }
  #chamada.aberta { display: flex; }
  #chamada small { font-size: 17px; color: #9fd9bd; }
  #chamada h3 { margin: 0; font-size: 42px; }
  #chamada p { margin: 4px 0 0; font-size: 17px; color: #d5ded9; line-height: 1.5; }
  #indicadores { display: flex; justify-content: center; gap: 6px; margin-top: 5px; font-size: 11px; font-weight: 700; }
  #indicadores span { color: #3b4146; }
  #indicadores span.ligado { color: var(--destaque); }
  #arSaida { left: 1310px; top: 506px; width: 230px; height: 44px; overflow: hidden; opacity: 0; transition: opacity .4s; pointer-events: none; }
  #arSaida.ligado { opacity: .9; }
  #arSaida i { position: absolute; left: -40%; width: 40%; height: 3px; border-radius: 2px; background: var(--ar, #8fd0ff); animation: sopro var(--sopro, 1.2s) linear infinite; }
  #arSaida i:nth-child(1) { top: 7px; }
  #arSaida i:nth-child(2) { top: 21px; animation-delay: -.45s; }
  #arSaida i:nth-child(3) { top: 35px; animation-delay: -.8s; }
  @keyframes sopro { to { left: 110%; } }
  @media (prefers-reduced-motion: reduce) { #arSaida i { animation: none; left: 30%; } }'''),
    ('  #legenda { left: 1282px; top: 594px; width: 290px; font-size: 16px; color: #aab0b4; line-height: 1.75; }',
     '  #legenda { left: 1282px; top: 590px; width: 300px; font-size: 15px; color: #aab0b4; line-height: 1.7; }'),
    # ---- Botões físicos ----
    ('''      <g fill="#1d2125" stroke="#3e444a" stroke-width="2"><circle cx="62" cy="26" r="20"/><circle cx="458" cy="26" r="20"/></g>
      <path d="M62 11 V21 M458 11 V21" stroke="#c8ccd0" stroke-width="3"/>
      <path d="M120 16 H400 M120 26 H400 M120 36 H400" stroke="#20252a" stroke-width="4"/>
''', '''      <g id="botaoTemp"><circle cx="62" cy="26" r="20" fill="#1d2125" stroke="#3e444a" stroke-width="2"/><path d="M62 11 V21" stroke="#c8ccd0" stroke-width="3"/></g>
      <g id="botaoVent"><circle cx="458" cy="26" r="20" fill="#1d2125" stroke="#3e444a" stroke-width="2"/><path d="M458 11 V21" stroke="#c8ccd0" stroke-width="3"/></g>
      <text id="txtTemp" x="94" y="33" fill="#d8dcde" font-size="20" font-weight="700">22°C</text>
      <text id="txtVent" x="426" y="33" fill="#d8dcde" font-size="20" font-weight="700" text-anchor="end">Vent 2</text>
      <path d="M190 16 H330 M190 26 H330 M190 36 H330" stroke="#20252a" stroke-width="4"/>
'''),
    ('''    <div id="contador"><b id="nDisparos">0</b> comandos · <b id="nMs">0</b> ms</div>
''', '''    <div id="contador"><b id="nDisparos">0</b> comandos · <b id="nMs">0</b> ms</div>
    <div id="indicadores"><span id="indAr">A/C</span><span id="indLimp">LIMP</span><span id="indFarol">FAROL</span></div>
'''),
    # ---- Páginas ----
    ('''    <div class="faixa">
      <div id="capa"''', '''    <div id="abas">
      <button data-pagina="musica">Música</button><button data-pagina="clima">Clima</button><button data-pagina="telefone">Telefone</button><button data-pagina="limpador">Limpador</button><button data-pagina="farois">Faróis</button>
    </div>
    <div class="pagina" id="pag_musica">
    <div class="faixa">
      <div id="capa"'''),
    ('''    <div id="volumeCaixa" aria-hidden="true"><div id="volumeBarra"></div></div>
''', '''    <div id="volumeCaixa" aria-hidden="true"><div id="volumeBarra"></div></div>
    </div>
    <div class="pagina" id="pag_clima">
      <div class="campos">
        <div class="campo" data-campo="temperatura"><small>Temperatura</small><b id="cTemp">22°C</b></div>
        <div class="campo" data-campo="ventilacao"><small>Ventilação</small><b id="cVent">2</b></div>
        <div class="campo" data-campo="direcao"><small>Direção do ar</small><b class="texto" id="cDir">Rosto</b></div>
      </div>
      <div class="situacao"><span>Ar-condicionado: <b id="cEstado">desligado</b></span><span>Simulado</span></div>
    </div>
    <div class="pagina" id="pag_telefone">
      <div class="destaque">
        <svg viewBox="0 0 24 24" fill="#e8eae8" aria-hidden="true"><path d="M6.6 10.8a15 15 0 0 0 6.6 6.6l2.2-2.2a1 1 0 0 1 1-.25 11 11 0 0 0 3.6.6 1 1 0 0 1 1 1V20a1 1 0 0 1-1 1A17 17 0 0 1 3 4a1 1 0 0 1 1-1h3.5a1 1 0 0 1 1 1 11 11 0 0 0 .6 3.6 1 1 0 0 1-.25 1z"/></svg>
        <h3>Telefone</h3>
        <p id="ultimaChamada">Nenhuma chamada recente</p>
        <p>Sinal de pare simula uma chamada</p>
      </div>
    </div>
    <div class="pagina" id="pag_limpador">
      <div class="destaque">
        <h3 id="limpTitulo">Desligado</h3>
        <p>Chuva e limpadores simulados</p>
        <div id="velocidades"><span>Desligado</span><span>Intermitente</span><span>Normal</span><span>Rápido</span></div>
      </div>
    </div>
    <div class="pagina" id="pag_farois">
      <div class="destaque">
        <svg id="iconeFarol" viewBox="0 0 24 24" fill="none" stroke="#6d7479" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M13 5c5 0 8 3.5 8 7s-3 7-8 7c-1.5-4-1.5-10 0-14z"/><path d="M9 7H3M9 12H3M9 17H3"/></svg>
        <h3 id="farolTitulo">Desligados</h3>
        <p>Com os faróis ligados, a cena escurece e a pista fica iluminada</p>
      </div>
    </div>
    <div id="chamada" role="alertdialog" aria-label="Chamada">
      <small id="chamadaEstado">Chamada recebida</small>
      <h3 id="chamadaNome">-</h3>
      <p id="chamadaAjuda"></p>
    </div>
'''),
    ('''  <div id="led" aria-hidden="true"></div>
''', '''  <div id="led" aria-hidden="true"></div>
  <div id="arSaida" aria-hidden="true"><i></i><i></i><i></i></div>
'''),
    # ---- Efeitos no para-brisa ----
    ('''      tela.fillStyle = "#e6523e"; tela.fillRect(x - h * .018, y - h * .3, h * .036, h * .05);
    }
  });
}
''', '''      tela.fillStyle = "#e6523e"; tela.fillRect(x - h * .018, y - h * .3, h * .036, h * .05);
    }
  });
  desenharFarois(dt);
  desenharChuva(dt);
}
let nivelFarol = 0;
function desenharFarois(dt) {
  nivelFarol += ((farois ? 1 : 0) - nivelFarol) * Math.min(1, dt * 2.5 || (REDUZIR ? 1 : 0));
  if (nivelFarol < 0.01) return;
  tela.fillStyle = `rgba(5, 8, 28, ${0.74 * nivelFarol})`;
  tela.fillRect(0, 0, LARG, ALT);
  const ponta = LARG / 2 + curvaAtual * 620 * Math.pow(0.72, 2.1), topo = HORIZONTE + 54;
  const luz = tela.createLinearGradient(0, ALT, 0, topo);
  luz.addColorStop(0, `rgba(255, 238, 190, ${0.5 * nivelFarol})`);
  luz.addColorStop(1, "rgba(255, 238, 190, 0)");
  tela.save();
  tela.globalCompositeOperation = "lighter";
  tela.fillStyle = luz;
  tela.beginPath(); tela.moveTo(160, ALT); tela.lineTo(ponta - 70, topo); tela.lineTo(ponta + 70, topo); tela.lineTo(1440, ALT); tela.fill();
  tela.restore();
}
const ESTEIRAS = [[560, 490], [1060, 490]], COMPRIMENTO = 440, GRAU = Math.PI / 180;
const REPOUSO = -176 * GRAU, MAXIMO = -62 * GRAU, RITMOS = [0, 1.1, 1.1, 0.65], PAUSAS = [0, 1.6, 0, 0];
let gotas = [], faseLimpador = 0, esperaLimpador = 0, anguloLimpador = REPOUSO;
function desenharChuva(dt) {
  const v = limpador;
  if (v > 0) {
    tela.fillStyle = "rgba(90, 105, 125, .2)"; tela.fillRect(0, 0, LARG, ALT);
    tela.strokeStyle = "rgba(220, 232, 245, .35)"; tela.lineWidth = 1.2; tela.beginPath();
    for (let i = 0; i < 70; i++) { const x = Math.random() * LARG, y = Math.random() * ALT; tela.moveTo(x, y); tela.lineTo(x - 3, y + 16); }
    tela.stroke();
    const novas = Math.floor(dt * 170 + Math.random());
    for (let i = 0; i < novas && gotas.length < 320; i++) gotas.push({ x: 90 + Math.random() * 1420, y: Math.random() * 480, r: 1.5 + Math.random() * 2.5 });
    if (esperaLimpador > 0) esperaLimpador -= dt;
    else {
      faseLimpador += dt / RITMOS[v];
      if (faseLimpador >= 1) { faseLimpador = 0; esperaLimpador = PAUSAS[v]; }
    }
  } else if (faseLimpador > 0) {
    // Ao desligar, termina a passada e volta ao repouso.
    faseLimpador = Math.min(1, faseLimpador + dt / 1.1);
    if (faseLimpador >= 1) faseLimpador = 0;
  } else if (gotas.length) gotas.splice(0, Math.ceil(dt * 60));
  anguloLimpador = REPOUSO + (MAXIMO - REPOUSO) * (0.5 - 0.5 * Math.cos(faseLimpador * 2 * Math.PI));
  gotas = gotas.filter(g => !ESTEIRAS.some(([px, py]) => {
    const dx = g.x - px, dy = g.y - py;
    return Math.hypot(dx, dy) < COMPRIMENTO && Math.abs(Math.atan2(dy, dx) - anguloLimpador) < 5 * GRAU;
  }));
  tela.fillStyle = "rgba(225, 238, 250, .55)";
  gotas.forEach(g => { tela.beginPath(); tela.arc(g.x, g.y, g.r, 0, 7); tela.fill(); });
  if (v > 0 || faseLimpador > 0) {
    ESTEIRAS.forEach(([px, py]) => {
      const x = px + Math.cos(anguloLimpador) * COMPRIMENTO, y = py + Math.sin(anguloLimpador) * COMPRIMENTO;
      tela.strokeStyle = "#0a0b0d"; tela.lineWidth = 7; tela.lineCap = "round";
      tela.beginPath(); tela.moveTo(px + Math.cos(anguloLimpador) * 40, py + Math.sin(anguloLimpador) * 40); tela.lineTo(x, y); tela.stroke();
    });
  }
}
'''),
    # ---- Estado ----
    ('historico = [], menuAberto = false, selecionada = 0, comandos = 0;',
     '''historico = [], menuAberto = false, selecionada = 0, comandos = 0;
const PAGINAS = ["musica", "clima", "telefone", "limpador", "farois"];
const NOMES_PAGINA = { musica: "Música", clima: "Clima", telefone: "Telefone", limpador: "Limpador", farois: "Faróis" };
const NOMES_LIMPADOR = ["Desligado", "Intermitente", "Normal", "Rápido"];
const CAMPOS = ["temperatura", "ventilacao", "direcao"], DIRECOES = ["Rosto", "Pés", "Rosto e pés", "Para-brisa"];
const CONTATOS = ["Casa", "Trabalho", "Oficina"];
const clima = { ligado: false, temperatura: 22, ventilacao: 2, direcao: 0, campo: 0 };
let pagina = "musica", farois = false, limpador = 0, ultimoLimpador = 2, chamada = null, indiceContato = 0, fatorAudio = 1, volumeToque = 0.1;'''),
    # ---- Volume ----
    ('''  mestre.gain.value = volume;
''', '''  mestre.gain.value = volume * fatorAudio;
'''),
    ('''  if (mestre) mestre.gain.setTargetAtTime(volume, ctx.currentTime, 0.05);
''', '''  aplicarGanho();
'''),
    ('''function mudarVolume(d) {''', '''function aplicarGanho() { if (mestre) mestre.gain.setTargetAtTime(volume * fatorAudio, ctx.currentTime, 0.05); }
function mudarVolume(d) {'''),
    # ---- Legenda ----
    ('''  const linhas = menuAberto
    ? [["Polegar para cima ou baixo", "navega"], ["Sinal de pare", "seleciona"], ["Tamborilar os dedos", "volta"]]
    : [["Para a direita", "próxima"], ["Para a esquerda", "última tocada"], ["Sinal de pare", tocando ? "pausa" : "toca"], ["Polegar", "volume"]];
  if (!menuAberto && !tocando) linhas.push(["Tamborilar os dedos", "abre o menu"]);
''', '''  const FUNCAO = ["Rolar a mão para trás", "troca de função"];
  let linhas;
  if (chamada && chamada.estado === "tocando") linhas = [["Manter polegar para cima", "atende"], ["Manter polegar para baixo", "recusa"]];
  else if (chamada) linhas = [["Sinal de pare", "encerra"]];
  else if (menuAberto) linhas = [["Polegar para cima ou baixo", "navega"], ["Sinal de pare", "seleciona"], ["Tamborilar os dedos", "volta"]];
  else if (pagina === "clima") linhas = [["Deslizar para os lados", "troca o ajuste"], ["Polegar para cima ou baixo", "muda o valor"], ["Sinal de pare", clima.ligado ? "desliga" : "liga"], FUNCAO];
  else if (pagina === "telefone") linhas = [["Sinal de pare", "simula uma chamada"], FUNCAO];
  else if (pagina === "limpador") linhas = [["Polegar para cima ou baixo", "muda a velocidade"], ["Sinal de pare", limpador ? "desliga" : "liga"], FUNCAO];
  else if (pagina === "farois") linhas = [["Sinal de pare", farois ? "desliga" : "liga"], FUNCAO];
  else {
    linhas = [["Para a direita", "próxima"], ["Para a esquerda", "última tocada"], ["Sinal de pare", tocando ? "pausa" : "toca"], ["Polegar", "volume"]];
    if (!tocando) linhas.push(["Tamborilar os dedos", "abre o menu"]);
    linhas.push(FUNCAO);
  }
'''),
    # ---- Comandos ----
    ('''function executar(c) {
  if (!faixas.length || !ligado) return;
  let feito = true;
  if (menuAberto) {''', '''function mostrarPagina(nova) {
  pagina = nova;
  PAGINAS.forEach(p => $("pag_" + p).classList.toggle("ativa", p === nova));
  document.querySelectorAll("#abas button").forEach(b => b.classList.toggle("ativa", b.dataset.pagina === nova));
  atualizarLegenda();
}
function atualizarClima() {
  $("cTemp").textContent = $("txtTemp").textContent = clima.temperatura + "°C";
  $("cVent").textContent = clima.ventilacao; $("txtVent").textContent = "Vent " + clima.ventilacao;
  $("cDir").textContent = DIRECOES[clima.direcao];
  $("cEstado").textContent = clima.ligado ? "ligado" : "desligado";
  document.querySelectorAll(".campo").forEach(c => c.classList.toggle("ativo", c.dataset.campo === CAMPOS[clima.campo]));
  $("botaoTemp").setAttribute("transform", `rotate(${(clima.temperatura - 23) * 22} 62 26)`);
  $("botaoVent").setAttribute("transform", `rotate(${(clima.ventilacao - 3) * 45} 458 26)`);
  $("indAr").classList.toggle("ligado", clima.ligado);
  $("arSaida").classList.toggle("ligado", clima.ligado);
  $("arSaida").style.setProperty("--ar", clima.temperatura <= 22 ? "#8fd0ff" : clima.temperatura >= 25 ? "#ffab7a" : "#e5e9ec");
  $("arSaida").style.setProperty("--sopro", (1.9 - clima.ventilacao * 0.3) + "s");
}
function ajustarClima(d) {
  const campo = CAMPOS[clima.campo];
  if (campo === "temperatura") clima.temperatura = Math.max(18, Math.min(28, clima.temperatura + d));
  else if (campo === "ventilacao") clima.ventilacao = Math.max(1, Math.min(5, clima.ventilacao + d));
  else clima.direcao = (clima.direcao + d + DIRECOES.length) % DIRECOES.length;
  atualizarClima();
  avisar(campo === "temperatura" ? clima.temperatura + "°C" : campo === "ventilacao" ? "Ventilação " + clima.ventilacao : DIRECOES[clima.direcao]);
}
function atualizarLimpador() {
  $("limpTitulo").textContent = NOMES_LIMPADOR[limpador];
  [...$("velocidades").children].forEach((s, i) => s.classList.toggle("ativo", i === limpador));
  $("indLimp").classList.toggle("ligado", limpador > 0);
}
function definirLimpador(v) {
  limpador = v; if (v > 0) ultimoLimpador = v;
  atualizarLimpador(); atualizarLegenda(); avisar("Limpador: " + NOMES_LIMPADOR[v]);
}
function atualizarFarois() {
  $("farolTitulo").textContent = farois ? "Ligados" : "Desligados";
  $("iconeFarol").setAttribute("stroke", farois ? "#ffe29a" : "#6d7479");
  $("indFarol").classList.toggle("ligado", farois);
}
// ---- Chamada simulada: atender ou recusar exige manter o polegar, para evitar disparo por engano ----
function tocarToque() {
  if (!ctx) return;
  [0, 0.28].forEach(atraso => {
    const o = ctx.createOscillator(), g = ctx.createGain(), t = ctx.currentTime + atraso;
    o.frequency.value = 880;
    g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(Math.max(volumeToque, 0.0002), t + 0.02); g.gain.exponentialRampToValueAtTime(0.0001, t + 0.22);
    o.connect(g); g.connect(ctx.destination); o.start(t); o.stop(t + 0.25);
  });
}
function mostrarChamada() {
  const c = chamada;
  $("chamada").classList.toggle("aberta", !!c);
  if (c) {
    $("chamadaNome").textContent = c.nome;
    $("chamadaEstado").textContent = c.estado === "tocando" ? "Chamada recebida (simulada)" : "Em chamada · " + formatar((performance.now() - c.inicio) / 1000);
    $("chamadaAjuda").innerHTML = c.estado === "tocando" ? "Manter polegar para cima: atende<br>Manter polegar para baixo: recusa" : "Sinal de pare: encerra";
  }
  atualizarLegenda();
}
function iniciarChamada() {
  const nome = CONTATOS[indiceContato++ % CONTATOS.length];
  chamada = { nome, estado: "tocando", pendente: null, inicio: 0, retomar: false };
  if (ctx) ctx.resume();
  fatorAudio = 0.25; aplicarGanho();
  tocarToque();
  chamada.toque = setInterval(tocarToque, 2000);
  chamada.limite = setTimeout(() => terminarChamada("Chamada perdida: " + nome), 20000);
  mostrarChamada(); avisar("Chamada de " + nome);
}
function atenderChamada() {
  clearInterval(chamada.toque); clearTimeout(chamada.limite);
  chamada.retomar = tocando;
  if (tocando) pausar();
  chamada.estado = "atendida"; chamada.inicio = performance.now();
  mostrarChamada(); avisar("Chamada atendida");
}
function terminarChamada(texto) {
  clearInterval(chamada.toque); clearTimeout(chamada.limite);
  const retomar = chamada.retomar;
  chamada = null;
  fatorAudio = 1; aplicarGanho();
  if (retomar) tocar();
  else if (!tocando && !audioEl && ctx) ctx.suspend();
  $("ultimaChamada").textContent = texto;
  mostrarChamada(); avisar(texto);
}
function comandoChamada(c, repeticao) {
  if (chamada.estado === "atendida") {
    if (c !== "alternar" || repeticao) return false;
    terminarChamada("Última chamada: " + chamada.nome + ", " + formatar((performance.now() - chamada.inicio) / 1000));
    return true;
  }
  if (c !== "volume_mais" && c !== "volume_menos") return false;
  if (!repeticao) {
    chamada.pendente = c;
    avisar(c === "volume_mais" ? "Mantenha para atender" : "Mantenha para recusar");
    return false;
  }
  if (chamada.pendente !== c) return false;
  if (c === "volume_mais") atenderChamada();
  else terminarChamada("Chamada recusada: " + chamada.nome);
  return true;
}
function comandoPagina(c, repeticao) {
  if (pagina === "clima") {
    if (c === "volume_mais" || c === "volume_menos") { ajustarClima(c === "volume_mais" ? 1 : -1); return true; }
    if (repeticao) return false;
    if (c === "proxima" || c === "anterior") {
      clima.campo = (clima.campo + (c === "proxima" ? 1 : -1) + CAMPOS.length) % CAMPOS.length;
      atualizarClima(); avisar(["Temperatura", "Ventilação", "Direção do ar"][clima.campo]); return true;
    }
    if (c === "alternar") { clima.ligado = !clima.ligado; atualizarClima(); atualizarLegenda(); avisar("Ar-condicionado " + (clima.ligado ? "ligado" : "desligado")); return true; }
    return false;
  }
  if (repeticao) return false;
  if (pagina === "telefone") { if (c === "alternar") { iniciarChamada(); return true; } return false; }
  if (pagina === "limpador") {
    if (c === "volume_mais") { if (limpador < 3) definirLimpador(limpador + 1); return true; }
    if (c === "volume_menos") { if (limpador > 0) definirLimpador(limpador - 1); return true; }
    if (c === "alternar") { definirLimpador(limpador ? 0 : ultimoLimpador); return true; }
    return false;
  }
  if (pagina === "farois") {
    if (c === "alternar") { farois = !farois; atualizarFarois(); atualizarLegenda(); avisar("Faróis " + (farois ? "ligados" : "desligados")); return true; }
    return false;
  }
  return null;
}
function executar(c, repeticao = false) {
  if (!faixas.length || !ligado) return;
  let feito = true;
  if (chamada) feito = comandoChamada(c, repeticao);
  else if (c === "funcao" && !menuAberto) {
    if (repeticao) return;
    const nova = PAGINAS[(PAGINAS.indexOf(pagina) + 1) % PAGINAS.length];
    mostrarPagina(nova); avisar(NOMES_PAGINA[nova]);
  }
  else if (!menuAberto && pagina !== "musica") feito = comandoPagina(c, repeticao);
  else if (repeticao && !menuAberto) feito = false;
  else if (menuAberto) {'''),
    # ---- Teclado e modelo ----
    ('''  if (e.key === "ArrowUp") { e.preventDefault(); executar("volume_mais"); }
  if (e.key === "ArrowDown") { e.preventDefault(); executar("volume_menos"); }
''', '''  if (e.key === "ArrowUp") { e.preventDefault(); executar("volume_mais", e.repeat); }
  if (e.key === "ArrowDown") { e.preventDefault(); executar("volume_menos", e.repeat); }
  if (e.key.toLowerCase() === "r") executar("funcao");
'''),
    ('''  // Manter o polegar só repete o comando dentro do menu; o volume muda um passo por gesto.
  if (e.tipo === "comando" && (!e.repeticao || menuAberto)) executar(e.comando);
''', '''  // Manter o polegar repete no menu e no clima, e confirma a chamada; o volume muda um passo por gesto.
  if (e.tipo === "comando") executar(e.comando, !!e.repeticao);
'''),
    ('''    relogioTempo = agora; atualizarTempo();
''', '''    relogioTempo = agora; atualizarTempo();
    if (chamada && chamada.estado === "atendida") $("chamadaEstado").textContent = "Em chamada · " + formatar((performance.now() - chamada.inicio) / 1000);
'''),
    ('''atualizarLegenda();
carregarFaixas();
''', '''document.querySelectorAll("#abas button").forEach(b => b.onclick = () => { if (ligado && !chamada && !menuAberto) mostrarPagina(b.dataset.pagina); });
mostrarPagina("musica"); atualizarClima(); atualizarLimpador(); atualizarFarois();
carregarFaixas();
'''),
])
print('ok')
