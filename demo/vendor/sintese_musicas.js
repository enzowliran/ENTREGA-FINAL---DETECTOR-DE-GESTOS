// Músicas originais, geradas no navegador: não há arquivos de áudio nem obras de terceiros.
const nota = n => 440 * Math.pow(2, (n - 69) / 12);
const ESCALAS = { menor: [0, 2, 3, 5, 7, 8, 10], maior: [0, 2, 4, 5, 7, 9, 11], dorico: [0, 2, 3, 5, 7, 9, 10] };
const S = (nome, compassos, partes) => ({ nome, compassos, ...partes });
const MUSICAS = [
  { titulo: "Estrada Azul", ganho: 1.15, estilo: "Lo-fi", bpm: 78, swing: 0.28, tom: 57, escala: "dorico", cor: "#245a8f", semente: 11,
    progressao: [0, 3, 5, 4], teclas: "suave", ritmoMelodia: "x..x..x...x.x...x.....x.x..x....",
    kick: "x.....x...x.....", caixa: "....x.......x...", chimbal: "x.x.x.x.x.x.x.x.", baixo: "x.....x...x.....", acordes: "x.......x.......",
    secoes: [S("intro", 4, { teclas: 1, chimbal: 1 }), S("verso", 8, { teclas: 1, kick: 1, caixa: 1, chimbal: 1, baixo: 1 }),
      S("refrão", 8, { teclas: 1, kick: 1, caixa: 1, chimbal: 1, baixo: 1, melodia: 1 }), S("pausa", 4, { teclas: 1, melodia: 1 }),
      S("refrão", 8, { teclas: 1, kick: 1, caixa: 1, chimbal: 1, baixo: 1, melodia: 1 }), S("fim", 4, { teclas: 1 })] },
  { titulo: "Noite na Via", ganho: 0.7, estilo: "Synthwave", bpm: 104, swing: 0, tom: 50, escala: "menor", cor: "#6b3f9e", semente: 23,
    progressao: [0, 5, 2, 6], teclas: "pad", ritmoMelodia: "x...x.x...x...x.x...x.x.x.......",
    kick: "x...x...x...x...", caixa: "....x.......x...", chimbal: "..x...x...x...x.", baixo: "x.xxx.xxx.xxx.xx", arpejo: 2, acordes: "x...............",
    secoes: [S("intro", 4, { teclas: 1, arpejo: 1 }), S("verso", 8, { teclas: 1, arpejo: 1, kick: 1, baixo: 1, chimbal: 1 }),
      S("refrão", 8, { teclas: 1, arpejo: 1, kick: 1, caixa: 1, chimbal: 1, baixo: 1, melodia: 1 }), S("pausa", 4, { teclas: 1, melodia: 1 }),
      S("refrão", 8, { teclas: 1, arpejo: 1, kick: 1, caixa: 1, chimbal: 1, baixo: 1, melodia: 1 }), S("fim", 4, { teclas: 1, arpejo: 1 })] },
  { titulo: "Sinal Verde", ganho: 0.95, estilo: "House", bpm: 122, swing: 0.08, tom: 57, escala: "menor", cor: "#1f7d5c", semente: 37,
    progressao: [0, 5, 3, 4], teclas: "stab", ritmoMelodia: "..x...x...x.x.....x...x.x...x...",
    kick: "x...x...x...x...", caixa: "....x.......x...", chimbal: "..x...x...x...x.", baixo: "..x..x..x..x..x.", acordes: "..x..x.....x..x.",
    secoes: [S("intro", 4, { kick: 1, chimbal: 1 }), S("verso", 8, { kick: 1, chimbal: 1, baixo: 1, teclas: 1 }),
      S("refrão", 8, { kick: 1, caixa: 1, chimbal: 1, baixo: 1, teclas: 1, melodia: 1 }), S("pausa", 4, { teclas: 1, chimbal: 1 }),
      S("refrão", 8, { kick: 1, caixa: 1, chimbal: 1, baixo: 1, teclas: 1, melodia: 1 }), S("fim", 4, { kick: 1, chimbal: 1 })] },
  { titulo: "Chuva Fina", ganho: 1.3, estilo: "Piano ambiente", bpm: 66, swing: 0, tom: 52, escala: "maior", cor: "#3e5a66", semente: 41,
    progressao: [0, 4, 5, 3], teclas: "piano", ritmoMelodia: "x.......x...x...x...........x...",
    kick: "", caixa: "", chimbal: "", baixo: "x...............", arpejo: 4, acordes: "x.......",
    secoes: [S("início", 4, { teclas: 1, arpejo: 1 }), S("tema", 8, { teclas: 1, arpejo: 1, baixo: 1, melodia: 1 }),
      S("meio", 8, { teclas: 1, baixo: 1, melodia: 1 }), S("tema", 8, { teclas: 1, arpejo: 1, baixo: 1, melodia: 1 }), S("fim", 4, { teclas: 1 })] },
  { titulo: "Quilômetro Zero", ganho: 0.85, estilo: "Funk retrô", bpm: 112, swing: 0.16, tom: 55, escala: "dorico", cor: "#a4522b", semente: 53,
    progressao: [0, 0, 3, 4], teclas: "stab", ritmoMelodia: "x.x...x.x.....x...x.x...x...x.x.",
    kick: "x..x..x...x..x..", caixa: "....x..x.x..x...", chimbal: "x.x.x.x.x.x.x.x.", baixo: "x..x.xx...x.x.x.", acordes: ".x...x.x...x.x..",
    secoes: [S("intro", 4, { kick: 1, chimbal: 1, baixo: 1 }), S("verso", 8, { kick: 1, caixa: 1, chimbal: 1, baixo: 1, teclas: 1 }),
      S("refrão", 8, { kick: 1, caixa: 1, chimbal: 1, baixo: 1, teclas: 1, melodia: 1 }), S("solo", 4, { kick: 1, chimbal: 1, baixo: 1, melodia: 1 }),
      S("refrão", 8, { kick: 1, caixa: 1, chimbal: 1, baixo: 1, teclas: 1, melodia: 1 }), S("fim", 2, { kick: 1, baixo: 1 })] },
];
MUSICAS.forEach(m => {
  m.compassos = m.secoes.reduce((t, s) => t + s.compassos, 0);
  m.mapa = [];
  m.secoes.forEach(s => { for (let i = 0; i < s.compassos; i++) m.mapa.push(s); });
  m.motivo = criarMotivo(m);
});
function aleatorio(semente) {
  let x = semente >>> 0;
  return () => { x = (x * 1664525 + 1013904223) >>> 0; return x / 4294967296; };
}
// A melodia repete um motivo de 2 compassos sobre as notas do acorde, para soar como um tema.
function criarMotivo(m) {
  const r = aleatorio(m.semente), saida = [];
  let posicao = 1;
  for (const c of m.ritmoMelodia) {
    if (c === "x") { posicao = Math.max(0, Math.min(5, posicao + Math.round((r() - 0.45) * 3))); saida.push(posicao); }
    else saida.push(null);
  }
  return saida;
}
function acordeDe(m, grau) {
  const e = ESCALAS[m.escala];
  const n = i => m.tom + e[i % 7] + 12 * Math.floor(i / 7);
  return [n(grau), n(grau + 2), n(grau + 4), n(grau + 6)];
}

let ctx, mestre, analisador, ruido, envioEco, envioSala, faixaBus, ganchoPad;
let faixa = 0, tocando = false, proximoTempo = 0, passo = 0, relogio = null;

function iniciarAudio() {
  if (ctx) return;
  ctx = new (window.AudioContext || window.webkitAudioContext)();
  const compressor = ctx.createDynamicsCompressor();
  compressor.threshold.value = -16; compressor.ratio.value = 3; compressor.attack.value = 0.01; compressor.release.value = 0.2;
  mestre = ctx.createGain(); mestre.gain.value = 0.7;
  analisador = ctx.createAnalyser(); analisador.fftSize = 64;
  compressor.connect(mestre); mestre.connect(analisador); analisador.connect(ctx.destination);
  ctx.saida = compressor;

  ruido = ctx.createBuffer(1, ctx.sampleRate, ctx.sampleRate);
  const d = ruido.getChannelData(0);
  for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;

  // Sala: resposta ao impulso gerada aqui, com queda exponencial.
  const sala = ctx.createConvolver(), tamanho = Math.floor(ctx.sampleRate * 2.2), resposta = ctx.createBuffer(2, tamanho, ctx.sampleRate);
  for (let c = 0; c < 2; c++) {
    const canal = resposta.getChannelData(c);
    for (let i = 0; i < tamanho; i++) canal[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / tamanho, 3);
  }
  sala.buffer = resposta;
  envioSala = ctx.createGain(); envioSala.gain.value = 0.22;
  envioSala.connect(sala); sala.connect(compressor);

  const eco = ctx.createDelay(1.5), retorno = ctx.createGain();
  eco.delayTime.value = 0.33; retorno.gain.value = 0.32;
  envioEco = ctx.createGain(); envioEco.gain.value = 0.2;
  envioEco.connect(eco); eco.connect(retorno); retorno.connect(eco); eco.connect(compressor); retorno.connect(envioSala);
  ctx.eco = eco;
  novaFaixaBus();
}
function novaFaixaBus() {
  if (faixaBus) {
    const antigo = faixaBus;
    antigo.gain.setTargetAtTime(0, ctx.currentTime, 0.04);
    setTimeout(() => antigo.disconnect(), 400);
  }
  faixaBus = ctx.createGain(); faixaBus.gain.value = MUSICAS[faixa].ganho; faixaBus.connect(ctx.saida);
  ganchoPad = ctx.createGain(); ganchoPad.gain.value = 1; ganchoPad.connect(faixaBus);
}
function conectar(no, destino, eco, sala) {
  no.connect(destino || faixaBus);
  if (eco) { const g = ctx.createGain(); g.gain.value = eco; no.connect(g); g.connect(envioEco); }
  if (sala) { const g = ctx.createGain(); g.gain.value = sala; no.connect(g); g.connect(envioSala); }
}
function envelope(inicio, ataque, sustento, queda, vol) {
  const g = ctx.createGain();
  g.gain.setValueAtTime(0.0001, inicio);
  g.gain.exponentialRampToValueAtTime(vol, inicio + ataque);
  g.gain.setValueAtTime(vol, inicio + ataque + sustento);
  g.gain.exponentialRampToValueAtTime(0.0001, inicio + ataque + sustento + queda);
  return g;
}
function oscilador(tipo, freq, desafinar, inicio, fim, destino) {
  const o = ctx.createOscillator();
  o.type = tipo; o.frequency.value = freq; o.detune.value = desafinar;
  o.connect(destino); o.start(inicio); o.stop(fim + 0.05);
  return o;
}
function pad(notas, t, dur) {
  const filtro = ctx.createBiquadFilter(); filtro.type = "lowpass"; filtro.Q.value = 0.7;
  filtro.frequency.setValueAtTime(500, t); filtro.frequency.linearRampToValueAtTime(1400, t + dur * 0.6);
  const g = envelope(t, dur * 0.25, dur * 0.5, dur * 0.3, 0.05);
  filtro.connect(g); conectar(g, ganchoPad, 0, 0.8);
  notas.forEach(n => [-7, 7].forEach(d => oscilador("sawtooth", nota(n), d, t, t + dur * 1.05, filtro)));
}
function eletrico(notas, t, dur, vol) {
  // Piano elétrico por FM: um oscilador rápido modula a frequência da nota.
  notas.forEach(n => {
    const g = envelope(t, 0.006, 0.04, dur, vol), mod = ctx.createOscillator(), profundidade = ctx.createGain();
    mod.frequency.value = nota(n) * 2; profundidade.gain.setValueAtTime(nota(n) * 1.6, t);
    profundidade.gain.exponentialRampToValueAtTime(1, t + dur * 0.7);
    const o = ctx.createOscillator(); o.frequency.value = nota(n);
    mod.connect(profundidade); profundidade.connect(o.frequency); o.connect(g);
    conectar(g, null, 0.25, 0.5);
    mod.start(t); o.start(t); mod.stop(t + dur + 0.1); o.stop(t + dur + 0.1);
  });
}
function piano(notas, t, dur, vol) {
  notas.forEach(n => {
    const filtro = ctx.createBiquadFilter(); filtro.type = "lowpass";
    filtro.frequency.setValueAtTime(3200, t); filtro.frequency.exponentialRampToValueAtTime(700, t + dur);
    const g = envelope(t, 0.004, 0.02, dur, vol); filtro.connect(g); conectar(g, null, 0.2, 1);
    oscilador("triangle", nota(n), 0, t, t + dur, filtro);
    oscilador("sine", nota(n) * 2, 3, t, t + dur * 0.5, filtro);
  });
}
function stab(notas, t, vol) {
  const filtro = ctx.createBiquadFilter(); filtro.type = "lowpass"; filtro.Q.value = 3;
  filtro.frequency.setValueAtTime(2800, t); filtro.frequency.exponentialRampToValueAtTime(500, t + 0.18);
  const g = envelope(t, 0.004, 0.02, 0.2, vol); filtro.connect(g); conectar(g, ganchoPad, 0.5, 0.4);
  notas.forEach(n => oscilador("sawtooth", nota(n), 5, t, t + 0.26, filtro));
}
function baixo(n, t, dur, m) {
  const filtro = ctx.createBiquadFilter(); filtro.type = "lowpass"; filtro.Q.value = 5;
  filtro.frequency.setValueAtTime(m.escala === "maior" ? 500 : 900, t); filtro.frequency.exponentialRampToValueAtTime(160, t + dur);
  const g = envelope(t, 0.005, dur * 0.4, dur * 0.6, 0.3); filtro.connect(g); conectar(g);
  oscilador(m.escala === "maior" ? "triangle" : "sawtooth", nota(n), 0, t, t + dur, filtro);
  oscilador("sine", nota(n), 0, t, t + dur, g);
}
function arpejo(n, t, dur, vol) {
  const filtro = ctx.createBiquadFilter(); filtro.type = "lowpass"; filtro.Q.value = 6;
  filtro.frequency.setValueAtTime(3500, t); filtro.frequency.exponentialRampToValueAtTime(600, t + dur);
  const g = envelope(t, 0.003, 0, dur, vol); filtro.connect(g); conectar(g, null, 0.6, 0.3);
  oscilador("square", nota(n), 0, t, t + dur, filtro);
}
function melodia(n, t, dur, vol) {
  const g = envelope(t, 0.02, dur * 0.5, dur * 0.8, vol), filtro = ctx.createBiquadFilter();
  filtro.type = "lowpass"; filtro.frequency.value = 2600; filtro.connect(g); conectar(g, null, 0.7, 0.8);
  const o = oscilador("sawtooth", nota(n), 4, t, t + dur * 1.5, filtro);
  oscilador("triangle", nota(n) * 2, -4, t, t + dur * 1.5, filtro);
  const lfo = ctx.createOscillator(), vibrato = ctx.createGain();
  lfo.frequency.value = 5.2; vibrato.gain.value = 0;
  vibrato.gain.setValueAtTime(0, t + 0.12); vibrato.gain.linearRampToValueAtTime(nota(n) * 0.006, t + 0.3);
  lfo.connect(vibrato); vibrato.connect(o.frequency); lfo.start(t); lfo.stop(t + dur * 1.5 + 0.05);
}
function kick(t) {
  const o = ctx.createOscillator(), g = ctx.createGain();
  o.frequency.setValueAtTime(150, t); o.frequency.exponentialRampToValueAtTime(45, t + 0.1);
  g.gain.setValueAtTime(0.85, t); g.gain.exponentialRampToValueAtTime(0.001, t + 0.24);
  o.connect(g); conectar(g); o.start(t); o.stop(t + 0.3);
  // Abaixa os acordes a cada bumbo, para abrir espaço.
  ganchoPad.gain.cancelScheduledValues(t);
  ganchoPad.gain.setValueAtTime(0.45, t); ganchoPad.gain.linearRampToValueAtTime(1, t + 0.22);
}
function ruidoFiltrado(t, dur, tipo, freq, vol, sala) {
  const f = ctx.createBufferSource(), filtro = ctx.createBiquadFilter(), g = ctx.createGain();
  f.buffer = ruido; filtro.type = tipo; filtro.frequency.value = freq;
  g.gain.setValueAtTime(vol, t); g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
  f.connect(filtro); filtro.connect(g); conectar(g, null, 0, sala);
  f.start(t, Math.random() * 0.5, dur + 0.02);
}
function caixa(t) {
  ruidoFiltrado(t, 0.17, "bandpass", 1900, 0.34, 0.6);
  const o = ctx.createOscillator(), g = ctx.createGain();
  o.frequency.setValueAtTime(240, t); o.frequency.exponentialRampToValueAtTime(150, t + 0.07);
  g.gain.setValueAtTime(0.18, t); g.gain.exponentialRampToValueAtTime(0.001, t + 0.09);
  o.connect(g); conectar(g); o.start(t); o.stop(t + 0.12);
}
function chimbal(t, forte) { ruidoFiltrado(t, forte ? 0.1 : 0.04, "highpass", 7500, forte ? 0.12 : 0.075, 0); }

function tocarPasso(m, p, t, semicolcheia) {
  const compasso = Math.floor(p / 16), s = p % 16, secao = m.mapa[compasso];
  const acorde = acordeDe(m, m.progressao[compasso % m.progressao.length]);
  const barra = semicolcheia * 16;
  const marca = (padrao, i) => padrao && padrao[i % padrao.length] === "x";
  if (secao.teclas) {
    if (m.teclas === "pad" && s === 0) pad(acorde, t, barra);
    if (marca(m.acordes, s)) {
      const voz = acorde.slice(0, 4).map(n => n + 12);
      if (m.teclas === "suave") eletrico(voz, t, barra * 0.4, 0.05);
      if (m.teclas === "piano") piano(acorde.map(n => n + 12), t, barra * 0.5, 0.055);
      if (m.teclas === "stab") stab(voz.slice(0, 3), t, 0.075);
    }
  }
  if (secao.kick && marca(m.kick, s)) kick(t);
  if (secao.caixa && marca(m.caixa, s)) caixa(t);
  if (secao.chimbal && marca(m.chimbal, s)) chimbal(t, s % 8 === 6);
  if (secao.baixo && marca(m.baixo, s)) {
    const salto = s % 8 === 7 ? 7 : s === 14 ? 12 : 0;
    baixo(acorde[0] - 12 + salto, t, semicolcheia * 1.8, m);
  }
  if (secao.arpejo && m.arpejo && s % m.arpejo === 0) {
    const ordem = [0, 1, 2, 3, 2, 1, 2, 3];
    const n = acorde[ordem[(s / m.arpejo) % ordem.length]] + 12;
    if (m.teclas === "piano") piano([n + 12], t, barra * 0.3, 0.05);
    else arpejo(n, t, semicolcheia * 1.6, 0.04);
  }
  if (secao.melodia) {
    const i = p % m.motivo.length, posicao = m.motivo[i];
    if (posicao !== null) {
      const tons = [acorde[0], acorde[1], acorde[2], acorde[3], acorde[1] + 12, acorde[2] + 12];
      melodia(tons[posicao] + 12, t, semicolcheia * (m.bpm < 90 ? 3 : 2), 0.055);
    }
  }
}
