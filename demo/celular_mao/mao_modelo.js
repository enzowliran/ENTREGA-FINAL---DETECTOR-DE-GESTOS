// Mão realista: um modelo de mão com malha e ossos (demo/mao/mao_left.glb e mao_right.glb, a "generic-hand" do pacote
// @webxr-input-profiles/assets, do W3C Immersive Web, licença MIT) movido pelos 21 pontos do MediaPipe.
// Cada osso do modelo é posto direto no ponto correspondente do MediaPipe, girado para apontar para o ponto seguinte
// e esticado até ele; assim a malha acompanha o tamanho e a pose da mão da pessoa.
// Este arquivo acrescenta ao modelo: pele (cor, poros, palma mais clara), unhas e antebraço.
// Usado por demo/hand3d.html e demo/carro3d.html (servido em /demo/mao_modelo.js).
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

// ponto do MediaPipe -> osso do modelo (os metacarpos dos quatro dedos não têm ponto: saem da pose da palma)
const NOMES = ["wrist",
  "thumb-metacarpal", "thumb-phalanx-proximal", "thumb-phalanx-distal", "thumb-tip",
  ...["index", "middle", "ring", "pinky"].flatMap(d => [`${d}-finger-phalanx-proximal`, `${d}-finger-phalanx-intermediate`, `${d}-finger-phalanx-distal`, `${d}-finger-tip`])];
const METACARPOS = ["index", "middle", "ring", "pinky"].map(d => `${d}-finger-metacarpal`);
const ARQUIVOS = { "-1": "./mao/mao_left.glb", "1": "./mao/mao_right.glb" };   // lado da palma -> arquivo
const ALEM_DA_PONTA = 0.008;   // o ponto "ponta" do MediaPipe fica ~8 mm além do osso "tip" do modelo
const COMP_ANTEBRACO = 0.11;

const V = () => new THREE.Vector3();
const v1 = V(), v2 = V(), vs = V(), vn = V(), vT = V(), vPos = V(), vDob = V(), vRef = V();
const mB = new THREE.Matrix4(), qA = new THREE.Quaternion(), qD = new THREE.Quaternion(), qPalma = new THREE.Quaternion();

// Direções da mão a partir de 21 pontos (serve para os pontos do MediaPipe e para a pose de repouso do modelo).
// lado: +1 ou -1, de que lado do produto vetorial (5-0) x (17-0) fica a palma.
function direcoes(p, lado, B) {
  B.lat.subVectors(p[5], p[17]).normalize();
  B.fd.subVectors(p[9], p[0]).normalize();
  B.nP.crossVectors(v1.subVectors(p[5], p[0]), v2.subVectors(p[17], p[0])).normalize().multiplyScalar(lado);
  B.u.subVectors(p[5], p[2]).normalize();
  quadro(B.nP, B.u, B.qPol);   // base do polegar: direção da palma e direção do polegar para o indicador
  return B;
}
const novaBase = () => ({ lat: V(), fd: V(), nP: V(), u: V(), qPol: new THREE.Quaternion() });
// Orientação de um osso: eixo T (para o ponto seguinte) e uma referência lateral. Falha se os dois forem paralelos.
function quadro(T, ref, q) {
  vs.copy(ref).addScaledVector(T, -ref.dot(T));
  if (vs.lengthSq() < 1e-8) return false;
  vs.normalize(); vn.crossVectors(T, vs);
  q.setFromRotationMatrix(mB.makeBasis(T, vs, vn));
  return true;
}

// ---- pele: mapa de normal com poros, feito num canvas ----
let texPoros = null;
function poros() {
  if (texPoros) return texPoros;
  const tam = 256, c = document.createElement("canvas"); c.width = c.height = tam;
  const g = c.getContext("2d");
  g.fillStyle = "#808080"; g.fillRect(0, 0, tam, tam);
  for (let i = 0; i < 1300; i++) {
    const x = Math.random() * tam, y = Math.random() * tam, r = 0.5 + Math.random() * 0.6;
    g.fillStyle = `rgba(0,0,0,${0.2 + Math.random() * 0.2})`;
    for (const ox of [-tam, 0, tam]) for (const oy of [-tam, 0, tam]) { g.beginPath(); g.arc(x + ox, y + oy, r, 0, Math.PI * 2); g.fill(); }
  }
  for (let i = 0; i < tam * tam / 4; i++) {
    g.fillStyle = Math.random() > 0.5 ? "rgba(255,255,255,.06)" : "rgba(0,0,0,.06)";
    g.fillRect(Math.random() * tam, Math.random() * tam, 1, 1);
  }
  const alt = g.getImageData(0, 0, tam, tam).data, out = g.createImageData(tam, tam), o = out.data, forca = 2.2;
  const H = (x, y) => alt[(((y + tam) % tam) * tam + ((x + tam) % tam)) * 4] / 255;
  for (let y = 0; y < tam; y++) for (let x = 0; x < tam; x++) {
    const dx = (H(x + 1, y) - H(x - 1, y)) * forca, dy = (H(x, y + 1) - H(x, y - 1)) * forca;
    const inv = 1 / Math.sqrt(dx * dx + dy * dy + 1), k = (y * tam + x) * 4;
    o[k] = (-dx * inv * 0.5 + 0.5) * 255; o[k + 1] = (dy * inv * 0.5 + 0.5) * 255; o[k + 2] = (inv * 0.5 + 0.5) * 255; o[k + 3] = 255;
  }
  g.putImageData(out, 0, 0);
  texPoros = new THREE.CanvasTexture(c);
  texPoros.wrapS = texPoros.wrapT = THREE.RepeatWrapping; texPoros.repeat.set(10, 10); texPoros.anisotropy = 4;
  return texPoros;
}
function materialPele(cor) {
  const c = new THREE.Color(cor);
  return new THREE.MeshPhysicalMaterial({
    color: c, vertexColors: true, roughness: 0.55, metalness: 0, ior: 1.4,
    sheen: 0.4, sheenColor: new THREE.Color(0xffc8ae), sheenRoughness: 0.55,
    normalMap: poros(), normalScale: new THREE.Vector2(0.3, 0.3),
    emissive: c.clone().multiply(new THREE.Color(1, 0.42, 0.32)).multiplyScalar(0.09),   // calor nas sombras
  });
}

// ---- malha mais lisa: subdivisão de Loop (cada triângulo vira quatro e a superfície é arredondada) ----
// O modelo original tem poucos triângulos (as pontas dos dedos ficam facetadas). Os pesos dos ossos e o UV dos
// vértices novos são a média dos dois vértices da aresta.
function subdividir(geo) {
  const pos = geo.attributes.position, uv = geo.attributes.uv, si = geo.attributes.skinIndex, sw = geo.attributes.skinWeight, ind = geo.index.array;
  const chave = new Map(), canon = new Int32Array(pos.count), rep = [];
  for (let i = 0; i < pos.count; i++) {   // vértices repetidos (costuras do UV) viram um só
    const k = `${Math.round(pos.getX(i) * 1e5)},${Math.round(pos.getY(i) * 1e5)},${Math.round(pos.getZ(i) * 1e5)}`;
    if (!chave.has(k)) { chave.set(k, rep.length); rep.push(i); }
    canon[i] = chave.get(k);
  }
  const P = rep.map(i => V().fromBufferAttribute(pos, i)), U = rep.map(i => [uv.getX(i), uv.getY(i)]);
  const W = rep.map(i => { const m = new Map(); for (let k = 0; k < 4; k++) { const w = sw.getComponent(i, k); if (w > 0) m.set(si.getComponent(i, k), (m.get(si.getComponent(i, k)) || 0) + w); } return m; });
  const tris = [];
  for (let t = 0; t < ind.length; t += 3) { const a = canon[ind[t]], b = canon[ind[t + 1]], c = canon[ind[t + 2]]; if (a !== b && b !== c && a !== c) tris.push([a, b, c]); }
  const arestas = new Map(), viz = P.map(() => new Set()), borda = P.map(() => []);
  const aresta = (a, b) => { const k = a < b ? a + "," + b : b + "," + a; let e = arestas.get(k); if (!e) { e = { a, b, op: [], id: -1 }; arestas.set(k, e); } return e; };
  for (const [a, b, c] of tris) { aresta(a, b).op.push(c); aresta(b, c).op.push(a); aresta(c, a).op.push(b); viz[a].add(b).add(c); viz[b].add(a).add(c); viz[c].add(a).add(b); }
  for (const e of arestas.values()) if (e.op.length === 1) { borda[e.a].push(e.b); borda[e.b].push(e.a); }
  const NP = [], NU = [], NW = [];
  P.forEach((v, i) => {
    const q = V();
    if (borda[i].length === 2) q.copy(v).multiplyScalar(0.75).addScaledVector(P[borda[i][0]], 0.125).addScaledVector(P[borda[i][1]], 0.125);
    else if (borda[i].length) q.copy(v);
    else { const n = viz[i].size, beta = n === 3 ? 3 / 16 : 3 / (8 * n); q.copy(v).multiplyScalar(1 - n * beta); for (const j of viz[i]) q.addScaledVector(P[j], beta); }
    NP.push(q); NU.push(U[i]); NW.push(W[i]);
  });
  for (const e of arestas.values()) {
    const q = V().addVectors(P[e.a], P[e.b]);
    if (e.op.length === 2) q.multiplyScalar(0.375).addScaledVector(P[e.op[0]], 0.125).addScaledVector(P[e.op[1]], 0.125); else q.multiplyScalar(0.5);
    const m = new Map();
    for (const w of [W[e.a], W[e.b]]) for (const [k, x] of w) m.set(k, (m.get(k) || 0) + x / 2);
    e.id = NP.length; NP.push(q); NU.push([(U[e.a][0] + U[e.b][0]) / 2, (U[e.a][1] + U[e.b][1]) / 2]); NW.push(m);
  }
  const n = NP.length, p = new Float32Array(n * 3), u = new Float32Array(n * 2), ji = new Uint16Array(n * 4), jw = new Float32Array(n * 4), idx = [];
  for (let i = 0; i < n; i++) {
    NP[i].toArray(p, i * 3); u[2 * i] = NU[i][0]; u[2 * i + 1] = NU[i][1];
    const maiores = [...NW[i]].sort((x, y) => y[1] - x[1]).slice(0, 4), soma = maiores.reduce((t, x) => t + x[1], 0) || 1;
    maiores.forEach(([k, x], c) => { ji[4 * i + c] = k; jw[4 * i + c] = x / soma; });
  }
  for (const [a, b, c] of tris) { const ab = aresta(a, b).id, bc = aresta(b, c).id, ca = aresta(c, a).id; idx.push(a, ab, ca, b, bc, ab, c, ca, bc, ab, bc, ca); }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(p, 3)); g.setAttribute("uv", new THREE.BufferAttribute(u, 2));
  g.setAttribute("skinIndex", new THREE.BufferAttribute(ji, 4)); g.setAttribute("skinWeight", new THREE.BufferAttribute(jw, 4));
  g.setIndex(idx); g.computeVertexNormals();
  return g;
}

// ---- montagem de um modelo (uma vez por mão e por lado) ----
function montar(gltf, cor) {
  const raiz = gltf.scene;
  let pele = null;
  raiz.traverse(o => { if (o.isSkinnedMesh) pele = o; });
  raiz.updateMatrixWorld(true);
  pele.geometry = subdividir(subdividir(pele.geometry));
  const osso = n => raiz.getObjectByName(n);
  const rp = NOMES.map(n => osso(n).getWorldPosition(V()));   // os 21 pontos na pose de repouso do modelo
  // de que lado fica a palma no modelo: o polegar fica do lado da palma
  const cruz = V().crossVectors(v1.subVectors(rp[5], rp[0]), v2.subVectors(rp[17], rp[0]));
  const lado = Math.sign(v1.subVectors(rp[4], rp[0]).dot(cruz)) || 1;
  const B0 = direcoes(rp, lado, novaBase());
  const M = { raiz, lado, segs: [], palma0: rp[0].distanceTo(rp[9]) };

  const seg = (bone, a, b, ref, extra = {}) => {
    const pa = typeof a === "number" ? rp[a] : a, T0 = V().subVectors(rp[b], pa), len0 = T0.length();
    T0.normalize();
    const qf = new THREE.Quaternion(); quadro(T0, extra.ref0 || B0[ref], qf);
    const s = { bone, a, b, ref, T0, len0, qInv: qf.invert(), q0: bone.quaternion.clone(), ...extra };
    M.segs.push(s); return s;
  };
  const comPonta = (s, tip) => { s.tip = tip; s.qTip0 = tip.quaternion.clone(); s.lenTip0 = s.len0; s.len0 += ALEM_DA_PONTA; return s; };
  // o dorso de cada osso é um dos sentidos do eixo Y local; qual deles, vê-se num dedo (contra a direção da palma)
  const eixoY = b => V().set(0, 1, 0).applyQuaternion(b.getWorldQuaternion(new THREE.Quaternion()));
  const sinalDorso = eixoY(osso(NOMES[7])).dot(B0.nP) < 0 ? 1 : -1;
  // no polegar a referência é para onde a polpa olha; guardada também em relação à palma (padLoc), para a mão aberta
  const polpa = b => ({ ref0: eixoY(b).multiplyScalar(-sinalDorso), padLoc: eixoY(b).multiplyScalar(-sinalDorso).applyQuaternion(B0.qPol.clone().invert()) });
  M.punho = seg(osso("wrist"), 0, 9, "lat");
  M.segs.pop();   // o punho é tratado à parte (define a pose da palma)
  seg(osso(NOMES[1]), 1, 2, "pol", polpa(osso(NOMES[1]))); seg(osso(NOMES[2]), 2, 3, "pol", polpa(osso(NOMES[2])));
  comPonta(seg(osso(NOMES[3]), 3, 4, "pol", polpa(osso(NOMES[3]))), osso(NOMES[4]));
  for (let f = 0; f < 4; f++) {
    const m = 5 + 4 * f, meta = osso(METACARPOS[f]), pm = meta.getWorldPosition(V());
    seg(meta, pm, m, "lat", { desloc: pm.clone().sub(rp[0]) });   // metacarpo: nasce da palma, aponta para o nó do dedo
    seg(osso(NOMES[m]), m, m + 1, "lat"); seg(osso(NOMES[m + 1]), m + 1, m + 2, "lat");
    comPonta(seg(osso(NOMES[m + 2]), m + 2, m + 3, "lat"), osso(NOMES[m + 3]));
  }

  // pele: material, e cor por vértice (palma mais clara e rosada; pontas dos dedos um pouco mais vermelhas)
  const geo = pele.geometry, pos = geo.attributes.position, nor = geo.attributes.normal, n = pos.count;
  const cores = new Float32Array(n * 3), nomesOssos = pele.skeleton.bones.map(b => b.name);
  const si = geo.attributes.skinIndex, sw = geo.attributes.skinWeight;
  for (let i = 0; i < n; i++) {
    const palma = THREE.MathUtils.smoothstep(v1.fromBufferAttribute(nor, i).dot(B0.nP), -0.25, 0.6);
    let ponta = 0;
    for (let k = 0; k < 4; k++) if (/tip$|distal$/.test(nomesOssos[si.getComponent(i, k)] || "")) ponta += sw.getComponent(i, k);
    const t = palma * 0.9, q = ponta * palma * 0.7;
    cores[3 * i] = (1 + 0.07 * t) * (1 + 0.03 * q); cores[3 * i + 1] = (1 - 0.02 * t) * (1 - 0.09 * q); cores[3 * i + 2] = (1 - 0.05 * t) * (1 - 0.1 * q);
  }
  geo.setAttribute("color", new THREE.BufferAttribute(cores, 3));
  const mat = materialPele(cor);
  pele.material = mat; pele.castShadow = true; pele.frustumCulled = false;

  // alvo dos raios na pose de repouso (para achar a superfície onde vão as unhas)
  const alvo = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ side: THREE.DoubleSide }));
  alvo.updateMatrixWorld(true);
  const raio = new THREE.Raycaster();
  const bate = (o, d, longe) => { raio.set(o, d); raio.far = 0.06; const h = raio.intersectObject(alvo, false); return h.length ? (longe ? h[h.length - 1] : h[0]) : null; };

  // unhas: uma grade colada na superfície do dorso da última falange, filha do osso (acompanha o dedo)
  const matUnha = new THREE.MeshPhysicalMaterial({ color: new THREE.Color(cor).lerp(new THREE.Color(0xf0b4ac), 0.6), roughness: 0.28, ior: 1.5,
    clearcoat: 0.6, clearcoatRoughness: 0.2, side: THREE.DoubleSide });
  const NU = 10, NV = 9;
  for (const s of M.segs) {
    if (!s.tip) continue;
    const o = rp[s.a], T = s.T0, polegar = s.ref === "pol";
    const dorso = V().set(0, sinalDorso, 0).applyQuaternion(s.bone.getWorldQuaternion(new THREE.Quaternion()));
    dorso.addScaledVector(T, -dorso.dot(T)).normalize();
    const ladoU = V().crossVectors(T, dorso);
    const fim = bate(v1.copy(o).addScaledVector(T, 0.002), T, true);
    if (!fim) continue;
    const L = fim.distance + 0.002, u0 = polegar ? 0.4 : 0.44, u1 = 0.93, pts = [];
    let ok = true;
    for (let u = 0; u < NU && ok; u++) {
      const uu = u / (NU - 1), eixo = V().copy(o).addScaledVector(T, L * (u0 + (u1 - u0) * uu));
      // largura ao longo da unha: cutícula arredondada, meio largo, borda livre levemente arredondada
      const larg = (polegar ? 0.82 : 0.98) * Math.pow(Math.min(1, uu * 3 + 0.25), 0.5) * (1 - 0.3 * Math.pow(Math.max(0, (uu - 0.72) / 0.28), 2));
      for (let v = 0; v < NV; v++) {
        const a = (v / (NV - 1) * 2 - 1) * larg, d = V().copy(dorso).multiplyScalar(Math.cos(a)).addScaledVector(ladoU, Math.sin(a));
        const h = bate(eixo, d, false);
        if (!h) { ok = false; break; }
        const sobe = 0.00035 + 0.00025 * (1 - Math.abs(a) / larg) + 0.0003 * uu * uu;   // um pouco acima da pele, mais no meio e na borda livre
        pts.push(s.bone.worldToLocal(h.point.clone().addScaledVector(d, sobe)));
      }
    }
    if (!ok) continue;
    const g = new THREE.BufferGeometry(), idx = [];
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(pts.flatMap(q => [q.x, q.y, q.z])), 3));
    for (let u = 0; u < NU - 1; u++) for (let v = 0; v < NV - 1; v++) { const a = u * NV + v, b = a + NV; idx.push(a, a + 1, b, a + 1, b + 1, b); }
    g.setIndex(idx); g.computeVertexNormals();
    // normais para fora (o lado do dorso), qualquer que seja o sentido dos triângulos
    const dl = dorso.clone().transformDirection(new THREE.Matrix4().copy(s.bone.matrixWorld).invert()), nn = g.attributes.normal;
    if (v1.fromBufferAttribute(nn, Math.floor(NU / 2) * NV + Math.floor(NV / 2)).dot(dl) < 0) for (let i = 0; i < nn.count; i++) nn.setXYZ(i, -nn.getX(i), -nn.getY(i), -nn.getZ(i));
    const unha = new THREE.Mesh(g, matUnha); unha.frustumCulled = false;
    s.bone.add(unha);
  }

  // antebraço: continua a borda aberta do punho do modelo (filho do osso do punho)
  antebraco(pele, osso("wrist"), mat);
  return M;
}

function antebraco(pele, punho, mat) {
  const geo = pele.geometry, pos = geo.attributes.position, nor = geo.attributes.normal, ind = geo.index.array;
  // vértices repetidos (costuras do UV) contam como um só
  const chave = new Map(), canon = new Int32Array(pos.count);
  for (let i = 0; i < pos.count; i++) {
    const k = `${Math.round(pos.getX(i) * 1e5)},${Math.round(pos.getY(i) * 1e5)},${Math.round(pos.getZ(i) * 1e5)}`;
    if (!chave.has(k)) chave.set(k, i);
    canon[i] = chave.get(k);
  }
  // arestas de borda: as que pertencem a um triângulo só
  const conta = new Map();
  for (let t = 0; t < ind.length; t += 3) for (let e = 0; e < 3; e++) {
    const a = canon[ind[t + e]], b = canon[ind[t + (e + 1) % 3]], k = a < b ? `${a},${b}` : `${b},${a}`;
    conta.set(k, (conta.get(k) || 0) + 1);
  }
  const viz = new Map();
  for (const [k, c] of conta) if (c === 1) { const [a, b] = k.split(",").map(Number); (viz.get(a) || viz.set(a, []).get(a)).push(b); (viz.get(b) || viz.set(b, []).get(b)).push(a); }
  if (!viz.size) return;
  // maior laço fechado
  let melhor = [];
  const visto = new Set();
  for (const ini of viz.keys()) {
    if (visto.has(ini)) continue;
    const laco = [ini]; visto.add(ini);
    let ant = -1, at = ini;
    for (;;) { const prox = viz.get(at).find(x => x !== ant && !visto.has(x)); if (prox === undefined) break; laco.push(prox); visto.add(prox); ant = at; at = prox; }
    if (laco.length > melhor.length) melhor = laco;
  }
  const n = melhor.length;
  if (n < 6) return;
  const inv = new THREE.Matrix4().copy(punho.matrixWorld).invert();
  const P = melhor.map(i => V().fromBufferAttribute(pos, i).applyMatrix4(inv));
  const Nn = melhor.map(i => V().fromBufferAttribute(nor, i).transformDirection(inv));
  const c = P.reduce((s, q) => s.add(q), V()).multiplyScalar(1 / n);
  // os dedos ficam em -Z local do punho; o antebraço vai para o outro lado. Confere com a posição da borda.
  const sentido = c.z >= 0 ? 1 : -1;
  const NR = 10, nv = (NR + 1) * n + 1, p = new Float32Array(nv * 3), uv = new Float32Array(nv * 2), cor = new Float32Array(nv * 3).fill(1), idx = [];
  // seção do antebraço no fim: elipse com os mesmos eixos médios da borda do punho, um pouco mais larga
  let rx = 0, ry = 0;
  for (const q of P) { rx = Math.max(rx, Math.abs(q.x - c.x)); ry = Math.max(ry, Math.abs(q.y - c.y)); }
  for (let k = 0; k <= NR; k++) {
    const t = k / NR, liso = t * t * (3 - 2 * t), cresce = 1 + 0.22 * t;
    for (let i = 0; i < n; i++) {
      const q = P[i], ang = Math.atan2((q.y - c.y) / ry, (q.x - c.x) / rx);
      const ex = c.x + Math.cos(ang) * rx * cresce, ey = c.y + Math.sin(ang) * ry * cresce, m = Math.min(1, liso * 1.6);
      const j = (k * n + i) * 3;
      p[j] = q.x + (ex - q.x) * m; p[j + 1] = q.y + (ey - q.y) * m; p[j + 2] = q.z + (c.z - q.z) * m + sentido * COMP_ANTEBRACO * t;
      uv[(k * n + i) * 2] = i / n * 3; uv[(k * n + i) * 2 + 1] = t * 4;
    }
  }
  const centro = (NR + 1) * n;
  p[centro * 3] = c.x; p[centro * 3 + 1] = c.y; p[centro * 3 + 2] = c.z + sentido * (COMP_ANTEBRACO + 0.004);
  for (let k = 0; k < NR; k++) for (let i = 0; i < n; i++) {
    const a = k * n + i, b = k * n + (i + 1) % n, d = a + n, e = b + n;
    idx.push(a, b, d, b, e, d);
  }
  for (let i = 0; i < n; i++) idx.push(NR * n + i, NR * n + (i + 1) % n, centro);
  // triângulos virados para fora
  const A = V().fromArray(p, 0), B = V().fromArray(p, 3), D = V().fromArray(p, n * 3);
  const face = V().crossVectors(B.clone().sub(A), D.clone().sub(A));
  if (face.dot(A.clone().sub(c).setZ(0)) < 0) for (let t = 0; t < idx.length; t += 3) { const x = idx[t + 1]; idx[t + 1] = idx[t + 2]; idx[t + 2] = x; }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(p, 3)); g.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
  g.setAttribute("color", new THREE.BufferAttribute(cor, 3)); g.setIndex(idx); g.computeVertexNormals();
  // na emenda, as mesmas normais da pele do modelo: a luz passa de um para o outro sem degrau
  for (let i = 0; i < n; i++) g.attributes.normal.setXYZ(i, Nn[i].x, Nn[i].y, Nn[i].z);
  const m = new THREE.Mesh(g, mat); m.castShadow = true; m.frustumCulled = false;
  punho.add(m);
}

// ---- interface ----
const arquivos = {};
function baixar(lado) { return arquivos[lado] || (arquivos[lado] = fetch(ARQUIVOS[lado]).then(r => { if (!r.ok) throw new Error("modelo da mão: " + r.status); return r.arrayBuffer(); })); }

// Cria a mão (as duas versões, esquerda e direita; a cada quadro aparece a que combina com a mão vista).
export function criarMao(cor, pai) {
  const r = { grupo: new THREE.Group(), modelos: {}, base: novaBase() };
  pai.add(r.grupo);
  for (const lado of [-1, 1]) {
    baixar(lado).then(buf => new GLTFLoader().parse(buf.slice(0), "", gltf => {
      const M = montar(gltf, cor);
      M.raiz.visible = false; r.grupo.add(M.raiz); r.modelos[M.lado] = M;
    }, e => console.error("modelo da mão", e))).catch(e => console.error(e));
  }
  return r;
}

// p: os 21 pontos (Vector3, no espaço do pai); lado: de que lado de (5-0) x (17-0) fica a palma.
export function atualizarMao(r, p, lado) {
  const M = r.modelos[lado];
  for (const k in r.modelos) r.modelos[k].raiz.visible = r.modelos[k] === M;
  if (!M) return;
  const B = direcoes(p, lado, r.base);
  // polpa do polegar: com o polegar reto, fica como no modelo em relação à palma; quando ele dobra, olha para dentro da dobra
  v1.subVectors(p[2], p[1]).normalize(); v2.subVectors(p[3], p[2]).normalize(); vPos.subVectors(p[4], p[3]).normalize();
  vDob.subVectors(vPos, v2).addScaledVector(v2, 0.6).addScaledVector(v1, -0.6);
  const dobra = vDob.length(), peso = Math.min(0.6, dobra * 1.5);
  const esc = p[0].distanceTo(p[9]) / M.palma0, gros = THREE.MathUtils.clamp(esc, 0.85, 1.2);   // tamanho da mão e grossura
  // punho: dá a pose da palma
  const W = M.punho;
  if (quadro(B.fd, B.lat, qA)) { qPalma.copy(qA).multiply(W.qInv); W.bone.quaternion.copy(qPalma).multiply(W.q0); }
  W.bone.position.copy(p[0]); W.bone.scale.setScalar(gros);
  for (const s of M.segs) {
    const a = s.desloc ? vPos.copy(s.desloc).applyQuaternion(qPalma).multiplyScalar(esc).add(p[0]) : p[s.a];
    vT.subVectors(p[s.b], a);
    const len = vT.length();
    if (len < 1e-5) continue;
    vT.multiplyScalar(1 / len);
    const ax = len / s.len0;
    s.bone.position.copy(a); s.bone.scale.set(gros, gros, ax);
    let ref = B[s.ref];
    if (s.padLoc) { ref = vRef.copy(s.padLoc).applyQuaternion(B.qPol); if (dobra > 1e-4) ref.multiplyScalar(1 - peso).addScaledVector(vDob, peso / dobra); }
    if (!quadro(vT, ref, qA)) continue;
    qD.copy(qA).multiply(s.qInv);
    s.bone.quaternion.copy(qD).multiply(s.q0);
    if (s.tip) {
      s.tip.position.copy(a).addScaledVector(vT, s.lenTip0 * ax);
      s.tip.quaternion.copy(qD).multiply(s.qTip0); s.tip.scale.set(gros, gros, ax);
    }
  }
}
