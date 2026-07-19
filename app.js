/* ══════════════════════════════════════════════════════════════════════
   Plataforma de Análisis Urbano — lógica del frontend general
   Consume la API Flask de servidor.py (/api/...)
   ══════════════════════════════════════════════════════════════════════ */

const $ = q => document.querySelector(q);
const api = async (u, opt) => {
  const r = await fetch(u, opt);
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
};
let ESTADO = null, MAPA = null, CAPA = null, chCurva = null, chComp = null, GEO_SAT = null;

/* ── navegación ── */
document.querySelectorAll('nav button').forEach(b => b.onclick = () => {
  document.querySelectorAll('nav button').forEach(x => x.classList.remove('act'));
  document.querySelectorAll('main section').forEach(x => x.classList.remove('act'));
  b.classList.add('act'); $('#' + b.dataset.s).classList.add('act');
  if (b.dataset.s === 'mando' && MAPA) setTimeout(() => MAPA.invalidateSize(), 80);
});

/* ── utilidades ── */
const nivel = s => s == null ? ['gris', 'Sin datos'] : s >= .5 ? ['alto', 'Alto'] : s >= .25 ? ['medio', 'Medio'] : ['bajo', 'Bajo'];
// Semáforo del S_RU: verde / ámbar / rojo
const colorScore = s => s == null ? '#9aa0a6' : s >= .5 ? '#d93025' : s >= .25 ? '#f9ab00' : '#188038';
function dz(drop, input) {
  drop.onclick = () => input.click();
  drop.ondragover = e => { e.preventDefault(); drop.classList.add('over'); };
  drop.ondragleave = () => drop.classList.remove('over');
  drop.ondrop = e => {
    e.preventDefault(); drop.classList.remove('over');
    input.files = e.dataTransfer.files; drop.textContent = 'Archivo: ' + input.files[0].name;
  };
  input.onchange = () => { if (input.files[0]) drop.textContent = 'Archivo: ' + input.files[0].name; };
}
dz($('#dropA'), $('#fileA')); dz($('#dropS'), $('#fileS')); dz($('#dropL'), $('#fileL'));

/* ── estado inicial ── */
async function initEstado() {
  ESTADO = await api('/api/estado');
  const chip = (txt, on) => `<span class="chip ${on === undefined ? '' : (on ? 'on' : 'off')}">${txt}</span>`;
  $('#chips').innerHTML =
    chip(`${ESTADO.gpus.length} GPU · ${ESTADO.device}`) +
    chip(`${ESTADO.manzanas} manzanas`) +
    chip(`${ESTADO.manzanas_evaluadas} evaluadas`) +
    chip('Detectron2', ESTADO.detectron2) +
    chip('SAM3', ESTADO.sam3) +
    chip(ESTADO.tomtom_key ? 'TomTom: en vivo' : 'TomTom: histórico', ESTADO.tomtom_key) +
    chip(`Verificación &lt; ${Math.round(ESTADO.umbral_oraculo * 100)} %`);
  $('#tiposA').innerHTML = Object.entries(ESTADO.tipos_fachada).map(([id, n]) =>
    `<label><input type="checkbox" value="${id}" ${['fachada', 'ventanas'].includes(id) ? 'checked' : ''}>${n}</label>`).join('');
  $('#motoresL').innerHTML = ESTADO.motores_comparador.map((m, i) =>
    `<label><input type="checkbox" value="${m}" ${i < 2 ? 'checked' : ''} onchange="promptVis()">${m}</label>`).join('');
  promptVis();
}
function promptVis() {
  const sam = [...document.querySelectorAll('#motoresL input:checked')].some(i => i.value.includes('SAM3'));
  $('#promptWrap').style.display = sam ? 'block' : 'none';
}
window.promptVis = promptVis;

/* ── mapa + ranking ── */
async function refrescarMando() {
  const gj = await (await fetch('/api/manzanas-geojson')).json();
  if (!MAPA) {
    MAPA = L.map('mapa').setView([19.410, -99.172], 15.4);
    // Base clara y neutra: los polígonos del semáforo de riesgo resaltan solos
    L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
      { attribution: '© OpenStreetMap · CARTO', maxZoom: 19 }).addTo(MAPA);
  }
  if (CAPA) MAPA.removeLayer(CAPA);
  CAPA = L.geoJSON(gj, {
    style: f => ({
      color: colorScore(f.properties.score_riesgo), weight: 1.2,
      fillColor: colorScore(f.properties.score_riesgo), fillOpacity: .35
    }),
    onEachFeature: (f, l) => {
      const p = f.properties, s = p.score_riesgo;
      l.bindPopup(`<b class="mono">${p.CVEGEO}</b><br>` +
        (s != null ? `Score <b>${s.toFixed(3)}</b> · congestión ${(p.congestion * 100 || 0).toFixed(0)} %` +
          `<br>Pisos ${p.altura_promedio_pisos ?? '—'} · muestras ${p.num_fotos ?? 0}` +
          `<br>Población ${p.poblacion_estimada ?? '—'} · fuente: ${p.fuente_congestion ?? ''}` : 'Sin datos'));
      l.on('click', () => { $('#selManzana').value = p.CVEGEO; pintaKpis(); });
    }
  }).addTo(MAPA);

  const rk = (await api('/api/riesgo')).manzanas;
  window.RIESGOS = Object.fromEntries(rk.map(r => [r.cvegeo, r]));
  const sel = $('#selManzana'), prev = sel.value;
  sel.innerHTML = gj.features.map(f => {
    const cv = f.properties.CVEGEO;
    const r = window.RIESGOS[cv];
    return `<option value="${cv}">${cv}${r ? '  ·  ' + r.score_riesgo.toFixed(3) : ''}</option>`;
  }).join('');
  if (prev) sel.value = prev;
  sel.onchange = pintaKpis; pintaKpis();

  $('#tblRank tbody').innerHTML = rk.map(r => {
    const [c, t] = nivel(r.score_riesgo);
    return `<tr><td class="mono">${r.cvegeo}</td>
      <td><b>${r.score_riesgo.toFixed(3)}</b></td><td><span class="b ${c}">${t}</span></td>
      <td>${r.danos_ponderados}</td><td>${((r.congestion || 0) * 100).toFixed(0)} %</td>
      <td>${r.altura_promedio_pisos}</td><td>${r.num_fotos}</td>
      <td>${((r.confianza || 0) * 100).toFixed(0)} %</td><td>${r.poblacion_estimada ?? '—'}</td>
      <td><span class="b ${r.fuente_congestion === 'tomtom' ? 'pri' : 'gris'}">${r.fuente_congestion || '—'}</span></td></tr>`;
  }).join('');

  const selS = $('#selSim'), prevS = selS.value;
  selS.innerHTML = rk.map(r => `<option>${r.cvegeo}</option>`).join('');
  if (prevS) selS.value = prevS;
  if (rk.length) simular();

  /* hero global + dona de niveles */
  $('#hManz').textContent = `${rk.length} / ${gj.features.length}`;
  $('#hEdif').textContent = rk.reduce((a, r) => a + (r.num_fotos || 0), 0).toLocaleString();
  $('#hPob').textContent = rk.reduce((a, r) => a + (r.poblacion_estimada || 0), 0).toLocaleString() || '—';
  $('#hScore').textContent = rk.length ? Math.max(...rk.map(r => r.score_riesgo)).toFixed(3) : '—';
  const niveles = { Bajo: 0, Medio: 0, Alto: 0, ['Sin datos']: gj.features.length - rk.length };
  rk.forEach(r => { niveles[nivel(r.score_riesgo)[1]]++; });
  if (window.chDona) window.chDona.destroy();
  window.chDona = new Chart($('#chDona'), {
    type: 'doughnut', data: {
      labels: Object.keys(niveles),
      datasets: [{
        data: Object.values(niveles),
        backgroundColor: ['#188038', '#f9ab00', '#d93025', '#dadce0'],
        borderColor: '#ffffff', borderWidth: 2
      }]
    },
    options: {
      cutout: '62%', plugins: {
        legend: {
          position: 'right',
          labels: { color: '#202124', font: { family: 'Roboto', size: 12 }, boxWidth: 14, usePointStyle: true }
        }
      }
    }
  });
}
function pintaKpis() {
  const r = window.RIESGOS?.[$('#selManzana').value];
  $('#kScore').textContent = r ? r.score_riesgo.toFixed(3) : '—';
  $('#kCong').textContent = r ? ((r.congestion || 0) * 100).toFixed(0) + ' %' : '—';
  $('#kPisos').textContent = r ? r.altura_promedio_pisos : '—';
  $('#kPob').textContent = r ? (r.poblacion_estimada ?? '—') : '—';
  $('#kMeta').textContent = r ? `Confianza ${(r.confianza * 100).toFixed(0)} % · ${r.num_fotos} muestras · daños ponderados ${r.danos_ponderados} · congestión: ${r.fuente_congestion}` : 'Sin evaluación — ejecuta el pipeline.';
}

/* ── acciones ── */
async function accion(btn, sp, fn) {
  btn.disabled = true; if (sp) sp.classList.add('on');
  try { await fn(); } catch (e) { alert(e.message); }
  finally { btn.disabled = false; if (sp) sp.classList.remove('on'); }
}
$('#btnAnalizar').onclick = () => accion($('#btnAnalizar'), $('#spMando'), async () => {
  await api('/api/analizar', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ cvegeo: $('#selManzana').value })
  }); await refrescarMando();
});
$('#btnTodas').onclick = () => accion($('#btnTodas'), $('#spMando'), async () => {
  await api('/api/analizar-todas', { method: 'POST' }); await refrescarMando();
});
$('#btnImportar').onclick = () => accion($('#btnImportar'), $('#spMando'), async () => {
  const r = await api('/api/importar-sam3', { method: 'POST' });
  alert(`CSV importado: ${r.insertadas} edificios (${r.con_manzana} con manzana). ` +
    `${r.manzanas_reevaluadas} manzanas reevaluadas.`);
  await refrescarMando();
});

/* Descarga de PDF: abre el diálogo nativo "Guardar como…" (File System Access
   API, Chrome/Edge) para elegir carpeta y nombre; en Firefox/Safari cae a la
   descarga normal del navegador. */
async function guardarPdf(url, nombreSugerido) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error('El servidor no pudo generar el reporte');
  const blob = await resp.blob();
  if (window.showSaveFilePicker) {
    const handle = await window.showSaveFilePicker({
      suggestedName: nombreSugerido,
      types: [{ description: 'Documento PDF', accept: { 'application/pdf': ['.pdf'] } }]
    });
    const writable = await handle.createWritable();
    await writable.write(blob); await writable.close();
  } else {
    const objUrl = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = objUrl; a.download = nombreSugerido;
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(objUrl);
  }
}
$('#btnPdf').onclick = () => accion($('#btnPdf'), null, async () => {
  const cvegeo = $('#selManzana').value;
  try { await guardarPdf('/api/pdf?cvegeo=' + cvegeo, `reporte_${cvegeo}.pdf`); }
  catch (e) { if (e.name !== 'AbortError') throw e; } // AbortError = el usuario canceló el diálogo
});
$('#btnGeo').onclick = () => {
  fetch('/api/manzanas-geojson').then(r => r.blob()).then(b => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(b); a.download = 'riesgo_manzanas.geojson'; a.click();
  });
};

/* ── bitácora ── */
async function pollTerm() {
  try {
    const { bitacora } = await api('/api/bitacora');
    $('#term').innerHTML = bitacora.map(e => {
      const t = e.ts?.slice(11, 19) || ''; const { etapa, ts, ...rest } = e;
      return `<div><span class="t">[${t}]</span> <span class="e">${etapa}</span> ` +
        `<span class="d">${Object.entries(rest).map(([k, v]) => k + '=' + JSON.stringify(v)).join(' ')}</span></div>`;
    }).join('');
    $('#term').scrollTop = $('#term').scrollHeight;
  } catch (e) { }
  setTimeout(pollTerm, 4000);
}

/* ── análisis de fachada ── */
$('#confA').oninput = e => $('#confAv').textContent = e.target.value;
$('#btnDetectar').onclick = () => accion($('#btnDetectar'), $('#spA'), async () => {
  const f = $('#fileA').files[0]; if (!f) throw new Error('Selecciona una imagen');
  const fd = new FormData(); fd.append('imagen', f);
  document.querySelectorAll('#tiposA input:checked').forEach(i => fd.append('tipo', i.value));
  fd.append('conf', $('#confA').value);
  $('#cardsA').innerHTML = '';
  const r = await api('/api/detectar', { method: 'POST', body: fd });
  $('#kpisA').style.display = 'grid';
  $('#aVent').textContent = r.ventanas; $('#aPisos').textContent = r.numero_pisos;
  $('#aConf').textContent = (r.confianza_ventanas * 100).toFixed(0) + ' %';
  $('#aOra').textContent = r.correccion ? r.correccion.motor + ' ✓' : 'No requerida';
  $('#cardsA').innerHTML = Object.values(r.resultados).map(x => `
    <div class="card"><div class="h">${ESTADO.tipos_fachada[x.tipo] || x.tipo} — confianza ${(x.confianza_promedio * 100).toFixed(0)} %</div>
    ${x.imagen_base64 ? `<img src="data:image/jpeg;base64,${x.imagen_base64}">` : ''}
    <div class="meta">${Object.entries(x.conteo_clases).map(([k, v]) => `${k}: ${v}`).join(' · ') || 'Sin detecciones'}
    ${x.total_danos ? `<br><span class="b alto">${x.total_danos} daños detectados (ponderados: ${x.danos_ponderados})</span>` : ''}
    </div></div>`).join('');
});

/* ── satélite ── */
$('#tileS').oninput = e => $('#tileV').textContent = e.target.value;
$('#overS').oninput = e => $('#overV').textContent = e.target.value;
$('#btnSat').onclick = () => accion($('#btnSat'), $('#spS'), async () => {
  const f = $('#fileS').files[0]; if (!f) throw new Error('Selecciona una imagen satelital');
  const fd = new FormData(); fd.append('imagen', f);
  ['lat_nw|latNW', 'lon_nw|lonNW', 'lat_se|latSE', 'lon_se|lonSE', 'escala|escala', 'conf|confS',
    'tile_size|tileS', 'overlap|overS']
    .forEach(p => { const [k, id] = p.split('|'); fd.append(k, $('#' + id).value); });
  const r = await api('/api/satelite', { method: 'POST', body: fd });
  GEO_SAT = r.geojson; $('#btnSatGeo').style.display = 'inline-block';
  $('#kpisS').style.display = 'grid';
  $('#sNum').textContent = r.num_detecciones;
  $('#sArea').textContent = r.area_total_m2.toLocaleString() + ' m²';
  $('#cardsS').innerHTML = `<div class="card"><div class="h">Polígonos detectados</div>
    <img src="data:image/jpeg;base64,${r.imagen_base64}"></div>`;
  const tb = $('#tblSat'); tb.style.display = 'table';
  tb.querySelector('tbody').innerHTML = r.detecciones.map(d =>
    `<tr><td>${d.id}</td><td>${d.clase}</td><td>${(d.confianza * 100).toFixed(0)} %</td>
     <td>${d.area_m2}</td><td class="mono">${d.lat}</td><td class="mono">${d.lon}</td></tr>`).join('');
});
$('#btnSatGeo').onclick = () => {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(GEO_SAT)], { type: 'application/geo+json' }));
  a.download = 'deteccion_satelital.geojson'; a.click();
};

/* ── laboratorio ── */
$('#confL').oninput = e => $('#confLv').textContent = e.target.value;
$('#btnLab').onclick = () => accion($('#btnLab'), $('#spL'), async () => {
  const f = $('#fileL').files[0]; if (!f) throw new Error('Selecciona una imagen');
  const fd = new FormData(); fd.append('imagen', f);
  document.querySelectorAll('#motoresL input:checked').forEach(i => fd.append('motor', i.value));
  fd.append('conf', $('#confL').value); fd.append('prompt', $('#promptL').value);
  $('#cardsL').innerHTML = '';
  const r = await api('/api/comparador', { method: 'POST', body: fd });
  $('#cardsL').innerHTML = Object.entries(r.resultados).map(([m, x]) => `
    <div class="card"><div class="h">${m}</div>
    ${x.error ? `<div class="meta" style="color:var(--bad)">Error: ${x.error}</div>`
      : `<img src="data:image/jpeg;base64,${x.imagen_base64}">
        <div class="meta">${x.n_detecciones} ventanas · confianza promedio ${(x.confianza_promedio * 100).toFixed(0)} %</div>`}
    </div>`).join('');
});

/* ── simulador (series: score azul sólido / congestión ámbar punteada) ── */
const gridCfg = { color: '#e8eaed' }, tickCfg = { color: '#5f6368', font: { family: 'Roboto', size: 11 } };
const legendCfg = { labels: { color: '#202124', font: { family: 'Roboto', size: 12 }, boxWidth: 16, usePointStyle: true } };
async function simular() {
  const cv = $('#selSim').value; if (!cv) return;
  const q = new URLSearchParams({
    cvegeo: cv, dia: $('#diaSim').value,
    hora: $('#horaSim').value, factor: $('#factSim').value
  });
  try {
    const r = await api('/api/simular?' + q);
    $('#simAct').textContent = r.score_actual.toFixed(3);
    $('#simSim').textContent = r.score_simulado.toFixed(3);
    $('#simCong').textContent = (r.congestion_simulada * 100).toFixed(0) + ' %';
    const horas = [...Array(24).keys()].map(h => h + ':00');
    if (chCurva) chCurva.destroy();
    chCurva = new Chart($('#chCurva'), {
      type: 'line', data: {
        labels: horas, datasets: [
          {
            label: 'Score de riesgo', data: r.curva_scores, borderColor: '#1a73e8', borderWidth: 2,
            pointRadius: 2, pointHoverRadius: 5, tension: .35, fill: false,
            pointBackgroundColor: '#1a73e8'
          },
          {
            label: 'Congestión (perfil histórico)', data: r.curva_congestion, borderColor: '#e37400', borderWidth: 2,
            borderDash: [6, 4], pointRadius: 0, pointHoverRadius: 5, tension: .35, fill: false
          }]
      },
      options: {
        responsive: true, interaction: { mode: 'index', intersect: false },
        plugins: { legend: legendCfg },
        scales: { y: { min: 0, max: 1, grid: gridCfg, ticks: tickCfg }, x: { grid: gridCfg, ticks: tickCfg } }
      }
    });
    if (chComp) chComp.destroy();
    const top = r.comparativa.slice(0, 12);
    chComp = new Chart($('#chComp'), {
      type: 'bar', data: {
        labels: top.map(x => x.cvegeo.slice(-5)), datasets: [
          { label: 'Score simulado', data: top.map(x => x.simulado), backgroundColor: '#d93025', borderRadius: 3, maxBarThickness: 18 },
          { label: 'Score actual', data: top.map(x => x.actual), backgroundColor: '#9aa0a6', borderRadius: 3, maxBarThickness: 18 }]
      },
      options: {
        responsive: true,
        plugins: { legend: legendCfg, tooltip: { callbacks: { title: i => '…' + i[0].label } } },
        scales: { y: { min: 0, max: 1, grid: gridCfg, ticks: tickCfg }, x: { grid: { display: false }, ticks: tickCfg } }
      }
    });
  } catch (e) {/* manzana sin evaluación */ }
}
['selSim', 'diaSim'].forEach(id => $('#' + id).onchange = simular);
$('#horaSim').oninput = e => { $('#horaSimV').textContent = e.target.value; simular(); };
$('#factSim').oninput = e => { $('#factSimV').textContent = '×' + (+e.target.value).toFixed(1); simular(); };

/* ── tráfico ── */
$('#btnTraf').onclick = () => accion($('#btnTraf'), $('#spT'), async () => {
  const r = await api('/api/trafico', { method: 'POST' });
  const tb = $('#tblTraf'); tb.style.display = 'table';
  tb.querySelector('tbody').innerHTML = r.calles.map(c =>
    `<tr><td>${c.vialidad}</td><td>${c.congestion != null ? (c.congestion * 100).toFixed(0) + ' %' : 's/d'}</td>
     <td><span class="b ${c.fuente === 'tomtom' ? 'pri' : 'gris'}">${c.fuente}</span></td>
     <td class="mono">${c.cvegeo || '—'}</td></tr>`).join('');
  await refrescarMando();
});

/* ── inicio ── */
initEstado().then(refrescarMando).then(pollTerm).catch(e => {
  $('#chips').innerHTML = `<span class="chip" style="color:var(--bad);border-color:#f3c6c6">Sin conexión: ${e.message}</span>`;
});
