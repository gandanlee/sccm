/* ============================================================
   SCCM project page — 3-panel reconstruction gallery viewer
   GT / EDM / SCCM side by side with shared orbit camera.
   three.js (module) + a tiny custom orbit controller.
   ============================================================ */
import * as THREE from 'three';

(function () {
  'use strict';

  var DATA_DIR = 'static/data/';
  var MODELS = ['gt', 'edm', 'sccm'];

  var elPrev = document.getElementById('pcPrev');
  var elNext = document.getElementById('pcNext');
  var elCounter = document.getElementById('pcCounter');
  var elScene = document.getElementById('pcScene');

  function pad(n) { return (n < 10 ? '0' : '') + n; }

  /* ---- parse .bin: uint32 N | float32 xyz[N*3] | uint8 rgb[N*3] ---- */
  function parseCloud(buf) {
    var dv = new DataView(buf);
    var N = dv.getUint32(0, true);
    var pos = new Float32Array(buf.slice(4, 4 + N * 12));
    var rgb = new Uint8Array(buf.slice(4 + N * 12, 4 + N * 12 + N * 3));
    var col = new Float32Array(N * 3);
    for (var i = 0; i < N * 3; i++) col[i] = rgb[i] / 255;
    return { N: N, pos: pos, col: col };
  }
  function bounds(pos) {
    var lo = [1e9, 1e9, 1e9], hi = [-1e9, -1e9, -1e9], k;
    for (var i = 0; i < pos.length; i += 3)
      for (k = 0; k < 3; k++) {
        if (pos[i + k] < lo[k]) lo[k] = pos[i + k];
        if (pos[i + k] > hi[k]) hi[k] = pos[i + k];
      }
    return { radius: Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) * 0.5 || 1 };
  }
  function makePoints(cloud, ptSize) {
    var g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(cloud.pos, 3));
    g.setAttribute('color', new THREE.BufferAttribute(cloud.col, 3));
    return new THREE.Points(g, new THREE.PointsMaterial(
      { size: ptSize, vertexColors: true, sizeAttenuation: true }));
  }

  /* ---- shared camera state across all 3 viewers ---- */
  var target = new THREE.Vector3(0, 0, 0);
  var sph = { r: 6, theta: 0.7, phi: 1.12 };
  var autoRotate = true;

  /* ---- per-viewer setup ---- */
  var viewers = MODELS.map(function (m) {
    var host = document.getElementById('pcViewer-' + m);
    if (!host) return null;
    var renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    host.appendChild(renderer.domElement);

    var loadEl = document.createElement('div');
    loadEl.className = 'pc-loading'; loadEl.style.display = 'none';
    host.appendChild(loadEl);

    var scene = new THREE.Scene();
    var camera = new THREE.PerspectiveCamera(48, 1, 0.01, 5000);
    return { model: m, host: host, renderer: renderer, scene: scene,
             camera: camera, loadEl: loadEl, current: null };
  }).filter(Boolean);

  if (viewers.length === 0) return;

  function applyCamera() {
    var s = Math.sin(sph.phi);
    var px = sph.r * s * Math.sin(sph.theta);
    var py = sph.r * Math.cos(sph.phi);
    var pz = sph.r * s * Math.cos(sph.theta);
    viewers.forEach(function (v) {
      v.camera.position.set(px, py, pz);
      v.camera.lookAt(target);
    });
  }
  function resize() {
    viewers.forEach(function (v) {
      var w = v.host.clientWidth, h = v.host.clientHeight;
      if (!w || !h) return;
      v.renderer.setSize(w, h, false);
      v.camera.aspect = w / h; v.camera.updateProjectionMatrix();
    });
  }
  window.addEventListener('resize', resize);

  /* ---- shared orbit: attach handlers to every canvas ---- */
  viewers.forEach(function (v) {
    var el = v.renderer.domElement;
    el.style.touchAction = 'none';
    var dragging = false, lastX = 0, lastY = 0;
    el.addEventListener('pointerdown', function (e) {
      dragging = true; autoRotate = false;
      lastX = e.clientX; lastY = e.clientY;
      try { el.setPointerCapture(e.pointerId); } catch (_) {}
    });
    el.addEventListener('pointermove', function (e) {
      if (!dragging) return;
      sph.theta -= (e.clientX - lastX) * 0.007;
      sph.phi = Math.max(0.15, Math.min(Math.PI - 0.15,
                sph.phi - (e.clientY - lastY) * 0.007));
      lastX = e.clientX; lastY = e.clientY;
    });
    function endDrag() { dragging = false; }
    el.addEventListener('pointerup', endDrag);
    el.addEventListener('pointercancel', endDrag);
    el.addEventListener('wheel', function (e) {
      e.preventDefault();
      sph.r = Math.max(0.4, Math.min(80, sph.r * (e.deltaY > 0 ? 1.1 : 0.9)));
    }, { passive: false });
  });

  function loop() {
    requestAnimationFrame(loop);
    if (autoRotate) sph.theta += 0.0024;
    applyCamera();
    viewers.forEach(function (v) { v.renderer.render(v.scene, v.camera); });
  }

  /* ---- gallery state ---- */
  var manifest = [], idx = 0, cache = {}, started = false;

  var elMetricEdm = document.getElementById('pcMetric-edm');
  var elMetricSccm = document.getElementById('pcMetric-sccm');

  function updateBar() {
    var m = manifest[idx] || {};
    elCounter.textContent = (idx + 1) + ' / ' + manifest.length;
    elScene.innerHTML = m.scene
      ? 'scene <b>' + m.scene + '</b> &nbsp;overlap ' + (m.overlap != null ? m.overlap.toFixed(2) : '—')
      : '';
    if (elMetricEdm)  elMetricEdm.innerHTML  = formatMetric(m, 'edm');
    if (elMetricSccm) elMetricSccm.innerHTML = formatMetric(m, 'sccm');
  }
  function formatMetric(m, tag) {
    if (m['acc_' + tag] == null) return '&nbsp;';
    return 'Acc&darr; <b>' + m['acc_' + tag].toFixed(2) + 'm</b>'
         + ' &nbsp;F@10cm&uarr; <b>' + m['f10_' + tag].toFixed(2) + '</b>';
  }

  function ensure(mdl, i) {
    var k = mdl + '_' + pad(manifest[i].i);
    if (cache[k]) return Promise.resolve(cache[k]);
    return fetch(DATA_DIR + k + '.bin')
      .then(function (r) { return r.ok ? r.arrayBuffer() : Promise.reject(); })
      .then(function (buf) {
        var c = parseCloud(buf), b = bounds(c.pos);
        var e = { pts: makePoints(c, Math.max(b.radius * 0.006, 0.004)), b: b };
        cache[k] = e; return e;
      });
  }

  function render(i, keepCam) {
    idx = i; updateBar();
    var promises = viewers.map(function (v) {
      v.loadEl.textContent = 'loading…'; v.loadEl.style.display = 'flex';
      return ensure(v.model, i)
        .then(function (e) {
          v.loadEl.style.display = 'none';
          if (v.current) v.scene.remove(v.current);
          v.current = e.pts; v.scene.add(v.current);
          return e.b.radius;
        })
        .catch(function () {
          v.loadEl.textContent = 'unavailable';
          if (v.current) { v.scene.remove(v.current); v.current = null; }
          return 1;
        });
    });
    Promise.all(promises).then(function (radii) {
      if (!keepCam) {
        sph.r = Math.max.apply(null, radii) * 2.6;
        sph.theta = 0.7; sph.phi = 1.12; autoRotate = true;
      }
      if (!started) { started = true; resize(); loop(); }
    });
  }

  function show(i) { render((i + manifest.length) % manifest.length, false); }

  elPrev.addEventListener('click', function () { show(idx - 1); });
  elNext.addEventListener('click', function () { show(idx + 1); });
  document.addEventListener('keydown', function (e) {
    if (e.key === 'ArrowLeft')  show(idx - 1);
    if (e.key === 'ArrowRight') show(idx + 1);
  });

  /* ---- boot ---- */
  fetch(DATA_DIR + 'manifest.json?v=' + Date.now())
    .then(function (r) { return r.ok ? r.json() : Promise.reject(); })
    .then(function (list) {
      if (!list || !list.length) { return; }
      manifest = list;
      render(0, false);
    })
    .catch(function () {
      viewers.forEach(function (v) {
        v.loadEl.className = 'pc-fail';
        v.loadEl.textContent = 'gallery unavailable';
        v.loadEl.style.display = 'flex';
      });
    });
})();
