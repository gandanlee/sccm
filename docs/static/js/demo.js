/* ERP metric-stretch demo: a fixed-size pixel window on the ERP chart vs. its footprint on the sphere.
   Latitude is signed, -90 (south pole) .. +90 (north pole). */
(function () {
  var cv = document.getElementById('erpCanvas');
  if (!cv || !cv.getContext) return;
  var ctx = cv.getContext('2d');
  var W = cv.width, H = cv.height;
  var slider = document.getElementById('latSlider');
  var roLat = document.getElementById('roLat');
  var roCos = document.getElementById('roCos');
  var roInv = document.getElementById('roInv');
  var DEG = Math.PI / 180;
  var C = {
    panel: '#ffffff', border: '#d9d9d9', grid: '#ececec', faint: '#8a8a8a',
    ink: '#363636', accent: '#3273dc', sphere1: '#f6f6f7', sphere2: '#dcdce2'
  };

  function roundRect(x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }
  function rgba(hex, a) {
    var n = parseInt(hex.slice(1), 16);
    return 'rgba(' + (n >> 16 & 255) + ',' + (n >> 8 & 255) + ',' + (n & 255) + ',' + a + ')';
  }
  function fmtLat(p) {
    if (p === 0) return '0°';
    return (p > 0 ? '+' : '−') + Math.abs(p).toFixed(0) + '°';
  }

  function draw() {
    var phi = parseFloat(slider.value);
    var cosp = Math.max(0, Math.cos(phi * DEG));
    ctx.clearRect(0, 0, W, H);

    /* ---- left: ERP chart, true 2:1 aspect, -90..+90 ---- */
    var cx0 = 78, cw = 440, ch = 220, cy0 = (H - ch) / 2 + 10;
    ctx.fillStyle = C.panel; ctx.strokeStyle = C.border; ctx.lineWidth = 1.5;
    roundRect(cx0, cy0, cw, ch, 6); ctx.fill(); ctx.stroke();

    ctx.fillStyle = C.ink; ctx.font = '600 22px "Noto Sans", sans-serif'; ctx.textAlign = 'left';
    ctx.fillText('ERP image', cx0, cy0 - 18);

    ctx.strokeStyle = C.grid; ctx.lineWidth = 1; ctx.beginPath();
    for (var lat = -60; lat <= 60; lat += 30) {
      var gy = cy0 + ch * (0.5 - lat / 180);
      ctx.moveTo(cx0, gy); ctx.lineTo(cx0 + cw, gy);
    }
    for (var k = 1; k < 8; k++) {
      var gx = cx0 + cw * k / 8;
      ctx.moveTo(gx, cy0); ctx.lineTo(gx, cy0 + ch);
    }
    ctx.stroke();
    ctx.strokeStyle = C.border; ctx.setLineDash([2, 3]); ctx.beginPath();
    ctx.moveTo(cx0, cy0 + ch / 2); ctx.lineTo(cx0 + cw, cy0 + ch / 2); ctx.stroke(); ctx.setLineDash([]);

    ctx.fillStyle = C.faint; ctx.font = '17px "Noto Sans", sans-serif'; ctx.textAlign = 'right';
    ctx.fillText('+90°', cx0 - 10, cy0 + 14);
    ctx.fillText('0°', cx0 - 10, cy0 + ch / 2 + 6);
    ctx.fillText('−90°', cx0 - 10, cy0 + ch);

    var selY = cy0 + ch * (0.5 - phi / 180);
    ctx.strokeStyle = C.accent; ctx.lineWidth = 1.5; ctx.setLineDash([5, 4]);
    ctx.beginPath(); ctx.moveTo(cx0, selY); ctx.lineTo(cx0 + cw, selY); ctx.stroke(); ctx.setLineDash([]);

    var ws = 44;
    var wx = cx0 + cw * 0.5 - ws / 2;
    var wy = Math.max(cy0 + 2, Math.min(cy0 + ch - ws - 2, selY - ws / 2));
    ctx.fillStyle = rgba(C.accent, 0.15); ctx.strokeStyle = C.accent; ctx.lineWidth = 2.5;
    roundRect(wx, wy, ws, ws, 4); ctx.fill(); ctx.stroke();
    ctx.fillStyle = C.accent; ctx.font = '600 17px "Noto Sans", sans-serif'; ctx.textAlign = 'center';
    ctx.fillText('fixed pixel window', wx + ws / 2, (phi < 0 ? wy - 10 : wy + ws + 22));

    /* ---- right: sphere ---- */
    var gx2 = 730, gy2 = H / 2 + 14, R = 150;
    var grad = ctx.createRadialGradient(gx2 - 46, gy2 - 52, 24, gx2, gy2, R);
    grad.addColorStop(0, C.sphere1); grad.addColorStop(1, C.sphere2);
    ctx.fillStyle = grad; ctx.beginPath(); ctx.arc(gx2, gy2, R, 0, 2 * Math.PI); ctx.fill();
    ctx.strokeStyle = C.border; ctx.lineWidth = 1.5; ctx.stroke();

    ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
    for (var L = -75; L <= 75; L += 15) {
      var ry = gy2 - R * Math.sin(L * DEG), rx = R * Math.cos(L * DEG);
      ctx.beginPath(); ctx.ellipse(gx2, ry, rx, rx * 0.17, 0, 0, 2 * Math.PI); ctx.stroke();
    }

    var syR = gy2 - R * Math.sin(phi * DEG), sxR = R * cosp;
    if (sxR > 0.5) {
      ctx.strokeStyle = rgba(C.accent, 0.5); ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.ellipse(gx2, syR, sxR, sxR * 0.17, 0, 0, 2 * Math.PI); ctx.stroke();
      ctx.strokeStyle = C.accent; ctx.lineCap = 'round'; ctx.lineWidth = Math.max(4, 12 * cosp + 2);
      ctx.beginPath(); ctx.ellipse(gx2, syR, sxR, sxR * 0.17, 0, Math.PI * 0.5 - 0.62, Math.PI * 0.5 + 0.62);
      ctx.stroke(); ctx.lineCap = 'butt';
    } else {
      ctx.fillStyle = C.accent; ctx.beginPath(); ctx.arc(gx2, syR, 4, 0, 2 * Math.PI); ctx.fill();
    }

    ctx.fillStyle = C.ink; ctx.font = '600 22px "Noto Sans", sans-serif'; ctx.textAlign = 'center';
    ctx.fillText('Sphere', gx2, gy2 - R - 18);
    ctx.fillStyle = C.accent; ctx.font = '600 17px "Noto Sans", sans-serif';
    if (Math.abs(phi) > 62) { ctx.textAlign = 'left'; ctx.fillText('true footprint', gx2 + Math.max(sxR, 4) + 12, syR + 6); }
    else { ctx.fillText('true footprint', gx2, phi < 0 ? syR - sxR * 0.17 - 14 : syR + sxR * 0.17 + 26); }

    ctx.strokeStyle = C.faint; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(wx + ws, wy + ws / 2); ctx.lineTo(gx2 - Math.max(sxR, 4) - 6, syR); ctx.stroke();
    ctx.setLineDash([]);

    roLat.textContent = fmtLat(phi);
    roCos.textContent = cosp.toFixed(2) + '×';
    roInv.textContent = (cosp < 0.02 ? '∞' : (1 / cosp).toFixed(2) + '×');
  }

  slider.addEventListener('input', draw);
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(draw);
  draw();
})();
