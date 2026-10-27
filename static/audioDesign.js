// Visualiseur audio d'Aria : anneaux concentriques animés par le spectre de la voix.
// Repris de la branche WebHMI-tmp (v2.6). Les courbes restent TOUJOURS dans le cadre du canvas (trait compris).
//  - p.fitFrame = true  : les anneaux épousent tout le cadre (ellipses largeur x hauteur)
//  - p.fitFrame = false : cercle centré, limité par la plus petite dimension
(() => {
    // Géométrie pure (sans DOM) : testable sous node. Renvoie les segments à tracer pour le spectre `data` dans un cadre w x h.
    function ariaVisuGeometry(p, data, w, h) {
        const margin = 2, half = p.stroke / 2;
        const fit = p.fitFrame === true;
        // Demi-axes disponibles : le trait (demi-épaisseur) et une marge restent à l'intérieur du cadre
        const bx = Math.max(1, (fit ? w : Math.min(w, h)) / 2 - half - margin);
        const by = Math.max(1, (fit ? h : Math.min(w, h)) / 2 - half - margin);
        const cx = w / 2, cy = h / 2;
        // Un cadre large demande plus de points pour que les courbes restent lisses
        const points = Math.max(3, Math.floor(p.resolution * (fit ? Math.max(1, Math.sqrt(w / h)) : 1)));
        const step = (Math.PI * 2) / points;
        const level = (idx) => Math.min(1, Math.max(0, ((data[idx] || 0) / 255) * (p.gain / 5)));
        const segs = [];

        for (let rIdx = 0; rIdx < p.nbRings; rIdx++) {
            const hue = (rIdx / p.nbRings) * 260;
            const freqStart = Math.floor((rIdx / p.nbRings) * 400);
            const baseFrac = Math.min(0.3 + (rIdx * p.spacing) / Math.min(bx, by), 1);   // part du rayon occupée au repos
            const availFrac = 1 - baseFrac;                                               // part restante pour les pics

            for (let i = 0; i < points; i++) {
                const j = (i + 1) % points;
                const idx1 = freqStart + Math.floor((i / points) * 50);
                const idx2 = freqStart + Math.floor((j / points) * 50);
                // f <= 1 : un pic touche le bord sans le dépasser
                const f1 = baseFrac + level(idx1) * availFrac, f2 = baseFrac + level(idx2) * availFrac;
                const a1 = i * step - Math.PI / 2, a2 = j * step - Math.PI / 2;
                const intensity = Math.min(1, (data[idx1] || 0) / 255);
                segs.push({
                    x1: cx + Math.cos(a1) * bx * f1, y1: cy + Math.sin(a1) * by * f1,
                    x2: cx + Math.cos(a2) * bx * f2, y2: cy + Math.sin(a2) * by * f2,
                    hue: hue,
                    alpha: p.alphaMin + intensity * (p.alphaMax - p.alphaMin),
                });
            }
        }
        return segs;
    }
    window.ariaVisuGeometry = ariaVisuGeometry;

    if (window.ariaVisualizerReady) return;
    const canvas = document.getElementById('audio_visualizer');
    if (!canvas) { console.warn("[ARIA-VISU] canvas #audio_visualizer introuvable"); return; }
    window.ariaVisualizerReady = true;

    const ctx = canvas.getContext('2d');
    let audioCtx, analyser, dataArray, source, running = false;

    // Relie l'élément audio qui va être joué à l'analyseur (appelée à chaque bloc audio, événement aria_audio_start).
    window.setupAriaVisualizer = async function (audioElement) {
        try {
            if (!audioCtx) {
                audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                analyser = audioCtx.createAnalyser();
                analyser.fftSize = 2048;
                dataArray = new Uint8Array(analyser.frequencyBinCount);
            }
            if (audioCtx.state === 'suspended') await audioCtx.resume();
            if (source) source.disconnect();
            source = audioCtx.createMediaElementSource(audioElement);
            source.connect(analyser);
            analyser.connect(audioCtx.destination);
            if (!running) { running = true; animate(); }
        } catch (e) {
            console.warn("[ARIA-VISU] Visualiseur indisponible pour ce bloc :", e);   // le son reste audible
        }
    };

    function animate() {
        requestAnimationFrame(animate);
        if (!analyser) return;
        analyser.getByteFrequencyData(dataArray);
        const p = window.visualParams || { nbRings: 8, resolution: 30, spacing: 2, gain: 3.0, stroke: 2.0, alphaMin: 0.0, alphaMax: 1.0 };
        // Résolution interne = taille affichée, pour éviter le flou
        if (canvas.width !== canvas.clientWidth || canvas.height !== canvas.clientHeight) {
            canvas.width = canvas.clientWidth;
            canvas.height = canvas.clientHeight;
        }
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.lineWidth = p.stroke;
        for (const s of ariaVisuGeometry(p, dataArray, canvas.width, canvas.height)) {
            ctx.beginPath();
            ctx.strokeStyle = `hsla(${s.hue}, 90%, 60%, ${s.alpha})`;
            ctx.moveTo(s.x1, s.y1);
            ctx.lineTo(s.x2, s.y2);
            ctx.stroke();
        }
    }
})();
