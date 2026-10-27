// VERSION : 2.6 - FIX SCOPE & DYNAMIC SCALE
const canvas = document.getElementById('audio_visualizer');
const ctx = canvas.getContext('2d');
let audioCtx, analyser, dataArray, source;

/**
 * Rendre la fonction accessible globalement pour le TTS
 */
window.setupAriaVisualizer = async function(audioElement) {
    if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioCtx.createAnalyser();
        analyser.fftSize = 2048; 
        dataArray = new Uint8Array(analyser.frequencyBinCount);
    }

    // Débloque le contexte si suspendu par le navigateur
    if (audioCtx.state === 'suspended') {
        await audioCtx.resume();
    }

    if (source) source.disconnect();
    
    source = audioCtx.createMediaElementSource(audioElement);
    source.connect(analyser);
    analyser.connect(audioCtx.destination);
    
    animate();
};

function animate() {
    requestAnimationFrame(animate);
    if(!analyser) return;
    
    analyser.getByteFrequencyData(dataArray);
    
    // Paramètres par défaut si window.visualParams est absent
    const p = window.visualParams || { 
        nbRings: 8, 
        resolution: 30, 
        spacing: 2, 
        gain: 3.0, 
        stroke: 2.0, 
        alphaMin: 0.0, 
        alphaMax: 1.0 
    };
    
    // Ajustement de la résolution interne pour éviter le flou
    if (canvas.width !== canvas.clientWidth || canvas.height !== canvas.clientHeight) {
        canvas.width = canvas.clientWidth;
        canvas.height = canvas.clientHeight;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    
    // --- CALCUL POUR TENIR DANS LE CHAMP (100px) ---
    // On définit le rayon maximum (45% de la hauteur pour laisser une petite marge)
    const maxSafeRadius = (canvas.height / 2) * 0.9;
    
    for (let rIdx = 0; rIdx < p.nbRings; rIdx++) {
        const hue = (rIdx / p.nbRings) * 260; 
        const freqStart = Math.floor((rIdx / p.nbRings) * 400);
        
        // Le rayon de base est réduit pour laisser de la place à l'amplitude
        const baseRadius = (maxSafeRadius * 0.3) + (rIdx * p.spacing);
        // L'amplitude maximale possible pour ne pas sortir du canvas
        const maxAvailableAmplitude = maxSafeRadius - baseRadius;

        const pointsCount = Math.floor(p.resolution);
        const angleStep = (Math.PI * 2) / pointsCount;
        
        for (let i = 0; i < pointsCount; i++) {
            let sampleIdx = freqStart + Math.floor((i / pointsCount) * 50);
            let intensity = dataArray[sampleIdx] / 255;
            let alpha = p.alphaMin + (intensity * (p.alphaMax - p.alphaMin));
            
            // On bride l'amplitude ici pour qu'elle respecte maxAvailableAmplitude
            const amplitude = intensity * maxAvailableAmplitude * (p.gain / 5);
            const r = baseRadius + amplitude;
            
            const angle = i * angleStep - Math.PI / 2;
            const nextAngle = ((i + 1) % pointsCount) * angleStep - Math.PI / 2;

            const x = centerX + Math.cos(angle) * r;
            const y = centerY + Math.sin(angle) * r;

            let nextSampleIdx = freqStart + Math.floor(((i + 1) % pointsCount / pointsCount) * 50);
            let nextIntensity = dataArray[nextSampleIdx] / 255;
            const nextR = baseRadius + (nextIntensity * maxAvailableAmplitude * (p.gain / 5));
            const nextX = centerX + Math.cos(nextAngle) * nextR;
            const nextY = centerY + Math.sin(nextAngle) * nextR;

            ctx.beginPath();
            ctx.lineWidth = p.stroke;
            ctx.strokeStyle = `hsla(${hue}, 90%, 60%, ${alpha})`;
            ctx.moveTo(x, y);
            ctx.lineTo(nextX, nextY);
            ctx.stroke();
        }
    }
}