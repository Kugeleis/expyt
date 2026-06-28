// DRY Consumable Pac-Man loader animator
export function animatePacman({
    stage,
    textRow,
    pacman,
    pacBody,
    text,
    cycleTime = 3000,
    pacWidth = 20,
    letterFontSize = '16px'
}) {
    if (!textRow || !pacman || !pacBody || !stage) return;

    const spans = [];
    textRow.innerHTML = '';
    for (const ch of text) {
        const span = document.createElement('span');
        span.className = 'letter';
        if (letterFontSize) span.style.fontSize = letterFontSize;
        span.textContent = ch;
        textRow.appendChild(span);
        spans.push(span);
    }

    function setX(x) {
        if (pacman) pacman.style.left = x + 'px';
    }

    function setEatingMode() {
        if (pacman && pacBody) {
            pacman.style.transform = 'translateY(-50%) scaleX(1)';
            pacBody.className = 'pac-body chomping';
        }
    }

    function setPoopingMode() {
        if (pacman && pacBody) {
            pacman.style.transform = 'translateY(-50%) scaleX(-1)';
            pacBody.className = 'pac-body pooping';
        }
    }

    function runCycle() {
        if (!pacman || !pacman.isConnected) return;
        spans.forEach((s) => s.classList.remove('eaten'));
        setEatingMode();

        const stageRect = stage.getBoundingClientRect();
        const textRowRect = textRow.getBoundingClientRect();

        const textLeft = textRowRect.left - stageRect.left;
        const textRight = textRowRect.right - stageRect.left;

        const startX = textLeft - pacWidth - 10;
        const endX = textRight + 10;
        const halfCycle = cycleTime / 2;

        let startTime = null;
        let phase = 'forward';
        let pauseStart = null;

        function tick(ts) {
            if (!pacman || !pacman.isConnected) return;
            if (!startTime) startTime = ts;
            const elapsed = ts - startTime;

            if (phase === 'forward') {
                const t = Math.min(elapsed / halfCycle, 1);
                const x = startX + (endX - startX) * t;
                setX(x);

                const mouthX = x + pacWidth;
                spans.forEach((span) => {
                    const r = span.getBoundingClientRect();
                    const mid = r.left - stageRect.left + r.width / 2;
                    if (mid < mouthX) span.classList.add('eaten');
                });

                if (t >= 1) {
                    phase = 'pause';
                    pauseStart = ts;
                    setPoopingMode();
                }
            } else if (phase === 'pause') {
                if (ts - pauseStart > 350) {
                    phase = 'backward';
                    startTime = ts;
                }
            } else if (phase === 'backward') {
                const t = Math.min(elapsed / halfCycle, 1);
                const x = endX + (startX - endX) * t;
                setX(x);

                const buttX = x + pacWidth;
                spans.forEach((span) => {
                    const r = span.getBoundingClientRect();
                    const mid = r.left - stageRect.left + r.width / 2;
                    if (mid > buttX) span.classList.remove('eaten');
                });

                if (t >= 1) {
                    setEatingMode();
                    setTimeout(() => {
                        if (pacman && pacman.isConnected) {
                            runCycle();
                        }
                    }, 400);
                    return;
                }
            }

            requestAnimationFrame(tick);
        }

        requestAnimationFrame(tick);
    }

    runCycle();
}
