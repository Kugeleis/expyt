import { state } from './state.js';
import { els } from './elements.js';
import { showError, setSessionStatus } from './helpers.js';
import { navigateToStep } from './navigation.js';
import {
    renderResultsTable,
    updatePlotsFilter,
    updatePlotsCounter,
    validateStep1Next,
    renderSubgroupsList
} from './ui.js';

// Start a new wizard session
export async function startNewSession() {
    try {
        setSessionStatus('Initializing session...', 'waiting');

        // 1. Create a session on backend
        const response = await fetch('/wizard/sessions', { method: 'POST' });
        if (!response.ok) throw new Error('Could not create wizard session.');

        const data = await response.json();
        state.sessionId = data.session_id;
        state.currentStep = data.current_step;

        setSessionStatus(`Session: ${state.sessionId.substring(0, 8)}...`, 'active');

        // 2. Move to the step indicated by session status (usually dataset_selection)
        navigateToStep(state.currentStep);
    } catch (err) {
        showError(err.message);
        setSessionStatus('Initialization failed', 'error');
    }
}

// Fetch applicable statistical methods
export async function fetchApplicableMethods() {
    try {
        const response = await fetch(`/wizard/sessions/${state.sessionId}/methods`);
        if (!response.ok) throw new Error('Failed to retrieve applicable methods list.');

        const rawMethods = await response.json();
        state.applicableMethods = rawMethods;
        els.methodsList.innerHTML = '';
        if (els.btnSidebarNext) els.btnSidebarNext.disabled = true;
        state.selectedMethod = '';
        state.selectedDiscreteMethod = '';

        const hasContinuous = state.selectedValueColumns.size > 0;
        const hasDiscrete = state.selectedDiscreteColumns.size > 0;

        const continuousMethods = rawMethods.filter(m => m.variable_type === 'continuous');
        const discreteMethods = rawMethods.filter(m => m.variable_type === 'discrete');

        let totalContinuousApplicable = continuousMethods.length;
        let totalDiscreteApplicable = discreteMethods.length;

        if ((hasContinuous && totalContinuousApplicable === 0) || (hasDiscrete && totalDiscreteApplicable === 0)) {
            els.methodsList.innerHTML = '<p class="no-filters-msg">No statistical methods are applicable to your current dataset properties. Please check your data or adjust preprocessing filters.</p>';
            return;
        }

        const checkStep3NextState = () => {
            const continuousValid = !hasContinuous || state.selectedMethod !== '';
            const discreteValid = !hasDiscrete || state.selectedDiscreteMethod !== '';
            if (els.btnSidebarNext) els.btnSidebarNext.disabled = !(continuousValid && discreteValid);
        };

        // Render Continuous Methods
        if (hasContinuous && totalContinuousApplicable > 0) {
            const header = document.createElement('h4');
            header.textContent = 'Continuous Columns Method';
            header.style.marginTop = '1rem';
            els.methodsList.appendChild(header);

            const grid = document.createElement('div');
            grid.className = 'methods-grid';

            continuousMethods.forEach(method => {
                const card = document.createElement('article');
                card.className = 'method-card continuous-method-card';
                card.dataset.name = method.name;

                card.innerHTML = `
                    <div class="method-title">${method.name}</div>
                    <div class="method-desc">${method.description}</div>
                `;

                card.addEventListener('click', (e) => {
                    els.methodsList.querySelectorAll('.continuous-method-card').forEach(c => c.classList.remove('selected'));
                    const activeCard = e.currentTarget;
                    activeCard.classList.add('selected');

                    state.selectedMethod = activeCard.dataset.name;
                    checkStep3NextState();
                });

                grid.appendChild(card);
            });

            els.methodsList.appendChild(grid);
        }

        // Render Discrete Methods
        if (hasDiscrete && totalDiscreteApplicable > 0) {
            const header = document.createElement('h4');
            header.textContent = 'Discrete Columns Method';
            header.style.marginTop = '1.5rem';
            els.methodsList.appendChild(header);

            const grid = document.createElement('div');
            grid.className = 'methods-grid';

            discreteMethods.forEach(method => {
                const card = document.createElement('article');
                card.className = 'method-card discrete-method-card';
                card.dataset.name = method.name;

                card.innerHTML = `
                    <div class="method-title">${method.name}</div>
                    <div class="method-desc">${method.description}</div>
                `;

                card.addEventListener('click', (e) => {
                    els.methodsList.querySelectorAll('.discrete-method-card').forEach(c => c.classList.remove('selected'));
                    const activeCard = e.currentTarget;
                    activeCard.classList.add('selected');

                    state.selectedDiscreteMethod = activeCard.dataset.name;
                    checkStep3NextState();
                });

                grid.appendChild(card);
            });

            els.methodsList.appendChild(grid);
        }
    } catch (err) {
        showError(err.message);
    }
}

// Run the evaluation endpoint
export async function executeStatisticalMethod() {
    try {
        const response = await fetch(`/wizard/sessions/${state.sessionId}/results`);
        if (!response.ok) throw new Error('Statistical evaluation run failed.');

        const data = await response.json();
        state.statResults = data;

        // Default sort: p-value asc
        state.sortColumn = 'p_value';
        state.sortAsc = true;

        // Sort initial data
        state.statResults.sort((a, b) => {
            if (a.p_value === null || a.p_value === undefined) return 1;
            if (b.p_value === null || b.p_value === undefined) return -1;
            return a.p_value - b.p_value;
        });

        // Update significance filter count & state.plotsTopN
        updatePlotsFilter();

        // Render
        renderResultsTable();
    } catch (err) {
        showError(err.message);
    }
}

// Fetch applicable plots list
export async function fetchApplicablePlots() {
    try {
        const response = await fetch(`/wizard/sessions/${state.sessionId}/plots`);
        if (!response.ok) throw new Error('Failed to fetch applicable plot generators.');

        state.applicablePlots = await response.json();
        els.plotsSelector.innerHTML = '';
        if (els.btnSidebarNext) els.btnSidebarNext.disabled = true;
        state.selectedPlots = [];
        els.plotsDisplay.innerHTML = '<span class="no-plots-msg">Select plot types and click Generate Plots above.</span>';

        if (state.applicablePlots.length === 0) {
            els.plotsSelector.innerHTML = '<p class="no-filters-msg">No visualizations applicable.</p>';
            if (els.plotsGenerationCounter) {
                els.plotsGenerationCounter.textContent = 'Will generate 0 plots';
            }
            if (els.btnGeneratePlots) {
                els.btnGeneratePlots.disabled = true;
            }
            return;
        }

        state.applicablePlots.forEach(plot => {
            const card = document.createElement('article');
            card.className = 'plot-select-item';
            card.dataset.name = plot.name;

            // Check if this is the boxplot and preselect it
            const isBoxplot = plot.name === 'boxplot';
            if (isBoxplot) {
                state.selectedPlots.push(plot.name);
                card.classList.add('selected');
            }

            card.innerHTML = `
                <input type="checkbox" id="chk-plot-${plot.name}" ${isBoxplot ? 'checked' : ''}>
                <div class="plot-select-info">
                    <h5>${plot.name}</h5>
                    <p>${plot.description}</p>
                </div>
            `;

            // Toggle selection logic
            const checkbox = card.querySelector('input');
            const toggleSelection = () => {
                checkbox.checked = !checkbox.checked;
                card.classList.toggle('selected', checkbox.checked);

                if (checkbox.checked) {
                    state.selectedPlots.push(plot.name);
                } else {
                    state.selectedPlots = state.selectedPlots.filter(p => p !== plot.name);
                }
                updatePlotsCounter();
            };

            card.addEventListener('click', (e) => {
                // Prevent double trigger when clicking checkbox directly
                if (e.target !== checkbox) {
                    toggleSelection();
                }
            });
            checkbox.addEventListener('change', () => {
                card.classList.toggle('selected', checkbox.checked);
                if (checkbox.checked) {
                    state.selectedPlots.push(plot.name);
                } else {
                    state.selectedPlots = state.selectedPlots.filter(p => p !== plot.name);
                }
                updatePlotsCounter();
            });

            els.plotsSelector.appendChild(card);
        });

        updatePlotsCounter();
    } catch (err) {
        showError(err.message);
    }
}

// Helper for chomping/pooping Pac-Man loading animation
function startPacmanAnimation() {
    const textRow = document.getElementById('textRow');
    const pacman  = document.getElementById('pacman');
    const pacBody = document.getElementById('pacBody');
    const stage   = document.querySelector('.stage');
    if (!textRow || !pacman || !pacBody || !stage) return;

    const TEXT = 'Loading data ....';
    const CYCLE = 4000;
    const spans = [];

    for (const ch of TEXT) {
        const span = document.createElement('span');
        span.className = 'letter';
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
        spans.forEach(s => s.classList.remove('eaten'));
        setEatingMode();

        const stageRect   = stage.getBoundingClientRect();
        const textRowRect = textRow.getBoundingClientRect();

        const textLeft  = textRowRect.left - stageRect.left;
        const textRight = textRowRect.right - stageRect.left;

        const pacW = 30;
        const startX = textLeft - pacW - 10;
        const endX   = textRight + 10;
        const halfCycle = CYCLE / 2;

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

                const mouthX = x + pacW;
                spans.forEach(span => {
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

                const buttX = x + pacW;
                spans.forEach(span => {
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

        setX(startX);
        requestAnimationFrame(tick);
    }

    setTimeout(runCycle, 100);
}

// Generate plots client side preview (updates live as they check boxes)
export async function generatePlotsPreview() {
    if (state.selectedPlots.length === 0) {
        els.plotsDisplay.innerHTML = '<span class="no-plots-msg">Select one or more plots to generate.</span>';
        return;
    }

    try {
        els.plotsDisplay.innerHTML = `
            <div class="pacman-loader-container">
                <div class="stage">
                    <div class="text-row" id="textRow"></div>
                    <div class="pacman-wrapper" id="pacman">
                        <div class="pac-body chomping" id="pacBody"></div>
                        <div class="pac-eye" id="pacEye"></div>
                    </div>
                </div>
            </div>
        `;
        startPacmanAnimation();
        els.btnGeneratePlots.disabled = true;
        if (els.btnSidebarNext) els.btnSidebarNext.disabled = true;

        const response = await fetch(`/wizard/sessions/${state.sessionId}/plots`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                selected_plots: state.selectedPlots,
                top_n_columns: state.plotsTopN
            })
        });

        if (!response.ok) throw new Error('Plot rendering failed.');

        const data = await response.json();
        els.plotsDisplay.innerHTML = '';

        // Group plots by variable (column_name)
        const plotsByVar = {};
        data.plot_results.forEach(plot => {
            const col = plot.column_name || 'General';
            if (!plotsByVar[col]) {
                plotsByVar[col] = [];
            }
            plotsByVar[col].push(plot);
        });

        Object.entries(plotsByVar).forEach(([colName, plots]) => {
            const card = document.createElement('article');
            card.className = 'variable-plots-card';
            card.style.margin = '0 0 1rem 0';
            card.style.width = '100%';

            const header = document.createElement('header');
            header.style.padding = '0.5rem 0.75rem';
            header.style.marginBottom = '0.75rem';
            header.style.backgroundColor = 'rgba(255, 255, 255, 0.02)';
            header.style.borderBottom = '1px solid var(--pico-border-color)';

            const title = document.createElement('h4');
            title.style.margin = '0';
            title.style.fontSize = '0.95rem';
            title.style.fontWeight = '600';
            title.style.display = 'flex';
            title.style.justifyContent = 'space-between';
            title.style.alignItems = 'center';
            title.style.flexWrap = 'wrap';
            title.style.gap = '0.5rem';
            title.style.width = '100%';
            title.textContent = colName;

            // Fetch statistical properties for this column to display as badges next to the title
            const statResult = state.statResults ? state.statResults.find(res => res.column_name === colName) : null;
            if (statResult) {
                const statsContainer = document.createElement('div');
                statsContainer.className = 'card-header-stats';
                statsContainer.style.display = 'flex';
                statsContainer.style.gap = '0.4rem';
                statsContainer.style.fontSize = '0.75rem';
                statsContainer.style.flexWrap = 'wrap';
                statsContainer.style.fontWeight = 'normal'; // Reset font-weight from h4 defaults

                // Method badge
                const methodBadge = document.createElement('span');
                methodBadge.style.backgroundColor = 'rgba(255, 255, 255, 0.04)';
                methodBadge.style.padding = '0.1rem 0.4rem';
                methodBadge.style.borderRadius = '4px';
                methodBadge.style.border = '1px solid var(--pico-border-color)';
                methodBadge.style.fontWeight = '500';
                methodBadge.textContent = statResult.method_name;
                statsContainer.appendChild(methodBadge);

                // Statistic badge
                if (statResult.test_statistic !== null && statResult.test_statistic !== undefined) {
                    const statBadge = document.createElement('span');
                    statBadge.style.backgroundColor = 'rgba(255, 255, 255, 0.04)';
                    statBadge.style.padding = '0.1rem 0.4rem';
                    statBadge.style.borderRadius = '4px';
                    statBadge.style.border = '1px solid var(--pico-border-color)';
                    statBadge.innerHTML = `Stat: <strong>${Number(statResult.test_statistic).toFixed(4)}</strong>`;
                    statsContainer.appendChild(statBadge);
                }

                // p-value badge
                if (statResult.p_value !== null && statResult.p_value !== undefined) {
                    const pBadge = document.createElement('span');
                    pBadge.style.backgroundColor = 'rgba(255, 255, 255, 0.04)';
                    pBadge.style.padding = '0.1rem 0.4rem';
                    pBadge.style.borderRadius = '4px';
                    pBadge.style.border = '1px solid var(--pico-border-color)';
                    pBadge.innerHTML = `p-val: <strong>${Number(statResult.p_value).toFixed(6)}</strong>`;

                    const filterInput = document.getElementById('plots-sig-filter');
                    const threshold = filterInput ? parseFloat(filterInput.value) : 0.05;
                    if (statResult.p_value <= threshold) {
                        pBadge.style.color = 'var(--pico-primary)';
                        pBadge.style.borderColor = 'rgba(16, 185, 129, 0.3)';
                        pBadge.style.backgroundColor = 'rgba(16, 185, 129, 0.05)';
                    }
                    statsContainer.appendChild(pBadge);
                }

                // Effect Size badge
                if (statResult.effect_size !== null && statResult.effect_size !== undefined) {
                    const esBadge = document.createElement('span');
                    esBadge.style.backgroundColor = 'rgba(255, 255, 255, 0.04)';
                    esBadge.style.padding = '0.1rem 0.4rem';
                    esBadge.style.borderRadius = '4px';
                    esBadge.style.border = '1px solid var(--pico-border-color)';
                    esBadge.innerHTML = `ES: <strong>${Number(statResult.effect_size).toFixed(4)}</strong>`;
                    statsContainer.appendChild(esBadge);
                }

                title.appendChild(statsContainer);
            }

            header.appendChild(title);
            card.appendChild(header);

            // Grid row for plots
            const row = document.createElement('div');
            row.className = 'plots-grid';

            plots.forEach(plot => {
                const plotWrapper = document.createElement('div');
                plotWrapper.className = 'plot-image-wrapper';
                plotWrapper.style.textAlign = 'center';

                const img = document.createElement('img');
                img.src = `data:${plot.content_type};base64,${plot.image_base64}`;
                img.alt = `${plot.plot_type} for ${colName}`;
                img.style.maxWidth = '100%';
                img.style.height = 'auto';
                img.style.borderRadius = '8px';
                img.style.border = '1px solid var(--pico-border-color)';

                const label = document.createElement('div');
                label.style.marginTop = '0.25rem';
                label.style.fontSize = '0.75rem';
                label.style.color = 'var(--pico-muted-color)';
                label.textContent = plot.plot_type.toUpperCase();

                plotWrapper.appendChild(img);
                plotWrapper.appendChild(label);
                row.appendChild(plotWrapper);
            });

            card.appendChild(row);
            els.plotsDisplay.appendChild(card);
        });

        if (els.btnSidebarNext) els.btnSidebarNext.disabled = false;
    } catch (err) {
        els.plotsDisplay.innerHTML = `<span class="no-plots-msg text-error">Failed to render plots: ${err.message}</span>`;
    } finally {
        els.btnGeneratePlots.disabled = false;
    }
}

// Fetch unique values for a group column (subgroups) from the dataset
export async function updateSubgroupsList() {
    const datasetId = state.selectedDatasetId;
    const groupCol = els.groupColSelect.value;

    if (!groupCol) {
        els.subgroupsSection.classList.add('hidden');
        state.selectedGroups = new Set();
        state.availableGroups = [];
        validateStep1Next();
        return;
    }

    try {
        const response = await fetch(`/wizard/datasets/${datasetId}/columns/${groupCol}/unique`);
        if (!response.ok) throw new Error('Failed to fetch unique values for group column.');

        const groups = await response.json();

        state.availableGroups = groups;
        state.selectedGroups = new Set(groups);

        if (els.subgroupsSearch) {
            els.subgroupsSearch.value = '';
        }

        renderSubgroupsList();
    } catch (err) {
        showError(err.message);
        els.subgroupsSection.classList.add('hidden');
    }
}
