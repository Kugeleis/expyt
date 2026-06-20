/**
 * ExpYT — Evaluation Wizard Frontend Orchestrator.
 * 
 * Handles step transitions, state management, and backend communication.
 */

// Global State
const state = {
    sessionId: null,
    currentStep: 'dataset_selection',
    selectedDatasetId: '',
    selectedDatasetColumns: [],
    activeFilters: [],
    applicableMethods: [],
    selectedMethod: '',
    applicablePlots: [],
    selectedPlots: [],
    selectedExportFormat: 'pdf'
};

// UI Elements
const els = {
    sessionInfo: document.getElementById('session-info'),
    fileUpload: document.getElementById('file-upload'),
    btnUpload: document.getElementById('btn-upload'),
    uploadStatus: document.getElementById('upload-status'),
    datasetDetails: document.getElementById('dataset-details'),
    detailDesc: document.getElementById('detail-desc'),
    groupColSelect: document.getElementById('group-col-select'),
    valueColSelect: document.getElementById('value-col-select'),
    btnStep1Next: document.getElementById('btn-step-1-next'),
    
    activeFilters: document.getElementById('active-filters'),
    filterType: document.getElementById('filter-type'),
    filterCol: document.getElementById('filter-col'),
    fieldsNumericRange: document.getElementById('fields-numeric_range'),
    fieldsCategoryFilter: document.getElementById('fields-category_filter'),
    filterNumMin: document.getElementById('filter-num-min'),
    filterNumMax: document.getElementById('filter-num-max'),
    filterCatValues: document.getElementById('filter-cat-values'),
    filterCatExclude: document.getElementById('filter-cat-exclude'),
    btnAddFilterAction: document.getElementById('btn-add-filter-action'),
    btnStep2Next: document.getElementById('btn-step-2-next'),
    
    methodsList: document.getElementById('methods-list'),
    btnStep3Next: document.getElementById('btn-step-3-next'),
    
    resultMethodName: document.getElementById('result-method-name'),
    resultStatistic: document.getElementById('result-statistic'),
    resultPValue: document.getElementById('result-p-value'),
    resultSummaryText: document.getElementById('result-summary-text'),
    btnStep4Next: document.getElementById('btn-step-4-next'),
    
    plotsSelector: document.getElementById('plots-selector'),
    plotsDisplay: document.getElementById('plots-display'),
    btnStep5Next: document.getElementById('btn-step-5-next'),
    
    btnExportDownload: document.getElementById('btn-export-download'),
    btnRestart: document.getElementById('btn-restart'),
    btnStep2Back: document.getElementById('btn-step-2-back'),
    btnStep3Back: document.getElementById('btn-step-3-back'),
    btnStep4Back: document.getElementById('btn-step-4-back'),
    btnStep5Back: document.getElementById('btn-step-5-back'),
    btnStep6Back: document.getElementById('btn-step-6-back'),
    
    errorToast: document.getElementById('error-toast'),
    toastMsg: document.getElementById('toast-msg'),
    btnToastClose: document.getElementById('btn-toast-close')
};

// Map step identifier strings to nav and panel elements
const stepsConfig = [
    { key: 'dataset_selection', navId: 'nav-step-1', panelId: 'panel-step-1' },
    { key: 'filters', navId: 'nav-step-2', panelId: 'panel-step-2' },
    { key: 'stat_method', navId: 'nav-step-3', panelId: 'panel-step-3' },
    { key: 'results', navId: 'nav-step-4', panelId: 'panel-step-4' },
    { key: 'plot_selection', navId: 'nav-step-5', panelId: 'panel-step-5' },
    { key: 'export', navId: 'nav-step-6', panelId: 'panel-step-6' }
];

// Initialize application
window.addEventListener('DOMContentLoaded', async () => {
    initEventListeners();
    await startNewSession();
});

// Start a new wizard session
async function startNewSession() {
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

// Set session status bar state
function setSessionStatus(text, type) {
    const textEl = els.sessionInfo.querySelector('.text');
    const dotEl = els.sessionInfo.querySelector('.dot');
    textEl.textContent = text;
    
    dotEl.style.backgroundColor = 
        type === 'active' ? 'var(--success-green)' : 
        type === 'error' ? 'var(--error-red)' : 'var(--text-secondary)';
    dotEl.style.boxShadow = 
        type === 'active' ? '0 0 8px var(--success-green)' : 
        type === 'error' ? '0 0 8px var(--error-red)' : 'none';
}

// Navigate to a specific step panel
function navigateToStep(stepKey) {
    state.currentStep = stepKey;
    
    let activeIndex = stepsConfig.findIndex(s => s.key === stepKey);
    if (activeIndex === -1) activeIndex = 0;
    
    stepsConfig.forEach((step, idx) => {
        const navEl = document.getElementById(step.navId);
        const panelEl = document.getElementById(step.panelId);
        
        // Manage Panel Views
        if (idx === activeIndex) {
            panelEl.classList.add('active');
            navEl.className = 'step-nav-item active';
        } else {
            panelEl.classList.remove('active');
            if (idx < activeIndex) {
                navEl.className = 'step-nav-item completed';
            } else {
                navEl.className = 'step-nav-item';
            }
        }
        
        // Remove old click listeners by cloning
        const newNavEl = navEl.cloneNode(true);
        navEl.parentNode.replaceChild(newNavEl, navEl);
        
        // Add click handler for completed steps
        if (idx < activeIndex) {
            newNavEl.addEventListener('click', () => {
                goToStep(step.key);
            });
        }
    });
}

// Navigate back to a previously completed step via backend
async function goToStep(stepKey) {
    try {
        const response = await fetch(`/wizard/sessions/${state.sessionId}/go-to/${stepKey}`, {
            method: 'POST'
        });
        
        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || 'Failed to navigate to step.');
        }
        
        const data = await response.json();
        
        // Update local state from server response
        state.currentStep = data.current_step;
        state.selectedMethod = data.selected_method || '';
        state.activeFilters = data.filters_config || [];
        state.selectedPlots = data.selected_plots || [];
        
        // Re-render UI for the target step
        navigateToStep(data.current_step);
        
        // Restore step-specific UI state
        if (stepKey === 'filters') {
            renderActiveFilters();
        } else if (stepKey === 'stat_method') {
            await fetchApplicableMethods();
            // Re-select previously selected method if it's still there
            if (state.selectedMethod) {
                const card = document.querySelector(`.method-card[data-name="${state.selectedMethod}"]`);
                if (card) {
                    card.classList.add('selected');
                    els.btnStep3Next.disabled = false;
                }
            }
        } else if (stepKey === 'plot_selection') {
            await fetchApplicablePlots();
        }
    } catch (err) {
        showError(err.message);
    }
}

// Setup Event Listeners
function initEventListeners() {
    // Restart session
    els.btnRestart.addEventListener('click', async () => {
        if (confirm('Are you sure you want to restart the session? All configuration will be lost.')) {
            // Reset state
            state.activeFilters = [];
            state.selectedMethod = '';
            state.selectedPlots = [];
            
            // Clean active filters panel
            renderActiveFilters();
            els.plotsDisplay.innerHTML = '<span class="no-plots-msg">No plots generated yet.</span>';
            els.btnStep1Next.disabled = true;
            els.btnStep3Next.disabled = true;
            els.btnStep5Next.disabled = true;
            els.fileUpload.value = '';
            els.uploadStatus.textContent = '';
            els.datasetDetails.classList.add('hidden');
            
            await startNewSession();
        }
    });
    
    // Toast close
    els.btnToastClose.addEventListener('click', () => {
        els.errorToast.classList.add('hidden');
    });

    // Step 1: Upload Data
    els.btnUpload.addEventListener('click', async () => {
        const file = els.fileUpload.files[0];
        if (!file) {
            showError('Please select a file to upload.');
            return;
        }

        els.uploadStatus.textContent = 'Uploading...';
        els.uploadStatus.style.color = 'var(--text-secondary)';
        els.btnUpload.disabled = true;

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/wizard/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || 'Upload failed.');
            }

            const dataset = await response.json();
            
            state.selectedDatasetId = dataset.id;
            state.selectedDatasetColumns = dataset.columns;
            els.detailDesc.textContent = dataset.description;
            
            // Populate group and value columns
            els.groupColSelect.innerHTML = '';
            els.valueColSelect.innerHTML = '';
            els.filterCol.innerHTML = '';

            dataset.columns.forEach(col => {
                const opt1 = document.createElement('option');
                opt1.value = col.name;
                opt1.textContent = `${col.name} (${col.dtype})`;
                els.groupColSelect.appendChild(opt1);

                // For values, suggest numeric columns primarily
                const opt2 = opt1.cloneNode(true);
                els.valueColSelect.appendChild(opt2);

                const opt3 = opt1.cloneNode(true);
                els.filterCol.appendChild(opt3);
            });

            els.datasetDetails.classList.remove('hidden');
            els.btnStep1Next.disabled = false;
            els.uploadStatus.textContent = 'Upload successful!';
            els.uploadStatus.style.color = 'var(--success-green)';
        } catch (err) {
            showError(err.message);
            els.uploadStatus.textContent = 'Upload failed.';
            els.uploadStatus.style.color = 'var(--error-red)';
            els.datasetDetails.classList.add('hidden');
            els.btnStep1Next.disabled = true;
        } finally {
            els.btnUpload.disabled = false;
        }
    });
    
    // Step 1: Submit dataset
    els.btnStep1Next.addEventListener('click', async () => {
        try {
            const payload = {
                dataset_id: state.selectedDatasetId,
                group_column: els.groupColSelect.value,
                selected_value_columns: []
            };
            
            const response = await fetch(`/wizard/sessions/${state.sessionId}/dataset`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            
            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || 'Failed to select dataset.');
            }
            
            const data = await response.json();
            navigateToStep(data.current_step);
        } catch (err) {
            showError(err.message);
        }
    });

    // Step 2: Toggle Filter Fields based on type selected
    els.filterType.addEventListener('change', (e) => {
        const type = e.target.value;
        if (type === 'numeric_range') {
            els.fieldsNumericRange.classList.remove('hidden');
            els.fieldsCategoryFilter.classList.add('hidden');
        } else {
            els.fieldsNumericRange.classList.add('hidden');
            els.fieldsCategoryFilter.classList.remove('hidden');
        }
    });

    // Step 2: Add filter action
    els.btnAddFilterAction.addEventListener('click', () => {
        const type = els.filterType.value;
        const col = els.filterCol.value;
        
        if (!col) return;
        
        let filterObj = { name: type, params: { column: col } };
        
        if (type === 'numeric_range') {
            const min = els.filterNumMin.value ? parseFloat(els.filterNumMin.value) : null;
            const max = els.filterNumMax.value ? parseFloat(els.filterNumMax.value) : null;
            
            if (min === null && max === null) {
                showError('You must specify at least a minimum or maximum value.');
                return;
            }
            if (min !== null && max !== null && min > max) {
                showError('Minimum value cannot exceed maximum value.');
                return;
            }
            
            if (min !== null) filterObj.params.min = min;
            if (max !== null) filterObj.params.max = max;
            
            // Clear inputs
            els.filterNumMin.value = '';
            els.filterNumMax.value = '';
        } else {
            const valsStr = els.filterCatValues.value.trim();
            if (!valsStr) {
                showError('Please input at least one categorical value.');
                return;
            }
            
            const values = valsStr.split(',').map(v => v.trim()).filter(v => v.length > 0);
            filterObj.params.categories = values;
            filterObj.params.exclude = els.filterCatExclude.checked;
            
            // Clear inputs
            els.filterCatValues.value = '';
            els.filterCatExclude.checked = false;
        }
        
        state.activeFilters.push(filterObj);
        renderActiveFilters();
    });
    
    // Step 2: Submit filters
    els.btnStep2Next.addEventListener('click', async () => {
        try {
            const response = await fetch(`/wizard/sessions/${state.sessionId}/filters`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filters_config: state.activeFilters })
            });
            
            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || 'Filter configuration failed validation.');
            }
            
            const data = await response.json();
            
            // Immediately fetch applicable methods for step 3
            await fetchApplicableMethods();
            navigateToStep(data.current_step);
        } catch (err) {
            showError(err.message);
        }
    });

    // Step 3: Submit selected statistical method
    els.btnStep3Next.addEventListener('click', async () => {
        try {
            const response = await fetch(`/wizard/sessions/${state.sessionId}/method`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ selected_method: state.selectedMethod })
            });
            
            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || 'Method selection rejected.');
            }
            
            const data = await response.json();
            
            // Execute results immediately to show in Step 4
            await executeStatisticalMethod();
            navigateToStep(data.current_step);
        } catch (err) {
            showError(err.message);
        }
    });

    // Step 4: Confirm statistical results and go to Plots
    els.btnStep4Next.addEventListener('click', async () => {
        // Fetch applicable plots before navigation
        await fetchApplicablePlots();
        navigateToStep('plot_selection');
    });

    // Step 5: Submit generated plots
    els.btnStep5Next.addEventListener('click', async () => {
        try {
            const response = await fetch(`/wizard/sessions/${state.sessionId}/plots`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ selected_plots: state.selectedPlots })
            });
            
            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || 'Failed to register selected plots.');
            }
            
            const data = await response.json();
            navigateToStep(data.current_step);
        } catch (err) {
            showError(err.message);
        }
    });

    // Step 6: Exporter select card
    document.querySelectorAll('.exporter-card').forEach(card => {
        card.addEventListener('click', (e) => {
            document.querySelectorAll('.exporter-card').forEach(c => c.classList.remove('active'));
            const activeCard = e.currentTarget;
            activeCard.classList.add('active');
            state.selectedExportFormat = activeCard.dataset.format;
        });
    });

    // Step 6: Download Report
    els.btnExportDownload.addEventListener('click', async () => {
        try {
            const response = await fetch(`/wizard/sessions/${state.sessionId}/export`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ export_format: state.selectedExportFormat })
            });
            
            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || 'Export compilation failed.');
            }
            
            // Trigger browser download dialog
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            
            // Extract filename from headers if possible
            const disposition = response.headers.get('content-disposition');
            let filename = `experiment_report.${state.selectedExportFormat}`;
            if (disposition && disposition.indexOf('attachment') !== -1) {
                const filenameRegex = /filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/;
                const matches = filenameRegex.exec(disposition);
                if (matches != null && matches[1]) {
                    filename = matches[1].replace(/['"]/g, '');
                }
            }
            
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            a.remove();
            window.URL.revokeObjectURL(url);
        } catch (err) {
            showError(err.message);
        }
    });

    // Back buttons
    els.btnStep2Back.addEventListener('click', () => goToStep('dataset_selection'));
    els.btnStep3Back.addEventListener('click', () => goToStep('filters'));
    els.btnStep4Back.addEventListener('click', () => goToStep('stat_method'));
    els.btnStep5Back.addEventListener('click', () => goToStep('results'));
    els.btnStep6Back.addEventListener('click', () => goToStep('plot_selection'));
}

// Render active filter badges
function renderActiveFilters() {
    if (state.activeFilters.length === 0) {
        els.activeFilters.innerHTML = '<p class="no-filters-msg">No filters configured. Click below to add a filter, or proceed directly.</p>';
        return;
    }
    
    els.activeFilters.innerHTML = '';
    state.activeFilters.forEach((filter, idx) => {
        const badge = document.createElement('div');
        badge.className = 'filter-badge';
        
        let descText = '';
        if (filter.name === 'numeric_range') {
            const hasMin = filter.params.min !== undefined;
            const hasMax = filter.params.max !== undefined;
            if (hasMin && hasMax) descText = `${filter.params.min} <= val <= ${filter.params.max}`;
            else if (hasMin) descText = `val >= ${filter.params.min}`;
            else descText = `val <= ${filter.params.max}`;
        } else {
            const categories = filter.params.categories.join(', ');
            descText = `${filter.params.exclude ? 'Exclude' : 'Include'}: [${categories}]`;
        }

        badge.innerHTML = `
            <div class="filter-info">
                <h5>${filter.params.column}</h5>
                <p>${filter.name === 'numeric_range' ? 'Numeric Range' : 'Category Filter'}: ${descText}</p>
            </div>
            <button class="btn-remove-filter" data-index="${idx}">&times;</button>
        `;
        
        // Remove handler
        badge.querySelector('.btn-remove-filter').addEventListener('click', (e) => {
            const removeIdx = parseInt(e.target.dataset.index);
            state.activeFilters.splice(removeIdx, 1);
            renderActiveFilters();
        });
        
        els.activeFilters.appendChild(badge);
    });
}

// Fetch applicable statistical methods
async function fetchApplicableMethods() {
    try {
        const response = await fetch(`/wizard/sessions/${state.sessionId}/methods`);
        if (!response.ok) throw new Error('Failed to retrieve applicable methods list.');
        
        state.applicableMethods = await response.json();
        els.methodsList.innerHTML = '';
        els.btnStep3Next.disabled = true;
        state.selectedMethod = '';
        
        if (state.applicableMethods.length === 0) {
            els.methodsList.innerHTML = '<p class="no-filters-msg">No statistical methods are applicable to your current dataset properties. Please check your data or adjust preprocessing filters.</p>';
            return;
        }

        state.applicableMethods.forEach(method => {
            const card = document.createElement('div');
            card.className = 'method-card';
            card.dataset.name = method.name;
            
            card.innerHTML = `
                <div class="method-title">${method.name}</div>
                <div class="method-desc">${method.description}</div>
            `;
            
            card.addEventListener('click', (e) => {
                document.querySelectorAll('.method-card').forEach(c => c.classList.remove('selected'));
                const activeCard = e.currentTarget;
                activeCard.classList.add('selected');
                
                state.selectedMethod = activeCard.dataset.name;
                els.btnStep3Next.disabled = false;
            });
            
            els.methodsList.appendChild(card);
        });
    } catch (err) {
        showError(err.message);
    }
}

// Run the evaluation endpoint
async function executeStatisticalMethod() {
    try {
        const response = await fetch(`/wizard/sessions/${state.sessionId}/results`);
        if (!response.ok) throw new Error('Statistical evaluation run failed.');
        
        const data = await response.json();
        
        const container = document.getElementById('statResultsContainer');
        container.innerHTML = '';

        if (data.length === 0) {
            container.textContent = 'No statistical results generated.';
            return;
        }

        const table = document.createElement('table');
        table.className = 'results-table';
        const thead = document.createElement('thead');
        thead.innerHTML = `
            <tr>
                <th>Column</th>
                <th>Method</th>
                <th>Statistic</th>
                <th>p-value</th>
                <th>Effect Size</th>
                <th>Summary</th>
            </tr>
        `;
        table.appendChild(thead);

        const tbody = document.createElement('tbody');
        data.forEach(res => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${res.column_name || ''}</td>
                <td>${res.method_name}</td>
                <td>${Number(res.test_statistic).toFixed(4)}</td>
                <td>${Number(res.p_value).toFixed(6)}</td>
                <td>${res.effect_size !== null ? Number(res.effect_size).toFixed(4) : ''}</td>
                <td>${res.summary}</td>
            `;
            tbody.appendChild(tr);
        });
        table.appendChild(tbody);
        container.appendChild(table);
    } catch (err) {
        showError(err.message);
    }
}

// Fetch applicable plots list
async function fetchApplicablePlots() {
    try {
        const response = await fetch(`/wizard/sessions/${state.sessionId}/plots`);
        if (!response.ok) throw new Error('Failed to fetch applicable plot generators.');
        
        state.applicablePlots = await response.json();
        els.plotsSelector.innerHTML = '';
        els.btnStep5Next.disabled = true;
        state.selectedPlots = [];
        els.plotsDisplay.innerHTML = '<span class="no-plots-msg">Select one or more plots to generate.</span>';
        
        if (state.applicablePlots.length === 0) {
            els.plotsSelector.innerHTML = '<p class="no-filters-msg">No visualizations applicable.</p>';
            return;
        }

        state.applicablePlots.forEach(plot => {
            const card = document.createElement('div');
            card.className = 'plot-select-item';
            card.dataset.name = plot.name;
            
            card.innerHTML = `
                <input type="checkbox" id="chk-plot-${plot.name}">
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
                
                els.btnStep5Next.disabled = state.selectedPlots.length === 0;
                generatePlotsPreview();
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
                els.btnStep5Next.disabled = state.selectedPlots.length === 0;
                generatePlotsPreview();
            });
            
            els.plotsSelector.appendChild(card);
        });
    } catch (err) {
        showError(err.message);
    }
}

// Generate plots client side preview (updates live as they check boxes)
async function generatePlotsPreview() {
    if (state.selectedPlots.length === 0) {
        els.plotsDisplay.innerHTML = '<span class="no-plots-msg">Select one or more plots to generate.</span>';
        return;
    }
    
    try {
        els.plotsDisplay.innerHTML = '<span class="no-plots-msg">Generating plots...</span>';
        
        const response = await fetch(`/wizard/sessions/${state.sessionId}/plots`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ selected_plots: state.selectedPlots })
        });
        
        if (!response.ok) throw new Error('Plot rendering failed.');
        
        const data = await response.json();
        els.plotsDisplay.innerHTML = '';
        
        data.plot_results.forEach(plot => {
            const container = document.createElement('div');
            container.className = 'plot-image-wrapper';
            container.style.textAlign = 'center';
            container.style.width = '100%';
            
            const img = document.createElement('img');
            img.src = `data:${plot.content_type};base64,${plot.image_base64}`;
            img.alt = plot.plot_type;
            
            const title = document.createElement('div');
            title.style.marginTop = '0.5rem';
            title.style.fontSize = '0.8rem';
            title.style.color = 'var(--text-secondary)';
            title.textContent = `${plot.plot_type.toUpperCase()} Generator`;
            
            container.appendChild(img);
            container.appendChild(title);
            els.plotsDisplay.appendChild(container);
        });
    } catch (err) {
        els.plotsDisplay.innerHTML = `<span class="no-plots-msg text-error">Failed to render plots: ${err.message}</span>`;
    }
}

// Show Error Toast UI
function showError(message) {
    els.toastMsg.textContent = message;
    els.errorToast.classList.remove('hidden');
    
    // Automatically hide after 5 seconds
    setTimeout(() => {
        els.errorToast.classList.add('hidden');
    }, 5000);
}
