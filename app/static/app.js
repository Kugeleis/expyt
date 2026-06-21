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
    selectedValueColumns: new Set(),
    selectedGroups: new Set(),
    availableGroups: [],
    activeFilters: [],
    applicableMethods: [],
    selectedMethod: '',
    applicablePlots: [],
    selectedPlots: [],
    selectedExportFormat: 'pdf',
    statResults: [],
    plotsTopN: 1,
    sortColumn: null,
    sortAsc: true
};

// UI Elements
const els = {
    sessionInfo: document.getElementById('session-info'),
    fileUpload: document.getElementById('file-upload'),
    uploadStatus: document.getElementById('upload-status'),
    datasetDetails: document.getElementById('dataset-details'),
    detailDesc: document.getElementById('detail-desc'),
    groupColSelect: document.getElementById('group-col-select'),
    valueColSearch: document.getElementById('value-col-search'),
    valueColumnsList: document.getElementById('value-columns-list'),
    subgroupsSection: document.getElementById('subgroups-section'),
    subgroupsSearch: document.getElementById('subgroups-search'),
    subgroupsList: document.getElementById('subgroups-list'),
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
    plotsSigFilter: document.getElementById('plots-sig-filter'),
    filteredPlotsCounter: document.getElementById('filtered-plots-counter'),
    btnStep4Next: document.getElementById('btn-step-4-next'),
    
    plotsSelector: document.getElementById('plots-selector'),
    plotsDisplay: document.getElementById('plots-display'),
    btnGeneratePlots: document.getElementById('btn-generate-plots'),
    plotsGenerationCounter: document.getElementById('plots-generation-counter'),
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
            state.selectedValueColumns = new Set();
            state.selectedGroups = new Set();
            state.availableGroups = [];
            if (els.valueColSearch) {
                els.valueColSearch.value = '';
            }
            if (els.subgroupsSearch) {
                els.subgroupsSearch.value = '';
            }
            
            // Clean active filters panel
            renderActiveFilters();
            els.plotsDisplay.innerHTML = '<span class="no-plots-msg">No plots generated yet.</span>';
            els.btnStep1Next.disabled = true;
            els.btnStep3Next.disabled = true;
            els.btnStep5Next.disabled = true;
            els.fileUpload.value = '';
            els.uploadStatus.textContent = '';
            els.datasetDetails.classList.add('hidden');
            els.subgroupsSection.classList.add('hidden');
            els.subgroupsList.innerHTML = '';
            
            await startNewSession();
        }
    });
    
    // Toast close
    els.btnToastClose.addEventListener('click', () => {
        els.errorToast.classList.add('hidden');
    });

    els.groupColSelect.addEventListener('change', async () => {
        const selectedGroupCol = els.groupColSelect.value;
        if (selectedGroupCol) {
            state.selectedValueColumns.delete(selectedGroupCol);
        }
        updateValueColumnsList();
        await updateSubgroupsList();
    });

    if (els.valueColSearch) {
        els.valueColSearch.addEventListener('input', () => {
            updateValueColumnsList();
        });
    }

    if (els.subgroupsSearch) {
        els.subgroupsSearch.addEventListener('input', () => {
            renderSubgroupsList();
        });
    }

    const selectAllBtn = document.getElementById('btn-select-all-cols');
    const deselectAllBtn = document.getElementById('btn-deselect-all-cols');

    if (selectAllBtn) {
        selectAllBtn.addEventListener('click', () => {
            const checkboxes = els.valueColumnsList.querySelectorAll('input[type="checkbox"]');
            checkboxes.forEach(cb => {
                cb.checked = true;
                state.selectedValueColumns.add(cb.value);
            });
            validateStep1Next();
        });
    }

    if (deselectAllBtn) {
        deselectAllBtn.addEventListener('click', () => {
            const checkboxes = els.valueColumnsList.querySelectorAll('input[type="checkbox"]');
            checkboxes.forEach(cb => {
                cb.checked = false;
                state.selectedValueColumns.delete(cb.value);
            });
            validateStep1Next();
        });
    }

    const selectAllGroupsBtn = document.getElementById('btn-select-all-groups');
    const deselectAllGroupsBtn = document.getElementById('btn-deselect-all-groups');

    if (selectAllGroupsBtn) {
        selectAllGroupsBtn.addEventListener('click', () => {
            const checkboxes = els.subgroupsList.querySelectorAll('input[type="checkbox"]');
            checkboxes.forEach(cb => {
                cb.checked = true;
                state.selectedGroups.add(cb.value);
            });
            validateStep1Next();
        });
    }

    if (deselectAllGroupsBtn) {
        deselectAllGroupsBtn.addEventListener('click', () => {
            const checkboxes = els.subgroupsList.querySelectorAll('input[type="checkbox"]');
            checkboxes.forEach(cb => {
                cb.checked = false;
                state.selectedGroups.delete(cb.value);
            });
            validateStep1Next();
        });
    }


    // Step 1: Upload Data automatically when file is selected
    els.fileUpload.addEventListener('change', async () => {
        const file = els.fileUpload.files[0];
        if (!file) return;

        els.uploadStatus.textContent = 'Uploading...';
        els.uploadStatus.style.color = 'var(--text-secondary)';
        els.fileUpload.disabled = true;

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
            els.valueColumnsList.innerHTML = '';
            els.filterCol.innerHTML = '';

            dataset.columns.forEach(col => {
                if (col.is_discrete) {
                    const opt1 = document.createElement('option');
                    opt1.value = col.name;
                    opt1.textContent = `${col.name} (${col.dtype})`;
                    els.groupColSelect.appendChild(opt1);
                }

                const opt3 = document.createElement('option');
                opt3.value = col.name;
                opt3.textContent = `${col.name} (${col.dtype})`;
                els.filterCol.appendChild(opt3);
            });

            // Initialize state.selectedValueColumns
            state.selectedValueColumns = new Set();
            const selectedGroupCol = els.groupColSelect.value;
            dataset.columns.forEach(col => {
                if (col.is_numeric && col.name !== selectedGroupCol) {
                    state.selectedValueColumns.add(col.name);
                }
            });

            if (els.valueColSearch) {
                els.valueColSearch.value = '';
            }

            updateValueColumnsList();
            await updateSubgroupsList();

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
            els.fileUpload.disabled = false;
        }
    });
    
    // Step 1: Submit dataset
    els.btnStep1Next.addEventListener('click', async () => {
        try {
            const selectedCols = Array.from(state.selectedValueColumns);

            const payload = {
                dataset_id: state.selectedDatasetId,
                group_column: els.groupColSelect.value,
                selected_value_columns: selectedCols,
                selected_groups: Array.from(state.selectedGroups)
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
                body: JSON.stringify({
                    selected_plots: state.selectedPlots,
                    top_n_columns: state.plotsTopN
                })
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

    // Step 5: Click Generate Plots button
    els.btnGeneratePlots.addEventListener('click', async () => {
        await generatePlotsPreview();
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

    // Step 4: Significance filter change
    if (els.plotsSigFilter) {
        els.plotsSigFilter.addEventListener('input', () => {
            updatePlotsFilter();
        });
    }

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
            const card = document.createElement('article');
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

// Client-side sorting for Step 4 statistical results
function sortResults(field) {
    if (state.sortColumn === field) {
        state.sortAsc = !state.sortAsc;
    } else {
        state.sortColumn = field;
        state.sortAsc = true;
    }
    
    state.statResults.sort((a, b) => {
        let valA = a[field];
        let valB = b[field];
        
        // Keep null/undefined values always at the bottom of the table
        if (valA === null || valA === undefined) {
            if (valB === null || valB === undefined) return 0;
            return 1;
        }
        if (valB === null || valB === undefined) {
            return -1;
        }
        
        if (typeof valA === 'number' && typeof valB === 'number') {
            return state.sortAsc ? valA - valB : valB - valA;
        } else {
            // String comparison
            const strA = String(valA).toLowerCase();
            const strB = String(valB).toLowerCase();
            if (strA < strB) return state.sortAsc ? -1 : 1;
            if (strA > strB) return state.sortAsc ? 1 : -1;
            return 0;
        }
    });
    
    renderResultsTable();
}

// Render the results table
function renderResultsTable() {
    const container = document.getElementById('statResultsContainer');
    if (!container) return;
    container.innerHTML = '';

    if (!state.statResults || state.statResults.length === 0) {
        container.textContent = 'No statistical results generated.';
        return;
    }

    const wrapper = document.createElement('div');
    wrapper.className = 'overflow-auto';

    const table = document.createElement('table');
    table.className = 'results-table striped';
    
    const thead = document.createElement('thead');
    const tr = document.createElement('tr');
    
    const headers = [
        { label: 'Column', field: 'column_name' },
        { label: 'Method', field: 'method_name' },
        { label: 'Statistic', field: 'test_statistic' },
        { label: 'p-value', field: 'p_value' },
        { label: 'Effect Size', field: 'effect_size' }
    ];
    
    headers.forEach(h => {
        const th = document.createElement('th');
        th.setAttribute('scope', 'col');
        th.style.cursor = 'pointer';
        th.style.userSelect = 'none';
        th.dataset.field = h.field;
        
        let indicator = '';
        if (state.sortColumn === h.field) {
            indicator = state.sortAsc ? ' ▲' : ' ▼';
        }
        th.textContent = h.label + indicator;
        
        th.addEventListener('click', () => {
            sortResults(h.field);
        });
        
        tr.appendChild(th);
    });
    
    thead.appendChild(tr);
    table.appendChild(thead);

    const tbody = document.createElement('tbody');
    state.statResults.forEach(res => {
        const trRow = document.createElement('tr');
        trRow.innerHTML = `
            <td>${res.column_name || ''}</td>
            <td>${res.method_name}</td>
            <td>${res.test_statistic !== null && res.test_statistic !== undefined ? Number(res.test_statistic).toFixed(4) : ''}</td>
            <td>${res.p_value !== null && res.p_value !== undefined ? Number(res.p_value).toFixed(6) : ''}</td>
            <td>${res.effect_size !== null && res.effect_size !== undefined ? Number(res.effect_size).toFixed(4) : ''}</td>
        `;
        tbody.appendChild(trRow);
    });
    table.appendChild(tbody);
    wrapper.appendChild(table);
    container.appendChild(wrapper);
}

// Calculate the number of variables with p-value <= significance threshold
function updatePlotsFilter() {
    const filterInput = els.plotsSigFilter;
    if (!filterInput) return;
    
    let threshold = parseFloat(filterInput.value);
    if (isNaN(threshold) || threshold < 0) {
        threshold = 0.05;
    }
    
    const matchingResults = state.statResults.filter(res => res.p_value !== null && res.p_value !== undefined && res.p_value <= threshold);
    const count = matchingResults.length;
    
    state.plotsTopN = count;
    
    if (els.filteredPlotsCounter) {
        els.filteredPlotsCounter.textContent = `Matches ${count} variable${count !== 1 ? 's' : ''}`;
    }
    
    // Update step 5 plots count if needed
    updatePlotsCounter();
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

// Generate plots client side preview (updates live as they check boxes)
async function generatePlotsPreview() {
    if (state.selectedPlots.length === 0) {
        els.plotsDisplay.innerHTML = '<span class="no-plots-msg">Select one or more plots to generate.</span>';
        return;
    }
    
    try {
        els.plotsDisplay.innerHTML = '<span class="no-plots-msg">Generating plots...</span>';
        els.btnGeneratePlots.disabled = true;
        els.btnStep5Next.disabled = true;
        
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
            title.textContent = colName;
            header.appendChild(title);
            card.appendChild(header);
            
            // Grid row for plots - up to 3 columns
            const row = document.createElement('div');
            row.style.display = 'grid';
            row.style.gridTemplateColumns = 'repeat(auto-fit, minmax(200px, 1fr))';
            row.style.gap = '1rem';
            row.style.width = '100%';
            
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
        
        els.btnStep5Next.disabled = false;
    } catch (err) {
        els.plotsDisplay.innerHTML = `<span class="no-plots-msg text-error">Failed to render plots: ${err.message}</span>`;
    } finally {
        els.btnGeneratePlots.disabled = false;
    }
}

function updatePlotsCounter() {
    const numVariables = state.plotsTopN;
    const numPlots = state.selectedPlots.length;
    const totalPlots = numVariables * numPlots;
    
    if (els.plotsGenerationCounter) {
        els.plotsGenerationCounter.textContent = `Will generate ${totalPlots} plot${totalPlots !== 1 ? 's' : ''} (${numVariables} variable${numVariables !== 1 ? 's' : ''} × ${numPlots} plot type${numPlots !== 1 ? 's' : ''})`;
    }
    
    if (els.btnGeneratePlots) {
        els.btnGeneratePlots.disabled = totalPlots === 0;
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

// Helpers for Multi-column select checklist
// Helpers for Multi-column select checklist
function isNumericDtype(col) {
    return col && col.is_numeric;
}

function isDiscreteDtype(col) {
    return col && col.is_discrete;
}

function updateValueColumnsList() {
    const selectedGroupCol = els.groupColSelect.value;
    const filterText = (els.valueColSearch?.value || '').toLowerCase().trim();

    els.valueColumnsList.innerHTML = '';
    
    let hasNumeric = false;
    let hasVisibleNumeric = false;
    
    state.selectedDatasetColumns.forEach(col => {
        if (col.is_numeric && col.name !== selectedGroupCol) {
            hasNumeric = true;
            
            // Check if column name matches filter text
            if (filterText && !col.name.toLowerCase().includes(filterText)) {
                return;
            }
            
            hasVisibleNumeric = true;
            
            const item = document.createElement('label');
            item.className = 'value-column-item';
            
            const cb = document.createElement('input');
            cb.type = 'checkbox';
            cb.value = col.name;
            cb.checked = state.selectedValueColumns.has(col.name);
            cb.addEventListener('change', (e) => {
                if (e.target.checked) {
                    state.selectedValueColumns.add(col.name);
                } else {
                    state.selectedValueColumns.delete(col.name);
                }
                validateStep1Next();
            });
            
            const span = document.createElement('span');
            span.textContent = `${col.name} (${col.dtype})`;
            
            item.appendChild(cb);
            item.appendChild(span);
            els.valueColumnsList.appendChild(item);
        }
    });
    
    if (!hasNumeric) {
        els.valueColumnsList.innerHTML = '<span class="no-columns-msg" style="color: var(--text-secondary); font-size: 0.95rem;">No numeric columns available.</span>';
    } else if (!hasVisibleNumeric) {
        els.valueColumnsList.innerHTML = '<span class="no-columns-msg" style="color: var(--text-secondary); font-size: 0.95rem;">No columns match search.</span>';
    }
    
    validateStep1Next();
}

function validateStep1Next() {
    const selectedGroupCol = els.groupColSelect.value;
    if (!selectedGroupCol || state.selectedValueColumns.size === 0 || state.selectedGroups.size === 0) {
        els.btnStep1Next.disabled = true;
    } else {
        els.btnStep1Next.disabled = false;
    }
}

async function updateSubgroupsList() {
    const datasetId = state.selectedDatasetId;
    const groupCol = els.groupColSelect.value;
    
    if (!datasetId || !groupCol) {
        els.subgroupsSection.classList.add('hidden');
        els.subgroupsList.innerHTML = '';
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

function renderSubgroupsList() {
    els.subgroupsList.innerHTML = '';
    const filterText = (els.subgroupsSearch?.value || '').toLowerCase().trim();
    
    let hasVisibleGroups = false;
    
    state.availableGroups.forEach(groupVal => {
        if (filterText && !groupVal.toLowerCase().includes(filterText)) {
            return;
        }
        
        hasVisibleGroups = true;
        
        const item = document.createElement('label');
        item.className = 'value-column-item';
        
        const cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.value = groupVal;
        cb.checked = state.selectedGroups.has(groupVal);
        cb.addEventListener('change', (e) => {
            if (e.target.checked) {
                state.selectedGroups.add(groupVal);
            } else {
                state.selectedGroups.delete(groupVal);
            }
            validateStep1Next();
        });
        
        const span = document.createElement('span');
        span.textContent = groupVal;
        
        item.appendChild(cb);
        item.appendChild(span);
        els.subgroupsList.appendChild(item);
    });
    
    if (state.availableGroups.length === 0) {
        els.subgroupsList.innerHTML = '<span class="no-columns-msg" style="color: var(--text-secondary); font-size: 0.95rem;">No values available in this column.</span>';
    } else if (!hasVisibleGroups) {
        els.subgroupsList.innerHTML = '<span class="no-columns-msg" style="color: var(--text-secondary); font-size: 0.95rem;">No subgroups match search.</span>';
    }
    
    els.subgroupsSection.classList.remove('hidden');
    validateStep1Next();
}



