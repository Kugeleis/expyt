import { state } from './state.js';
import { els } from './elements.js';
import { goToStep, navigateToStep } from './navigation.js';
import {
    startNewSession,
    executeStatisticalMethod,
    fetchApplicableMethods,
    fetchApplicablePlots,
    generatePlotsPreview,
    updateSubgroupsList,
    updateClustersList
} from './api.js';
import {
    renderActiveFilters,
    updateValueColumnsList,
    updatePlotsFilter,
    validateStep1Next,
    renderSubgroupsList,
    renderClustersList,
    renderResultsTable,
    populateHierarchyDropdowns,
    startHeaderPacman,
    stopHeaderPacman
} from './ui.js';
import { showError } from './helpers.js';

// Setup Event Listeners
export function initEventListeners() {
    // Restart session
    if (els.btnRestart) {
        els.btnRestart.addEventListener('click', async () => {
            if (confirm('Are you sure you want to restart the session? All configuration will be lost.')) {
                // Reset state
                state.activeFilters = [];
                state.selectedMethod = '';
                state.selectedDiscreteMethod = '';
                state.selectedPlots = [];
                state.selectedValueColumns = new Set();
                state.selectedDiscreteColumns = new Set();
                state.selectedGroups = new Set();
                state.selectedClusters = new Set();
                state.availableClusters = [];
                state.availableGroups = [];
                if (els.valueColSearch) {
                    els.valueColSearch.value = '';
                }
                if (els.discreteColSearch) {
                    els.discreteColSearch.value = '';
                }
                if (els.subgroupsSearch) {
                    els.subgroupsSearch.value = '';
                }

                // Clean active filters panel
                renderActiveFilters();
                els.plotsDisplay.innerHTML = '<span class="no-plots-msg">No plots generated yet.</span>';
                if (els.btnSidebarNext) els.btnSidebarNext.disabled = true;
                els.fileUpload.value = '';
                els.uploadStatus.textContent = '';
                els.datasetDetails.classList.add('hidden');
                els.subgroupsSection.classList.add('hidden');
                els.subgroupsList.innerHTML = '';

                state.isHierarchical = false;
                els.hierarchyConfigSection.classList.add('hidden');
                if (els.clustersSection) els.clustersSection.classList.add('hidden');
                if (els.clustersList) els.clustersList.innerHTML = '';
                if (els.optClusterExclusion) els.optClusterExclusion.classList.add('hidden');
                if (els.tabFlat) {
                    els.tabFlat.classList.add('active');
                    els.tabFlat.style.borderBottomColor = 'var(--pico-primary)';
                    els.tabFlat.style.color = 'var(--pico-color)';
                }
                if (els.tabHierarchical) {
                    els.tabHierarchical.classList.remove('active');
                    els.tabHierarchical.style.borderBottomColor = 'transparent';
                    els.tabHierarchical.style.color = 'var(--pico-muted-color)';
                    els.tabHierarchical.disabled = false;
                    els.tabHierarchical.style.opacity = '';
                    els.tabHierarchical.style.cursor = 'pointer';
                    els.tabHierarchical.title = '';
                }

                // Show flat tab content and hide hierarchical tab content
                const flatTabContent = document.getElementById('flat-tab-content');
                if (flatTabContent) flatTabContent.classList.remove('hidden');

                // Move Group Column Card to Flat container
                const groupColCard = document.getElementById('group-column-card');
                const flatGroupContainer = document.getElementById('flat-group-container');
                if (groupColCard && flatGroupContainer) {
                    flatGroupContainer.appendChild(groupColCard);
                }

                if (els.enableHierarchy) els.enableHierarchy.checked = false;

                await startNewSession();
            }
        });
    }

    // Toast close
    if (els.btnToastClose) {
        els.btnToastClose.addEventListener('click', () => {
            els.errorToast.classList.add('hidden');
        });
    }

    if (els.groupColSelect) {
        els.groupColSelect.addEventListener('change', async () => {
            const prevGroupCol = els.groupColSelect.dataset.prevValue;
            const selectedGroupCol = els.groupColSelect.value;
            if (prevGroupCol && prevGroupCol !== selectedGroupCol) {
                const colMeta = state.selectedDatasetColumns.find((c) => c.name === prevGroupCol);
                if (colMeta) {
                    if (colMeta.is_numeric) {
                        state.selectedValueColumns.add(prevGroupCol);
                    } else if (colMeta.is_discrete) {
                        state.selectedDiscreteColumns.add(prevGroupCol);
                    }
                }
            }
            if (selectedGroupCol) {
                state.selectedValueColumns.delete(selectedGroupCol);
                state.selectedDiscreteColumns.delete(selectedGroupCol);
            }
            els.groupColSelect.dataset.prevValue = selectedGroupCol;

            updateValueColumnsList();
            await updateSubgroupsList();

            if (state.isHierarchical) {
                populateHierarchyDropdowns();
                await updateClustersList();
            }
        });
    }

    if (els.valueColSearch) {
        els.valueColSearch.addEventListener('input', () => {
            updateValueColumnsList();
        });
    }

    if (els.discreteColSearch) {
        els.discreteColSearch.addEventListener('input', () => {
            updateValueColumnsList();
        });
    }

    if (els.clusterColSelect) {
        els.clusterColSelect.addEventListener('change', async () => {
            updateValueColumnsList();
            await updateClustersList();
        });
    }

    if (els.unitColSelect) {
        els.unitColSelect.addEventListener('change', () => {
            updateValueColumnsList();
        });
    }

    if (els.xColSelect) {
        els.xColSelect.addEventListener('change', () => {
            updateValueColumnsList();
        });
    }

    if (els.yColSelect) {
        els.yColSelect.addEventListener('change', () => {
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
            checkboxes.forEach((cb) => {
                cb.checked = true;
                state.selectedValueColumns.add(cb.value);
            });
            validateStep1Next();
        });
    }

    if (deselectAllBtn) {
        deselectAllBtn.addEventListener('click', () => {
            const checkboxes = els.valueColumnsList.querySelectorAll('input[type="checkbox"]');
            checkboxes.forEach((cb) => {
                cb.checked = false;
                state.selectedValueColumns.delete(cb.value);
            });
            validateStep1Next();
        });
    }

    const selectAllDiscreteBtn = document.getElementById('btn-select-all-discrete');
    const deselectAllDiscreteBtn = document.getElementById('btn-deselect-all-discrete');

    if (selectAllDiscreteBtn) {
        selectAllDiscreteBtn.addEventListener('click', () => {
            const checkboxes = els.discreteColumnsList.querySelectorAll('input[type="checkbox"]');
            checkboxes.forEach((cb) => {
                cb.checked = true;
                state.selectedDiscreteColumns.add(cb.value);
            });
            validateStep1Next();
        });
    }

    if (deselectAllDiscreteBtn) {
        deselectAllDiscreteBtn.addEventListener('click', () => {
            const checkboxes = els.discreteColumnsList.querySelectorAll('input[type="checkbox"]');
            checkboxes.forEach((cb) => {
                cb.checked = false;
                state.selectedDiscreteColumns.delete(cb.value);
            });
            validateStep1Next();
        });
    }

    if (els.clustersSearch) {
        els.clustersSearch.addEventListener('input', () => {
            renderClustersList();
        });
    }

    const selectAllClustersBtn = document.getElementById('btn-select-all-clusters');
    const deselectAllClustersBtn = document.getElementById('btn-deselect-all-clusters');

    if (selectAllClustersBtn) {
        selectAllClustersBtn.addEventListener('click', () => {
            const checkboxes = els.clustersList.querySelectorAll('input[type="checkbox"]');
            checkboxes.forEach((cb) => {
                cb.checked = true;
                state.selectedClusters.add(cb.value);
            });
            validateStep1Next();
        });
    }

    if (deselectAllClustersBtn) {
        deselectAllClustersBtn.addEventListener('click', () => {
            const checkboxes = els.clustersList.querySelectorAll('input[type="checkbox"]');
            checkboxes.forEach((cb) => {
                cb.checked = false;
                state.selectedClusters.delete(cb.value);
            });
            validateStep1Next();
        });
    }

    const selectAllGroupsBtn = document.getElementById('btn-select-all-groups');
    const deselectAllGroupsBtn = document.getElementById('btn-deselect-all-groups');

    if (selectAllGroupsBtn) {
        selectAllGroupsBtn.addEventListener('click', () => {
            const checkboxes = els.subgroupsList.querySelectorAll('input[type="checkbox"]');
            checkboxes.forEach((cb) => {
                cb.checked = true;
                state.selectedGroups.add(cb.value);
            });
            validateStep1Next();
        });
    }

    if (deselectAllGroupsBtn) {
        deselectAllGroupsBtn.addEventListener('click', () => {
            const checkboxes = els.subgroupsList.querySelectorAll('input[type="checkbox"]');
            checkboxes.forEach((cb) => {
                cb.checked = false;
                state.selectedGroups.delete(cb.value);
            });
            validateStep1Next();
        });
    }

    // Step 1: Upload Data automatically when file is selected
    if (els.fileUpload) {
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
                state.selectedDatasetColumns = dataset.columns.map((col) => ({
                    ...col,
                    is_numeric: col.is_numeric === true || col.is_numeric === 'true',
                    is_discrete: col.is_discrete === true || col.is_discrete === 'true'
                }));
                els.detailDesc.textContent = dataset.description;

                // Populate group and value columns
                els.groupColSelect.innerHTML = '';
                els.valueColumnsList.innerHTML = '';
                els.filterCol.innerHTML = '';

                state.selectedDatasetColumns.forEach((col) => {
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

                // Initialize state.selectedValueColumns and state.selectedDiscreteColumns
                state.selectedValueColumns = new Set();
                state.selectedDiscreteColumns = new Set();
                const firstDiscreteCol = state.selectedDatasetColumns.find((col) => col.is_discrete);
                const selectedGroupCol = firstDiscreteCol ? firstDiscreteCol.name : '';
                if (firstDiscreteCol) {
                    els.groupColSelect.value = firstDiscreteCol.name;
                }
                els.groupColSelect.dataset.prevValue = selectedGroupCol;
                state.selectedDatasetColumns.forEach((col) => {
                    if (col.name !== selectedGroupCol) {
                        if (col.is_numeric) {
                            state.selectedValueColumns.add(col.name);
                        } else if (col.is_discrete) {
                            state.selectedDiscreteColumns.add(col.name);
                        }
                    }
                });

                if (els.valueColSearch) {
                    els.valueColSearch.value = '';
                }
                if (els.discreteColSearch) {
                    els.discreteColSearch.value = '';
                }

                updateValueColumnsList();
                await updateSubgroupsList();

                state.isHierarchical = false;
                els.hierarchyConfigSection.classList.add('hidden');
                if (els.clustersSection) els.clustersSection.classList.add('hidden');
                if (els.optClusterExclusion) els.optClusterExclusion.classList.add('hidden');
                if (els.tabFlat) {
                    els.tabFlat.classList.add('active');
                    els.tabFlat.style.borderBottomColor = 'var(--pico-primary)';
                    els.tabFlat.style.color = 'var(--pico-color)';
                }
                if (els.tabHierarchical) {
                    els.tabHierarchical.classList.remove('active');
                    els.tabHierarchical.style.borderBottomColor = 'transparent';
                    els.tabHierarchical.style.color = 'var(--pico-muted-color)';
                }

                // Show flat tab content and hide hierarchical tab content
                const flatTabContent = document.getElementById('flat-tab-content');
                if (flatTabContent) flatTabContent.classList.remove('hidden');

                // Move Group Column Card to Flat container
                const groupColCard = document.getElementById('group-column-card');
                const flatGroupContainer = document.getElementById('flat-group-container');
                if (groupColCard && flatGroupContainer) {
                    flatGroupContainer.appendChild(groupColCard);
                }

                if (els.enableHierarchy) els.enableHierarchy.checked = false;

                // Grey out Hierarchical tab if only one discrete column is present
                const discreteCount = state.selectedDatasetColumns.filter((col) => col.is_discrete).length;
                if (els.tabHierarchical) {
                    if (discreteCount <= 1) {
                        els.tabHierarchical.disabled = true;
                        els.tabHierarchical.style.opacity = '0.5';
                        els.tabHierarchical.style.cursor = 'not-allowed';
                        els.tabHierarchical.title =
                            'Hierarchical mode requires at least two discrete columns (one for group, one for cluster).';
                    } else {
                        els.tabHierarchical.disabled = false;
                        els.tabHierarchical.style.opacity = '';
                        els.tabHierarchical.style.cursor = 'pointer';
                        els.tabHierarchical.title = '';
                    }
                }

                els.datasetDetails.classList.remove('hidden');
                if (els.btnSidebarNext) els.btnSidebarNext.disabled = false;
                els.uploadStatus.textContent = 'Upload successful!';
                els.uploadStatus.style.color = 'var(--success-green)';
            } catch (err) {
                showError(err.message);
                els.uploadStatus.textContent = 'Upload failed.';
                els.uploadStatus.style.color = 'var(--error-red)';
                els.datasetDetails.classList.add('hidden');
                if (els.btnSidebarNext) els.btnSidebarNext.disabled = true;
            } finally {
                els.fileUpload.disabled = false;
            }
        });
    }

    // Tab event handlers
    if (els.tabFlat && els.tabHierarchical) {
        els.tabFlat.addEventListener('click', () => {
            state.isHierarchical = false;
            els.tabFlat.classList.add('active');
            els.tabFlat.style.borderBottomColor = 'var(--pico-primary)';
            els.tabFlat.style.color = 'var(--pico-color)';

            els.tabHierarchical.classList.remove('active');
            els.tabHierarchical.style.borderBottomColor = 'transparent';
            els.tabHierarchical.style.color = 'var(--pico-muted-color)';

            // Move Group Column Card to Flat container
            const groupColCard = document.getElementById('group-column-card');
            const flatGroupContainer = document.getElementById('flat-group-container');
            if (groupColCard && flatGroupContainer) {
                flatGroupContainer.appendChild(groupColCard);
            }

            // Toggle tab content visibility
            const flatTabContent = document.getElementById('flat-tab-content');
            if (flatTabContent) flatTabContent.classList.remove('hidden');
            els.hierarchyConfigSection.classList.add('hidden');

            if (els.enableHierarchy) els.enableHierarchy.checked = false;

            if (els.clustersSection) els.clustersSection.classList.add('hidden');
            if (els.optClusterExclusion) els.optClusterExclusion.classList.add('hidden');
            if (els.filterType.value === 'cluster_exclusion') {
                els.filterType.value = 'numeric_range';
                els.filterType.dispatchEvent(new Event('change'));
            }
            updateValueColumnsList();
            validateStep1Next();
        });

        els.tabHierarchical.addEventListener('click', async () => {
            if (els.tabHierarchical.disabled) return;
            state.isHierarchical = true;
            els.tabHierarchical.classList.add('active');
            els.tabHierarchical.style.borderBottomColor = 'var(--pico-primary)';
            els.tabHierarchical.style.color = 'var(--pico-color)';

            els.tabFlat.classList.remove('active');
            els.tabFlat.style.borderBottomColor = 'transparent';
            els.tabFlat.style.color = 'var(--pico-muted-color)';

            // Move Group Column Card to Hierarchical container
            const groupColCard = document.getElementById('group-column-card');
            const hierarchicalGroupContainer = document.getElementById('hierarchical-group-container');
            if (groupColCard && hierarchicalGroupContainer) {
                hierarchicalGroupContainer.appendChild(groupColCard);
            }

            // Toggle tab content visibility
            const flatTabContent = document.getElementById('flat-tab-content');
            if (flatTabContent) flatTabContent.classList.add('hidden');
            els.hierarchyConfigSection.classList.remove('hidden');

            if (els.enableHierarchy) els.enableHierarchy.checked = true;

            if (els.optClusterExclusion) els.optClusterExclusion.classList.remove('hidden');

            populateHierarchyDropdowns();
            await updateClustersList();
            updateValueColumnsList();
            validateStep1Next();
        });
    }

    // Step 2: Add filter action
    if (els.btnAddFilterAction) {
        els.btnAddFilterAction.addEventListener('click', () => {
            const type = els.filterType.value;
            const col = els.filterCol.value;

            if (!col && type !== 'cluster_exclusion') return;

            let filterObj = { name: type, params: {} };
            if (type !== 'cluster_exclusion') {
                filterObj.params.column = col;
            }

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
            } else if (type === 'category_filter') {
                const valsStr = els.filterCatValues.value.trim();
                if (!valsStr) {
                    showError('Please input at least one categorical value.');
                    return;
                }

                const values = valsStr
                    .split(',')
                    .map((v) => v.trim())
                    .filter((v) => v.length > 0);
                filterObj.params.categories = values;
                filterObj.params.exclude = els.filterCatExclude.checked;

                // Clear inputs
                els.filterCatValues.value = '';
                els.filterCatExclude.checked = false;
            } else if (type === 'cluster_exclusion') {
                const clusterId = els.filterClusterId.value.trim();
                const reason = els.filterClusterReason.value.trim();

                if (!clusterId) {
                    showError('Please input a cluster ID.');
                    return;
                }
                if (!reason) {
                    showError('Please input a non-empty reason for exclusion.');
                    return;
                }

                filterObj.params = {
                    exclusions: [{ cluster_id: clusterId, reason: reason }]
                };

                // Clear inputs
                els.filterClusterId.value = '';
                els.filterClusterReason.value = '';
            }

            state.activeFilters.push(filterObj);
            renderActiveFilters();
        });
    }

    // Step 5: Click Generate Plots button
    if (els.btnGeneratePlots) {
        els.btnGeneratePlots.addEventListener('click', async () => {
            await generatePlotsPreview();
        });
    }

    // Step 6: Exporter select card
    document.querySelectorAll('.exporter-card').forEach((card) => {
        card.addEventListener('click', (e) => {
            document.querySelectorAll('.exporter-card').forEach((c) => c.classList.remove('active'));
            const activeCard = e.currentTarget;
            activeCard.classList.add('active');
            state.selectedExportFormat = activeCard.dataset.format;
        });
    });

    // Step 6: Download Report
    if (els.btnExportDownload) {
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
    }

    // Step 4: Significance filter change
    if (els.plotsSigFilter) {
        els.plotsSigFilter.addEventListener('input', () => {
            updatePlotsFilter();
            renderResultsTable(); // Re-render table and chart with new limit
        });
    }

    // Handle centralized Sidebar Next Button logic
    if (els.btnSidebarNext) {
        els.btnSidebarNext.addEventListener('click', async () => {
            try {
                startHeaderPacman();
                if (state.currentStep === 'dataset_selection') {
                    const payload = {
                        dataset_id: state.selectedDatasetId,
                        group_column: els.groupColSelect.value,
                        selected_value_columns: Array.from(state.selectedValueColumns),
                        selected_discrete_columns: Array.from(state.selectedDiscreteColumns),
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
                    state.selectedValueColumns = new Set(data.selected_value_columns || []);
                    state.selectedDiscreteColumns = new Set(data.selected_discrete_columns || []);

                    if (state.isHierarchical) {
                        const hierarchyPayload = {
                            group_col: els.groupColSelect.value,
                            cluster_col: els.clusterColSelect.value,
                            selected_clusters: Array.from(state.selectedClusters),
                            unit_col: els.unitColSelect.value || null,
                            x_col: els.xColSelect.value || null,
                            y_col: els.yColSelect.value || null
                        };

                        const hierResponse = await fetch(`/wizard/sessions/${state.sessionId}/hierarchy`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(hierarchyPayload)
                        });

                        if (!hierResponse.ok) {
                            const errData = await hierResponse.json();
                            throw new Error(errData.detail || 'Failed to configure hierarchical settings.');
                        }
                        const hierData = await hierResponse.json();
                        state.selectedValueColumns = new Set(hierData.session.selected_value_columns || []);
                        state.selectedClusters = new Set(hierData.session.hierarchy.selected_clusters || []);
                        state.selectedDiscreteColumns = new Set(hierData.session.selected_discrete_columns || []);
                    }

                    navigateToStep(data.current_step);
                } else if (state.currentStep === 'filters') {
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
                    await fetchApplicableMethods();
                    navigateToStep(data.current_step);
                } else if (state.currentStep === 'stat_method') {
                    const payload = {};
                    if (state.selectedValueColumns.size > 0) {
                        payload.selected_method = state.selectedMethod;
                    }
                    if (state.selectedDiscreteColumns.size > 0) {
                        payload.selected_discrete_method = state.selectedDiscreteMethod;
                    }

                    const response = await fetch(`/wizard/sessions/${state.sessionId}/method`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });

                    if (!response.ok) {
                        const errData = await response.json();
                        throw new Error(errData.detail || 'Method selection rejected.');
                    }
                    const data = await response.json();
                    await executeStatisticalMethod();
                    navigateToStep(data.current_step);
                } else if (state.currentStep === 'results') {
                    await fetchApplicablePlots();
                    navigateToStep('plot_selection');
                } else if (state.currentStep === 'plot_selection') {
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
                } else if (state.currentStep === 'export') {
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
                }
            } catch (err) {
                showError(err.message);
            } finally {
                stopHeaderPacman();
            }
        });
    }

    // Handle centralized Sidebar Back Button logic
    if (els.btnSidebarBack) {
        els.btnSidebarBack.addEventListener('click', () => {
            switch (state.currentStep) {
                case 'filters':
                    goToStep('dataset_selection');
                    break;
                case 'stat_method':
                    goToStep('filters');
                    break;
                case 'results':
                    goToStep('stat_method');
                    break;
                case 'plot_selection':
                    goToStep('results');
                    break;
                case 'export':
                    goToStep('plot_selection');
                    break;
            }
        });
    }
}
