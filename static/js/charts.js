// ============================================
// Disease Prediction System - Chart.js Visualizations
// ============================================

const CHART_COLORS = {
    mobilenet_v3: { bg: 'rgba(67, 97, 238, 0.7)', border: '#4361ee' },
    resnet50: { bg: 'rgba(114, 9, 183, 0.7)', border: '#7209b7' },
    vgg16: { bg: 'rgba(247, 37, 133, 0.7)', border: '#f72585' },
};

const METRIC_LABELS = {
    accuracy: 'Accuracy',
    precision: 'Precision',
    recall: 'Recall',
    f1_score: 'F1 Score',
    auc_roc: 'AUC-ROC',
};

// --- Grouped Bar Chart: All Metrics Per Model ---
function renderMetricsBarChart(canvasId, metricsData) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    const metricKeys = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc'];
    const datasets = [];

    Object.entries(metricsData).forEach(([modelKey, data]) => {
        const colors = CHART_COLORS[modelKey] || { bg: 'rgba(100,100,100,0.7)', border: '#666' };
        datasets.push({
            label: data.model_name || modelKey,
            data: metricKeys.map((k) => ((data.metrics || {})[k] || 0) * 100),
            backgroundColor: colors.bg,
            borderColor: colors.border,
            borderWidth: 1,
            borderRadius: 4,
        });
    });

    new Chart(canvas, {
        type: 'bar',
        data: {
            labels: metricKeys.map((k) => METRIC_LABELS[k]),
            datasets,
        },
        options: {
            responsive: true,
            plugins: {
                legend: { position: 'top' },
                tooltip: {
                    callbacks: {
                        label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(2)}%`,
                    },
                },
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: { callback: (v) => v + '%' },
                },
            },
        },
    });
}

// --- Radar Chart: Accuracy Comparison ---
function renderAccuracyRadar(canvasId, metricsData) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    const metricKeys = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc'];
    const datasets = [];

    Object.entries(metricsData).forEach(([modelKey, data]) => {
        const colors = CHART_COLORS[modelKey] || { bg: 'rgba(100,100,100,0.2)', border: '#666' };
        datasets.push({
            label: data.model_name || modelKey,
            data: metricKeys.map((k) => ((data.metrics || {})[k] || 0) * 100),
            backgroundColor: colors.bg.replace('0.7', '0.15'),
            borderColor: colors.border,
            borderWidth: 2,
            pointBackgroundColor: colors.border,
        });
    });

    new Chart(canvas, {
        type: 'radar',
        data: {
            labels: metricKeys.map((k) => METRIC_LABELS[k]),
            datasets,
        },
        options: {
            responsive: true,
            scales: {
                r: {
                    beginAtZero: true,
                    max: 100,
                    ticks: { stepSize: 20 },
                },
            },
            plugins: {
                legend: { position: 'top' },
            },
        },
    });
}

// --- Confusion Matrix Heatmaps ---
function renderConfusionMatrices(metricsData, classLabels) {
    Object.entries(metricsData).forEach(([modelKey, data]) => {
        const cm = data.confusion_matrix;
        if (!cm) return;

        const canvasId = `cm_${modelKey}`;
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;

        // Flatten confusion matrix into scatter data for heatmap
        const chartData = [];
        const maxVal = Math.max(...cm.flat());

        cm.forEach((row, i) => {
            row.forEach((val, j) => {
                chartData.push({ x: j, y: i, v: val });
            });
        });

        // Use a table-based visualization since Chart.js doesn't natively support heatmaps
        const container = canvas.parentElement;
        canvas.style.display = 'none';

        const table = document.createElement('div');
        table.className = 'cm-grid';
        table.innerHTML = buildConfusionMatrixHTML(cm, classLabels, maxVal);
        container.appendChild(table);
    });
}

function buildConfusionMatrixHTML(cm, labels, maxVal) {
    let html = '<table class="table table-sm text-center mb-0" style="font-size: 0.8rem;">';

    // Header row
    html += '<thead><tr><th></th>';
    labels.forEach((l) => {
        html += `<th class="text-muted">${l.length > 6 ? l.substring(0, 6) + '.' : l}</th>`;
    });
    html += '</tr></thead><tbody>';

    // Data rows
    cm.forEach((row, i) => {
        html += `<tr><td class="fw-bold text-muted">${labels[i].length > 6 ? labels[i].substring(0, 6) + '.' : labels[i]}</td>`;
        row.forEach((val, j) => {
            const intensity = maxVal > 0 ? val / maxVal : 0;
            const isDiag = i === j;
            const bg = isDiag
                ? `rgba(6, 214, 160, ${0.2 + intensity * 0.6})`
                : `rgba(247, 37, 133, ${intensity * 0.4})`;
            html += `<td style="background: ${bg}; font-weight: 600;">${val}</td>`;
        });
        html += '</tr>';
    });

    html += '</tbody></table>';
    html += '<div class="text-center mt-1"><small class="text-muted">Predicted &rarr; | Actual &darr;</small></div>';
    return html;
}

// --- Cross-Disease Comparison Chart ---
function renderCrossDiseaseChart(canvasId, comparisons, metricKey, metricLabel) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    const diseases = Object.keys(comparisons);
    const diseaseLabels = diseases.map((k) => comparisons[k].name);

    // Collect unique model keys
    const modelKeys = new Set();
    diseases.forEach((dk) => {
        Object.keys(comparisons[dk].metrics).forEach((mk) => modelKeys.add(mk));
    });

    const datasets = [];
    modelKeys.forEach((modelKey) => {
        const colors = CHART_COLORS[modelKey] || { bg: 'rgba(100,100,100,0.7)', border: '#666' };
        const data = diseases.map((dk) => {
            const m = comparisons[dk].metrics[modelKey];
            return m ? (m.metrics[metricKey] || 0) * 100 : 0;
        });

        // Get display name from first available entry
        let displayName = modelKey;
        for (const dk of diseases) {
            if (comparisons[dk].metrics[modelKey]) {
                displayName = comparisons[dk].metrics[modelKey].model_name || modelKey;
                break;
            }
        }

        datasets.push({
            label: displayName,
            data,
            backgroundColor: colors.bg,
            borderColor: colors.border,
            borderWidth: 1,
            borderRadius: 4,
        });
    });

    new Chart(canvas, {
        type: 'bar',
        data: { labels: diseaseLabels, datasets },
        options: {
            responsive: true,
            plugins: {
                legend: { position: 'top' },
                title: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(2)}%`,
                    },
                },
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: { display: true, text: metricLabel + ' (%)' },
                    ticks: { callback: (v) => v + '%' },
                },
            },
        },
    });
}
