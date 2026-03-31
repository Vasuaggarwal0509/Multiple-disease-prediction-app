// ============================================
// Disease Prediction System - Main JS
// ============================================

document.addEventListener('DOMContentLoaded', function () {
    setupImageUpload();
    setupTabularForm();
});

// --- Image Upload & Prediction ---

function setupImageUpload() {
    const uploadZone = document.getElementById('uploadZone');
    const imageInput = document.getElementById('imageInput');
    const imagePreview = document.getElementById('imagePreview');
    const previewImg = document.getElementById('previewImg');
    const clearBtn = document.getElementById('clearImage');
    const predictBtn = document.getElementById('predictBtn');
    const imageForm = document.getElementById('imageForm');

    if (!uploadZone) return;

    // Click to browse
    uploadZone.addEventListener('click', () => imageInput.click());

    // Drag & drop
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('dragover');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            imageInput.files = e.dataTransfer.files;
            showPreview(e.dataTransfer.files[0]);
        }
    });

    // File selected
    imageInput.addEventListener('change', () => {
        if (imageInput.files.length) {
            showPreview(imageInput.files[0]);
        }
    });

    // Clear image
    if (clearBtn) {
        clearBtn.addEventListener('click', () => {
            imageInput.value = '';
            imagePreview.style.display = 'none';
            uploadZone.style.display = 'block';
            predictBtn.disabled = true;
            document.getElementById('imageResults').style.display = 'none';
            document.getElementById('resultsPlaceholder').style.display = 'block';
        });
    }

    // Form submit
    if (imageForm) {
        imageForm.addEventListener('submit', (e) => {
            e.preventDefault();
            runImagePrediction();
        });
    }

    function showPreview(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            imagePreview.style.display = 'block';
            uploadZone.style.display = 'none';
            predictBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }
}

async function runImagePrediction() {
    const form = document.getElementById('imageForm');
    const diseaseKey = form.dataset.disease;
    const imageInput = document.getElementById('imageInput');
    const loading = document.getElementById('predictionLoading');
    const results = document.getElementById('imageResults');
    const placeholder = document.getElementById('resultsPlaceholder');
    const grid = document.getElementById('modelResultsGrid');

    if (!imageInput.files.length) return;

    // Show loading
    loading.style.display = 'block';
    results.style.display = 'none';
    placeholder.style.display = 'none';

    const formData = new FormData();
    formData.append('disease_key', diseaseKey);
    formData.append('image', imageInput.files[0]);

    try {
        const response = await fetch('/predict/image', {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();

        if (data.error) {
            grid.innerHTML = `<div class="col-12"><div class="alert alert-danger">${data.error}</div></div>`;
        } else {
            renderImageResults(grid, data.results);
        }

        loading.style.display = 'none';
        results.style.display = 'block';
    } catch (err) {
        loading.style.display = 'none';
        grid.innerHTML = `<div class="col-12"><div class="alert alert-danger">Error: ${err.message}</div></div>`;
        results.style.display = 'block';
    }
}

function renderImageResults(container, results) {
    container.innerHTML = '';

    results.forEach((result) => {
        const col = document.createElement('div');
        col.className = 'col-md-4';

        if (!result.available) {
            col.innerHTML = `
                <div class="model-result-card unavailable">
                    <div class="model-name">${result.model_name}</div>
                    <p class="text-muted small mt-2">Model not downloaded yet</p>
                    <p class="small">Run <code>python scripts/download_models.py</code></p>
                </div>`;
        } else {
            let probBars = '';
            if (result.probabilities) {
                Object.entries(result.probabilities).forEach(([cls, prob]) => {
                    const pct = (prob * 100).toFixed(1);
                    const isTop = cls === result.label;
                    probBars += `
                        <div class="prob-row">
                            <span class="prob-label">${cls}</span>
                            <div class="prob-bar-bg">
                                <div class="prob-bar-fill" style="width: ${pct}%; ${isTop ? 'background: var(--success)' : ''}"></div>
                            </div>
                            <span class="prob-value">${pct}%</span>
                        </div>`;
                });
            }

            col.innerHTML = `
                <div class="model-result-card">
                    <div class="model-name">${result.model_name}</div>
                    <div class="prediction-label">${result.label}</div>
                    <div class="mb-2">
                        <small class="text-muted">Confidence</small>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${(result.confidence * 100).toFixed(1)}%"></div>
                        </div>
                        <small class="fw-bold">${(result.confidence * 100).toFixed(1)}%</small>
                    </div>
                    <hr>
                    <small class="text-muted fw-semibold">Class Probabilities</small>
                    <div class="mt-1">${probBars}</div>
                </div>`;
        }

        container.appendChild(col);
    });
}

// --- Tabular Prediction ---

function setupTabularForm() {
    const form = document.getElementById('tabularForm');
    if (!form) return;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const diseaseKey = form.dataset.disease;
        const inputs = form.querySelectorAll('.feature-input');
        const values = Array.from(inputs).map((input) => parseFloat(input.value) || 0);

        const resultDiv = document.getElementById('tabularResult');
        const contentDiv = document.getElementById('tabularResultContent');

        try {
            const response = await fetch('/predict/tabular', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ disease_key: diseaseKey, values }),
            });

            const data = await response.json();

            if (data.error) {
                contentDiv.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
            } else {
                const isPositive = data.prediction === 1;
                contentDiv.innerHTML = `
                    <div class="d-flex align-items-center gap-3 mt-2">
                        <span class="badge ${isPositive ? 'bg-danger' : 'bg-success'} fs-6 px-3 py-2">
                            ${data.label}
                        </span>
                        <span class="text-muted">via ${data.model}</span>
                    </div>`;
            }

            resultDiv.style.display = 'block';
        } catch (err) {
            contentDiv.innerHTML = `<div class="alert alert-danger">Error: ${err.message}</div>`;
            resultDiv.style.display = 'block';
        }
    });
}
