# Training Notebooks

Each notebook trains 3 CNN models (MobileNet V3, ResNet-50, VGG-16) on a disease dataset and exports:
- Trained `.keras` model files to `models/<disease>/`
- Metrics `.json` files to `metrics/<disease>/`

## How to Use

1. Open the notebook in **Google Colab** or **Kaggle** (GPU recommended)
2. Upload/mount the dataset
3. Run all cells
4. Download the output model files and metrics JSONs
5. Place them in the correct directories

## Notebooks

| Notebook | Disease | Dataset |
|----------|---------|---------|
| `eye_disease_comparison.ipynb` | Eye Disease (OCT) | Kermany OCT (84K images) |
| `brain_tumor_comparison.ipynb` | Brain Tumor (MRI) | Brain Tumor MRI (7K images) |
| `pneumonia_comparison.ipynb` | Pneumonia (X-Ray) | Chest X-Ray (5.8K images) |
| `malaria_comparison.ipynb` | Malaria (Cell) | NIH Malaria Cells (27K images) |

## Alternative: Download Pre-trained Models

Instead of training, you can download pre-trained models from Kaggle notebooks.
Run `python scripts/download_models.py --list` for links.
