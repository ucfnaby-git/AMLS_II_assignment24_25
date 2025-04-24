# README

# 📸 Real Image Denoising with MWCNN

### NTIRE 2020 SIDD+ (Track 2) – AMLS II Coursework

This project was developed as part of the AMLS II module at UCL. It tackles the NTIRE 2020 Real Image Denoising Challenge (Track 2: sRGB) using a Multi-level Wavelet Convolutional Neural Network (MWCNN). The model was trained from scratch and demonstrates competitive performance on the official SIDD+ benchmark.

------

## 📁 Project Structure

-   `main.py` – Entrypoint for training and validation
-   `submit_srgb.py` – Script for generating the submission file
-   `A/` – Core codebase
    -   `model.py` – MWCNN architecture
    -   `myssim.py` – Official SSIM scoring script
    -   `trainingset.py` – Dataset class for full training images
    -   `valmatset.py` – Dataset class for validation `mat` patches
    -   `indexcalc.py` – PSNR and SSIM evaluation functions
    -   `results/` – Trained weights, submission CSV, and training curve
-   `B/` – (Empty folder reserved for future components)
-   `Datasets/` – Input data for training, validation, and testing
    -   `train/`, `valid/`, and `test/` folders
-   `env/` – Reproducibility files
    -   `environment.yml`, `requirements.txt`

------

## ⚙️ How to Run

1.  **Set up the environment.**
    Use `conda env create -f env/environment.yml` to install the conda environment.

    If failed try `pip install -r env/requirements.txt` to fix it.

1.  **Train the model**
     Run `main.py` to begin training. Training outputs are saved to `A/results/`. 

     The code will output Training loss, Learning rate, Validation loss, PSNR and SIMM. All these results will be saved in train_curve.csv.

2.  **Generate Submission**
     Run `submit_srgb.py` to produce `SubmitSrgb.csv`, ready for Kaggle SIDD Benchmark upload.

------

## 📝 Report

All architectural design, experimental settings, training methodology, and results are documented in the report:

**AMLS_II_Denoising_Report.pdf**

------

## 🏆 Benchmark Scores (Kaggle)

-   **PSNR**: 38.4238 (Rank **29** / 96 Teams)
-   **SSIM**: 0.897506 (Rank **30** / 83 Teams)

These results outperform the NTIRE 2020 winner in PSNR by over 5 dB using a single MWCNN model.

------

## 🔁 Reproducibility

All model checkpoints, logs, and results are stored in `A/results/`. The submission file is generated from the exact 1250-epoch checkpoint used for benchmarking.

------

## ⚠️ Notes

-   Please manually place the SIDD+ dataset in the `Datasets/` folder before training or testing.
-   `myssim.py` must remain unchanged to ensure compatibility with official evaluation standards.

## 📁 More detailed folder structure
```
\AMLSII_24-25_SN24070891
│  
│  main.py
│  README.md
│  submit_srgb.py
│  
├─A
│  │  indexcalc.py
│  │  model.py
│  │  myssim.py
│  │  submit_srgb.py
│  │  trainingset.py
│  │  valmatset.py
│  │  __init__.py
│  │  
│  └─results
│          mwcnn_trained.pth
│          SubmitSrgb.csv
│          training-carve.csv
│          training_metrics_curve.png
│          
├─B
├─Datasets
│  ├─test
│  │      BenchmarkNoisyBlocksSrgb.mat
│  │      
│  ├─train
│  │  └─sRGB
│  │      ├─0001_001_S6_00100_00060_3200_L
│  │      │      GT_SRGB_010.PNG
│  │      │      NOISY_SRGB_010.PNG
│  │      │      
│  │      ├─0002_001_S6_00100_00020_3200_N
│  │      │      GT_SRGB_010.PNG
│  │      │      NOISY_SRGB_010.PNG
│  │      ├─ ......
│  │      │      
│  │      └─0200_010_GP_01600_03200_5500_N
│  │              GT_SRGB_010.PNG
│  │              NOISY_SRGB_010.PNG
│  │              
│  └─valid
│          siddplus_valid_gt_srgb.mat
│          siddplus_valid_noisy_srgb.mat
│          
└─env
        environment.yml
        requirements.txt
```