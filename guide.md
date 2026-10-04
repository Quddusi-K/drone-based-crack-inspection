You are an experienced computer vision and machine learning research engineer. Build a complete, research-oriented computer vision project for **UAV/drone-based visual inspection of chemical tank walls**, with a particular focus on **crack detection, semantic segmentation, morphological refinement, and quantitative crack characterization**.
Use venv ~/Music/wheel/.venv for laptop. Some of the instruction/ steps below may be redundant or unnecessary so you may skip them
The eventual application is inspection of industrial chemical tanks using drone imagery. However, the actual drone imagery is currently confidential and cannot be used in this prototype. Therefore, for development and experimentation, use a **publicly available concrete-wall crack dataset from Kaggle** as a proxy dataset.

The concrete-wall dataset should be treated as a **methodology-development dataset**, not as a direct representation of chemical-tank imagery.

The project should investigate both:

1. Classical computer vision approaches
2. Deep-learning-based semantic segmentation

The project should specifically investigate the effects of:

* Image preprocessing
* Photometric/illumination variation
* Classical feature-based crack detection
* Deep-learning semantic segmentation
* Morphological refinement
* Crack characterization
* Data augmentation
* Quantitative evaluation
* Error analysis

Do not fabricate experimental results. All metrics and conclusions must be generated from experiments performed on the actual dataset.

---

# 1. PROJECT OBJECTIVE

Develop an end-to-end computer-vision pipeline conceptually structured as:

Image acquisition
→ preprocessing
→ crack detection
→ semantic segmentation
→ morphological refinement
→ crack characterization
→ quantitative evaluation

The eventual target scenario is:

UAV image of chemical tank wall
→ image preprocessing
→ defect/crack localization
→ segmentation
→ refinement
→ quantitative characterization

For the current implementation, the input will instead be images from the public Kaggle concrete-wall crack dataset.

The research questions should include:

* How effectively can cracks be detected using classical computer vision?
* How does preprocessing affect crack visibility?
* How robust are crack-detection methods to illumination and photometric variation?
* How does deep-learning segmentation compare with classical approaches?
* Does morphological refinement improve segmentation quality?
* Can the resulting crack masks be used for quantitative crack characterization?
* Which components of the pipeline contribute most to performance?

---

# 2. DATASET

The project will use a **Kaggle concrete-wall crack dataset**.
the current crack_sample folder only contain few samples for classical approaches, the whole dataset for deep learning approch is in kaggle
---
All the image proprocessing, refinement, loss function and approaches  and others are just some exhaustive suggestion which you may reduce or change if you think there is a better and more time saving approach
# 4. IMAGE PREPROCESSING

Create a modular preprocessing pipeline.

The preprocessing stage should specifically investigate the problem of **photometric variation**.

Although the current concrete-wall dataset is a proxy, the eventual tank imagery may contain:

* uneven illumination
* shadows
* bright reflections
* low-light regions
* changes in exposure
* changes in camera orientation
* low contrast between crack and surface

Implement configurable preprocessing methods such as:

### Normalization

* Min-max normalization
* Standardization
* RGB normalization

### Contrast enhancement

* Histogram equalization
* CLAHE

### Denoising

* Gaussian filtering
* Median filtering
* Bilateral filtering

### Illumination correction

Investigate methods such as:

* Gamma correction
* Local contrast normalization
* Background illumination estimation
* Retinex-style illumination correction

Do not automatically apply every preprocessing technique.

Create experiments that compare:

1. Raw images
2. Normalized images
3. Contrast-enhanced images
4. Illumination-corrected images
5. Combined preprocessing

Generate before/after visualizations.

The purpose is to investigate whether preprocessing improves the visibility and separability of crack structures.

---

# 5. PHOTOMETRIC VARIATION EXPERIMENT

Create a dedicated experiment to investigate robustness to changes in image appearance.

Simulate realistic photometric variations such as:

* brightness changes
* contrast changes
* gamma changes
* shadows
* local illumination changes
* mild color/intensity shifts

Do not change the crack geometry when performing these transformations.

Evaluate the model under different photometric conditions.

Compare:

Original
vs.
Photometrically altered

and measure:

* IoU
* Dice
* Precision
* Recall
* F1

This should provide an experimental investigation into **robustness to photometric variation**.

---

# 6. CLASSICAL COMPUTER VISION

Implement the classical computer-vision pipeline separately from the deep-learning pipeline.

This component should be designed to run efficiently on a **local laptop without requiring GPU acceleration**.

The classical CV pipeline should include configurable combinations of:

Input image
→ preprocessing
→ edge/gradient extraction
→ thresholding
→ morphological operations
→ connected-component filtering
→ crack mask

Implement:

### Edge detection

* Canny
* Sobel
* Laplacian

### Thresholding

* Global thresholding
* Otsu thresholding
* Adaptive thresholding

### Morphological operations

* Erosion
* Dilation
* Opening
* Closing

### Connected components

Remove small isolated candidate regions using configurable area thresholds.

The classical pipeline should output:

* binary crack mask
* overlay
* quantitative metrics
* processing time

Measure the computational cost of each classical approach.

---

# 7. CLASSICAL CV EXPERIMENTS

Create experiments comparing different classical approaches.

For example:

Experiment 1:
Raw image + Canny

Experiment 2:
CLAHE + Canny

Experiment 3:
Illumination correction + Canny

Experiment 4:
CLAHE + adaptive thresholding

Experiment 5:
Preprocessing + edge detection + morphology

Experiment 6:
Preprocessing + thresholding + morphology + connected components

Do not assume any method is superior.

Generate an experimental comparison table containing:

Method
Preprocessing
Edge/Threshold Method
Morphology
IoU
Dice
Precision
Recall
F1
Processing Time

---

# 8. DEEP LEARNING

All deep-learning model training should be designed to run on **Kaggle notebooks using GPU acceleration**.

Use PyTorch.

The project should include a Kaggle-compatible training notebook/script.

Start with:

### U-Net

U-Net should be the primary baseline for semantic segmentation.

Structure the implementation so that additional models can be added later.

Potential extensions include:

* U-Net++
* DeepLabV3
* SegFormer

Do not unnecessarily implement all models at once.

---

# 9. TRAINING PIPELINE

Implement:

* train/validation/test split
* configurable image size
* configurable batch size
* reproducible random seed
* optimizer
* learning-rate scheduler
* early stopping
* checkpointing
* training logs

Use Adam or AdamW.

Save the best model according to validation Dice or IoU.

The test set must only be used for final evaluation.

The training notebook should automatically detect whether a Kaggle GPU is available and report:

* GPU name
* CUDA availability
* device being used

---

# 10. LOSS FUNCTIONS

Implement configurable losses:

### Binary Cross Entropy

BCE

### Dice Loss

Dice Loss = 1 - Dice

### Combined BCE + Dice

Support:

Loss = BCE + Dice Loss

Optionally investigate:

* Focal Loss
* Tversky Loss

The purpose is to investigate whether losses designed for class imbalance improve thin-crack segmentation.

---

# 11. DATA AUGMENTATION

Implement realistic augmentation.

Geometric transformations:

* Horizontal flip
* Vertical flip where appropriate
* Small rotations
* Scaling
* Cropping
* Mild perspective transformation

Photometric transformations:

* Brightness
* Contrast
* Gamma
* Gaussian noise
* Blur
* Mild color/intensity changes

Geometric transformations must be applied consistently to image and mask.

Photometric transformations must only affect the image.

Make augmentation configurable.

---

# 12. SEMANTIC SEGMENTATION

The deep-learning system should perform binary semantic segmentation:

Input:

H × W × 3

Output:

H × W × 1

where each pixel represents the probability of being a crack.

Use sigmoid activation for the binary output.

Generate:

* probability map
* thresholded binary mask
* ground-truth comparison
* overlay visualization

Investigate appropriate probability thresholds rather than blindly assuming 0.5 is optimal.

---

# 13. EVALUATION METRICS

Evaluate segmentation using:

### Primary metrics

* IoU / Jaccard
* Dice coefficient

### Additional metrics

* Precision
* Recall
* F1
* Pixel accuracy
* Specificity

Calculate:

TP
FP
TN
FN

at the pixel level.

Use:

IoU = TP / (TP + FP + FN)

Dice = 2TP / (2TP + FP + FN)

Precision = TP / (TP + FP)

Recall = TP / (TP + FN)

F1 = 2 × Precision × Recall / (Precision + Recall)

Because cracks can occupy a small fraction of an image, explicitly discuss why pixel accuracy alone is insufficient.

---

# 14. MORPHOLOGICAL REFINEMENT

After obtaining the neural-network segmentation mask, implement configurable morphological post-processing.

Pipeline:

Probability map
→ threshold
→ binary mask
→ morphological refinement
→ connected-component filtering
→ optional skeletonization

Implement:

* Opening
* Closing
* Hole filling
* Small-component removal
* Gap closing
* Optional dilation/erosion

The purpose is to:

* suppress isolated false positives
* remove segmentation noise
* fill small holes
* connect fragmented crack regions
* improve structural continuity

Compare:

Raw neural-network segmentation

vs.

Neural-network segmentation + morphological refinement.

Evaluate both using:

* IoU
* Dice
* Precision
* Recall
* F1

---

# 15. CRACK CHARACTERIZATION

After obtaining the final crack mask, extract quantitative properties.

Calculate:

### Crack area

Number of crack pixels.

### Crack length

Use skeletonization to estimate crack centerline length.

### Crack width

Use distance-transform-based estimation or another appropriate image-processing technique.

### Orientation

Estimate dominant crack orientation.

### Connected components

Estimate the number of individual crack regions.

### Crack density

Crack area relative to inspected image area.

### Branching

Analyze skeleton junction points.

### Aspect ratio

Characterize the geometry of connected crack regions.

Return measurements in a structured format such as:

{
"crack_area_pixels": ...,
"crack_length_pixels": ...,
"mean_width_pixels": ...,
"max_width_pixels": ...,
"orientation_degrees": ...,
"num_components": ...,
"crack_density": ...,
"branch_points": ...
}

Unless calibration information is available, report measurements in **pixels/image coordinates**, not millimeters or centimeters.

---

# 16. CRACK CHARACTERIZATION VISUALIZATION

Generate visualizations containing:

Original image

*

Predicted crack mask

*

Ground-truth mask

*

Prediction overlay

*

Skeleton

*

Connected components

*

Crack measurements

Display:

* crack area
* estimated length
* estimated mean width
* maximum width
* orientation
* number of components

Save the visualizations automatically.

---

# 17. CRACK CHARACTERIZATION EVALUATION

If manually measured ground-truth crack properties are available, evaluate:

### Crack length

* MAE
* RMSE

### Crack width

* MAE
* RMSE

### Crack area

* absolute error
* relative error

### Orientation

* mean absolute angular error

Clearly distinguish between:

1. segmentation accuracy
2. crack measurement accuracy

---

# 18. EXPERIMENTAL DESIGN

Create an experiment framework comparing:

### Experiment A

Raw images + classical CV

### Experiment B

Preprocessed images + classical CV

### Experiment C

Raw images + U-Net

### Experiment D

Preprocessed images + U-Net

### Experiment E

Preprocessed + augmentation + U-Net

### Experiment F

U-Net + morphological refinement

The framework should automatically record:

Method
Preprocessing
Augmentation
Loss
Morphology
IoU
Dice
Precision
Recall
F1
Inference time

Do not fabricate any values.

---

# 19. ABLATION STUDIES

Create an ablation-study framework.

Investigate:

* preprocessing
* CLAHE
* illumination correction
* augmentation
* loss function
* morphological refinement

For example:

Baseline U-Net

vs.

U-Net + CLAHE

vs.

U-Net + illumination correction

vs.

U-Net + augmentation

vs.

U-Net + augmentation + morphology

The purpose is to identify which components actually contribute to segmentation performance.

---

# 20. ERROR ANALYSIS

Create an automated error-analysis module.

Identify examples containing:

### False positives

Predicted crack but no ground-truth crack.

### False negatives

Actual crack missed by the model.

### Boundary errors

Prediction does not accurately follow the crack boundary.

### Fragmentation

One continuous crack predicted as multiple disconnected regions.

### Merging

Multiple cracks incorrectly merged.

Generate visualizations of representative failure cases.

Include both quantitative and qualitative error analysis.

---

# 21. VISUALIZATIONS

Generate:

* Training loss curves
* Validation loss curves
* Dice curves
* IoU curves
* Precision/Recall curves
* Confusion matrix
* Method comparison plots
* Before/after preprocessing
* Before/after morphological refinement
* Segmentation overlays
* Failure cases
* Crack skeletons
* Crack characterization visualizations

Use matplotlib.

Make figures suitable for inclusion in a research report or presentation.

---

# 22. LAPTOP VS KAGGLE WORKFLOW

Clearly separate the project into two computational environments.

## Local Laptop

The laptop should be used for:

* Dataset exploration
* Image preprocessing experiments
* Classical CV
* Annotation utilities
* Visualization
* Morphological operations
* Crack characterization
* Lightweight evaluation

These components should work without requiring a GPU.

## Kaggle

Kaggle should be used for:

* Deep-learning training
* GPU-accelerated inference
* U-Net experiments
* Data augmentation experiments
* Loss-function experiments
* Ablation studies
* Model evaluation

Create Kaggle-compatible notebooks/scripts for the deep-learning workflow.

The code should automatically use:

CUDA if available

otherwise:

CPU

---

# 23. PROJECT STRUCTURE

Do not impose a structure on the downloaded Kaggle dataset.

Instead, maintain a clean project-code structure such as:

project/
│
├── configs/
│   └── config.yaml
│
├── src/
│   ├── data/
│   ├── preprocessing/
│   ├── classical_cv/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   ├── postprocessing/
│   ├── characterization/
│   └── visualization/
│
├── experiments/
│
├── checkpoints/
│
├── results/
│
├── notebooks/
│   └── kaggle_training.ipynb
│
├── scripts/
│
├── requirements.txt
│
└── README.md

The dataset path should be configurable rather than hard-coded.

---

# 24. CONFIGURATION

Use a YAML configuration file.

Example:

dataset:
path: ""
image_size: 512
batch_size: 8

preprocessing:
normalization: true
clahe: true
illumination_correction: false

augmentation:
enabled: true

model:
name: unet

training:
epochs: 50
learning_rate: 0.0001
optimizer: adamw

loss:
name: dice_bce

postprocessing:
morphology: true
min_component_area: 20

Make all important parameters configurable.

---

# 25. COMMAND-LINE INTERFACE

Provide commands such as:

python preprocess.py

python classical_cv.py

python train.py --config configs/config.yaml

python evaluate.py --checkpoint checkpoints/best.pt

python predict.py --input image.jpg

python characterize.py --input prediction.png

python run_experiment.py --experiment unet_clahe

The prediction pipeline should generate:

* input image
* segmentation probability map
* binary mask
* refined mask
* overlay
* crack measurements

---

# 26. REPRODUCIBILITY

Implement:

* random seeds
* configuration saving
* model checkpointing
* experiment IDs
* training logs
* metric logging

Every experiment should save its configuration and results.

For example:

results/
└── experiment_001/
├── config.yaml
├── metrics.json
├── predictions/
├── plots/
└── qualitative_examples/

---

# 27. README

Create a comprehensive README explaining:

1. Problem statement
2. Motivation
3. Confidentiality limitation and proxy dataset
4. Dataset setup
5. Annotation procedure
6. Preprocessing
7. Photometric variation
8. Classical CV methodology
9. Deep-learning methodology
10. Semantic segmentation
11. Morphological refinement
12. Crack characterization
13. Evaluation metrics
14. Experimental setup
15. Ablation studies
16. Error analysis
17. Local laptop workflow
18. Kaggle GPU workflow
19. Limitations
20. Future work

Include a Mermaid diagram showing:

Drone/tank imagery
→ preprocessing
→ classical CV / deep learning
→ segmentation
→ morphological refinement
→ crack characterization
→ quantitative evaluation

Clearly state that the current experiments use a public concrete-wall dataset because the actual industrial drone imagery is confidential.

---

# 28. DEVELOPMENT APPROACH

Do not attempt to create everything as one giant script.

Develop the project incrementally.

First:

1. Inspect the Kaggle dataset.
2. Understand its labels and structure.
3. Build the dataset loader.
4. Visualize sample images.
5. Implement preprocessing.
6. Implement classical CV.
7. Implement annotation/mask handling.
8. Implement evaluation.
9. Implement U-Net.
10. Implement Kaggle training.
11. Implement morphological refinement.
12. Implement crack characterization.
13. Implement experiments and ablations.
14. Create final visualizations and README.

After each major component, test it before proceeding.

If the dataset lacks pixel-level masks, clearly identify this early and implement the necessary annotation workflow rather than silently treating image-level labels as segmentation ground truth.
