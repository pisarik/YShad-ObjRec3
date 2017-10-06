# Kaggle competition yshad-objrec3

My code for classifying images from brodazt database with different transformations\distortions

## Organization of code and data

All work organized by folders and subfolders, separated in different levels. Depth of folder is level. Folders which not related to next level has '\_' prefix.

#### Levels
0. Contains common data, imgs;
1. Types of experiments (one against all, multi-classification, ...);
2. Types of preprocessings (raw, standardize, tiling)
3. Type of employed model, here we can see history of development of model, current results for tuning on train | valid.

## Data

111 classes of different textures. Each class represented with 5 grayscale images with size 200x200.

* comp1_test - shift??
* comp2_test - scale changed > 1
* comp3_test - scale changed < 1
* comp4_test - rotate


## Results