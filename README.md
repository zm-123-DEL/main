# Test Description

This code is used for testing the model.

The model weights (`model.pth`) and test sample data (`test.npy`) can be downloaded from the following link:  
👉 [Download Link](https://pan.com)

## Data Description

- `test.npy` contains 20 test samples, divided into 4 classes with 5 samples per class.
- Each test sample undergoes a similarity selection process:
  - The 15 most similar training samples are selected from the training set.
  - These selected samples are used to construct a **batch-wise population information graph** for testing.
