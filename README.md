# Test Description

This code is used for testing the model.

The model weights (`model.pth`) and test sample data (`test.npy`) can be downloaded from the following link:  
👉 [Download Link](https://drive.google.com/file/d/1gHg7nT_cQ_VIj39XD3a2yONUVcuGOlsS/view?usp=drive_link)
👉 [Download Link](https://drive.google.com/file/d/1w-AheEjmhSbdfnuVMEBpJxGrM0rwVLYR/view?usp=drive_link)

## Data Description

- `test.npy` contains 20 test samples, divided into 4 classes with 5 samples per class.
- Each test sample undergoes a similarity selection process:
  - The 15 most similar training samples are selected from the training set.
  - These selected samples are used to construct a **batch-wise population information graph** for testing.
