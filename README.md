# AMLNet
This code implements AMLNet from the manuscript "A deep-learning pipeline for the diagnosis and discrimination of acute myeloid leukemia from bone marrow smears"
## Prerequisites
Any operating system on which you can run GPU-accelerated PyTorch. Python >=3.8. For packages see requirements.txt.
### Training
Directory structure for training and test data:

#### For 5-fold cross-validation train
	DATA
	└── All
	│   ├──M1
	│   ├──M2a
	│   ├──...

Code in `./train `
- 1.train_kfold.py: it uses early stop、label smoothing、5-fold cross-validation、various data augmentation methods、cosine annealing with warm restart and so on for training

Configuring hyperparameters via the `./experiments ` directory

Nvidia TITAN RTX with 24GB of memory for GPU is used to accelerate training.
### Test
#### For test at the image-level
	DATA
	└── Dual_center_test
	│   ├──M1
	│   ├──M2a
	│   ├──...
		...


#### For test at the patient-level
	DATA
	└── Dual_center_test
	│   ├──M1
		│   ├──Patient1
		│   ├──Patient2
		│   ├──...
	│   ├──M2a
		│   ├──Patient1
		│   ├──Patient2
		│   ├──...
		...
Code in `./test `
- 1.test.py: evaluate model performance at the image level
- 2.test_patient.py: evaluate model performance at the patient level





