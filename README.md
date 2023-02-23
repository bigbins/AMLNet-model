# AMLNet
This code implements AMLNet from the manuscript "<font size="3">***A deep-learning pipeline for the diagnosis and discrimination of acute myeloid leukemia from bone marrow smears***</font>".

## Prerequisites
Any operating system on which you can run GPU-accelerated PyTorch. Python >=3.8. For packages see requirements.txt.
## Training
Directory structure for training and test data:

	DATA
	└── Train
	│   ├──M1
	│   ├──M2a
	│   ├──...
	└── Test
	│   ├──M1
	│   ├──M2a
	│   ├──...

#### For 5-fold cross-validation train


	DATA
	└── All
	│   ├──M1
	│   ├──M2a
	│   ├──...
Code in `./train `
- 1.train.py: it uses early stop, label smoothing, various data augmentation methods, cosine annealing with warm restart and so on for training.
- 2.train_kfold.py: based on the above, including 5-fold cross-validation for training.


Configuring hyperparameters via the `./experiments ` directory
-  `./train `: compare different models.
-  `./kfold `: implement 5-fold cross-validation, add DROPOUT under the MODEL directory.

Nvidia TITAN RTX with 24GB of memory for GPU is used to accelerate training.
## Test
#### For test at the image-level
	DATA
	└── Dual_center_test
	│   ├──M1
	│   ├──M2a
	│   ├──...


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
- 1.test.py: evaluate model performance at the image level.
- 2.test_patient.py: evaluate model performance at the patient level.





