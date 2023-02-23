# AMLNet
This code implements AMLNet from the manuscript "A deep-learning pipeline for the diagnosis and discrimination of acute myeloid leukemia from bone marrow smears"
## Prerequisites
Any operating system on which you can run GPU-accelerated PyTorch. Python >=3.8. For packages see requirements.txt.
### Training
Directory structure for training and test data:
#### For train
	DATA
	└── Train
	│   ├──M1
	│   ├──M2a
	│   ├──...
	├── Test
	│   ├──M1
	│   ├──M2a
	│   ├──...
#### For 5-fold cross-validation
		DATA
	└── All
	│   ├──M1
	│   ├──M2a
	│   ├──...
#### For test at the patient-level
		DATA
	└── Dual_center_test
	│   ├──M1
		│   ├──Patien1
		│   ├──Patien2
		│   ├──...
	│   ├──M2a
		│   ├──Patien1
		│   ├──Patien2
		│   ├──...
		...
Code in ./training
- 1.train.py
- 2.train_5_fold.py

Nvidia TITAN RTX with 24GB of memory for each GPU is used to accelerate training.
### Test
Code in ./test
- 1.test.py
- 2.test_patient.py













