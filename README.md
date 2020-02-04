### Preprocessing

Build either a factor model and/or ICA model and/or Support vector Machines to extract information from speech using librosa library provided in Feature Extraction documentation. Find the ground truth from the CREMA-D database. 

[https://github.com/CheyneyComputerScience/CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)

### Speaker Verification

Use Policy Gradient techniques and DQN to map from features of speech in RAVDESS dataset to each actor discretised using log softmax function. 

Policy Gradient Methods: 

[https://pdfs.semanticscholar.org/86ef/8541139f5ba2bbc9964c194841d5f757dd63.pdf](https://pdfs.semanticscholar.org/86ef/8541139f5ba2bbc9964c194841d5f757dd63.pdf)

### Reward System

The loss function will be as per the speaker verification system loss function, used in d-vector. Build a state space model using the ground truth and use likelihood ratio methods to score the ground truth against actors.

### Training

Train the model against the actor in RAVDESS dataset
