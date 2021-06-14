# OptimAL-DL
deep learning models for biomedical relation classification.

Please download the pretrainde models named pytorch_model.bin and put them in the corresponding files

run "python run.py" in scripts file.

scripts_v1.0: Bert + fully connected layer. 
best acc: 80.43% (66.90%, 80.43%, 70.12%, 75.09%,76.52%. 75.09%, 73.67%, 73.67%, 72.60%, 72.95%)

scripts_v2.0: Bert + bi-lstm layer (1 layer) + fully connected layer. 
best acc:77.94% (70.10%, 72.60%, 74.73%, 71.53%, 77.94%, 72.24%, 74.02%, 73.31%, 74.73%)

scripts_v2.1: Bert + bi-lstm layer (2 layer) + fully connected layer. 
beat acc: 75.80%(67.97%, 75.80%, 69.75%, 76.51%, 76.87%, 75.44%, 75.80%, 75.09%, 75.80%, 74.38%)


Ingore below.

============================

'''
pip install -r requirements.txt
'''

This project consisted of the following three part.
- define the model
- load the dataset
- train and test

The file architecture as shown below.
```
# Save the trained model
├── checkpoints/ 
# data related operations, such as data processing.
├── inputs/ 
│   ├── pretrained_models
│   └── datasets
# Definition deep learning models
├── models/ 
│   ├── __init__.py
│   ├── textCNN.py
│   └── Bi-lstm.py
# Training and evaluation functions
├── engine.py
# Tool functions, suche visulize
└── utils/
│   ├── __init__.py
│   └── visualize.py
# Code execution
├── train.py
# Configurable parameters and provide default values
├── config.py
# The third-party libraries that the program depends on
├── requirements.txt
# Instructions for this project.
├── README.md
```

