# OptimAL-DL
deep learning models for biomedical relation classification.

Please download the pytorch_model.bin and put it in inputs/biobert-v1.1 file.
https://huggingface.co/dmis-lab/biobert-v1.1/tree/main

run "python train.py" in src_transformer file.


Ingore below.


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

