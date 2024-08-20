'''
To run, first run these two commands in the terminal:

1. pip install cython pillow>=7.0.0 numpy>=1.18.1 opencv-python>=4.1.2 torch>=1.9.0 --extra-index-url https://download.pytorch.org/whl/cpu torchvision>=0.10.0 --extra-index-url https://download.pytorch.org/whl/cpu pytest==7.1.3 tqdm==4.64.1 scipy>=1.7.3 matplotlib>=3.4.3 mock==4.0.3
2. pip install imageai --upgrade

'''

# Importing necessary modules
from imageai.Classification import ImageClassification
import os

# Receiving user file input

file = input("Please type file name and extension: ")

# Accessing current working directory
execution_path = os.getcwd()
 
# Selecting and loading prediction model
prediction = ImageClassification()
prediction.setModelTypeAsMobileNetV2()
prediction.setModelPath(os.path.join(execution_path, 'mobilenet_v2-b0353104.pth'))
prediction.loadModel()
 
# Running model and obtaining prediction
predictions, probabilities = prediction.classifyImage(os.path.join(execution_path, file), result_count=5)
for eachPred, eachProb in zip(predictions, probabilities):
    print(f'{eachPred} : {eachProb}')