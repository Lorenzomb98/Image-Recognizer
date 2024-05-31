# Importing necessary modules
from imageai.Classification import ImageClassification
import os

# Accessing current working directory
execution_path = os.getcwd()
 
# Selecting and loading prediction model
prediction = ImageClassification()
prediction.setModelTypeAsMobileNetV2()
prediction.setModelPath(os.path.join(execution_path, 'mobilenet_v2-b0353104.pth'))
prediction.loadModel()
 
# Running model and obtaining prediction
predictions, probabilities = prediction.classifyImage(os.path.join(execution_path,'<image>.jpg'), result_count=5)
for eachPred, eachProb in zip(predictions, probabilities):
    print(f'{eachPred} : {eachProb}')