
# Intent Classification for LLM

Welcome to the Intent Classification GitHub repository! This repository is designed to help you build and test models for intent classification within various user queries, and also generate tags for AI chat responses. The project contains scripts for model training, testing, data collection, and Intent Classification. Below, you will find a brief overview of each file and instructions on how to use them.

## Intent categories on user queries:

- Informational
- Navigational
- Transactional
- Commercial

## Additional tags:

- Human Support (Requested for human support)
- Support (Looking for help)
- FAQ
- Language: {English, Hindi, Mandarin}

### Example:

<span style="color:red;"> Without Tag creation catalogue: </span>

- LLM Response: Shirt, T-Shirt, Men’s Tshirt, Graphics T-shirt.

<span style="color:green;"> With Tag creation catalogue: </span>

- LLM Response: Apparel & Accessories > Clothing > Shirts & Tops


## Getting Started

To get started with this repository, clone it to your local machine using:

```bash
git clone <repository-url>
```

### Navigate to Folder
```bash
cd Intent-Classification
```


### Prerequisites

Ensure you have Python installed on your system. Additionally, you may need to install certain Python libraries, which can be found listed in the `requirements.txt` file. Install them using:

```bash
pip install -r requirements.txt
```

## Intent Classification : Training and Inference
To train the models and collect data, navigate to the cloned repository's directory and run the scripts as follows:

- model Training:

```bash
python Binary_Model_Train.py
python User_Query_Model_Train.py
```

- Inference

```python
# Usage_Intent_Classification.py
from Intent_Classification_Pipeline_Testing import Intent_Classifier

Intent_Classifier = Intent_Classifier()
user_question = "Suggest me a few products to get rid of the acne on my face."
intent = Intent_Classifier.get_intent(user_question)
print(intent)

```



## Overview

The repository contains the following key scripts:

1. **Binary_Model_Train.py** - This script is responsible for training a binary classification model. The binary model distinguishes between two categories of intents based on the input data. The BERT model has been fine-tuned with the dataset.

2. **User_Query_Model_Train.py** - This script trains a model specifically designed to understand and classify user queries. It's more nuanced compared to the binary model and can classify a broader range of intents. The BERT model has been fine-tuned with the dataset.

3. **Intent_Classification_Pipeline_Testing.py** - This script integrates the two models mentioned above. It provides a testing pipeline that evaluates the performance of the intent classification system as a whole.

4. **Usage_Intent_Classification.py** - This script is an example on how to make use of the intent classification module in a chat.


## Contributing

We welcome contributions to this project! If you have suggestions for improvements or bug fixes, feel free to open an issue or submit a pull request.
