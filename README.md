
# Intent Classification Repository

Welcome to the Intent Classification GitHub repository! This repository is designed to help you build and test models for intent classification within various user queries, and also generate tags for AI chat responses. The project contains scripts for model training, testing, data collection, and tag generation. Below, you will find a brief overview of each file and instructions on how to use them.

## Overview

The repository contains the following key scripts:

1. **Binary_Model_Train.py** - This script is responsible for training a binary classification model. The binary model distinguishes between two categories of intents based on the input data. The BERT model has been fine-tuned with the dataset.

2. **User_Query_Model_Train.py** - This script trains a model specifically designed to understand and classify user queries. It's more nuanced compared to the binary model and can classify a broader range of intents. The BERT model has been fine-tuned with the dataset.

3. **Intent_Classification_Pipeline_Testing.py** - This script integrates the two models mentioned above. It provides a testing pipeline that evaluates the performance of the intent classification system as a whole.

4. **Tag_Generation_Data_Collection.py** - This script is used to collect data necessary for generating tags for AI chat responses. It should be run to store the required data before using the tag generation script. It uses OpenAI embeddings to find teh embeddings for each catalogue.

5. **AI_Response_Tag_Generation.py** - This script generates tags for AI chat responses based on the data collected by `Tag_Generation_Data_Collection.py`. It should be imported at the time of chatting with the AI to dynamically generate relevant tags. OpenAI LLM is used to identify the best fit catalogue catagory for the given response. 

6. **Usage_Intent_Classification.py** - This script is an example on how to make use of the intent classification module in a chat.

7. **Usage_Tag_generation.py** - This script is an example on how to implement the Tag generation in a chat.

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

## Tag generation : Training and Inference

- Collecting tag generation data:

```bash
python Tag_Generation_Data_Collection.py
```
- Inference Tag generation

``` python
# Usage_Intent_Classification.py
from AI_Response_Tag_Generation import Tag_generator
import openai

# Modify it to your openai API key
openai.api_key = ""

Tag_generator = Tag_generator(openai)

# AI suggested solution
AI_suggested_solution = "Ah, wrinkles on the fingers can be due to various reasons such as aging, dehydration, or prolonged exposure to water. To help with this, it's important to keep your skin moisturized. Using a hydrating hand cream regularly can improve the skin's elasticity. Also, ensure you're staying well-hydrated by drinking plenty of water throughout the day.\nHave you noticed any changes in your skin texture or is it just the wrinkles that are concerning you?"
c0, c1, c2, c3, t0, t1, t2, t3, product_features = Tag_generator.main(AI_suggested_solution)

tags_for_Product = c0 + " -> " + c1 + " -> " + c2 + " -> " + c3
print("Tags generated: ", tags_for_Product)
print("Product Features: ", product_features)
```

## Contributing

We welcome contributions to this project! If you have suggestions for improvements or bug fixes, feel free to open an issue or submit a pull request.
