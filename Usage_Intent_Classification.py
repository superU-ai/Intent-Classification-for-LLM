from Intent_Classification_Pipeline_Testing import Intent_Classifier

Intent_Classifier = Intent_Classifier()
user_question = "Suggest me a few products to get rid of the acne on my face."
intent = Intent_Classifier.get_intent(user_question)
print(intent)
