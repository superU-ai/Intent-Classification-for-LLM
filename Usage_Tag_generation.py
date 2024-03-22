from AI_Response_Tag_Generation import Tag_generator
import openai

openai.api_type = ""
openai.api_base = ""
openai.api_version = ""
openai.api_key = ""

Tag_generator = Tag_generator(openai)
AI_Response = "Ah, wrinkles on the fingers can be due to various reasons such as aging, dehydration, or prolonged exposure to water. To help with this, it's important to keep your skin moisturized. Using a hydrating hand cream regularly can improve the skin's elasticity. Also, ensure you're staying well-hydrated by drinking plenty of water throughout the day.\nHave you noticed any changes in your skin texture or is it just the wrinkles that are concerning you?"
c0, c1, c2, c3, t0, t1, t2, t3, product_features = Tag_generator.main(AI_Response)

tags_for_ai_response = c0 + " -> " + c1 + " -> " + c2 + " -> " + c3
print("Tags Generates: ", tags_for_ai_response)
print("Product Features: ", product_features)