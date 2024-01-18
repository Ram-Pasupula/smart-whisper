from transformers import pipeline

classifier = pipeline(task="text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest", top_k=None)

sentences = ["""I am not having a great day"""]

model_outputs = classifier(sentences)
print(model_outputs[0][0]['label'])
# produces a list of dicts for each of the labels
