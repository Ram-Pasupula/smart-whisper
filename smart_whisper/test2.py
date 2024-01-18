from datasets import load_dataset
from transformers import pipeline

dataset = load_dataset("anton-l/superb_demo", "si", split="test")

classifier = pipeline("audio-classification", model="superb/wav2vec2-base-superb-sid")
labels = classifier(dataset[0]["file"], top_k=5)
print(labels)