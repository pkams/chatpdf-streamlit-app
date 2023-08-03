from transformers import pipeline

checkpoint = "MBZUAI/LaMini-Flan-T5-77M"

model = pipeline('text2text-generation', model = checkpoint)