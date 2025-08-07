from google.cloud import aiplatform

aiplatform.init(project="finetuning-gemini-on-psl", location="us-central1")
datasets = aiplatform.MultimodalDataset.list()
print(datasets)
