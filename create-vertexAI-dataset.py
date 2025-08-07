from google.cloud import aiplatform

aiplatform.init(project="finetuning-gemini-on-psl", location="us-central1")

from google.cloud.aiplatform.preview import datasets


gcs_uri_of_jsonl_file = "gs://psl-video-captions/video_finetune.jsonl"

my_dataset = datasets.MultimodalDataset.from_gemini_request_jsonl(
  gcs_uri = gcs_uri_of_jsonl_file,
)

print("\n\nupload complete of data with len: ", len(my_dataset.list_data_items()))

