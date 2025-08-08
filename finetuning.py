
import time

import vertexai
from vertexai.tuning import sft

# TODO(developer): Update and un-comment below line
# PROJECT_ID = "your-project-id"
vertexai.init(project="finetuning-gemini-on-psl", location="us-central1")

# sft_tuning_job = sft.train(
#     source_model="gemini-2.5-flash",
#     # 1.5 and 2.0 models use the same JSONL format
#     train_dataset="gs://psl-video-captions/video_finetune.jsonl",
# )

sft_tuning_job = sft.train(
    source_model="gemini-2.5-flash",
    train_dataset="gs://psl-video-captions/video_finetune.jsonl",
    epochs=60,
    adapter_size=16,  ## used 8 by default
    # learning_rate_multiplier=2.0,
    tuned_model_display_name = "PSL finetuning r16e60",
)

# Polling for job completion
while not sft_tuning_job.has_ended:
    time.sleep(60)
    sft_tuning_job.refresh()

print(sft_tuning_job.tuned_model_name)
print(sft_tuning_job.tuned_model_endpoint_name)
print(sft_tuning_job.experiment)
# Example response:
# projects/123456789012/locations/us-central1/models/1234567890@1
# projects/123456789012/locations/us-central1/endpoints/123456789012345
# <google.cloud.aiplatform.metadata.experiment_resources.Experiment object at 0x7b5b4ae07af0>