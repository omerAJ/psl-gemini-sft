import os
import json

video_dir = r"E:\estudy\PSL finetuning\Words"
gcs_prefix = "gs://psl-video-captions/Words/"
jsonl_path = "video_finetune.jsonl"

# First, collect all possible words (labels)
all_words = [
    os.path.splitext(f)[0]
    for f in os.listdir(video_dir)
    if f.endswith(".mp4")
]
all_words_str = ", ".join(f'"{w}"' for w in sorted(all_words))  # alphabetically, for consistency

# New, clearer, more constrained PSL prompt
PROMPT_TEMPLATE = (
    "You are an expert in *Pakistani Sign Language (PSL)* recognition. "
    "The person in the video is performing the PSL sign for ONE of the following words:\n"
    f"[{all_words_str}]\n"
    "The person repeats the word two times. "
    "Identify exactly which PSL word is being signed.\n"
    "Output format:\n"
    "[{{\n  \"word\": \"<WORD>\"\n}}]\n"
    "Respond only with the exact JSON, with no explanation or extra text."
)

with open(jsonl_path, "w", encoding="utf-8") as out_file:
    for filename in os.listdir(video_dir):
        if filename.endswith(".mp4"):
            word = os.path.splitext(filename)[0]
            sample = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "fileData": {
                                    "fileUri": f"{gcs_prefix}{filename}",
                                    "mimeType": "video/mp4"
                                }
                            },
                            {
                                "text": PROMPT_TEMPLATE
                            }
                        ]
                    },
                    {
                        "role": "model",
                        "parts": [
                            {
                                "text": f"```json\n[{{\"word\": \"{word}\"}}]\n```"
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "mediaResolution": "MEDIA_RESOLUTION_LOW"
                }
            }
            out_file.write(json.dumps(sample) + "\n")
