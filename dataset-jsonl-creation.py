import os
import json

video_dir = r"E:\estudy\PSL finetuning\Words"
gcs_prefix = "gs://psl-video-captions/Words/"
jsonl_path = "video_finetune.jsonl"

# Build word list
all_words = [
    os.path.splitext(f)[0]
    for f in os.listdir(video_dir)
    if f.endswith(".mp4")
]
all_words_str = ", ".join(f'"{w}"' for w in sorted(all_words))

# System instructions as a separate field
SYSTEM_INSTRUCTIONS = (
    "You are an expert in Pakistani Sign Language (PSL) recognition. "
    "The person in the video is performing the PSL sign for ONE of the following words:\n"
    f"[{all_words_str}]\n"
    "The person repeats the word two times. "
    "Your task is to identify exactly which PSL word is being signed. "
    "Respond with only the word, nothing else."
)

with open(jsonl_path, "w", encoding="utf-8") as out_file:
    for filename in os.listdir(video_dir):
        if filename.endswith(".mp4"):
            word = os.path.splitext(filename)[0]
            sample = {
                "systemInstruction": {
                    "role": "system",
                    "parts": [
                        {"text": SYSTEM_INSTRUCTIONS}
                    ]
                },
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
                                "text": f"Which PSL word is being signed in this video?"
                            }
                        ]
                    },
                    {
                        "role": "model",
                        "parts": [
                            {"text": word}
                        ]
                    }
                ],
                "generationConfig": {
                    "mediaResolution": "MEDIA_RESOLUTION_LOW"
                }
            }
            out_file.write(json.dumps(sample) + "\n")
