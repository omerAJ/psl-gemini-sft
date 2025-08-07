from google import genai
from google.genai import types
import os

video_dir = r"E:\estudy\PSL finetuning\Words"
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

def generate_for_all_videos():
    client = genai.Client(
        vertexai=True,
        project="737222604798",
        location="us-central1",
    )

    model = "projects/737222604798/locations/us-central1/endpoints/7833311162904084480"
    video_dir = r"E:\estudy\PSL finetuning\Words"
    gcs_prefix = "gs://psl-video-captions/Words/"   # Change to your GCS bucket path

    for filename in os.listdir(video_dir):
        if filename.endswith(".mp4"):
            word = os.path.splitext(filename)[0]
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            file_data=types.FileData(
                                file_uri=f"{gcs_prefix}{filename}",
                                mime_type="video/mp4"
                            )
                        ),
                        types.Part(
                            text=PROMPT_TEMPLATE
                        )
                    ]
                )
            ]

            generate_content_config = types.GenerateContentConfig(
                temperature=1,
                top_p=1,
                seed=0,
                max_output_tokens=65535,
                safety_settings=[
                    types.SafetySetting(
                        category="HARM_CATEGORY_HATE_SPEECH",
                        threshold="OFF"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="OFF"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        threshold="OFF"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT",
                        threshold="OFF"
                    )
                ],
                thinking_config=types.ThinkingConfig(
                    thinking_budget=-1,
                ),
                media_resolution="MEDIA_RESOLUTION_LOW"
            )

            output = ""
            for chunk in client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=generate_content_config,
            ):
                output += chunk.text

            print(f"{word} -> {output.strip()}")

if __name__ == "__main__":
    generate_for_all_videos()