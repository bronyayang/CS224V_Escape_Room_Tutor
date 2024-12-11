import os
import re
import base64
import argparse
from tqdm import tqdm
import json
from openai import OpenAI

client = OpenAI(
  api_key="2b1b281a79b652b8036eb4b0896cf8ce936caa934e8ca350b5135de602288c78",
  base_url="https://api.together.xyz/v1",
)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


SYS_PROMPT_1 = """Your task is to provide concise, accurate descriptions of the game screenshot presented. Only describe objects, features, or elements in the scene that are clearly visible and confidently identifiable. Avoid guessing or hallucinating details. At the end of your response, list any tools or items shown in the right bar. If no tools are present, write "None". Split the answer into two lines: the first line is the description, and the second line is the tool. The first line should start with "Description:", and the second line should start with "Tool:".
""".strip()

example_image_1 = encode_image("CS224V/1-3_screenshots/Oct 25, 2024-2.png")
example_image_2 = encode_image("CS224V/1-3_screenshots/Oct 25, 2024-13.png")

ASSISTANT_1 = """The scene depicts a well-decorated room with blue wallpaper featuring yellow patterns, a wooden door in the center, a white desk with shelving on the right, and a blue sofa on the left. The room also includes potted plants with orange fruits near the door, a modern ceiling light, and a decorative rug with yellow crescent and circular designs.

**Tools:** None
""".strip()

ASSISTANT_2 = """The scene shows a close-up of a light pink cabinet or shelf with a rectangular metallic panel featuring a grid of oval-shaped holes. To the left, part of a wall is visible with blue wallpaper featuring stars and a painting.

**Tools:** Key
""".strip()

SYS_PROMPT_2 = """Your task is to update the player's progress by truthfully and concisely merging their current view description into their past progress.
Your output format should be:
<directly write merged player progress here>

**Tools:** <directly write merged tools here>
""".strip()

SYS_PROMPT_3 = """Your task is to give a game hint to the user based on the player progress and escape game walkthrough. You need to first locate the step to tell the user based on the player's progress. Next, you should only tell at most one step after the player's current progress.
The input format is:
**Player Progress:**
<player progress here>

**Tools:** <player accuired tools>
**Walkthrough:**
<solution to escape the room>
""".strip()


def load_images(folder_name):
    valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}

    # Helper function to extract the number after the last hyphen for sorting
    def numerical_sort_key(filename):
        match = re.search(r'-([\d]+)\.png$', filename)  # Match the number before `.png`
        return int(match.group(1)) if match else float('inf')  # Place files without a number at the end

    image_files = sorted(
        [os.path.join(folder_name, f) for f in os.listdir(folder_name)
         if os.path.splitext(f)[1].lower() in valid_extensions],
        key=numerical_sort_key
    )
    
    return image_files



def tutor_prepare(t, image_path, level, summary, internal_state_record):

    ## Screenshot understanding
    image = encode_image(image_path)

    if str(t) not in internal_state_record.keys():
        internal_state_record[str(t)] = {}
        description = client.chat.completions.create(
            model='meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo',
            messages=[
                {"role": "system", "content": SYS_PROMPT_1},
                {"role": "user", "content": [
                        {
                        "type": "text",
                        "text": 'Split the answer into two lines: the first line is the description, and the second line is the tool. The first line should start with \"Description:\", and the second line should start with \"Tool:\".',
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image}",
                            }
                        },
                    ],
                },
            ]
        )
        
        description = description.choices[0].message.content
        #print(description)
        internal_state_record[str(t)]["description"] = description
    else:
        description = internal_state_record[str(t)]["description"]

    if "summary" not in internal_state_record[str(t)].keys():
        user_prompt = f"Merge current view into provided past progress.\nPast progress:\n{summary}\n\nPlayer's current view:\n{description}"
        summary_ans = client.chat.completions.create(
            model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
            messages=[
                {"role": "system", "content": SYS_PROMPT_2},
                {"role": "assistant", "content": user_prompt},
            ]
        )
        new_summary = summary_ans.choices[0].message.content
        internal_state_record[str(t)]["summary"] = new_summary
    else:
        new_summary = None

    return new_summary



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default='CS224V/1-1_screenshots', help="Path to the folder containing images")
    parser.add_argument("--user_query", type=str, default='Give me a hint.', help="User's query for the hint")
    parser.add_argument("--level", type=str, default='1-1', help="Game level identifier")

    args = parser.parse_args()

    # level = args.level
    # image_folder = args.image_folder
    model_name = 'Llama-3.2-90B-Vision-Instruct-Turbo'
    if not os.path.exists(model_name):
        os.makedirs(model_name)
    hints = {}
    for i in range(1, 26):
        level = f'1-{i}'
        print(f'Level: {level}')
        image_folder = f'CS224V/{level}_screenshots'
    

        print("Setting up game tutor...")
        # When user initiate a query, we assume that we receive a folder, containing all history screenshots
        image_list = load_images(image_folder)

        print("Reading the game state...")
        summary = ""
        if os.path.exists(f"{model_name}/internal_state_record_{level}.json"):
            # Load the JSON file
            with open(f"{model_name}/internal_state_record_{level}.json", 'r') as file:
                internal_state_record = json.load(file)
        else:
            # Initialize as an empty dictionary
            internal_state_record = {}

        for t, image_path in enumerate(tqdm(image_list, desc="Processing Images")):
            summary = tutor_prepare(t, image_path, level, summary, internal_state_record)

        with open(f"{model_name}/internal_state_record_{level}.json", "w") as json_file:
            json.dump(internal_state_record, json_file, indent=4)
