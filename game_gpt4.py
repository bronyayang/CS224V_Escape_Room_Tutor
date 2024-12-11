from openai import AzureOpenAI
import os
import re
import base64
import argparse
from tqdm import tqdm
import json

client = AzureOpenAI(
        api_key=os.environ.get('OAI'),
        api_version="2024-07-01-preview",
        azure_endpoint="https://sfc-ml-sweden.openai.azure.com/",
    )

def encode_image(image):
    if isinstance(image, str):
        with open(image, 'rb') as image_file:  
            byte_data = image_file.read() 
    else:
        output_buffer = BytesIO()
        image.save(output_buffer, format="PNG")
        byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data).decode("utf-8")
    return base64_str

SYS_PROMPT_1 = """Your task is to provide concise, accurate descriptions of the game screenshot presented. Only describe objects, features, or elements in the scene that are clearly visible and confidently identifiable. Avoid guessing or hallucinating details. At the end of your response, list any tools or items shown in the right bar. If no tools are present, write "None".
""".strip()

example_image_1 = encode_image("/code/users/shiyang/Conversational_Agent/1-3_screenshots/Oct 25, 2024-2.png")
example_image_2 = encode_image("/code/users/shiyang/Conversational_Agent/1-3_screenshots/Oct 25, 2024-13.png")

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

SYS_PROMPT_3 = """Your task is to give a game hint to the user based on the player progress and escape game walkthrough. You need to first locate the step to tell the user based on the player's progress. Next, you should only tell at most one step after the player's current progress. Do not directly tell password or code from the walkthrough. Instead, focuses on the action that user should take. You should also try to engage with user's question.
The input format is:
**Player Progress:**
<player progress here>

**Tools:** <player accuired tools>
**Walkthrough:**
<solution to escape the room>
**Question:**
<user's question>

The output format is:
**Hint:**
""".strip()

# USER_PROMPT_2_1 = """
# Merge current view into provided past progress.
# Past progress:

# Player's current view:
# The room is stylishly decorated with blue walls featuring yellow patterns and a wooden door at the center. The furniture includes a blue sofa, a white desk with shelving units, some of which are pink. There are potted plants with orange fruits near the door and a decorative lamp hanging from the ceiling. The floor has a rug with yellow crescent and sun designs. There are colorful paintings on the walls, including one above the sofa and three smaller ones to the right of the door.

# **Tools:** None
# """.strip()

# PROGRESS_1 = """The player is in a well-decorated room with blue wallpaper featuring yellow patterns, a wooden door in the center, a white desk with shelving on the right, and a blue sofa on the left. The room also includes potted plants with orange fruits near the door, a modern ceiling light, and a decorative rug with yellow crescent and circular designs.

# **Tools:** None
# """.strip()

# USER_PROMPT_2_2 = """
# Merge current view into provided past progress.
# Past progress:
# The player is in a well-decorated room with blue wallpaper featuring yellow patterns, a wooden door in the center, a white desk with shelving on the right, and a blue sofa on the left. The room also includes potted plants with orange fruits near the door, a modern ceiling light, and a decorative rug with yellow crescent and circular designs.

# **Tools:** None

# Player's current view:
# The scene is a slightly zoomed-out view similar to the previous one, showing a light pink cabinet or shelf with a rectangular metallic panel featuring a grid of oval-shaped holes. To the left, part of a wall with blue wallpaper patterned with stars is visible along with a decorative painting. Part of the room's floor and rug is also visible at the bottom of the image.

# **Tools:** None
# """.strip()

# PROGRESS_2 = """The player is in a well-decorated room with blue wallpaper featuring yellow patterns, a wooden door in the center, a white desk with shelving on the right, and a blue sofa on the left. The player investigated the light pink cabinet or shelf, seeing a rectangular metallic panel featuring a grid of oval-shaped holes.

# **Tools:** None
# """.strip()

# USER_PROMPT_2_3 = """
# Merge current view into provided past progress.
# Past progress:
# The player is in a well-decorated room with blue wallpaper featuring yellow patterns, a wooden door in the center, a white desk with shelving on the right, and a blue sofa on the left. The player investigated the light pink cabinet or shelf, seeing a rectangular metallic panel featuring a grid of oval-shaped holes.

# **Tools:** None

# Player's current view:
# The scene depicts a close-up of a blue sofa with yellow and blue pillows. A metallic key is visible on the sofa. The background consists of blue wallpaper with a star pattern.

# **Tools:** Key
# """.strip()

# PROGRESS_3 = """The player is in a well-decorated room with blue wallpaper featuring yellow patterns, a wooden door in the center, a white desk with shelving on the right, and a blue sofa on the left. The player investigated the light pink cabinet or shelf, seeing a rectangular metallic panel featuring a grid of oval-shaped holes. Then, the player obtained the key on the blue sofa.

# **Tools:** Key
# """.strip()

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

def load_walkthrough(level):
    with open('./walkthrough.json', 'r') as file:
        walkthrough = json.load(file)
    walkthrough_level = walkthrough[level]
    return walkthrough_level


def tutor_prepare(t, image_path, level, summary, internal_state_record):

    ## Screenshot understanding
    image = encode_image(image_path)

    if str(t) not in internal_state_record.keys():
        internal_state_record[str(t)] = {}
        description = client.chat.completions.create(
            model="sfc-cortex-analyst-dev",
            messages=[
                {"role": "system", "content": SYS_PROMPT_1},
                {"role": "user", "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{example_image_1}",
                            }
                        },
                    ],
                },
                {"role": "assistant", "content": ASSISTANT_1},
                {"role": "user", "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{example_image_2}",
                            }
                        },
                    ],
                },
                {"role": "assistant", "content": ASSISTANT_2},
                {"role": "user", "content": [
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
        internal_state_record[str(t)]["description"] = description
    else:
        description = internal_state_record[str(t)]["description"]

    if "summary" not in internal_state_record[str(t)].keys():
        user_prompt = f"Merge current view into provided past progress.\nPast progress:\n{summary}\n\nPlayer's current view:\n{description}"
        summary_ans = client.chat.completions.create(
            model="sfc-cortex-analyst-dev",
            messages=[
                {"role": "system", "content": SYS_PROMPT_2},
                # {"role": "user", "content": USER_PROMPT_2_1},
                # {"role": "assistant", "content": PROGRESS_1},
                # {"role": "user", "content": USER_PROMPT_2_2},
                # {"role": "assistant", "content": PROGRESS_2},
                # {"role": "user", "content": USER_PROMPT_2_3},
                # {"role": "assistant", "content": PROGRESS_3},
                {"role": "assistant", "content": user_prompt},
            ]
        )
        new_summary = summary_ans.choices[0].message.content
        internal_state_record[str(t)]["summary"] = new_summary
    else:
        new_summary = None
    # prompt_summary = f"Update the tool lists by combining them. If both are None, write None. If one is None, ignore it.\nTool list 1: {description}\nTool list 2: {summary}."
    # summary = client.chat.completions.create(
    # model="sfc-ml-sweden-gpt4-managed",
    # messages=[
    #     {"role": "system", "content":},
    #     {"role": "user", "content": f'''{prompt_summary}'''}
    # ])

    # summary = summary.choices[0].message.content
    # internal_state_record[t]["summary"] = summary

    return new_summary

def tutor_answer(user_query, summary, walkthrough):
    ## User-Facing GPT-4o QA Agent
    user_prompt = f"{summary}\n**Walkthrough:**\n{walkthrough}\n**Question:**{user_query}"
    hint = client.chat.completions.create(
            model="sfc-cortex-analyst-dev",
            messages=[
                {"role": "system", "content": SYS_PROMPT_3},
                {"role": "assistant", "content": user_prompt},
            ]
        )
    hint = hint.choices[0].message.content

    return hint


       

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, required=True, help="Path to the folder containing images")
    parser.add_argument("--user_query", type=str, required=True, help="User's query for the hint")
    parser.add_argument("--level", type=str, required=True, help="Game level identifier")

    args = parser.parse_args()

    # print("Setting up game tutor...")
    # When user initiate a query, we assume that we receive a folder, containing all history screenshots
    image_list = load_images(args.image_folder)

    # print("Reading the game state...")
    summary = ""
    if os.path.exists(f"/code/users/shiyang/Conversational_Agent/internal_state_record_{args.level}.json"):
        # Load the JSON file
        with open(f"/code/users/shiyang/Conversational_Agent/internal_state_record_{args.level}.json", 'r') as file:
            internal_state_record = json.load(file)
    else:
        # Initialize as an empty dictionary
        internal_state_record = {}

    for t, image_path in enumerate(tqdm(image_list, desc="Processing Images")):
        summary = tutor_prepare(t, image_path, args.level, summary, internal_state_record)

    with open(f"internal_state_record_{args.level}.json", "w") as json_file:
        json.dump(internal_state_record, json_file, indent=4)

    # print("Answering the hint...")
    walkthrough = load_walkthrough(args.level)
    new_summary = internal_state_record[str(t)]["summary"]
    print(tutor_answer(args.user_query, new_summary, walkthrough))