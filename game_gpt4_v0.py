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
    internal_state_record[t] = {}

    ## Screenshot understanding
    prompt = f"This is the game Room Escape 50 I at {level}. Only answer the tool that the user acquired in this image. If no, write None."
    image = encode_image(image_path)
    description = client.chat.completions.create(
        model="sfc-ml-sweden-gpt4-managed",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image}",
                        }
                    },
                ],
            },
            {"role": "user", "content": f'''{prompt}'''}
        ]
    )
    
    description = description.choices[0].message.content
    internal_state_record[t]["description"] = description

    prompt_summary = f"Update the tool lists by combining them. If both are None, write None. If one is None, ignore it.\nTool list 1: {description}\nTool list 2: {summary}."
    summary = client.chat.completions.create(
    model="sfc-ml-sweden-gpt4-managed",
    messages=[
        {"role": "user", "content": f'''{prompt_summary}'''}
    ])

    summary = summary.choices[0].message.content
    internal_state_record[t]["summary"] = summary

    return summary

def tutor_answer(user_query, summary, walkthrough):
    ## User-Facing GPT-4o QA Agent
    prompt = f"Answer the question from user after **Question:**. Follow these rules:\n1. Cross-check items in **Tools** to avoid redundant suggestions. Do **NOT** instruct the player to acquire tools already listed after **Tools**.\n2. Use the **Walkthrough** to determine logical next steps, focusing on actions with items in **Tools** over acquisition suggestions.\nTools:{summary}\nWalkthrough: {walkthrough}\nQuestion: {user_query}"
    response = client.chat.completions.create(
    model="sfc-ml-sweden-gpt4-managed",
    messages=[
        {"role": "user", "content": f'''{prompt}'''}
    ])
    hint = response.choices[0].message.content

    return hint


       

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, required=True, help="Path to the folder containing images")
    parser.add_argument("--user_query", type=str, required=True, help="User's query for the hint")
    parser.add_argument("--level", type=str, required=True, help="Game level identifier")

    args = parser.parse_args()

    print("Setting up game tutor...")
    # When user initiate a query, we assume that we receive a folder, containing all history screenshots
    image_list = load_images(args.image_folder)

    print("Reading the game state...")
    summary = ""
    internal_state_record = {}
    for t, image_path in enumerate(tqdm(image_list, desc="Processing Images")):
        summary = tutor_prepare(t, image_path, args.level, summary, internal_state_record)

    with open("internal_state_record.json", "w") as json_file:
        json.dump(internal_state_record, json_file, indent=4)

    print("Answering the hint...")
    walkthrough = load_walkthrough(args.level)
    print(tutor_answer(args.user_query, summary, walkthrough))