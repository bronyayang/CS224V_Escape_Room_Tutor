# CS224V_Escape_Room_Tutor
## Dataset
Download our evaluation dataset from: https://drive.google.com/drive/folders/1hElzji5hQdY1y7CeGuInxdbZSjXdMee4?usp=sharing

## Use GPT4o as the Multimodal Scene Understanding Agent and Get a Hint!
Run `python game_gpt4.py --image_folder <screenshots from level 1-x> --user_query "Can you give me a hint?" --level "1-x"`
This command directly provide a hint to the player based on the progress captured in screenshots folder. Note that you might want to change the API calling of GPT4o.

## Use LLama 3.2 as the Multimodal Scene Understanding Agent
Run `python game_llama3-2.py --model_name`  
This command is used to generate descriptions of game screenshots for each timestep of each level using Llama-3.2 and store them for subsequent processing. Specify the model name in `model_name`, such as `Llama-3.2-90B-Vision-Instruct-Turbo`.

## Running Evaluation in a Batch

Run `python run.py --base_folder "./CS224V (path to the dataset provided)" --script_path "./game_gpt4.py"` 
This command runs 25 levels for evaluation.
