# CS224V_Escape_Room_Tutor
## Datset
Download our evaluation dataset from: https://drive.google.com/drive/folders/1hElzji5hQdY1y7CeGuInxdbZSjXdMee4?usp=sharing

## Use LLama 3.2 as the Multimodal Scene Understanding Agent
Run `python game_llama3-2.py --model_name`  
This command is used to generate descriptions of game screenshots for each timestep of each level using Llama-3.2 and store them for subsequent processing. Specify the model name in `model_name`, such as `Llama-3.2-90B-Vision-Instruct-Turbo`.
