# uv_agent
A small blender extension designed to automate the process of manually placing the seams when UV unwrapping a model. 

Streamlined workflow goes like this:
1. Click "Unwrap My Mesh" in Blender.
2. Go to uv_agent GitHub Repo Issue / Copilot Chat.
3. Drag and drop export_data.json and the .png files into the chat.
4. Copy the exact text from the generated prompt.txt and send the message.
5. Take the array it spits out ([21, 5, 8...]), paste it into a file named import_seams.json in that folder and save it.
6. Click "Apply AI Seams" back in Blender
