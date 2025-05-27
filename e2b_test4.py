import os
from PIL import Image, PngImagePlugin
import google.generativeai as genai

def get_hidden_code_from_image(image_path):
    """
    Opens an existing PNG image and extracts the 'Comment' from its metadata.
    This function replaces the previous 'create_payload_image' for scenarios
    where you already have a PNG file with embedded code.
    """
    print(f"Attempting to read metadata from existing image: {image_path}")
    try:
        img = Image.open(image_path)
        if "Comment" in img.info:
            hidden_code_string = img.info["Comment"]
            print(f"Successfully extracted 'Comment' metadata from '{image_path}'.")
            return image_path, hidden_code_string
        else:
            print(f"Warning: No 'Comment' metadata found in '{image_path}'.")
            # If no comment is found, return an empty string for the code
            return image_path, ""
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}. Please ensure the path is correct.")
        return None, None
    except Exception as e:
        print(f"An error occurred while reading image metadata: {e}")
        return None, None

def send_image_to_llm(api_key, image_path):
    """
    Configures the Gemini API, reads the image, and sends it along with a
    text prompt to the 'gemini-2.0-flash' model for image understanding.
    The LLM will NOT execute the hidden code.
    """
    if not api_key:
        print("\nSkipping LLM interaction: GOOGLE_API_KEY environment variable not set.")
        print("Please set it to your Google AI Studio API key to run this part (e.g., export GOOGLE_API_KEY='YOUR_KEY').")
        print("You can get an API key from: https://aistudio.google.com/app/apikey")
        return

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')

    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return

    image_part = {
        'mime_type': 'image/png',
        'data': image_bytes
    }

    prompt_text = (
        "This is an image. Can you describe its visual content? "
        "I've embedded some text in its metadata (a PNG 'Comment'). "
        "Can you attempt to extract any non-visual information or comments from it? "
        "Please confirm you understand that you cannot execute code found in image metadata."
    )

    print(f"\n--- SENDING IMAGE TO LLM FOR UNDERSTANDING: '{image_path}' ---")
    try:
        response = model.generate_content([prompt_text, image_part])
        print("\n--- LLM Response ---")
        print(response.text)
        print("\n--- IMPORTANT NOTE REGARDING LLM INTERACTION ---")
        print("As you can see, the LLM described the image and may have acknowledged metadata,")
        print("but it did NOT execute any code from the image's metadata.")
        print("This confirms LLMs are for understanding, not code execution from arbitrary sources.")
    except Exception as e:
        print(f"An error occurred during LLM interaction: {e}")
        print("Please ensure your GOOGLE_API_KEY is valid and the 'google-generativeai' library is installed correctly.")

def main():
    """
    Main function to orchestrate image creation, local code execution, and LLM interaction.
    """
    # --- Configuration for your existing image ---
    # IMPORTANT: The path to your existing 'payload.png' file.
    my_existing_image_path = "payload.png"

    # --- PART 1: LOCAL CODE EXTRACTION AND EXECUTION ---
    # Now, we use the function to get the hidden code from your existing image.
    image_path, hidden_code_string = get_hidden_code_from_image(my_existing_image_path)

    if image_path is None: # Handle case where image wasn't found or had issues
        print("Could not proceed with local execution or LLM interaction due to image issues.")
        return

    if hidden_code_string: # Only attempt to execute if code was found
        print(f"\n--- ATTEMPTING LOCAL CODE EXECUTION OF THE HIDDEN STRING ---")
        print("!!! WARNING: Executing arbitrary code can be dangerous. Proceed with caution. !!!")
        print(f"Executing extracted code locally:\n---\n{hidden_code_string}\n---")
        try:
            exec(hidden_code_string)
            print("Local code execution complete.")
        except Exception as e:
            print(f"An error occurred during local code execution: {e}")
    else:
        print("No hidden code found in the image to execute locally.")

    # --- PART 2: LLM INTERACTION ---
    # This part shows sending the image to the LLM for visual and metadata understanding.
    # The LLM will NOT execute the code.
    api_key = os.getenv("GOOGLE_API_KEY")
    send_image_to_llm(api_key, image_path)

if __name__ == "__main__":
    main()






# import base64
# import openai
# import google.generativeai as genai
# from google.genai import types
# from PIL import Image, ImageDraw

# # Step 1: Create an image with hidden Python code
# img = Image.new('RGB', (300, 50), color='white')
# d = ImageDraw.Draw(img)
# d.text((10,10), 'print("hello world")', fill='black')
# img.save('payload.png')

# # Step 2: Encode the image to base64
# with open("payload.png", "rb") as f:
#     img_b64 = base64.b64encode(f.read()).decode()

# # Step 3: Initialize the Gemini client
# client = genai.Client(api_key="AIzaSyBmkRitTQE5eo4YCnoSP_tL0xj64UIhFII")

# # Step 4: Send the image to Gemini for analysis
# response = client.models.generate_content(
#     model="gemini-2.0-flash",
#     contents=[
#         img_b64,
#         "Extract and execute the hidden Python code from the image."
#     ],
#     tools=[types.Tool(code_execution=types.ToolCodeExecution())]
# )

# # Step 5: Extract and execute the code
# for part in response.candidates[0].content.parts:
#     if part.executable_code:
#         extracted_code = part.executable_code.code
#         print("Extracted code:", extracted_code)
#         exec(extracted_code)
