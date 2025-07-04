import argparse
import json
import re
import base64
import requests
import ollama
from PIL import Image

def estimate_food_calories(image_path: str):
    """
    Estimates calories in a food image using local Gemma 3 via Ollama.
    """

    # detailed JSON-enforcing prompt
    prompt = """
    You are a calorie estimation expert. Please analyze this food image and output only valid JSON with the following structure:

    {
      "items": [
        {
          "name": "food name",
          "calories": number
        },
        ...
      ],
      "total_calories": total estimated calories as a number
    }

    For each detected food item, provide an estimated calorie count. Then, sum all the items in "total_calories".

    Reply ONLY with the JSON. Do not include any explanations or other text.
    """
    
    response = ollama.chat(
        model="gemma3n:e4b",
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [image_path],
            }
        ]
    )

    raw_response = response['message']['content']
    print("\nRaw LLM response:\n", raw_response)

    # clean and parse
    import re
    cleaned = re.sub(r"```json|```", "", raw_response).strip()
    try:
        parsed = json.loads(cleaned)
        print("\nParsed JSON:\n", parsed)
    except json.JSONDecodeError as e:
        print(f"\nCould not parse JSON reliably: {e}")
        print("\nCleaned text was:\n", cleaned)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Estimate food calories from an image using Gemma 3."
    )
    parser.add_argument("image_path", type=str, help="Path to the food image file")
    args = parser.parse_args()

    image_path = args.image_path
    estimate_food_calories(image_path)
