import argparse
import json
import os
import re

import dotenv
from google import generativeai as genai

dotenv.load_dotenv()


def estimate_food_calories(image_path: str):
    """
    Estimates calories in a food image using Gemma 3 via Gemini API.
    """

    # Configure once with your API key
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    # create a generative model
    model = genai.GenerativeModel("gemma-3-27b-it")

    # upload the file
    uploaded_file = genai.upload_file(image_path)

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

    # call generate_content
    response = model.generate_content([uploaded_file, prompt])

    print("\nRaw response from Gemma 3:\n")
    print(response.text)

    # try parsing
    cleaned_text = re.sub(r"```json|```", "", response.text).strip()

    try:
        parsed = json.loads(cleaned_text)
        print("\nParsed JSON:\n")
        print(parsed)
    except json.JSONDecodeError as e:
        print(f"\nCould not parse JSON reliably: {e}")
        print("\nCleaned text was:\n", cleaned_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Estimate food calories from an image using Gemma 3."
    )
    parser.add_argument("image_path", type=str, help="Path to the food image file")
    args = parser.parse_args()

    image_path = args.image_path
    estimate_food_calories(image_path)
