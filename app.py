
from flask import Flask, render_template, request, jsonify, session
import os
import time
import tempfile
from PIL import Image
from io import BytesIO
from google import genai
from dotenv import load_dotenv
from google.generativeai import configure
from crewai import Agent, Task, Crew, LLM
from flask_cors import CORS
import base64
import sys
import json
import uuid

sys.path.append(os.path.abspath(os.path.dirname(__file__)))




from app.ai_core.task1_food_analyzer.detector import detect_food_item
from app.ai_core.task1_food_analyzer.packaged_handler import analyze_packaged_food
from app.ai_core.task1_food_analyzer.unpackaged_handler import analyze_unpackaged_food
from app.ai_core.task2_food_finder.suggestion_engine import get_healthy_places_nearby
from app.ai_core.task3_myth_buster.gemini_response import get_chat_response


load_dotenv(dotenv_path=r".env")
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
configure(api_key=os.environ["GEMINI_API_KEY"])
app = Flask(__name__)
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'img')
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecret") 


def get_frame_description(frame):
    try:
        img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'captured.png')
        frame.save(img_filename)

        print("We are trying to get description.")

      
        contents = [
            {"text": "Can you describe the contents of the following image? make the disciption detailed and personlised"},
            {
                "inline_data": {
                    "mime_type": "image/png",
                    "data": open(img_filename, "rb").read()
                }
            }
        ]
        

        response = client.models.generate_content(
            model="gemini-2.0-flash-exp-image-generation",
            contents=contents,
            config=genai.types.GenerateContentConfig(
                response_modalities=["Text", "Image"]
            )
        )

        print(f"API response: {response}")

        description = "No description generated."
        
        
        for part in response.candidates[0].content.parts:
            if part.text:
                description = part.text
            elif part.inline_data:
                image_data = part.inline_data.data
                if image_data:
                    try:
                        image = Image.open(BytesIO(image_data))
                        image.show()  
                        description = "An image has been generated and displayed."
                    except Exception as e:
                        print(f"Error processing generated image: {e}")
                        description = "Error displaying the generated image."
                else:
                    print("Received image data is None.")
                    description = "No image generated."

        return description
    except Exception as e:
        print(f"Error in image description generation: {e}")
        return "Unable to generate description."


@app.route('/')
def home():
    return render_template('index.html')

def handle_detection(detection):
    if not detection.get("edible", False):
        print("❌ This item is not edible.")

        return "❌ This item is not edible."

    if detection["edible"] and detection.get("packaged", False):
        print("📦 Packaged Food Detected!")
        details = detection.get("details", {})
        company = details.get("company", "Unknown")
        product_name = details.get("product_name", "Unknown")
        specifics = details.get("specifics", {})
        

        
        verdict = analyze_packaged_food(product_name,company,detection)
        return verdict

    if detection["edible"] and not detection.get("packaged", False):
        print("🥗 Unpackaged Food Detected!")
        verdict = analyze_unpackaged_food(detection)

        return verdict
    
    
    
    
    
@app.route('/capture', methods=['POST'])
def capture():
    data = request.get_json()
    user_description = data.get("user_description", "")
    if not data or 'image_data' not in data:
        return jsonify({'error': 'No image data provided'}), 400

    image_data = data['image_data'].split(",")[1] 
    image_bytes = base64.b64decode(image_data)
    save_path = os.path.join('static', 'img', 'captured_pic.png')

    with open(save_path, 'wb') as f:
        f.write(image_bytes)
        
    img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'image.png')
    with open(img_filename, 'wb') as img_file:
        img_file.write(image_bytes)
    image = Image.open(img_filename)
    
    
    description = get_frame_description(image)
    print(description)
    
    detection = detect_food_item(description)
    
    print(detection['edible'])
    if user_description:
        detection['user_input'] = user_description

    verdict = handle_detection(detection)
    verdict = verdict.replace("```html", "").replace("```", "").strip()
    render_template("index.html", result=verdict)
    print(verdict)

    return jsonify({'comment': verdict})





@app.route('/location', methods=['POST'])
def handle_location():
    try:
        data = request.get_json()
        latitude = data.get('latitude')
        longitude = data.get('longitude')

        if latitude is None or longitude is None:
            return jsonify({'error': 'Missing coordinates'}), 400

        print(f"📍 Received location: ({latitude}, {longitude})")

        
        verdict = get_healthy_places_nearby(latitude,longitude)
        verdict = verdict.replace("```html", "").replace("```", "").strip()


        return jsonify({'comment': verdict})

    except Exception as e:
        print("❌ Error in /location:", e)
        return jsonify({'error': 'Something went wrong'}), 500




@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "").strip()

    if not user_input:
        return jsonify({"reply": "Please enter a message."})

    
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())

    try:
        reply = get_chat_response(user_input, session["session_id"])
        return jsonify({"reply": reply})
    except Exception as e:
        print("❌ Chat error:", e)
        return jsonify({"reply": "Sorry, something went wrong processing your request."})
    
    
    
if __name__ == "__main__":
    app.run(debug=True)
