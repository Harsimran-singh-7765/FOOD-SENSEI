import os
from dotenv import load_dotenv
from google.generativeai import configure
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
from crewai_tools import ScrapeWebsiteTool 
from langchain.tools import Tool
from langchain.agents import Tool
import json
load_dotenv(dotenv_path=r".env")
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
configure(api_key=os.environ["GEMINI_API_KEY"])
llm = LLM(model="gemini/gemini-1.5-flash")
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# Initialize the tools
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()



def analyze_packaged_food(product_name,detection, company_name=None):
    try:
        

    
        Analyzer = Agent(
            role="Packaged Food Investigator",
            goal=(
                "Use web search to extract detailed nutrition information about a packaged food product. mainly from official site of '{product_name}' from '{company_name or 'any brand'}' "
                "Focus especially on its ingredients, any preservatives, nutritional value, allergens, and certifications like FDA or FSSAI. "
                "Return all insights in a clean dictionary format for further use."
                "don't go over searching , search less but effective mainly on official site"
            ),
            backstory=(
                "You are a nutritional analysis agent who specializes in packaged food products. "
                "You investigate the ingredients and health impact of products using the web and product packaging information."
            ),
            verbose=True,
            llm=llm,
            tools=[search_tool, scrape_tool],
        )

        health_evaluator = Agent(
                role="Nutritional Health Analyst",
                goal="Give a health score and analysis based on nutrition data, identifying benefits and risks",
                backstory="You are a certified nutrition expert who evaluates food based on international health norms",
                tools=[],  # Optional: add Serper tool if you'd like it to cross-verify
                verbose=True,
                llm=llm
            )
        
        Editor = Agent(
            role="Frontend Developer and Content Formatter",
            goal=(
                "Take a structured dictionary response about a food item and generate a visually appealing HTML `<div>` layout. "
                "Use color-coded tags for key attributes (e.g., green for 'edible', red for 'not edible'), "
                "display product details using semantic HTML and styled tables, and ensure accessibility and responsiveness."
            ),
            backstory=(
                "You are a frontend-focused AI expert who specializes in converting raw data into beautiful and structured UI components. "
                "You ensure that nutritional and product data is easy to read and visually appealing for end users."
            ),
            verbose=True,
            llm=llm,
        )
        
        analyzing = Task(
            description=(
                f"Find information on this product: '{product_name}' from '{company_name or 'any brand'}'. study more by {detection} "
                f"Search and extract:\n"
                "- Full list of ingredients\n"
                "- Type and amount of preservatives\n"
                "- Nutritional values (calories, protein, etc.)\n"
                "- Any certifications (FDA, organic, etc.)\n"
                "- Allergy warnings or health flags"
            ),
            expected_output=(
                "Return a dictionary in this format:\n"
                "{\n"
                "  'product': 'Product Name',\n"
                "  'company': 'Company Name',\n"
                "  'ingredients': [...],\n"
                "  'preservatives': [...],\n"
                "  'nutrition_facts': {\n"
                "      'calories': '...', 'protein': '...', ...\n"
                "  },\n"
                "  'certifications': [...],\n"
                "  'allergens': [...],\n"
                "  'warnings': '...'\n"
                "}"
            ),
            agent=Analyzer
        )
        
        consumption_advisor = Agent(
                role="Dietary Consumption Advisor",
                goal="Guide the user on how much and how often to eat the food based on health evaluation",
                backstory="You are a clinical dietitian helping people build balanced diets based on packaged food content",
                verbose=True,
                llm=llm
            )
        consumption_suggestion = Task(
            description=(
                "Using the nutrition data and health analysis, recommend appropriate consumption guidelines.\n"
                "- Suggest ideal serving size (in grams/ml)\n"
                "- Recommend frequency (e.g., daily/weekly/monthly)\n"
                "- Include special advice for kids, elderly, or fitness-conscious users\n"
                "- Add a short motivational health tip (e.g., 'Pair it with a salad to balance carbs')"
            ),
            expected_output=(
                "Return in this format:\n"
                "{\n"
                "  'serving_size': '50g',\n"
                "  'frequency': 'Twice a week',\n"
                "  'age_group_advice': {\n"
                "     'children': 'Small portions only, avoid daily use',\n"
                "     'elderly': 'Avoid if hypertensive',\n"
                "     'fitness': 'Acceptable post-workout snack'\n"
                "  },\n"
                "  'health_tip': 'Drink extra water with salty snacks'\n"
                "}"
            ),
            agent=consumption_advisor
        )
        health_analysis = Task(
            description=(
                "Review the nutrition dictionary for the food product.\n"
                "- Based on WHO/USDA/FSSAI norms, give a health score (out of 10)\n"
                "- List 2-3 health pros (e.g. 'low sugar', 'high fiber')\n"
                "- List 2-3 health risks (e.g. 'contains trans fats', 'excess sodium')\n"
                "- Suggest tags like: ['Healthy', 'Caution: Sodium', 'Avoid for Diabetics']\n"
                "- Mention if it's good for daily consumption, occasionally, or should be avoided"
            ),
            expected_output=(
                "Return in this format:\n"
                "{\n"
                "  'health_score': 7.5,\n"
                "  'health_pros': [...],\n"
                "  'health_risks': [...],\n"
                "  'health_tags': [...],\n"
                "  'consumption_advice': '...' \n"
                "}"
            ),
            agent=health_evaluator
        )
        formatting = Task(
            description=(
                f"""You are given a structured dictionary with nutrition and ingredient information of a packaged food item. 
        Use the data to generate a dark-themed, visually striking HTML layout inside a <div>. Stick to the exact format below:

        1. **Header Section**:
        - Display the 'product' and 'company' names prominently in white text.
        - Background: solid black.
        - Subtitle: "Health Analysis Summary" in cyan.
        - Use padding and center alignment.

        2. **Tag Section**:
        - Use health insights to decide which tag to show:
            - Green tag with label 'Healthy'
            - Purple tag with label 'Moderate'
            - Red or bright orange tag with label 'Unhealthy'
        - Tags must use inline CSS with:
            - Bold white text
            - Padding: 4–6px
            - Rounded corners
            - Backgrounds: green, purple, red/orange depending on the tag
            - Slight glow/shadow effect for modern look

        3. **Details Section**:
        - Use white headings with underline (e.g. Ingredients, Preservatives)
        - Text: light gray (#CCCCCC)
        - Ingredients, Preservatives, Allergens: shown as bullet lists
            - Use purple dots for bullets
            - Harmful ingredients (e.g. 'MSG', 'trans fat') must be shown in **bright red**
        - Certifications: comma-separated in cyan

        4. **Nutrition Table**:
        - Background: dark gray (#222)
        - Text: white
        - Borders: cyan
        - Header row: background black, text cyan
        - Alternate row shading with slightly lighter gray
        - Excessive nutrients like high sugar or trans fat: show in **bold red**

        5. **Consumption Advice**:
        - Below the table, add one of:
            - “✅ Safe for daily use in moderation.” – Green
            - “⚠️ Occasional use only due to high fat/sugar.” – Orange
        - Italicize advice and wrap in <p> with white or appropriate tag color

        6. **Styling and Layout**:
        - Entire content inside a <div> with:
            - `background-color: black`
            - `color: white`
            - `padding: 20px`
            - `border-radius: 15px`
            - `box-shadow: 0 0 15px cyan`
            - `font-family: 'Arial', sans-serif`
            - `max-width: 600px`
            - `margin: auto`

        Make it mobile responsive: text should wrap, avoid overflow.  
        Return **only the final HTML inside the <div> tag**. No explanation, no markdown.
        """
            ),
            expected_output=(
                "Return a full dark-themed HTML <div> using inline CSS, strictly following the design rules above."
            ),
            agent=Editor
        )


        crew = Crew(
            agents=[Analyzer, health_evaluator, consumption_advisor, Editor],
            tasks=[analyzing, health_analysis, consumption_suggestion, formatting]
            )

        result = crew.kickoff()
        return result.raw


    except Exception as e:
        print(f"Error analyzing packaged food: {e}")
        return {"error": str(e)}

