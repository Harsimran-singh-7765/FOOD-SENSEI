import os
from dotenv import load_dotenv
from google.generativeai import configure
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
from crewai_tools import ScrapeWebsiteTool 
from langchain.tools import Tool
from langchain.agents import Tool
import json

llm = LLM(model="gemini/gemini-1.5-flash")
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))


search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()



def analyze_unpackaged_food(description):
    try:
        
        Investigator = Agent(
            role="Descriptive Food Investigator",
            goal=(
                "Based on the description of a food item, guess what food it might be, search the web, "
                "and extract approximate information like ingredients, nutrients, risks, and benefits. "
                "Focus on reliable food blogs, nutrition databases, or official references."
            ),
            backstory=(
                "You are an AI expert who interprets vague or partial food descriptions (e.g., 'brown snack bar with nuts') "
                "and infers the likely product. You then search online to understand what it likely contains, its health profile, etc."
            ),
            verbose=True,
            llm=llm,
            tools=[search_tool, scrape_tool]
        )

        
        Summarizer = Agent(
            role="Nutritional Content Summarizer",
            goal=(
                "Take semi-structured information about a food (ingredients, likely name, nutrient guesses) and refine it "
                "into a structured dictionary suitable for UI display. Do not hallucinate. Be conservative if unsure."
            ),
            backstory=(
                "You are a food labeling agent that helps summarize possibly incomplete data into a standard schema "
                "for nutrition guidance apps."
            ),
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
        

    
        analyze_task = Task(
            description=(
                f"Analyze this food description: '{description}'. "
                "Guess the likely product (e.g., energy bar, cookie with chocolate chips, etc.) and search online for:\n"
                "- Ingredients\n"
                "- Nutritional info\n"
                "- Common allergens\n"
                "- Health risks or benefits"
            ),
            expected_output=(
                "Return a dictionary like:\n"
                "{\n"
                "  'name': 'Likely food name',\n"
                "  'ingredients': [...],\n"
                "  'nutrition_facts': {...},\n"
                "  'benefits': '...',\n"
                "  'warnings': '...',\n"
                "  'diet_tags': ['high-protein', 'vegan', etc.]\n"
                "}"
            ),
            agent=Investigator
        )

        
        refine_task = Task(
            description="Refine the extracted data into a clear and structured dictionary suitable for nutrition display.",
            expected_output="Clean dictionary as described above.",
            agent=Summarizer
        )

    
        formatting = Task(
            description=(
                f"""You are given a structured dictionary with nutrition and ingredient information of a packaged food item. 
        Use the data to generate a dark-themed, visually striking HTML layout inside a <div>. Stick to the exact format below:

        IMPORTANT:
        - All text values (like ingredients, allergens) must be escaped or wrapped inside valid HTML elements.
        - Never break inline CSS syntax (e.g., avoid any stray `'); color:` issues).
        - Every string inserted into HTML must be **properly closed and escaped**.

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
        - All values must be **wrapped in span tags or <li> properly** to avoid malformed styling.

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
                "Return a full dark-themed HTML <div> using inline CSS, strictly following the design rules above. "
                "Do NOT leave any malformed or broken CSS, and ensure all values are escaped properly."
            ),
            agent=Editor
        )



       
        crew = Crew(agents=[Investigator, Summarizer, Editor], tasks=[analyze_task, refine_task, formatting])
        result = crew.kickoff()

        return result.raw

    except Exception as e:
        print(f"Error analyzing food from description: {e}")
        return f"<p>Error: {str(e)}</p>"
