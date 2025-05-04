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
            role="Official Food Product Investigator",
            goal=(
                "Search only the *official website* of '{product_name}' from '{company_name or 'any brand'}'. "
                "Extract highly reliable and structured nutritional information. Prioritize:\n"
                "- Complete and accurate list of ingredients with their percentages , always return something \n"
                "- Preservatives, additives, and chemical contents\n"
                "- Nutritional values (e.g., calories, protein, fats)\n"
                "- Any health certifications like FDA, FSSAI, Organic\n"
                "- Allergy triggers or sensitive ingredients\n"
                "- Potential red flags like fake claims, suspicious pricing, or health risks\n\n"
                "Avoid unnecessary web crawling. Use official site URLs via search or known brand domains. "
                "Summarize in clean markdown format + provide structured dictionary for further use."
                "always return the full integrent full list proper with evrything inclusive alaways"
            ),
            backstory=(
                "You are a specialized nutrition inspector focusing on health safety and data authenticity. "
                "Your expertise lies in accurately interpreting product data from trusted sources, especially official company pages. "
                "You're cautious of marketing gimmicks and detect potential consumer risks in food items."
            ),
            verbose=True,
            llm=llm,
            tools=[search_tool, scrape_tool],  # You must configure search_tool to use Serper with site-restriction
        )


        health_evaluator = Agent(
            role="Nutritional Health Analyst",
            goal=(
                "Analyze provided nutrition data and return a health score, categorized pros and risks, and usage advice. "
                "Include the original nutrition information for future use."
            ),
            backstory=(
                "You are a certified nutrition expert. You evaluate food products based on international standards like WHO, USDA, and FSSAI, "
                "with a focus on health impact and dietary suitability."
            ),
            tools=[],  
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
        

        
        consumption_advisor = Agent(
            role="Dietary Consumption Advisor",
            goal=(
                "Provide clear, realistic eating guidelines based on health analysis and nutritional content. "
                "Make sure users understand how much and how often to consume the food safely."
            ),
            backstory=(
                "You are a clinical dietitian with experience in formulating consumption advice. "
                "You help people avoid risks and form healthy eating habits based on nutrition labels and expert analysis."
            ),
            verbose=True,
            llm=llm
        )

        
        
        analyzing = Task(
            description=(
                f"Study the product: '{product_name}' from '{company_name or 'any brand'}'. Detected as: {detection}.\n"
                f"Use smart Serper search queries like:\n"
                f"    site:{company_name}.com (do lowercase the name for proper search ) {product_name} nutrition OR ingredients\n"
                f"Goal:\n"
                "- Get official ingredient list (include percentage if available)\n"
                "- Detect preservatives and additives\n"
                "- Gather nutrition values (calories, proteins, carbs, fats, fiber, sugar, etc.)\n"
                "- Validate certifications (FSSAI, FDA, Organic, etc.)\n"
                "- Detect allergens, and mark potential health risks\n"
                "- Identify fake branding or overpromising claims (e.g., ultra-low price)\n\n"
                "Return output in:\n"
                "1. Markdown summary\n"
                "2. Dictionary format:\n"
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
                "  'warnings': '...',\n"
                "  'verified_source': 'https://officialsite.com/product-page'\n"
                "}"
            ),
            expected_output="Clean markdown summary + structured JSON dictionary with health relevance.",
            agent=Analyzer
        )
        health_analysis = Task(
            description=(
                "You are given a dictionary with nutrition information about a food product. "
                "Use this data to:\n"
                "- Evaluate the health impact based on WHO/USDA/FSSAI norms\n"
                "- Provide a health score (out of 10)\n"
                "- List 2-3 health pros (e.g. 'low sugar', 'high fiber')\n"
                "- List 2-3 health risks (e.g. 'contains trans fats', 'excess sodium')\n"
                "- Suggest health tags (e.g. 'Healthy', 'Caution: Allergens')\n"
                "- Advise on consumption frequency: Daily / Occasionally / Avoid\n\n"
                "Finally, include the original nutrition dictionary in the return so it can be used by another agent."
            ),
            expected_output=(
                "Return a dictionary in this format:\n"
                "{\n"
                "  'health_score': 8.2,\n"
                "  'health_pros': [...],\n"
                "  'health_risks': [...],\n"
                "  'health_tags': [...],\n"
                "  'consumption_advice': '...',\n"
                "  'original_nutrition_data': {  # full dictionary from Analyzer }\n"
                "}"
            ),
            agent=health_evaluator
        )

        consumption_suggestion = Task(
            description=(
                "You're given:\n"
                "- Detailed nutrition and ingredient data from a packaged food\n"
                "- A health analysis that includes score, pros, risks, and health tags\n\n"
                "Your job is to:\n"
                "- Recommend an ideal serving size (grams or ml)\n"
                "- Suggest how often it can be safely consumed (daily/weekly/etc.)\n"
                "- Offer tailored advice for 3 groups: children, elderly, fitness-conscious\n"
                "- Provide a motivational health tip (keep it relevant to the product)\n\n"
                "**Important:**\n"
                "- Use specific numbers where possible (e.g., 'Max 30g per day')\n"
                "- Consider sugar, preservatives, trans fats, allergens, protein content, etc.\n"
                "- If any expected information is missing (e.g., sugar level), assume average and mention it\n"
                "- Never return empty fields. Fill with safe estimates or advice like 'Consult a doctor if unsure.'\n"
                "- Return a full dictionary that also includes all data from previous agents"
            ),
            expected_output=(
                "Return a final dictionary like this:\n"
                "{\n"
                "  'serving_size': '50g',\n"
                "  'frequency': 'Once every 3 days',\n"
                "  'age_group_advice': {\n"
                "     'children': 'Limit to half serving, avoid daily intake',\n"
                "     'elderly': 'Avoid if diabetic or hypertensive',\n"
                "     'fitness': 'Good post-workout option, but watch sugar'\n"
                "  },\n"
                "  'health_tip': 'Add a fiber-rich fruit alongside to slow sugar absorption',\n"
                "  'full_health_analysis': { ... },\n"
                "  'original_nutrition_data': { ... }\n"
                "}"
            ),
            agent=consumption_advisor
        )


        formatting = Task(
            description=(
                f"""You are given a structured dictionary with nutrition and ingredient information of a packaged food item. 
        Use the data to generate a dark-themed, visually striking HTML layout inside a <div>. Stick to the exact format below:

        IMPORTANT:
        - All text values (like ingredients, allergens) must be properly escaped (e.g., use `html.escape()`) or wrapped inside safe HTML elements.
        - Do NOT insert raw user or model text directly into `style` or attribute areas.
        - Never break inline CSS syntax (e.g., avoid any stray `'); color:` issues).
        - Every string inserted into HTML must be **properly closed and escaped** to prevent layout or injection issues.
        - remove these characters also ' ');"> '

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

        Ensure all data values are sanitized to prevent malformed HTML.  
        Make it mobile responsive: text should wrap, avoid overflow.  
        add tables when there are list of ingridents or anything 
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

