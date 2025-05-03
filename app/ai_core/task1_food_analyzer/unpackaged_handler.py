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



def analyze_unpackaged_food(description):
    try:
        # Agent 1: Investigator
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

        # Agent 2: Nutrition Summarizer
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

        # Agent 3: Frontend Formatter
        Formatter = Agent(
            role="Frontend Developer and Content Formatter",
            goal=(
                "Take a food nutrition dictionary and convert it into a visually styled HTML <div>. "
                "Add color-coded tags, use good spacing, and format benefits and warnings clearly."
            ),
            backstory="You specialize in readable UI for food apps using nutrition data.",
            verbose=True,
            llm=llm
        )

        # Task 1: Analyze the vague food description
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

        # Task 2: Structure & refine
        refine_task = Task(
            description="Refine the extracted data into a clear and structured dictionary suitable for nutrition display.",
            expected_output="Clean dictionary as described above.",
            agent=Summarizer
        )

        # Task 3: Convert to HTML
        html_task = Task(
            description="Convert the structured nutrition dictionary into a clean HTML <div> block using inline CSS and accessible layout.",
            expected_output="HTML string only — styled, readable, and ready for browser rendering.",
            agent=Formatter
        )

        # Run crew
        crew = Crew(agents=[Investigator, Summarizer, Formatter], tasks=[analyze_task, refine_task, html_task])
        result = crew.kickoff()

        return result.raw

    except Exception as e:
        print(f"Error analyzing food from description: {e}")
        return f"<p>Error: {str(e)}</p>"
