import os
from dotenv import load_dotenv
from google.generativeai import configure
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
from crewai_tools import ScrapeWebsiteTool 
from langchain.tools import Tool
from langchain.agents import Tool
import json
import re
load_dotenv(dotenv_path=r".env")
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
configure(api_key=os.environ["GEMINI_API_KEY"])
llm = LLM(model="gemini/gemini-1.5-flash")
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# Initialize the tools
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

def detect_food_item(description):
    try:
        # Define the Agent
        agent = Agent(
            role="Nutrition Inspector AI",
            goal=(
                "Analyze the visual description of an object to determine if it is edible. "
                "If it is edible, classify whether it is a packaged product or not. "
                "For packaged foods, extract the company name, product name, and specific product details "
                "like ingredients, type, and flavor. For unpackaged items, describe the food in a helpful way, "
                "e.g., fresh fruit, cooked dish, vegetable, street food, etc."
            ),
            backstory=(
                "You are an expert in food classification and packaging analysis. "
                "You have a deep understanding of grocery items, food labeling, common brands, and "
                "types of consumables. You read product packaging and visual cues to determine the food type."
                
            ),
            verbose=True,
            llm=llm,
            tools=[search_tool, scrape_tool], 
        )

        # Define the Task
        task = Task(
            description=f"Analyze this visual scene from a webcam: {description}",
            expected_output=(
                "Return only a JSON object as your final answer. Use double quotes for all keys and string values, and lowercase `true`/`false` for booleans."

                "A dictionary with the following structure:\n"
                "{\n"
                "  'edible': True or False, \n"
                "  'packaged': True or False (if it is covered in a cartoon or a plastic or any type of cover),\n"
                "  'details': {\n"
                "     # If packaged: 'company', 'product_name', 'specifics' (like flavor, variant)\n"
                "     # If unpackaged: 'type' (fruit/vegetable/street food), 'description'\n"
                "     # If not edible : 'what is it', why not edible "
                "  }\n"
                "}"
            ),
            agent=agent
        )

        # Execute via Crew
        crew = Crew(agents=[agent], tasks=[task])
        result = crew.kickoff()

        raw = result.raw.strip()


        match = re.search(r'```json\s*(\{.*?\})\s*```', raw, re.DOTALL)
        if match:
            json_str = match.group(1)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                return {"edible": False, "error": "Malformed JSON in agent output"}
        else:
            print("No JSON block found in result")
            return json.loads(result.raw ) 
    except Exception as e:
        print(f"Error in food detection: {e}")
        return {"edible": False, "error": str(e)}
