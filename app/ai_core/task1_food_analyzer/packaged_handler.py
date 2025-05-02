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
        query = f"{product_name} {company_name or ''} ingredients preservatives nutrition label"

    
        agent = Agent(
            role="Packaged Food Investigator",
            goal=(
                "Use web search to extract detailed nutrition information about a packaged food product. mainly from official site of '{product_name}' from '{company_name or 'any brand'}' "
                "Focus especially on its ingredients, any preservatives, nutritional value, allergens, and certifications like FDA or FSSAI. "
                "Return all insights in a clean dictionary format for further use."
                "don't go over searching "
            ),
            backstory=(
                "You are a nutritional analysis agent who specializes in packaged food products. "
                "You investigate the ingredients and health impact of products using the web and product packaging information."
            ),
            verbose=True,
            llm=llm,
            tools=[search_tool, scrape_tool],
        )

        # Task definition
        task = Task(
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
            agent=agent
        )

        crew = Crew(agents=[agent], tasks=[task])
        result = crew.kickoff()
        return result.raw

    except Exception as e:
        print(f"Error analyzing packaged food: {e}")
        return {"error": str(e)}
