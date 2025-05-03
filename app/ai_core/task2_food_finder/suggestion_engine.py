# suggestion_engine.py

import os
from dotenv import load_dotenv,find_dotenv
from google.generativeai import configure
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from langchain.tools import Tool
import json

llm = LLM(model="gemini/gemini-1.5-flash")

 

load_dotenv(find_dotenv())


print("GEMINI_API_KEY:", os.getenv("GEMINI_API_KEY"))  # Check if key is loaded

search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

# Helper: Build location-specific query
def build_query(lat, lon):
    return f"top healthy food places within 5 km of latitude {lat}, longitude {lon}"



# Final orchestrator
def get_healthy_places_nearby(lat, lon):
    researcher = Agent(
        role="Focused Web Researcher",
        goal="Find accurate and popular healthy food places nearby the given coordinates.",
        backstory=(
            "You are a highly focused research assistant specializing in food and wellness. "
            "You are tasked with finding only high-quality, well-reviewed healthy food places near a location."
        ),
        tools=[search_tool,scrape_tool],
        verbose=True,
        llm=llm
    )
    
    
    Editor = Agent(
            role="Frontend Developer and Content Formatter",
            goal=(
                "Take  response  and generate a visually appealing HTML `<div>` layout. "
                "Use color-coded tags for key attributes , "
                "display in dark mode but appleaing view."
            ),
            backstory=(
                "You are a frontend-focused AI expert who specializes in converting raw data into beautiful and structured UI components. "
                "You ensure that nutritional and product data is easy to read and visually appealing for end users."
            ),
            verbose=True,
            llm=llm,
        )
    query = build_query(lat, lon)
    
    research = Task(
        description=f"Use the search tool to find healthy food restaurants: '{query}'",
        expected_output="A list of 3-5 healthy food places with name, rating, and website or source link.",
        agent=researcher
    )
    formatting = Task(
        description=(
            f"create a full HTML `<div>` that represents it nicely:"
            "- Add color-coded tags )\n"

           
        ),
        expected_output=(
            "Return only the full HTML string wrapped in a <div>. Use inline CSS for styling tags and layout if needed."
        ),
        agent=Editor
        )

    crew = Crew(
        agents=[researcher,Editor],
        tasks=[research,formatting],
        verbose=True,
        llm=llm
    )

    result = crew.kickoff({"query":build_query(lat,lon)})
    return result.raw 

