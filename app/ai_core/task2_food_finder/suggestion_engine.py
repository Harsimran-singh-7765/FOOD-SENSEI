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
            "Take the researched list of healthy restaurants and format it as a fixed HTML <div>. "
            "Use the exact structure below for consistency and visual appeal:\n\n"
            "<div style='background-color: #121212; color: #ffffff; padding: 20px; font-family: Arial, sans-serif;'>\n"
            "  <div style='background-color: #1f1f1f; border-radius: 12px; padding: 20px; margin-bottom: 20px;'>\n"
            "    <h2 style='margin: 0 0 10px; color: #00ffff;'>[Restaurant Name]</h2>\n"
            "    <p style='margin: 0 0 10px;'>Rating: ⭐ [4.5]</p>\n"
            "    <p style='margin: 0 0 10px;'><a href='[Website URL]' style='color: #7CFC00; text-decoration: none;'>Visit Website</a></p>\n"
            "    <h4 style='margin: 10px 0;'>Healthy Dishes:</h4>\n"
            "    <ul style='list-style-type: none; padding: 0; margin: 0;'>\n"
            "      <li style='margin-bottom: 8px;'>\n"
            "        <span style='font-weight: bold;'>[Dish Name]</span>\n"
            "        <span style='background-color: #000000; color: #ffffff; padding: 3px 8px; border-radius: 6px; margin-left: 6px;'>Grilled</span>\n"
            "        <span style='background-color: #32CD32; color: #ffffff; padding: 3px 8px; border-radius: 6px; margin-left: 6px;'>Low Carb</span>\n"
            "      </li>\n"
            "      <li style='margin-bottom: 8px;'>\n"
            "        <span style='font-weight: bold;'>[Dish Name]</span>\n"
            "        <span style='background-color: #800080; color: #ffffff; padding: 3px 8px; border-radius: 6px; margin-left: 6px;'>Vegan</span>\n"
            "        <span style='background-color: #00ffff; color: #000000; padding: 3px 8px; border-radius: 6px; margin-left: 6px;'>Fresh</span>\n"
            "      </li>\n"
            "    </ul>\n"
            "  </div>\n"
            "  <!-- Repeat above block for each restaurant -->\n"
            "</div>\n\n"
            "Replace the placeholders with actual data. Do not change the structure or styling. "
            "Use only inline CSS. Return ONLY the full HTML string starting with <div>."
        ),
        expected_output=(
            "A complete HTML string wrapped in a <div>, with each restaurant inside a card-like sub-div. "
            "Use only inline CSS. All tag labels must follow the given color scheme and structure."
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
