

import os
from dotenv import load_dotenv,find_dotenv
from google.generativeai import configure
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from langchain.tools import Tool
import json

llm = LLM(model="gemini/gemini-2.0-flash")

 

load_dotenv(find_dotenv())


print("GEMINI_API_KEY:", os.getenv("GEMINI_API_KEY"))  # Check if key is loaded

search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()


def build_query(lat, lon):
    return f"top healthy food places within 5 km of latitude {lat}, longitude {lon}"


def get_healthy_places_nearby(lat, lon):
    # AGENT 1: Reverse Geocoder
    geocoder = Agent(
        role="Geolocation Specialist",
        goal="Convert coordinates into a human-readable location name.",
        backstory=(
            "You are a geography-savvy assistant who excels at turning GPS coordinates into real-world place names "
            "so that other agents can work with a location-based query rather than raw numbers."
        ),
        tools=[search_tool],
        verbose=True,
        llm=llm
    )

    # AGENT 2: Focused Researcher
    researcher = Agent(
        role="Focused Web Researcher",
        goal="Find accurate and popular healthy food places nearby the given location name.",
        backstory=(
            "You are a highly focused research assistant specializing in food and wellness. "
            "You are tasked with finding only high-quality, well-reviewed healthy food places near a location."
        ),
        tools=[search_tool, scrape_tool],
        verbose=True,
        llm=llm
    )

    # AGENT 3: Menu Analyst
    analyst = Agent(
        role="Nutritional Analyst",
        goal="Analyze menu items from each restaurant and select healthier options.",
        backstory=(
            "You are a nutrition-focused assistant who understands food labels and menus. "
            "You identify meals that are lower in sugar, fried content, and saturated fats, "
            "and recommend options that are relatively healthier from the given menus."
            "don't do more than 3 online searches and just return somethiing u find out "
        ),
        tools=[search_tool, scrape_tool],
        verbose=True,
        llm=llm
    )

    # AGENT 4: Content Formatter
    editor = Agent(
        role="Frontend Developer and Content Formatter",
        goal="Take analyzed data and convert it into a visually appealing HTML layout with color-coded tags.",
        backstory=(
            "You are a frontend-focused AI expert who specializes in converting raw data into beautiful and structured UI components. "
            "You ensure that nutritional and product data is easy to read and visually appealing for end users."
            
            "don't do more than 3 online searches and just return somethiing u find out "
        ),
        verbose=True,
        llm=llm
    )

    # Task 1: Get location name
    reverse_geocode = Task(
        description=f"Convert coordinates ({lat}, {lon}) to a city or neighborhood name using reverse geocoding.",
        expected_output="Return only the location name (e.g., Indiranagar Bangalore or Midtown NYC).",
        agent=geocoder
    )

    # Task 2: Find healthy food spots
    search_places = Task(
        description="Search for healthy food restaurants near the location found in Task 1."
                    "don't do more than 3 online searches and just return somethiing u find out ",
        expected_output="List 3-5 healthy food places with name, rating, and source link.",
        agent=researcher
    )

    # Task 3: Analyze menus
    analyze_menu = Task(
        description="Check what each restaurant offers and identify relatively healthier options (e.g., salads, grilled, low sugar)."
                    "don't do more than 3 online searches and just return somethiing u find out ",
        expected_output="Return a list of each restaurant with 2-3 recommended healthier items from their menu.",
        agent=analyst
    )

    # Task 4: Format for display
    format_ui = Task(
        description=(
            "Take the restaurant names, ratings, menu highlights, and healthy items and format it as a full HTML <div> layout. "
            "Use dark mode colors, card-style layout, and inline CSS for visual tags like 'Low Carb', 'Grilled', 'Fresh'."
        ),
        expected_output="Return a full HTML string with a dark-themed layout, using color-coded tags for key highlights.",
        agent=editor
    )

    # CREW ASSEMBLY
    crew = Crew(
        agents=[geocoder, researcher, analyst, editor],
        tasks=[reverse_geocode, search_places, analyze_menu, format_ui],
        verbose=True,
        llm=llm
    )

    result = crew.kickoff({"lat": lat, "lon": lon})
    return result.raw


