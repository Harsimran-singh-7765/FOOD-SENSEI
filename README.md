


# <p align="center">
  <img src="static/img/logo.png" alt="Food Sensei Logo" width="150"/>
</p> FOOD SENSEI

**Food Sensei** is your AI-powered Nutrition & Food Guide built using Flask, CrewAI (multi-agent framework), and Gemini. It provides intelligent food insights, suggestions, and myth-busting knowledge based on images, location, and user queries.

---

## 🚀 Features

### 📷 Task 1: Food Analyzer
- Captures a live image using webcam.
- Determines whether the food is **packaged** or **unpackaged**.
- If packaged: Uses AI and search agents to extract **ingredients, preservatives, and nutrition info**.
- If unpackaged: Evaluates based on appearance and taste cues to provide a **health summary**.

### 📍 Task 2: Healthy Food Finder
- Uses your **geolocation** to find and recommend **healthier food alternatives** near you.
- Filters results through a smart suggestion engine with nutritional focus.

### 🧠 Task 3: Food Mythbuster
- A conversational agent that debunks **common food myths**.
- Fact-checks claims using Gemini-backed insights and logic.

---

## 🧠 Powered By

- **Python** & **Flask** for the backend
- **HTML/CSS/JavaScript** for frontend UI
- **CrewAI** for multi-agent orchestration
- **Gemini** for intelligent responses and analysis
- **Web scraping & search tools** for real-world food data

---

## 📁 Project Structure


food-sensei/
│
├── app/  
│   ├── __init__.py  
│   ├── routes.py  
│   ├── ai_core/  
│   │   ├── task1_food _analyzer/  
│   │   │   ├── __init__.py  
│   │   │   ├── controller.py  
│   │   │   ├── detector.py               # is_packaged(image) -> bool  
│   │   │   ├── unpackaged_handler.py    # uses Gemini + taste questions  
│   │   │   └── crew_agents.py          # Nutritionist, Critic etc.   
│   │   ├── task2_food_finder/  
│   │   │   ├── __init__.py  
│   │   │   ├── controller.py  
│   │   │   ├── location_parser.py  
│   │   │   ├── web_searcher.py  
│   │   │   └── suggestion_engine.py  
│   │   ├── task3_myth_buster/   
│   │   │   ├── __init__.py   
│   │   │   ├── controller.py   
│   │   │   ├── query_parser.py   
│   │   │   ├── gemini_response.py   
│   │   │   └── fact_checker.py     
│   ├── utils/   
│   │   ├── scraper.py  
│   │   ├── tag_generator.py    
│   │   └── image_utils.py   
│   └── templates/    
│       └── index.html   
│
├── static/   
│   ├── css/  
│   ├── js/  
│   └── img/   
│
├── .env   
├── .gitignore   
├── requirements.txt   
├── config.py   
├── run.py   
└── README.md   

---


---

## 🛠️ Setup Instructions

1. **Clone the repo**  
   ```bash
   git clone https://github.com/Harsimran-singh-7765/FOOD-SENSEI
   cd food-sensei
   ```
2. **Install dependencies**
  ```bash 
  pip install -r requirements.txt
  
  ```
3. ** create .env **
   ```bash
   GEMINI_API_KEY="GEMINI_kEY"
   SERPER_API_KEY = "SERPER_API_KEY"
   ```
   
5. **Run the apps**
```bash 
Python app.py

```
---

##DEMO 
![image](https://github.com/user-attachments/assets/57cb916e-d98a-4a42-ae67-537ac00e0d6a)

## Developed by Polardevs <p align="center">
  <img src="Polar_dev.png" alt="Food Sensei Logo" width="150"/>
</p>
