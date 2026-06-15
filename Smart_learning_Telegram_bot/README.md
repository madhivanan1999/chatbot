# 🎓 Smart Learning Telegram Bot

A **Telegram-based Smart Learning Assistant** that helps students follow a structured syllabus, track learning progress, and interact with AI-powered explanations.  
Built using **Python, PostgreSQL, and the Telegram Bot API**, with AI responses powered by **Groq LLM**.

---

## 🚀 Features

- 👤 **Student Registration & Profile Management**
  - Name, University, Stream, Department, Semester
- 📚 **Syllabus-Based Learning Flow**
  - Automatically assigns topics based on stream & semester
- ✅ **Task Progress Tracking**
  - Mark topics as completed and move to the next one
- 📊 **Learning Progress Dashboard**
  - Completed topics, remaining tasks, and mastery percentage
- 🤖 **AI-Powered Doubt Solving**
  - Context-aware responses using Groq LLM
- 🧠 **Short-Term Memory per User**
  - Maintains recent conversation context
- 🛠️ **Editable Profile**
  - Update academic details anytime
- 🧾 **Inline Keyboard Navigation**
  - Smooth Telegram UI experience

---

## 🏗️ Tech Stack

- **Language:** Python  
- **Bot Framework:** `python-telegram-bot` (v20+)  
- **Database:** PostgreSQL  
- **AI Model:** Groq LLM  
- **ORM / DB Driver:** `psycopg2`  
- **Environment Management:** `python-dotenv`  

---

## 📂 Project Structure
├── main.py # Telegram bot logic
|
├── groq_client.py # Groq LLM integration
|
├── requirements.txt # Python dependencies
|
├── .env.example # Environment variable template
|
├── README.md # Project documentation


---

## ⚙️ Environment Variables

Create a `.env` file in the root directory:

```env
TELEGRAM_BOT_TOKEN=your_telegram_bot_token

DB_NAME=your_database_name
DB_USER=your_database_user
DB_PASSWORD=your_database_password
DB_HOST=localhost
DB_PORT=5432

GROQ_API_KEY=your_groq_api_key


🤖 Telegram Commands
Command	Description
/start	Start the bot
/register	Create student profile
/edit	Edit profile details
/learn	Start learning topics
/profile	View profile
/syllabus	View syllabus
/status	View learning progress
/help	Show available commands


## Start the Bot
  python main.py