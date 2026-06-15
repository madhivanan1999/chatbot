import os 
from dotenv import load_dotenv
from collections import defaultdict
import psycopg2
from psycopg2.extras import RealDictCursor
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
from telegram.ext import Application, ConversationHandler, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
from groq_client import ask_groq
import warnings
from telegram.warnings import PTBUserWarning
warnings.filterwarnings(action="ignore", message=r".*CallbackQueryHandler", category=PTBUserWarning)

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
print("DB_NAME =", os.getenv("DB_NAME"))

# Define Conversation States
NAME, UNIVERSITY, STREAM, DEPARTMENT, SEM = range(5)

# --- Database Logic ---
def get_env(key):
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"Missing environment variable: {key}")
    return value

def get_conn():
    return psycopg2.connect(
        dbname= get_env("DB_NAME"),
        user= get_env("DB_USER"),
        password= get_env("DB_PASSWORD"),
        host= get_env("DB_HOST"),
        port= get_env("DB_PORT"))

def is_user_registered(telegram_id: int) -> bool:
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("select 1 from vs.user_info where telegram_id = %s", (telegram_id,))
                return cur.fetchone() is not None
    except Exception as e:
        print(f"DB Check Error: {e}")
        return False

def register_user_to_db(data):
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                # Update user info (your existing upsert logic)
                cur.execute(""" insert into vs.user_info(telegram_id, student_name, university, stream, department, sem) 
                            values(%s, %s, %s, %s, %s, %s)
                            on conflict(telegram_id) do update 
                            set student_name = excluded.student_name, university = excluded.university,
                            stream = excluded.stream, department = excluded.department, sem = excluded.sem;""",
                            (data['student_id'], data['student_name'], data['university'], data['stream'], data['department'], data['sem']))
                # Assign the first topic automatically
                cur.execute(""" insert into vs.user_progress (telegram_id, current_topic_id)
                            select %s, topic_id from vs.syllabus
                            where stream = %s AND department = %s AND sem = %s AND topic_order = 1
                            on conflict (telegram_id) do nothing;""", (data['student_id'], data['stream'],data['department'],data['sem']))
                conn.commit()
    except Exception as e:
        print(f"DB Insert Error: {e}")

# Fetches and displays the current task for a user
async def learn_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Use effective_chat to make it work for both messages and button clicks
    chat_id = update.effective_chat.id
    telegram_id = update.effective_user.id
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(""" select s.topic_name, s.content, s.topic_order
                        from  vs.user_progress p join vs.syllabus s on p.current_topic_id = s.topic_id
                        where p.telegram_id = %s""", (telegram_id,))
            task = cur.fetchone()
    
    if task:
        topic_name, content, order = task
        user_memories[telegram_id]= [{"role":"system", "content": f"The user is looking at Task{order}:{topic_name}.content:{content}"}]
        text = f"<b>📖 Task {order}: {topic_name}</b>\n\n{content}"
        # Inline button to trigger completion
        keyboard = [[InlineKeyboardButton("✅ Mark as Completed", callback_data= "task_done")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await context.bot.send_message(chat_id = chat_id, text=text, reply_markup=reply_markup, parse_mode = 'HTML')
    else:
        await update.message.reply_text("No active tasks found. Use /register to start!")

# Callback for when the user clicks the "Mark as Completed" button
async def task_completed_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    telegram_id = query.from_user.id
    await query.answer()

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(""" update vs.user_progress set current_topic_id = (
                        select next_s.topic_id 
                        from vs.syllabus next_s
                        join vs.syllabus current_s on next_s.topic_order = current_s.topic_order + 1
                        join vs.user_info u on u.telegram_id = %s
                        where current_s.topic_id = vs.user_progress.current_topic_id
                        and next_s.stream = u.stream
                        and next_s.department = u.department
                        and next_s.sem = u.sem)
                        where telegram_id =%s returning current_topic_id;""", (telegram_id, telegram_id))
            new_topic = cur.fetchone()
            conn.commit()
    if new_topic and new_topic[0]:
        await query.edit_message_text("✅ Task completed! \nType /learn to see your next task.")
    else:
        await query.edit_message_text("🎉 Congratulations! You have completed all tasks!")


#
async def syllabus_command(update:Update, context: ContextTypes.DEFAULT_TYPE):
    #Get the user's semester from your user_info table
    user_data = get_user_profile(update.effective_user.id)
    if not user_data:
        await update.message.reply_text("Please /register first!")
        return
    sem = user_data[4]
    # Query your syllabus table for this semester
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("Select topic_name from vs.syllabus where sem = %s", (sem,))
                rows = cur.fetchall()
        if rows:
            syllabus_text = f"📚 <b>Syllabus for Semester {sem}:</b>\n\n"
            topic_list = "\n".join([f"• {row[0]}" for row in rows])
            await update.message.reply_text(syllabus_text + topic_list,parse_mode = "HTML")
        else:
            await update.message.reply_text("No syllabus found for your semester.")
    except Exception as e:
        print(f"DB Error (Syllabus): {e}")
        await update.message.reply_text("⚠️ Sorry, I couldn't retrieve the syllabus right now.")


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    tg_id = update.effective_user.id

    with get_conn() as conn:
        with conn.cursor() as cur:

            # 1. Get current progress state
            cur.execute("""
                SELECT s.topic_order, s.sem, up.is_completed
                FROM vs.user_progress up
                JOIN vs.syllabus s ON up.current_topic_id = s.topic_id
                WHERE up.telegram_id = %s
            """, (tg_id,))
            row = cur.fetchone()

            if not row:
                await update.message.reply_text("Please /register or /learn first.")
                return

            current_order, sem, is_completed = row

            # 2. Total topics in semester
            cur.execute(
                "SELECT COUNT(*) FROM vs.syllabus WHERE sem = %s",
                (sem,)
            )
            total_topics = cur.fetchone()[0]

    # 3. Calculate completed topics
    completed = current_order if is_completed else current_order - 1
    completed = max(completed, 0)

    remaining = total_topics - completed
    percentage = (completed / total_topics * 100) if total_topics > 0 else 0

    status_msg = (
        f"📊 <b>Your Learning Progress</b>\n\n"
        f"✅ Completed: {completed}\n"
        f"⏳ Remaining: {remaining}\n"
        f"📈 Mastery: {percentage:.1f}%\n\n"
        "Keep going! Small steps every day lead to big results. 🚀"
    )

    await update.message.reply_text(status_msg, parse_mode="HTML")


#Create a memory storage
user_memories= defaultdict(list)
def user_memory(user_id, role, content):
    user_memories[user_id].append({"role": role, "content": content})
    if len(user_memories[user_id])> 6:
        user_memories[user_id].pop(0)

# Command list
async def post_init(application: Application):
    """ Sets the command list in the telegram menu button. """
    commands = [BotCommand("start", "start the bot"),
        BotCommand("register", "Create your student profile"),
        BotCommand("edit", "Update your profile details"),
        BotCommand("learn", "Start learning sessions"),
        BotCommand("profile", "View your current profile"),
        BotCommand("syllabus", "View your syllabus"),
        BotCommand("status", "View your current learning progress")]
    await application.bot.set_my_commands(commands)

async def help_command(update, context):
    help_text = (
        "📜Available Commands:\n\n"
        "/start - Welcome message\n"
        "/register - Setup your profile\n"
        "/edit - Modify your details\n"
        "/learn - Start learning sessions\n"
        "/profile - View your information\n"
        "/syllabus – View syllabus\n"
        "/status - vView your current learning progress"
    )
    await update.message.reply_text(help_text)
#----------------Helper for button------------------------

def build_menu(options, n_cols=2, prefix="", include_back=True):
    # Create buttons with unique prefixes: e.g., "univ_Anna_University"
    buttons = [InlineKeyboardButton(opt, callback_data=f"{prefix}{opt}") for opt in options]
    menu = [buttons[i:i + n_cols] for i in range(0, len(buttons), n_cols)]
    if include_back:
        # Special callback_data for the back button
        menu.append([InlineKeyboardButton("⬅️ Back", callback_data="back")])
    return InlineKeyboardMarkup(menu)

    
def get_user_profile(telegram_id: int):
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT student_name, university, stream, department, sem 
                    FROM vs.user_info 
                    WHERE telegram_id = %s
                """, (telegram_id,))
                return cur.fetchone()
    except Exception as e:
        print(f"Profile Fetch Error: {e}")
        return None


#------------------/start Command--------------------

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    tg_id =update.effective_user.id
    if is_user_registered(telegram_id= tg_id):
        text = ("✨Consistency beats cramming. Study a little every day!✨\n\n"
                "Let's begin your smart learning journey🚀📖\n"
                "Use /learn command to Start learning sessions")
    else:
        text = ("👋 Welcome to Smart Learning Bot!\n" 
                "Use /register command to set up your profile.")
    await update.message.reply_text(text)

#-----------------/register Command---------------

async def register_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles the /register command."""
    user_student_id = update.effective_user.id
    if is_user_registered(user_student_id):
        await update.message.reply_text("✅ You are already registered.\n\n"
        "👉 Use /help to continue 😊")
        return ConversationHandler.END
    await update.message.reply_text("🚀 Let's get started! \nwhat is your full name ?")
    return NAME
    
# Step-by-step handlers to collect data and store in context.user_data
async def get_name(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['student_name'] = update.message.text
    options = ["Pondicherry_University", "Anna_University", "Bharathiyar_University", "Alagappa_University"]
    await update.message.reply_text("which university you are ?", reply_markup= build_menu(options, prefix="univ_", include_back=False))
    return UNIVERSITY

async def get_university(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    if query.data =="back":
        await query.edit_message_text("Let's try again. What is your full name ?")
        return NAME
    if not query.data.startswith("univ_"):
        return UNIVERSITY
    # Remove the prefix "univ_" to get the actual value
    context.user_data['university'] = query.data.replace("univ_","")
    options= ["B.Tech", "B.Sc", "M.Tech", "PhD"]
    await query.edit_message_text(f"Selected: {context.user_data['university']}\nwhat is your stream ?", reply_markup= build_menu(options, prefix= "str_"))
    return STREAM

async def get_stream(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    if query.data == "back":
        options = ["Pondicherry_University", "Anna_University", "Bharathiyar_University", "Alagappa_University"]
        await query.edit_message_text("which university are you in ?", reply_markup= build_menu(options, prefix="univ_", include_back=False))
        return UNIVERSITY
    if not query.data.startswith("str_"):
        return STREAM
    context.user_data['stream'] = query.data.replace("str_", "")
    options = ["ECE", "EEE", "CSC", "MECH", "CIVIL", "IT"]
    await query.edit_message_text(text=f"Selected: {context.user_data['stream']}\nWhat is your Department ?", reply_markup= build_menu(options, n_cols=3, prefix="dept_"))
    return DEPARTMENT

async def get_dept(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    if query.data == "back":
        options= ["B.Tech", "B.Sc", "M.Tech", "PhD"]
        await query.edit_message_text(f"selected: {context.user_data['university']}\nWhat is your stream ?",reply_markup= build_menu(options, prefix= "str_"))
        return STREAM
    if not query.data.startswith("dept_"):
        return DEPARTMENT
    context.user_data['department'] = query.data.replace("dept_","")
    options = ["1", "2", "3", "4", "5", "6", "7", "8"]
    await query.edit_message_text(f"Selected: {context.user_data['department']}\nwhich semester are you in ?", reply_markup= build_menu(options, n_cols=4, prefix= "sem_"))
    return SEM

async def get_sem(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    if query.data == "back":
        options= ["ECE", "EEE", "CSC", "MECH", "CIVIL", "IT"]
        await query.edit_message_text(f"Selected: {context.user_data['stream']}\nWhat is your Department ?", reply_markup = build_menu(options, n_cols=3, prefix= "dept_"))
        return DEPARTMENT
    # Logic: only accept sem_ prefixes
    if not query.data.startswith("sem_"):
        return SEM
# Store the data from the button click
    context.user_data['sem'] = int(query.data.replace("sem_", ""))
    context.user_data['student_id'] = update.effective_user.id
    #save to DB
    register_user_to_db(context.user_data)

    await query.edit_message_text(
        text=( 
            "✅ <b>Registration Complete!</b>\n\n"
            f"👤 <b>Name:</b> {context.user_data['student_name']}\n"
            f"🏫 <b>University:</b> {context.user_data['university']}\n"
            f"🎓 <b>Stream:</b> {context.user_data['stream']}\n"
            f"🏢 <b>Dept:</b> {context.user_data['department']}\n"
            f"📅 <b>Semester:</b> {context.user_data['sem']}\n\n"
            "👉 <i>Use /learn command to Start learning sessions.</i>"
        ),parse_mode="HTML")
    context.user_data.clear()
    return ConversationHandler.END


async def edit_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🔄 <b>Profile Edit Mode</b>\n\n"
        "Let's update your info. \nWhat is your full name?",
        parse_mode="HTML"
    )
    return NAME

async def profile_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_data = get_user_profile(update.effective_user.id)
    
    if not user_data:
        await update.message.reply_text("❌ You are not registered yet. Use /register to start!")
        return

    name, uni, stream, dept, sem = user_data
    
    profile_text = (
        "👤 <b>Your Student Profile</b>\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📛 <b>Name:</b> {name}\n"
        f"🏫 <b>University:</b> {uni}\n"
        f"🎓 <b>Stream:</b> {stream}\n"
        f"🏢 <b>Dept:</b> {dept}\n"
        f"📅 <b>Semester:</b> {sem}\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "👉 <i>Use /edit command to change your details.</i>"
    )
    
    await update.message.reply_text(profile_text, parse_mode="HTML")


#------------------UNREGISTERED USERS------------------

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles all text messages."""
    user_student_id = update.effective_user.id
    if not is_user_registered(user_student_id):
        await update.message.reply_text("❌ Please /register first.")
        return
    user_input = update.message.text
    # BUSINESS LOGIC
    try:
        # Retrieve existing history for this specific user
        history = user_memories[user_student_id]
        # Call your groq_client function
        reply = ask_groq(user_student_id, user_input, history)
        # Update the memory with the latest exchange
        user_memories[user_student_id].append({"role": "user", "content": user_input})
        user_memories[user_student_id].append({"role": "assistant", "content": reply})

        if len(user_memories[user_student_id]) > 6:
            user_memories[user_student_id] = user_memories[user_student_id][-6:]

        await update.message.reply_text(reply)
    except Exception as e:
        print(f"Error calling Groq: {e}")
        await update.message.reply_text("⚠️ Error connecting to AI.")

def main():
    app = Application.builder().token(BOT_TOKEN).post_init(post_init).build()

    # Conversation Handler for Registration
    conv_handler = ConversationHandler(
        entry_points = [CommandHandler("register", register_command),
                        CommandHandler("edit", edit_command)],
        states= {
            NAME: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_name)],
            UNIVERSITY: [CallbackQueryHandler(get_university)],
            STREAM: [CallbackQueryHandler(get_stream)],
            DEPARTMENT: [CallbackQueryHandler(get_dept)],
            SEM: [CallbackQueryHandler(get_sem)],
        },
        fallbacks= [],
        conversation_timeout=600)

    app.add_handler(CommandHandler("start",start_command)) # 1. Start Command
    app.add_handler(conv_handler) # 2. Registration Flow
    app.add_handler(CommandHandler("learn", learn_command)) # 3. Learning
    app.add_handler(CallbackQueryHandler(task_completed_callback, pattern ="^task_done$")) # 4. Task Progression
    app.add_handler(CommandHandler("profile", profile_command))  # 5. View Profile
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("syllabus", syllabus_command))
    app.add_handler(CommandHandler("status", status_command))
    # 6. AI Message Handler (Must be last)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    print("Bot running... Press Ctrl+c to stop.")
    app.run_polling()

if __name__ == "__main__":
    main()
