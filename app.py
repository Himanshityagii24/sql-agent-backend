import os
import requests
import logging
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
try:
    from langgraph.prebuilt import create_react_agent
except ImportError:
    from langchain.agents import create_react_agent

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})  # Allow all origins for development

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'db', 'sqlite', 'sqlite3'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max

# Global variables
llm = None
db = None
agent = None
current_db_path = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def setup_llm():
    try:
        if os.getenv("OPENAI_API_KEY"):
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model="gpt-4", temperature=0)
        elif os.getenv("ANTHROPIC_API_KEY"):
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0)
        elif os.getenv("GOOGLE_API_KEY"):
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
        else:
            raise ValueError("No API key found")
    except ImportError as e:
        print(f"Missing package: {e}")
        return None

def download_database():
    if os.path.exists("Chinook.db"):
        return True
    
    try:
        url = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
        response = requests.get(url)
        if response.status_code == 200:
            with open("Chinook.db", "wb") as file:
                file.write(response.content)
            return True
        return False
    except Exception as e:
        print(f"Error downloading database: {e}")
        return False

def setup_database(db_path):
    global db, agent
    try:
        db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
        
        if llm and db:
            toolkit = SQLDatabaseToolkit(db=db, llm=llm)
            tools = toolkit.get_tools()
            
            system_prompt = f"""You are an agent designed to interact with a SQL database.
            Given an input question, create a syntactically correct {db.dialect} query to run,
            then look at the results and return the answer. Always limit your query to at most 10 results.
            You can order results by relevant columns. Only ask for relevant columns given the question.
            You MUST double check your query before executing it. If you get an error, rewrite the query.
            DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.).
            To start you should ALWAYS look at the tables in the database to see what you can query.
            Then query the schema of the most relevant tables."""
            
            agent = create_react_agent(llm, tools, prompt=system_prompt)
        
        return db
    except Exception as e:
        print(f"Error setting up database: {e}")
        return None

def initialize_app():
    global llm, current_db_path
    
    if not download_database():
        return False
    
    llm = setup_llm()
    if not llm:
        return False
    
    current_db_path = "Chinook.db"
    if not setup_database(current_db_path):
        return False
    
    return True

@app.route('/')
@app.route('/index.html')
def index():
    try:
        # Try to read index.html from current directory
        with open('index.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return """
        <html>
        <body>
        <h1>Error: index.html not found</h1>
        <p>Please make sure index.html is in the same folder as app.py</p>
        </body>
        </html>
        """, 404

@app.route('/api/health', methods=['GET'])
def health_check():
    try:
        logger.info("Health check called")
        response_data = {
            'status': 'healthy' if (llm and db and agent) else 'not_ready',
            'llm_ready': llm is not None,
            'db_ready': db is not None,
            'agent_ready': agent is not None,
            'current_db': os.path.basename(current_db_path) if current_db_path else None
        }
        logger.info(f"Health check response: {response_data}")
        return jsonify(response_data), 200
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/upload-db', methods=['POST'])
def upload_database():
    global current_db_path
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only .db, .sqlite, .sqlite3 allowed'}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Try to connect to the uploaded database
        test_db = setup_database(filepath)
        if not test_db:
            os.remove(filepath)
            return jsonify({'error': 'Invalid database file'}), 400
        
        current_db_path = filepath
        tables = db.get_usable_table_names()
        
        return jsonify({
            'success': True,
            'filename': filename,
            'tables': tables,
            'message': f'Database uploaded successfully! Found {len(tables)} tables.'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/tables', methods=['GET'])
def get_tables():
    if not db:
        return jsonify({'error': 'Database not initialized'}), 500
    
    try:
        tables = db.get_usable_table_names()
        return jsonify({'tables': tables})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/query', methods=['POST'])
def query_database():
    if not agent:
        return jsonify({'error': 'Agent not initialized'}), 500
    
    try:
        data = request.get_json()
        question = data.get('question', '')
        
        if not question.strip():
            return jsonify({'error': 'Question cannot be empty'}), 400
        
        messages = []
        final_response = ""
        
        for step in agent.stream(
            {"messages": [{"role": "user", "content": question}]},
            stream_mode="values",
        ):
            last_message = step["messages"][-1]
            messages.append({
                'type': last_message.__class__.__name__,
                'content': last_message.content,
                'tool_calls': getattr(last_message, 'tool_calls', [])
            })
            
            if hasattr(last_message, 'content') and last_message.content and not getattr(last_message, 'tool_calls', []):
                final_response = last_message.content
        
        return jsonify({
            'question': question,
            'answer': final_response,
            'messages': messages,
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/load-sample-db', methods=['POST'])
def load_sample_db():
    global current_db_path
    
    try:
        if not os.path.exists("Chinook.db"):
            if not download_database():
                return jsonify({'error': 'Failed to download sample database'}), 500
        
        current_db_path = "Chinook.db"
        if not setup_database(current_db_path):
            return jsonify({'error': 'Failed to load sample database'}), 500
        
        tables = db.get_usable_table_names()
        
        return jsonify({
            'success': True,
            'filename': 'Chinook.db',
            'tables': tables,
            'message': f'Sample database loaded! Found {len(tables)} tables.'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sample-questions', methods=['GET'])
def get_sample_questions():
    questions = [
        "Which genre on average has the longest tracks?",
        "What are the top 5 most popular artists by number of tracks?",
        "How many customers are there in each country?",
        "What is the total revenue for each genre?",
        "Which artist has the most albums?",
    ]
    return jsonify({'questions': questions})

if __name__ == '__main__':
    if initialize_app():
        print("🌐 Server running at: http://localhost:8080")
        app.run(debug=True, host='0.0.0.0', port=8080)
    else:
        print("❌ Failed to initialize app")