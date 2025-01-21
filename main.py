from flask import Flask, render_template, request, send_from_directory
import os
import subprocess
from langchain_experimental.sql import SQLDatabaseChain
from langchain.utilities import SQLDatabase
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate, SemanticSimilarityExampleSelector
from langchain.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
import mysql.connector
import json
import pandas as pd
from datetime import datetime
import time
import shutil
from pathlib import Path
import importlib
from contextlib import contextmanager
from typing import Optional
import grpc

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()

# Database connection config
db_user = "root"
db_password = "Anantgod100"
db_host = "localhost:3306"
db_name = "llm_db"

@contextmanager
def timeout_context(timeout_seconds: int = 30):
    """Context manager to handle query timeouts"""
    start_time = time.time()
    yield
    if time.time() - start_time > timeout_seconds:
        raise TimeoutError("Query execution timed out")

def ensure_static_directory():
    """Ensure static directory exists and is clean"""
    static_dir = Path("static")
    if static_dir.exists():
        # Clean the directory
        shutil.rmtree(static_dir)
    static_dir.mkdir(exist_ok=True)

def ensure_sql_server_active():
    """Check if SQL Server is running and attempt to start it if not."""
    try:
        # Try connecting to the database
        conn = mysql.connector.connect(
            host="localhost",
            port=3306,
            user=db_user,
            password=db_password,
        )
        conn.close()
        print("SQL Server is active.")
    except mysql.connector.Error as e:
        print("SQL Server is not active. Attempting to start it...")
        # Command to start SQL Server (adjust based on your system's configuration)
        start_command = "sudo systemctl start mysql"  # For Linux systems
        try:
            subprocess.run(start_command, shell=True, check=True)
            time.sleep(5)  # Give the server time to start
            print("SQL Server started successfully.")
        except subprocess.CalledProcessError as start_error:
            print(f"Failed to start SQL Server: {start_error}")
            raise RuntimeError("SQL Server could not be started. Please start it manually.")

# Ensure SQL Server is active before initializing the Flask app
ensure_sql_server_active()

# Initialize SQL database
db = SQLDatabase.from_uri(
    f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}",
    sample_rows_in_table_info=3,
)

def print_schema_info(database):
    schema_info = database.get_table_info()
    print("Database Schema Information:")
    print(schema_info)

# Call the function to print the schema info
print_schema_info(db)

# Initialize HuggingFace embeddings
model_name = "BAAI/bge-small-en-v1.5"

embeddings = HuggingFaceEmbeddings(
    model_name=model_name
)

def initialize_schema_embeddings(database):
    """Initialize and store schema embeddings in Chroma"""
    schema_info = database.get_table_info()
    schema_texts = []
    metadata = []
    
    for table in database.get_usable_table_names():
        columns = database.get_table_info_no_throw(table)
        schema_text = f"Table: {table}\nColumns: {columns}"
        schema_texts.append(schema_text)
        metadata.append({"table": table})
    
    vectorstore = Chroma.from_texts(
        texts=schema_texts,
        embedding=embeddings,
        metadatas=metadata,
        persist_directory="./schema_store"
    )
    return vectorstore

def get_relevant_tables(question, vectorstore, k=2):
    """Find relevant tables for the given question using similarity search"""
    results = vectorstore.similarity_search_with_relevance_scores(question, k=k)
    return [doc.metadata["table"] for doc, score in results if score > 0.5]

# Initialize schema embeddings
vectorstore = initialize_schema_embeddings(db)

# Set up the LLM model (Google Gemini)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2,
    api_key="AIzaSyDFhzGgknwIfC8_HDaoXZ6Py471c6sDivE",
)

# Enhanced query generation template
schema_aware_query_prompt = PromptTemplate.from_template(
    """Given the following user question and relevant tables, generate a SQL query that answers the question.
    
    Question: {question}
    Relevant Tables: {relevant_tables}
    Full Database Schema: {schema}
    
    Important Guidelines:
    1. Use ONLY the tables mentioned in the Relevant Tables list
    2. Ensure proper JOIN conditions if multiple tables are needed
    3. Include appropriate WHERE clauses and aggregations
    4. Format the query for readability
    
    SQL Query:"""
)

# Define the answer prompt template
answer_prompt = PromptTemplate.from_template(
    """
    You are a highly intelligent AI assistant tasked with analyzing SQL query results to provide precise and insightful answers to user questions.
    Format your response with clear table structure and analysis.

    Question: {question}
    SQL Query: {query}
    SQL Result: {result}

    Format your response in this exact structure:
    1. Create a markdown table with the results
    2. Provide analysis after the table
    3. Ensure the table and analysis are separated by blank lines
    
    Important:
    - Always start with the table header using | Column1 | Column2 | format
    - Use proper table formatting with | separator
    - Include a row of |---| separators after the header
    - Present all numerical data aligned right in the table
    - After the table, provide concise analysis of the key findings
    
    Response Format Example:
    | Department | Count |
    |------------|-------|
    | HR | 10 |
    | IT | 20 |

    Analysis shows that IT has twice as many employees as HR...

    Answer:
    """
)

# Define the visualization prompt template
viz_prompt = PromptTemplate.from_template(
    """Based on this data analysis question and result, recommend the most appropriate visualization type and explain why in 2-3 sentences.
    
    Question: {question}
    Result: {result}
    
    Return your response in this exact format:
    Recommended Graph: [graph type]
    Reasoning: [your explanation]
    
    Ensure your response can be parsed as a plain string and contains no special message objects."""
)

plot_prompt = PromptTemplate.from_template(
    """Given the following question and SQL result, generate Python plotting code using Plotly.
    Ensure the plot type aligns with the question's intent and the data provided:
    - **Bar Plot**: Use when comparing discrete categories (e.g., "What is the number of employees in each department?"). Sort the bars in ascending order by their values. Use the x-axis for categories (e.g., departments) and the y-axis for corresponding values (e.g., number of employees).
    - **Pie Chart**: Use when showing the proportion of parts to a whole (e.g., "What is the percentage of sales by region?"). Create a percentage pie chart with categories as labels and corresponding values for sizes, displaying percentages on the chart.
    - **Histogram**: Use for showing the frequency distribution of a continuous variable (e.g., "What is the distribution of ages among employees?"). Generate bins from the values and display their frequencies.
    - **Line Plot**: Use when visualizing trends or changes over a sequential variable like time or order (e.g., "How does the average salary vary over the years?"). Place the sequential variable (e.g., years) on the x-axis and the metric being tracked (e.g., average salary) on the y-axis. Add markers for each data point and a connecting line to show continuity.
    - **Scatter Plot**: Use when investigating relationships between two continuous variables (e.g., "Is there a relationship between overtime hours and monthly salary?"). Place one variable on the x-axis (e.g., overtime hours) and the other on the y-axis (e.g., monthly salary). Each data point represents one observation. If a third variable (e.g., department) is present, use it for color coding or sizing the markers to add another dimension.

    Question: {question}
    Result: {result}
    Columns: {columns}
    
    Return ONLY the Python code to create and save the plot. Do not include any explanations, markdown formatting, or unrelated plot types.
    The response should be plain Python code that can be directly executed.
    """
)



def safe_execute_query(query: str, db: SQLDatabase) -> Optional[str]:
    """Safely execute SQL query with timeout and error handling"""
    try:
        with timeout_context(30):  # 30 second timeout
            tool = QuerySQLDataBaseTool(db=db)
            result = tool.run(query)
            return result
    except Exception as e:
        print(f"Query execution error: {str(e)}")
        return None

def generate_query_with_schema(inputs):
    """Generate SQL query with schema awareness"""
    relevant_tables = get_relevant_tables(inputs["question"], vectorstore)
    schema_info = db.get_table_info()
    
    query = llm.invoke(
        schema_aware_query_prompt.format(
            question=inputs["question"],
            relevant_tables=relevant_tables,
            schema=schema_info
        )
    ).content
    return query

def clean_query(query):
    query = remove_limit_from_query(query)
    cleaned_query = query.replace("```sql", "").replace("```", "").strip()

    if os.path.exists("generated_sql_queries.sql"):
        os.remove("generated_sql_queries.sql")

    with open("generated_sql_queries.sql", "a") as f:
        f.write(f"-- Generated Query:\n{cleaned_query}\n\n")
    
    return cleaned_query

def remove_limit_from_query(query):
    if "LIMIT" in query.upper():
        query = query.rsplit("LIMIT", 1)[0]
    return query

def get_relevant_columns(db_config, question):
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    
    cursor.execute("""SELECT TABLE_NAME, COLUMN_NAME 
                      FROM INFORMATION_SCHEMA.COLUMNS 
                      WHERE TABLE_SCHEMA = %s""", (db_config['database'],))
    
    columns = cursor.fetchall()
    cursor.close()
    conn.close()
    
    return json.dumps(dict(columns))

def generate_plot_code(llm_response):
    """Generate plot code from LLM response with proper error handling"""
    try:
        # Ensure static directory is ready
        ensure_static_directory()
        
        # Extract the content if it's an AIMessage object
        if hasattr(llm_response, 'content'):
            plot_code = llm_response.content
        else:
            plot_code = str(llm_response)

        print("LLM Response for Plot Code:", plot_code)
        
        # Clean the code
        cleaned_response = plot_code.replace("```python", "").replace("```", "").strip()
        cleaned_response = cleaned_response.replace(
            "fig.write_html(",
            "# fig.write_html("
        ).replace(
            "fig.show()",
            "# fig.show()"
        )
        
        # Add the image saving code
        cleaned_response += '\nfig.write_image("static/plot.png")\n'

        # Remove old generate_plot.py if it exists
        if os.path.exists('generate_plot.py'):
            os.remove('generate_plot.py')

        # Write the complete code to file
        with open('generate_plot.py', 'w') as f:
            f.write("import plotly.express as px\n")
            f.write("import pandas as pd\n\n")
            f.write(cleaned_response)
            
        return cleaned_response
    except Exception as e:
        print(f"Error in generate_plot_code: {str(e)}")
        return None

def save_to_csv(question, answer):
    filename = "llm_answers.csv"
    if os.path.isfile(filename):
        os.remove(filename)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = {"timestamp": [timestamp], "question": [question], "answer": [answer]}
    df = pd.DataFrame(data)
    
    df.to_csv(filename, index=False)

def save_log(question, answer):
    """Save question and answer log to a CSV file"""
    filename = "question_answer_log.csv"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if os.path.isfile(filename):
        df = pd.read_csv(filename)
    else:
        df = pd.DataFrame(columns=["timestamp", "question", "answer"])
    
    new_data = pd.DataFrame({"timestamp": [timestamp], "question": [question], "answer": [answer]})
    df = pd.concat([df, new_data], ignore_index=True)
    df.to_csv(filename, index=False)

def generate_plot_file(plot_code):
    """Generate plot file with proper module reloading"""
    try:
        # Remove existing plot file
        plot_path = 'static/plot.png'
        if os.path.exists(plot_path):
            os.remove(plot_path)

        # Force Python to reload the generate_plot module
        import generate_plot
        importlib.reload(generate_plot)
        
        # Ensure the plot was generated
        if not os.path.exists(plot_path):
            raise Exception("Plot file was not generated")
            
        return plot_path
    except Exception as e:
        print(f"Error generating plot: {str(e)}")
        return None

# Database config
db_config = {
    'host': "localhost",
    'port': 3306,
    'user': "root",
    'password': "Anantgod100",
    'database': "llm_db"
}

# Initialize the complete chain
chain = (
    RunnablePassthrough.assign(query=lambda x: generate_query_with_schema(x))
    .assign(query=lambda context: clean_query(context["query"]))
    .assign(columns=lambda context: get_relevant_columns(db_config, context["question"]))
    .assign(
        result=lambda context: safe_execute_query(context["query"], db)
    )
    .assign(answer=answer_prompt | llm | StrOutputParser())
    .assign(visualization=viz_prompt | llm | StrOutputParser())
    .assign(plot_code=plot_prompt | llm | StrOutputParser() | generate_plot_code)
)

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_answer_and_plot():
    try:
        question = request.form.get('question')
        output_type = request.form.get('output')

        # Execute the chain with the question and handle potential errors
        try:
            result = chain.invoke({"question": question})
            if result.get('result') is None:
                raise Exception("Query execution failed")
        except Exception as e:
            return render_template(
                'result.html',
                error=f"Failed to execute query: {str(e)}",
                show_answer=False,
                show_plot=False
            )

        answer = result['answer'] if output_type in ['answer', 'both'] else None
        plot_code = result['plot_code'] if output_type in ['plot', 'both'] else None

        # Generate the plot file if requested
        plot_path = None
        if plot_code:
            plot_path = generate_plot_file(plot_code)
            if not plot_path:
                return render_template(
                    'result.html',
                    error="Failed to generate plot. Please try again.",
                    answer=answer,
                    show_answer=(output_type in ['answer', 'both']),
                    show_plot=False
                )

        # Save the answer to CSV if generated
        if answer:
            save_to_csv(question, answer)
        save_log(question, answer)

        return render_template(
            'result.html',
            answer=answer,
            plot_path=plot_path,
            show_answer=(output_type in ['answer', 'both']),
            show_plot=(output_type in ['plot', 'both'])
        )

    except Exception as e:
        return render_template(
            'result.html',
            error=f"An error occurred: {str(e)}",
            show_answer=False,
            show_plot=False
        )

@app.route('/history')
def view_history():
    """Render the question-answer history"""
    if os.path.isfile("question_answer_log.csv"):
        df = pd.read_csv("question_answer_log.csv")
        history = df.to_dict(orient="records")
    else:
        history = []
    return render_template('history.html', history=history)

@app.route('/schema')
def view_schemas():
    """Route to display database schemas."""
    schema_info = db.get_table_info()
    return render_template('schema.html', schema_info=schema_info)

@app.route('/static/<filename>')
def serve_plot(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)