from dotenv import load_dotenv
import gradio as gr
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent
import logging

# Load environment variables
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize database and LLM components
db = SQLDatabase.from_uri("sqlite:///meaningful_database_1.db")
llm = ChatOpenAI(temperature=0, model="gpt-4o")

# custom_prefix = """
# The names of the tables in this database are: 
# T1 = Patients
# T2 = Doctors
# T3 = Hospitals

# The names of the columns in this Patients table are: PatientID, Age, Height, Weight, BloodPressure, HeartRate, Cholesterol, BloodSugar, Temperature, OxygenSaturation, VisitDate, DiagnosisCode, TreatmentCode, InsuranceNumber, DoctorID, HospitalID, Allergies, Medications, Symptoms, Notes
# The names of the columns in the Doctors table are: DoctorID, Name, Specialization, ExperienceYears, PatientsTreated, AvailableDays, Rating, Salary, PhoneNumber, Email, HospitalID, ResearchPapers, Certifications, Awards, ShiftTiming, Country, State, City, PostalCode, LicenseNumber
# The names of the columns in the Hospitals table are: HospitalID, Name, Location, TotalBeds, OccupiedBeds, Departments, DoctorsCount, Rating, EmergencyAvailable, Ambulances, Contact, Email, Director, EstablishedYear, Revenue, City, State, PostalCode, Country, Website
# """

agent_executor = create_sql_agent(
    llm,
    db=db,
    agent_type="openai-tools",
    verbose=True,
    agent_executor_kwargs={"return_intermediate_steps": True},
    #prefix=custom_prefix
)

# Store chat history
chat_history = []

def respond(message, history):
    """
    Process user input and return response from the SQL agent
    """
    try:
        # Combine chat history into a single string for context
        context = "\n".join([f"User: {msg}\nAgent: {resp}" for msg, resp in chat_history])
        full_message = f"{context}\nUser: {message}"

        response = agent_executor.invoke(full_message)

        # Add to chat history
        chat_history.append((message, response["output"]))

        return response["output"]
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        return "An error occurred while processing your request. Please try again."

# Create Gradio interface with enhanced theming and examples
demo = gr.ChatInterface(
    respond,
    title="Talk to your Database",
    description="Ask questions about your database in natural language.",
    examples=[
        "How many tables there are in the database?",
        "How many beds there are in total?",
        "What is the weight of patient9?",
    ],
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="green",
    ),
    css="""
    body {
        background-color: #f0f0f0;
        font-family: Arial, sans-serif;
    }
    .gradio-container {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .gradio-header {
        background-color: #4CAF50;
        color: white;
        padding: 10px;
        border-radius: 10px 10px 0 0;
    }
    .gradio-footer {
        background-color: #4CAF50;
        color: white;
        padding: 10px;
        border-radius: 0 0 10px 10px;
    }
    """
)

if __name__ == "__main__":
    demo.launch(
        server_name="localhost",  # Makes the app accessible from other devices on the network
        server_port=7860,  # Specify a port
        share=True,  # Creates a public URL (optional)
    )