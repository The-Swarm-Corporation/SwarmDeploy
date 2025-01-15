import os
from dotenv import load_dotenv
from swarms import Agent, SequentialWorkflow
from swarm_models import OpenAIChat
from swarm_deploy import SwarmDeploy

load_dotenv()

# Get the OpenAI API key from the environment variable
api_key = os.getenv("GROQ_API_KEY")

# Model
model = OpenAIChat(
    openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key=api_key,
    model_name="llama-3.1-70b-versatile",
    temperature=0.1,
)


# Initialize specialized agents
data_extractor_agent = Agent(
    agent_name="Data-Extractor",
    system_prompt="""You are a data extraction specialist. Your role is to carefully analyze documents and extract key information, facts, and data points. Focus on:
- Financial metrics and KPIs
- Market statistics and trends
- Operational details and metrics
- Company background information
- Risk factors and challenges
Present the extracted information in a clear, structured format.""",
    llm=model,
    max_loops=1,
    autosave=True,
    verbose=True,
    dynamic_temperature_enabled=True,
    saved_state_path="data_extractor_agent.json",
    user_name="pe_firm",
    retry_attempts=1,
    context_length=200000,
    output_type="string",
)

summarizer_agent = Agent(
    agent_name="Document-Summarizer",
    system_prompt="""You are a document summarization expert. Your role is to:
- Create concise yet comprehensive summaries of documents
- Highlight the most important points and key takeaways
- Maintain the core message while reducing length
- Organize information in a logical structure
- Ensure all critical details are preserved
Focus on making complex information accessible while retaining all crucial insights.""",
    llm=model,
    max_loops=1,
    autosave=True,
    verbose=True,
    dynamic_temperature_enabled=True,
    saved_state_path="summarizer_agent.json",
    user_name="pe_firm",
    retry_attempts=1,
    context_length=200000,
    output_type="string",
)

financial_analyst_agent = Agent(
    agent_name="Financial-Analyst",
    system_prompt="""You are an expert financial analyst specializing in private equity analysis. Your role is to:
- Analyze financial statements and metrics
- Evaluate company valuation and growth potential
- Assess financial risks and opportunities
- Review cash flow and profitability trends
- Examine debt structure and financing
- Identify potential areas for financial optimization
Provide detailed insights and recommendations based on financial data.""",
    llm=model,
    max_loops=1,
    autosave=True,
    verbose=True,
    dynamic_temperature_enabled=True,
    saved_state_path="financial_analyst_agent.json",
    user_name="pe_firm",
    retry_attempts=1,
    context_length=200000,
    output_type="string",
)

market_analyst_agent = Agent(
    agent_name="Market-Analyst",
    system_prompt="""You are a market analysis expert. Your role is to:
- Analyze industry trends and market dynamics
- Evaluate competitive landscape
- Identify market opportunities and threats
- Assess market size and growth potential
- Review customer segments and needs
- Examine regulatory environment
Provide strategic insights on market positioning and growth opportunities.""",
    llm=model,
    max_loops=1,
    autosave=True,
    verbose=True,
    dynamic_temperature_enabled=True,
    saved_state_path="market_analyst_agent.json",
    user_name="pe_firm",
    retry_attempts=1,
    context_length=200000,
    output_type="string",
)

operational_analyst_agent = Agent(
    agent_name="Operational-Analyst",
    system_prompt="""You are an operational analysis expert. Your role is to:
- Evaluate operational efficiency and processes
- Identify operational risks and bottlenecks
- Assess supply chain and logistics
- Review technology infrastructure
- Examine organizational structure
- Identify potential areas for operational improvement
Provide detailed insights on operational optimization and scaling opportunities.""",
    llm=model,
    max_loops=1,
    autosave=True,
    verbose=True,
    dynamic_temperature_enabled=True,
    saved_state_path="operational_analyst_agent.json",
    user_name="pe_firm",
    retry_attempts=1,
    context_length=200000,
    output_type="string",
)

# Initialize the SwarmRouter
router = SequentialWorkflow(
    name="pe-document-analysis-swarm",
    description="Analyze documents for private equity due diligence and investment decision-making",
    max_loops=1,
    agents=[
        data_extractor_agent,
        summarizer_agent,
        financial_analyst_agent,
        market_analyst_agent,
        operational_analyst_agent,
    ],
    output_type="all",
)


# 5 lines of code to deploy your swarm
# Advanced usage with configuration
swarm = SwarmDeploy(
    router,
)

# Select your host and port!
swarm.start(
    host="0.0.0.0",
    port=8000,
)
