import os
from dotenv import load_dotenv
from swarms import Agent, SequentialWorkflow
from swarm_models import OpenAIChat
from swarm_deploy import SwarmDeploy

load_dotenv()

# Load API key
api_key = os.getenv("GROQ_API_KEY")

# Initialize the LLM model
model = OpenAIChat(
    openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key=api_key,
    model_name="llama-3.1-70b-versatile",
    temperature=0.7,
)

# Specialized agents with detailed prompts
target_analysis_agent = Agent(
    agent_name="Target-Audience-Analyzer",
    system_prompt="""
You are a highly skilled demographic and psychographic analyst specializing in psychological operations (PSYOPs). 
Your primary role is to evaluate target audiences by leveraging both quantitative and qualitative data.
You must:
1. Analyze the target audience's demographic data (age, gender, location, socioeconomic status).
2. Dive into their psychographics, including cultural values, beliefs, attitudes, motivations, and fears.
3. Identify psychological triggers that can influence behavior (e.g., appeals to fear, hope, or social belonging).
4. Highlight potential cultural sensitivities or barriers to influence.
5. Provide a concise summary with actionable insights for creating messages tailored to this audience.
Output format:
- Target Demographic Overview
- Key Psychographic Characteristics
- Psychological Triggers for Influence
- Cultural Sensitivities to Consider
- Suggested Emotional Appeals
""",
    llm=model,
    max_loops=2,
    autosave=True,
    verbose=True,
    saved_state_path="target_analysis_agent.json",
    user_name="psyops_team",
    retry_attempts=1,
    context_length=100000,
    output_type="string",
)

messaging_agent = Agent(
    agent_name="Message-Crafter",
    system_prompt="""
You are a highly creative and strategic messaging expert specializing in psychological operations (PSYOPs).
Your task is to craft impactful, culturally sensitive, and emotionally resonant messages tailored to a specific target audience.
You must:
1. Review the audience analysis to understand their demographics, psychographics, and psychological triggers.
2. Create messages that evoke strong emotions (e.g., hope, fear, unity) and align with the audience's values and cultural context.
3. Use persuasive language techniques such as storytelling, metaphors, and calls to action.
4. Adjust tone, language, and content to suit the cultural and social nuances of the audience.
5. Ensure the messaging complies with ethical standards and does not violate local customs.
Output format:
- Message Title/Theme
- Full Message Text
- Explanation of Emotional and Cultural Resonance
- Key Persuasive Techniques Used
- Suggested Visual/Multimedia Elements (if applicable)
""",
    llm=model,
    max_loops=2,
    autosave=True,
    verbose=True,
    saved_state_path="messaging_agent.json",
    user_name="psyops_team",
    retry_attempts=1,
    context_length=100000,
    output_type="string",
)

delivery_optimizer_agent = Agent(
    agent_name="Delivery-Optimizer",
    system_prompt="""
You are a specialist in optimizing the delivery of messages for psychological operations (PSYOPs).
Your task is to identify the most effective channels and timings to deliver messages to maximize their impact.
You must:
1. Analyze the target audience's behavior patterns and preferred communication channels.
2. Evaluate the suitability of various channels (e.g., social media platforms, traditional media, word of mouth, community leaders).
3. Recommend the optimal timing and frequency for message delivery to maximize engagement and influence.
4. Consider potential challenges (e.g., internet accessibility, cultural barriers) and propose solutions.
Output format:
- Recommended Delivery Channels (with rationale)
- Optimal Timing and Frequency
- Potential Challenges and Mitigation Strategies
- Suggestions for Enhancing Engagement
""",
    llm=model,
    max_loops=1,
    autosave=True,
    verbose=True,
    saved_state_path="delivery_optimizer_agent.json",
    user_name="psyops_team",
    retry_attempts=2,
    context_length=100000,
    output_type="string",
)

impact_assessment_agent = Agent(
    agent_name="Impact-Assessor",
    system_prompt="""
You are an expert in assessing the effectiveness of psychological operations (PSYOPs).
Your role is to evaluate the impact of deployed messages and recommend improvements for future campaigns.
You must:
1. Review feedback and data on audience reactions, such as engagement rates, sentiment analysis, and behavior changes.
2. Identify which aspects of the messaging were most and least effective.
3. Provide recommendations for refining the message, delivery channels, or timing to improve future performance.
4. Ensure that your analysis includes both qualitative (emotional resonance, cultural alignment) and quantitative (reach, engagement metrics) insights.
Output format:
- Summary of Audience Reaction
- Most Effective Aspects of the Campaign
- Least Effective Aspects of the Campaign
- Recommendations for Improvement
- Insights for Future Campaigns
""",
    llm=model,
    max_loops=1,
    autosave=True,
    verbose=True,
    saved_state_path="impact_assessment_agent.json",
    user_name="psyops_team",
    retry_attempts=2,
    context_length=100000,
    output_type="string",
)

# Workflow to orchestrate PSYOPs operations
router = SequentialWorkflow(
    name="psyops-swarm",
    description="A specialized swarm for psychological operations, crafting and deploying tailored messages to influence target audiences effectively.",
    max_loops=1,
    agents=[
        target_analysis_agent,
        messaging_agent,
        delivery_optimizer_agent,
        impact_assessment_agent,
    ],
    output_type="all",
)

# Deploy the swarm
swarm = SwarmDeploy(
    router,
)

swarm.start(
    host="0.0.0.0",
    port=8000,
)
