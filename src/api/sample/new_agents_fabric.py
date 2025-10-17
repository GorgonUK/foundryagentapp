import os
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.agents.models import FabricTool, ListSortOrder
import os
from dotenv import load_dotenv
load_dotenv()


project_client = AIProjectClient(
    endpoint=os.environ.get("AZURE_EXISTING_AIPROJECT_ENDPOINT"),
    credential=DefaultAzureCredential(),
)

# [START create_agent_with_fabric_tool]
fabric_connection_name = "MortgageModelSM"
print(f"Fabric Connection Name: {fabric_connection_name}")
conn_id = project_client.connections.get(fabric_connection_name).id
# proj_endpoint = os.environ.get("AZURE_EXISTING_AIPROJECT_ENDPOINT")
print(conn_id)

# Initialize an Agent Fabric tool and add the connection id
fabric = FabricTool(connection_id=conn_id)

# Create an Agent with the Fabric tool and process an Agent run
with project_client:
    agents_client = project_client.agents

    agent = agents_client.create_agent(
    model="gpt-4.1",
    name="my-agent",
    instructions="""
You are an AI Foundry agent designed to assist with analytics queries related to mortgage and loan book performance data. You interface directly with the Fabric Data Agent named 'Mortgage Model SM' to retrieve structured data, metadata, and performance metrics.

1. DATA SOURCE INTEGRATION:
- Connect to the Fabric Data Agent 'Mortgage Model SM'.
- Use this for mortgage accounts, loan applications, repayment schedules, interest rates, and performance metrics (delinquency, yield, risk).

2. QUERY HANDLING:
- For mortgage or loan book performance questions, formulate and execute a query against 'Mortgage Model SM'.
- Retrieve tables, relationships, measures, or KPIs and return results in the requested format.

3. OUTPUT FORMAT:
- If a 'data contract' is requested, return JSON with field names, data types, descriptions, and relationships.
- If an 'output template' is referenced, retrieve it from internal metadata and return with placeholders and mappings.
- If no format is specified, respond using rich formatted text.

4. METADATA AWARENESS:
- Be aware of schema, table relationships, column descriptions, measures, KPIs, and data lineage.

5. RESPONSE BEHAVIOUR:
- Always respond in the requested format (JSON, table, markdown).
- If ambiguous, use rich text.
- If a template is referenced by name, retrieve and apply it.

6. GOVERNANCE & SECURITY:
- Respect RLS and security constraints defined in Mortgage Model.
- Do not expose sensitive data unless explicitly permitted.
- Log queries and responses when required.

VOICE MODE (Speech Playground):
- Provide two outputs: 
  - voice_summary (≤ 3 sentences, ~40 words, natural language)
  - text_detail (tables, charts, or full results)
- Round numbers naturally and avoid digit-by-digit speech.
- Do not speak tables or code; only display them in text output.

ADDITIONAL RULES:
- If uncertain, clearly state assumptions.
- For any question about loan balances, mortgage values, loan book size, customer counts, arrears, or KPIs:
  - Query the Fabric Data Agent first.
- Run under the caller’s Entra ID (OBO).

""",
    tools=fabric.definitions,
)
    # [END create_agent_with_fabric_tool]
    print(f"Created Agent, ID: {agent.id}")

    # Create thread for communication
    thread = agents_client.threads.create()
    print(f"Created thread, ID: {thread.id}")

    # Helper to send a query, process run, and log the latest assistant response
    def send_and_log(query: str) -> None:
        msg = agents_client.messages.create(
            thread_id=thread.id,
            role="user",
            content=query,
        )
        print(f"Created message, ID: {msg.id}")

        run_local = agents_client.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)
        print(f"Run finished with status: {run_local.status}")

        if run_local.status == "failed":
            print(f"Run failed: {run_local.last_error}")
            return

        # Fetch most recent assistant message and log it
        recent_messages = agents_client.messages.list(thread_id=thread.id, order=ListSortOrder.DESCENDING)
        for m in recent_messages:
            if m.role == "assistant" and m.text_messages:
                latest_text = m.text_messages[-1]
                print(f"assistant: {latest_text.text.value}")
                break

    # Ask the same question twice and log responses for both runs
    prompt_text = "show me UPB by property type desc"
    send_and_log(prompt_text)
    send_and_log(prompt_text)

    # Delete the Agent when done
    agents_client.delete_agent(agent.id)
    print("Deleted agent")

    # Fetch and log all messages
    messages = agents_client.messages.list(thread_id=thread.id, order=ListSortOrder.ASCENDING)
    for msg in messages:
        if msg.text_messages:
            last_text = msg.text_messages[-1]
            print(f"{msg.role}: {last_text.text.value}")
