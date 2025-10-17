import os
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.agents.models import FabricTool, ListSortOrder
from dotenv import load_dotenv
load_dotenv()

project_client = AIProjectClient(
    endpoint=os.environ.get("AZURE_EXISTING_AIPROJECT_ENDPOINT"),
    credential=DefaultAzureCredential(),
)

# Fabric connection
fabric_connection_name = "MortgageModelSM"
print(f"Fabric Connection Name: {fabric_connection_name}")
conn_id = project_client.connections.get(fabric_connection_name).id
print(conn_id)

fabric = FabricTool(connection_id=conn_id)

# Retrieve existing agent
agent_id = os.environ.get("AZURE_EXISTING_AGENT_ID")
agents_client = project_client.agents
agent = agents_client.get_agent(agent_id)

# Create thread
thread = agents_client.threads.create()
print(f"Created thread, ID: {thread.id}")

def send_and_log(query: str) -> None:
    msg = agents_client.messages.create(
        thread_id=thread.id,
        role="user",
        content=query,
    )
    print(f"Created message, ID: {msg.id}")

    run_local = agents_client.runs.create_and_process(
        thread_id=thread.id,
        agent_id=agent.id
    )
    print(f"Run finished with status: {run_local.status}")

    if run_local.status == "failed":
        print(f"Run failed: {run_local.last_error}")
        return

    recent_messages = agents_client.messages.list(
        thread_id=thread.id,
        order=ListSortOrder.DESCENDING
    )
    for m in recent_messages:
        if m.role == "assistant" and m.text_messages:
            latest_text = m.text_messages[-1]
            print(f"assistant: {latest_text.text.value}")
            break

# Run twice
prompt_text = "show me UPB by property type desc"
send_and_log(prompt_text)
send_and_log(prompt_text)

# Delete agent (only if you really want to!)
# agents_client.delete_agent(agent.id)
# print("Deleted agent")

# Print all messages
messages = agents_client.messages.list(
    thread_id=thread.id,
    order=ListSortOrder.ASCENDING
)
for msg in messages:
    if msg.text_messages:
        last_text = msg.text_messages[-1]
        print(f"{msg.role}: {last_text.text.value}")
