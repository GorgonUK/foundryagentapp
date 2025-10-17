# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.

import asyncio
import json
import os
from typing import AsyncGenerator, Optional, Dict

import fastapi
from fastapi import Request, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse

import logging
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from azure.ai.agents.aio import AgentsClient
from azure.ai.agents.models import (
    Agent,
    MessageDeltaChunk,
    ThreadMessage,
    ThreadRun,
    AsyncAgentEventHandler,
    RunStep
)
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
   AgentEvaluationRequest,
   AgentEvaluationSamplingConfiguration,
   AgentEvaluationRedactionConfiguration,
   EvaluatorIds
)


# Create a logger for this module
logger = logging.getLogger("azureaiapp")

# Set the log level for the azure HTTP logging policy to WARNING (or ERROR)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)

from opentelemetry import trace
tracer = trace.get_tracer(__name__)

# Define the directory for your templates.
directory = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=directory)

# Create a new FastAPI router
router = fastapi.APIRouter()

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from typing import Optional
import secrets

security = HTTPBasic()

username = os.getenv("WEB_APP_USERNAME")
password = os.getenv("WEB_APP_PASSWORD")
basic_auth = username and password

def authenticate(credentials: Optional[HTTPBasicCredentials] = Depends(security)) -> None:

    if not basic_auth:
        logger.info("Skipping authentication: WEB_APP_USERNAME or WEB_APP_PASSWORD not set.")
        return
    
    correct_username = secrets.compare_digest(credentials.username, username)
    correct_password = secrets.compare_digest(credentials.password, password)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return

auth_dependency = Depends(authenticate) if basic_auth else None


def get_ai_project(request: Request) -> AIProjectClient:
    return request.app.state.ai_project

def get_agent_client(request: Request) -> AgentsClient:
    return request.app.state.agent_client

def get_agent(request: Request) -> Agent:
    return request.app.state.agent

def get_app_insights_conn_str(request: Request) -> str:
    if hasattr(request.app.state, "application_insights_connection_string"):
        return request.app.state.application_insights_connection_string
    else:
        return None

def serialize_sse_event(data: Dict) -> str:
    return f"data: {json.dumps(data)}\n\n"

@router.get("/speech/token")
async def get_speech_token(_ = auth_dependency):
    """Return short-lived Azure Speech token and region. Never expose the key to the client."""
    region = os.environ.get("AZURE_SPEECH_REGION", "").strip()
    key = os.environ.get("AZURE_SPEECH_KEY", "").strip()
    if not region or not key:
        raise HTTPException(status_code=400, detail="Speech region/key not configured")
    token_url = f"https://{region}.api.cognitive.microsoft.com/sts/v1.0/issueToken"
    headers = {"Ocp-Apim-Subscription-Key": key}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(token_url, headers=headers) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise HTTPException(status_code=500, detail=f"Failed to get token: {resp.status} {text}")
                token = await resp.text()
                return JSONResponse({"token": token, "region": region})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching speech token: {e}")
        raise HTTPException(status_code=500, detail="Failed to get speech token")

async def get_message_and_annotations(agent_client : AgentsClient, message: ThreadMessage) -> Dict:
    annotations = []
    # Get file annotations for the file search.
    for annotation in (a.as_dict() for a in message.file_citation_annotations):
        file_id = annotation["file_citation"]["file_id"]
        logger.info(f"Fetching file with ID for annotation {file_id}")
        openai_file = await agent_client.files.get(file_id)
        annotation["file_name"] = openai_file.filename
        logger.info(f"File name for annotation: {annotation['file_name']}")
        annotations.append(annotation)

    # Get url annotation for the index search.
    for url_annotation in message.url_citation_annotations:
        annotation = url_annotation.as_dict()
        annotation["file_name"] = annotation['url_citation']['title']
        logger.info(f"File name for annotation: {annotation['file_name']}")
        annotations.append(annotation)
            
    return {
        'content': message.text_messages[0].text.value,
        'annotations': annotations
    }

class MyEventHandler(AsyncAgentEventHandler[str]):
    def __init__(self, ai_project: AIProjectClient, app_insights_conn_str: str):
        super().__init__()
        self.agent_client = ai_project.agents
        self.ai_project = ai_project
        self.app_insights_conn_str = app_insights_conn_str
        # Toggle automated evaluation via env to avoid rate limiting
        self.eval_enabled = str(os.getenv("ENABLE_AGENT_EVALUATION", "")).lower() == "true"
        # Track processed runs to avoid duplicate evaluations on repeated events
        if not hasattr(MyEventHandler, "_evaluated_run_ids"):
            MyEventHandler._evaluated_run_ids = set()  # type: ignore[attr-defined]

    async def on_message_delta(self, delta: MessageDeltaChunk) -> Optional[str]:
        stream_data = {'content': delta.text, 'type': "message"}
        return serialize_sse_event(stream_data)

    async def on_thread_message(self, message: ThreadMessage) -> Optional[str]:
        try:
            logger.info(f"MyEventHandler: Received thread message, message ID: {message.id}, status: {message.status}")
            if message.status != "completed":
                return None

            logger.info("MyEventHandler: Received completed message")

            stream_data = await get_message_and_annotations(self.agent_client, message)
            stream_data['type'] = "completed_message"
            return serialize_sse_event(stream_data)
        except Exception as e:
            logger.error(f"Error in event handler for thread message: {e}", exc_info=True)
            return None

    async def on_thread_run(self, run: ThreadRun) -> Optional[str]:
        logger.info("MyEventHandler: on_thread_run event received")
        run_information = f"ThreadRun status: {run.status}, thread ID: {run.thread_id}"
        stream_data = {'content': run_information, 'type': 'thread_run'}
        if run.status == "failed":
            stream_data['error'] = run.last_error.as_dict()
        # Optionally run agent evaluation when the run is completed, with dedupe
        if run.status == "completed" and self.eval_enabled:
            evaluated: Set[str] = getattr(MyEventHandler, "_evaluated_run_ids")  # type: ignore[attr-defined]
            if run.id not in evaluated:
                evaluated.add(run.id)
                run_agent_evaluation(run.thread_id, run.id, self.ai_project, self.app_insights_conn_str)
        return serialize_sse_event(stream_data)

    async def on_error(self, data: str) -> Optional[str]:
        logger.error(f"MyEventHandler: on_error event received: {data}")
        stream_data = {'type': "stream_end"}
        return serialize_sse_event(stream_data)

    async def on_done(self) -> Optional[str]:
        logger.info("MyEventHandler: on_done event received")
        stream_data = {'type': "stream_end"}
        return serialize_sse_event(stream_data)

    async def on_run_step(self, step: RunStep) -> Optional[str]:
        logger.info(f"Step {step['id']} status: {step['status']}")
        step_details = step.get("step_details", {})
        tool_calls = step_details.get("tool_calls", [])

        if tool_calls:
            logger.info("Tool calls:")
            for call in tool_calls:
                azure_ai_search_details = call.get("azure_ai_search", {})
                if azure_ai_search_details:
                    logger.info(f"azure_ai_search input: {azure_ai_search_details.get('input')}")
                    logger.info(f"azure_ai_search output: {azure_ai_search_details.get('output')}")
        return None

@router.get("/", response_class=HTMLResponse)
async def index(request: Request, _ = auth_dependency):
    return templates.TemplateResponse(
        "index.html", 
        {
            "request": request,
        }
    )


async def get_result(
    request: Request, 
    thread_id: str, 
    agent_id: str, 
    ai_project: AIProjectClient,
    app_insight_conn_str: Optional[str], 
    carrier: Dict[str, str]
) -> AsyncGenerator[str, None]:
    ctx = TraceContextTextMapPropagator().extract(carrier=carrier)
    with tracer.start_as_current_span('get_result', context=ctx):
        logger.info(f"get_result invoked for thread_id={thread_id} and agent_id={agent_id}")
        try:
            agent_client = ai_project.agents
            async with await agent_client.runs.stream(
                thread_id=thread_id, 
                agent_id=agent_id,
                event_handler=MyEventHandler(ai_project, app_insight_conn_str),
            ) as stream:
                logger.info("Successfully created stream; starting to process events")
                async for event in stream:
                    _, _, event_func_return_val = event
                    logger.debug(f"Received event: {event}")
                    if event_func_return_val:
                        logger.info(f"Yielding event: {event_func_return_val}")
                        yield event_func_return_val
                    else:
                        logger.debug("Event received but no data to yield")
        except Exception as e:
            logger.exception(f"Exception in get_result: {e}")
            yield serialize_sse_event({'type': "error", 'message': str(e)})


@router.get("/chat/history")
async def history(
    request: Request,
    ai_project : AIProjectClient = Depends(get_ai_project),
    agent : Agent = Depends(get_agent),
	_ = auth_dependency
):
    with tracer.start_as_current_span("chat_history"):
        # Retrieve the thread ID from the cookies (if available).
        thread_id = request.cookies.get('thread_id')
        agent_id = request.cookies.get('agent_id')

        # Attempt to get an existing thread. If not found, create a new one.
        try:
            agent_client = ai_project.agents
            if thread_id and agent_id == agent.id:
                logger.info(f"Retrieving thread with ID {thread_id}")
                thread = await agent_client.threads.get(thread_id)
            else:
                logger.info("Creating a new thread")
                thread = await agent_client.threads.create()
        except Exception as e:
            logger.error(f"Error handling thread: {e}")
            raise HTTPException(status_code=400, detail=f"Error handling thread: {e}")

        thread_id = thread.id
        agent_id = agent.id

    # Create a new message from the user's input.
    try:
        content = []
        response = agent_client.messages.list(
            thread_id=thread_id,
        )
        async for message in response:
            formatteded_message = await get_message_and_annotations(agent_client, message)
            formatteded_message['role'] = message.role
            formatteded_message['created_at'] = message.created_at.astimezone().strftime("%m/%d/%y, %I:%M %p")
            content.append(formatteded_message)
                
                                        
        logger.info(f"List message, thread ID: {thread_id}")
        response = JSONResponse(content=content)
    
        # Update cookies to persist the thread and agent IDs.
        response.set_cookie("thread_id", thread_id)
        response.set_cookie("agent_id", agent_id)
        return response
    except Exception as e:
        logger.error(f"Error listing message: {e}")
        raise HTTPException(status_code=500, detail=f"Error list message: {e}")

@router.get("/agent")
async def get_chat_agent(
    request: Request
):
    return JSONResponse(content=get_agent(request).as_dict())  

@router.websocket("/voice/live")
async def voice_live(ws: WebSocket):
    """Bi-directional live voice session using Azure Voice Live.
    Client sends PCM16 mono frames at 24000 Hz as binary messages.
    Server streams partial/final transcripts back as JSON text frames.
    { type: "partial" | "final", text: string }
    """
    await ws.accept()

    endpoint = os.environ.get("AZURE_VOICELIVE_ENDPOINT", "").strip()
    model = os.environ.get("AZURE_VOICELIVE_MODEL", "").strip()
    api_key = os.environ.get("AZURE_VOICELIVE_API_KEY", "").strip()
    if not endpoint or not model:
        await ws.send_text(py_json.dumps({"type": "error", "message": "VoiceLive endpoint/model not configured"}))
        await ws.close()
        return

    # Prefer Azure token credential when no API key is provided
    credential = AzureKeyCredential(api_key) if api_key else DefaultAzureCredential()

    try:
        async with voicelive_connect(
            endpoint=endpoint,
            credential=credential,
            model=model,
            connection_options={
                "max_msg_size": 10 * 1024 * 1024,
                "heartbeat": 20,
                "timeout": 20,
            },
        ) as conn:
            # Configure session: audio in/out and simple VAD
            session = RequestSession(
                modalities=[Modality.TEXT, Modality.AUDIO],
                instructions=os.environ.get("AZURE_VOICELIVE_INSTRUCTIONS", "You are a helpful assistant."),
                voice=AzureStandardVoice(name=os.environ.get("AZURE_VOICELIVE_VOICE", "en-US-Ava:DragonHDLatestNeural"), type="azure-standard"),
                input_audio_format=VOICELIVE_INPUT_FMT,
                output_audio_format=VOICELIVE_OUTPUT_FMT,
                turn_detection=ServerVad(threshold=0.5, prefix_padding_ms=300, silence_duration_ms=500),
            )
            await conn.session.update(session=session)

            # Relay VoiceLive server events to client (partial/final transcripts only)
            async def relay_events():
                async for event in conn:
                    et = getattr(event, "type", None)
                    # Handle both enum and string event types
                    if et == getattr(ServerEventType, "RESPONSE_TRANSCRIPT_DELTA", object()) or et == "response.transcript.delta":
                        # partial transcript
                        await ws.send_text(py_json.dumps({"type": "partial", "text": getattr(event, "delta", "")}))
                    elif et == getattr(ServerEventType, "RESPONSE_TRANSCRIPT_COMPLETED", object()) or et == "response.transcript.completed":
                        await ws.send_text(py_json.dumps({"type": "final", "text": getattr(event, "transcript", "")}))
                    elif et == getattr(ServerEventType, "RESPONSE_CREATED", object()) or et == "response.created":
                        with contextlib.suppress(Exception):
                            await ws.send_text(py_json.dumps({"type": "info", "message": "response.created"}))
                    elif et == getattr(ServerEventType, "RESPONSE_AUDIO_DELTA", object()) or et == "response.audio.delta":
                        # stream audio frames to client as base64 to avoid binary WS framing issues
                        try:
                            audio_bytes = getattr(event, "delta", b"")
                            if isinstance(audio_bytes, (bytes, bytearray)) and audio_bytes:
                                audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                                await ws.send_text(py_json.dumps({"type": "audio", "data": audio_b64}))
                        except Exception:
                            pass
                    elif et == getattr(ServerEventType, "RESPONSE_AUDIO_DONE", object()) or et == "response.audio.done":
                        with contextlib.suppress(Exception):
                            await ws.send_text(py_json.dumps({"type": "audio_done"}))
                    elif et == getattr(ServerEventType, "RESPONSE_DONE", object()) or et == "response.done":
                        with contextlib.suppress(Exception):
                            await ws.send_text(py_json.dumps({"type": "done"}))
                    elif et == getattr(ServerEventType, "ERROR", object()) or et == "error":
                        with contextlib.suppress(Exception):
                            await ws.send_text(py_json.dumps({"type": "error", "message": getattr(event, "error", getattr(event, "message", ""))}))

            relay_task = asyncio.create_task(relay_events())

            try:
                while True:
                    message = await ws.receive()
                    if message.get("type") == "websocket.disconnect":
                        break
                    if "bytes" in message and message["bytes"] is not None:
                        # Binary PCM16 24kHz mono
                        # VoiceLive expects base64 audio chunks for input buffer
                        raw = message["bytes"]
                        b64 = base64.b64encode(raw).decode("utf-8")
                        await conn.input_audio_buffer.append(audio=b64)
                    elif "text" in message and message["text"] is not None:
                        try:
                            payload = py_json.loads(message["text"]) if message["text"] else {}
                        except Exception:
                            payload = {}
                        if payload.get("type") == "stop":
                            # Signal end of user turn and prompt model to respond, keep socket open for events
                            with contextlib.suppress(Exception):
                                await conn.input_audio_buffer.commit()
                            with contextlib.suppress(Exception):
                                await conn.response.create()
                            continue
            except WebSocketDisconnect:
                pass
            finally:
                with contextlib.suppress(Exception):
                    relay_task.cancel()
                with contextlib.suppress(Exception):
                    await ws.close()
    except Exception as e:
        logger.error(f"Error in /voice/live: {e}")
        with contextlib.suppress(Exception):
            await ws.send_text(py_json.dumps({"type": "error", "message": "Server error"}))
            await ws.close()

@router.post("/chat")
async def chat(
    request: Request,
    agent : Agent = Depends(get_agent),
    ai_project: AIProjectClient = Depends(get_ai_project),
    app_insights_conn_str : str = Depends(get_app_insights_conn_str),
	_ = auth_dependency
):
    # Retrieve the thread ID from the cookies (if available).
    thread_id = request.cookies.get('thread_id')
    agent_id = request.cookies.get('agent_id')

    with tracer.start_as_current_span("chat_request"):
        carrier = {}        
        TraceContextTextMapPropagator().inject(carrier)
        
        # Attempt to get an existing thread. If not found, create a new one.
        try:
            agent_client = ai_project.agents
            if thread_id and agent_id == agent.id:
                logger.info(f"Retrieving thread with ID {thread_id}")
                thread = await agent_client.threads.get(thread_id)
            else:
                logger.info("Creating a new thread")
                thread = await agent_client.threads.create()
        except Exception as e:
            logger.error(f"Error handling thread: {e}")
            raise HTTPException(status_code=400, detail=f"Error handling thread: {e}")

        thread_id = thread.id
        agent_id = agent.id

        # Parse the JSON from the request.
        try:
            user_message = await request.json()
        except Exception as e:
            logger.error(f"Invalid JSON in request: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid JSON in request: {e}")

        logger.info(f"user_message: {user_message}")

        # Create a new message from the user's input.
        try:
            message = await agent_client.messages.create(
                thread_id=thread_id,
                role="user",
                content=user_message.get('message', '')
            )
            logger.info(f"Created message, message ID: {message.id}")
        except Exception as e:
            logger.error(f"Error creating message: {e}")
            raise HTTPException(status_code=500, detail=f"Error creating message: {e}")

        # Set the Server-Sent Events (SSE) response headers.
        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
        logger.info(f"Starting streaming response for thread ID {thread_id}")

        # Create the streaming response using the generator.
        response = StreamingResponse(get_result(request, thread_id, agent_id, ai_project, app_insights_conn_str, carrier), headers=headers)

        # Update cookies to persist the thread and agent IDs.
        response.set_cookie("thread_id", thread_id)
        response.set_cookie("agent_id", agent_id)
        return response

def read_file(path: str) -> str:
    with open(path, 'r') as file:
        return file.read()


def run_agent_evaluation(
    thread_id: str, 
    run_id: str,
    ai_project: AIProjectClient,
    app_insights_conn_str: str):

    if app_insights_conn_str:
        agent_evaluation_request = AgentEvaluationRequest(
            run_id=run_id,
            thread_id=thread_id,
            evaluators={
                "Relevance": {"Id": EvaluatorIds.RELEVANCE.value},
                "TaskAdherence": {"Id": EvaluatorIds.TASK_ADHERENCE.value},
                "ToolCallAccuracy": {"Id": EvaluatorIds.TOOL_CALL_ACCURACY.value},
            },
            sampling_configuration=AgentEvaluationSamplingConfiguration(
                name="default",
                sampling_percent=100,
            ),
            redaction_configuration=AgentEvaluationRedactionConfiguration(
                redact_score_properties=False,
            ),
            app_insights_connection_string=app_insights_conn_str,
        )
        
        async def run_evaluation():
            try:        
                logger.info(f"Running agent evaluation on thread ID {thread_id} and run ID {run_id}")
                agent_evaluation_response = await ai_project.evaluations.create_agent_evaluation(
                    evaluation=agent_evaluation_request
                )
                logger.info(f"Evaluation response: {agent_evaluation_response}")
            except Exception as e:
                logger.error(f"Error creating agent evaluation: {e}")

        # Create a new task to run the evaluation asynchronously
        asyncio.create_task(run_evaluation())


@router.get("/config/azure")
async def get_azure_config(_ = auth_dependency):
    """Get Azure configuration for frontend use"""
    try:
        subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID", "")
        tenant_id = os.environ.get("AZURE_TENANT_ID", "")
        resource_group = os.environ.get("AZURE_RESOURCE_GROUP", "")
        ai_project_resource_id = os.environ.get("AZURE_EXISTING_AIPROJECT_RESOURCE_ID", "")
        
        # Extract resource name and project name from the resource ID
        # Format: /subscriptions/{sub}/resourceGroups/{rg}/providers/Microsoft.CognitiveServices/accounts/{resource}/projects/{project}
        resource_name = ""
        project_name = ""
        
        if ai_project_resource_id:
            parts = ai_project_resource_id.split("/")
            if len(parts) >= 8:
                resource_name = parts[8]  # accounts/{resource_name}
            if len(parts) >= 10:
                project_name = parts[10]  # projects/{project_name}
        
        return JSONResponse({
            "subscriptionId": subscription_id,
            "tenantId": tenant_id,
            "resourceGroup": resource_group,
            "resourceName": resource_name,
            "projectName": project_name,
            "wsid": ai_project_resource_id
        })
    except Exception as e:
        logger.error(f"Error getting Azure config: {e}")
        raise HTTPException(status_code=500, detail="Failed to get Azure configuration")