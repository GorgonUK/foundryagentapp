# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.

import contextlib
import os

from azure.ai.projects.aio import AIProjectClient
from azure.identity import DefaultAzureCredential

import fastapi
from fastapi.staticfiles import StaticFiles
from fastapi import Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from logging_config import configure_logging

enable_trace = False
logger = None

@contextlib.asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    agent = None
    ai_project = None

    proj_endpoint = os.environ.get("AZURE_EXISTING_AIPROJECT_ENDPOINT")
    agent_id = os.environ.get("AZURE_EXISTING_AGENT_ID")
    try:
        # Prefer service principal only if all three env vars are present and look valid; otherwise rely on CLI/VS Code
        sp_client_id = os.environ.get("AZURE_CLIENT_ID", "").strip()
        sp_tenant_id = os.environ.get("AZURE_TENANT_ID", "").strip()
        sp_client_secret = os.environ.get("AZURE_CLIENT_SECRET", "").strip()
        sp_enabled = bool(sp_client_id and sp_tenant_id and sp_client_secret)

        credential = DefaultAzureCredential(
            exclude_environment_credential=not sp_enabled,
            exclude_shared_token_cache_credential=True,
        )

        ai_project = AIProjectClient(
            credential=credential,
            endpoint=proj_endpoint,
            api_version = "2025-05-15-preview" # Evaluations yet not supported on stable (api_version="2025-05-01")
        )
        logger.info("Created AIProjectClient")

        if enable_trace:
            # Prefer explicit env connection string if provided
            application_insights_connection_string = os.getenv(
                "APPLICATION_INSIGHTS_CONNECTION_STRING", ""
            )
            try:
                # Fallback to AI Foundry project telemetry if available
                if not application_insights_connection_string and hasattr(ai_project, "telemetry"):
                    get_cs = getattr(ai_project.telemetry, "get_connection_string", None)
                    if callable(get_cs):
                        application_insights_connection_string = await get_cs()
            except Exception as e:
                logger.error(
                    "Failed to get Application Insights connection string, error: %s",
                    str(e),
                )
            if not application_insights_connection_string:
                logger.warning(
                    "Tracing requested but no connection string found. Continuing without tracing."
                )
            else:
                from azure.monitor.opentelemetry import configure_azure_monitor
                configure_azure_monitor(
                    connection_string=application_insights_connection_string
                )
                app.state.application_insights_connection_string = (
                    application_insights_connection_string
                )
                logger.info("Configured Application Insights for tracing.")

        if agent_id:
            try:
                agent = await ai_project.agents.get_agent(agent_id)
                logger.info("Agent already exists, skipping creation")
                logger.info(f"Fetched agent, agent ID: {agent.id}")
                logger.info(f"Fetched agent, model name: {agent.model}")
            except Exception as e:
                logger.error(f"Error fetching agent: {e}", exc_info=True)

        if not agent:
            # Fallback to searching by name
            agent_name = os.environ["AZURE_AI_AGENT_NAME"]
            agent_list = ai_project.agents.list_agents()
            if agent_list:
                async for agent_object in agent_list:
                    if agent_object.name == agent_name:
                        agent = agent_object
                        logger.info(f"Found agent by name '{agent_name}', ID={agent_object.id}")
                        break

        if not agent:
            raise RuntimeError("No agent found. Ensure qunicorn.py created one or set AZURE_EXISTING_AGENT_ID.")

        app.state.ai_project = ai_project
        app.state.agent = agent
        
        yield

    except Exception as e:
        logger.error(f"Error during startup: {e}", exc_info=True)
        raise RuntimeError(f"Error during startup: {e}")

    finally:
        try:
            if ai_project is not None:
                await ai_project.close()
                logger.info("Closed AIProjectClient")
        except Exception:
            logger.error("Error closing AIProjectClient", exc_info=True)


def create_app():
    if not os.getenv("RUNNING_IN_PRODUCTION"):
        load_dotenv(override=True)

    global logger
    logger = configure_logging(os.getenv("APP_LOG_FILE", ""))

    enable_trace_string = os.getenv("ENABLE_AZURE_MONITOR_TRACING", "")
    global enable_trace
    enable_trace = False
    if enable_trace_string == "":
        enable_trace = False
    else:
        enable_trace = str(enable_trace_string).lower() == "true"
    if enable_trace:
        logger.info("Tracing is enabled.")
        try:
            from azure.monitor.opentelemetry import configure_azure_monitor  # noqa: F401
        except ModuleNotFoundError:
            logger.warning("Tracing library not installed; continuing without tracing.")
            enable_trace = False
    else:
        logger.info("Tracing is not enabled")

    directory = os.path.join(os.path.dirname(__file__), "static")
    app = fastapi.FastAPI(lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=directory), name="static")
    
    # Mount React static files
    # Uncomment the following lines if you have a React frontend
    # react_directory = os.path.join(os.path.dirname(__file__), "static/react")
    # app.mount("/static/react", StaticFiles(directory=react_directory), name="react")

    from . import routes  # Import routes
    app.include_router(routes.router)

    # Global exception handler for any unhandled exceptions
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error("Unhandled exception occurred", exc_info=exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )
    
    return app
