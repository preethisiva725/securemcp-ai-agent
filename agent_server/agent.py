import logging
from typing import AsyncGenerator, Optional

import mlflow
from databricks.sdk import WorkspaceClient
from databricks_langchain import (
    ChatDatabricks,
    DatabricksMCPServer,
    DatabricksMultiServerMCPClient,
)
from langchain.agents import create_agent
from mlflow.genai.agent_server import invoke, stream
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    to_chat_completions_input,
)

from agent_server.utils import (
    get_session_id,
    process_agent_astream_events,
)


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logger = logging.getLogger(__name__)

mlflow.langchain.autolog()

logging.getLogger(
    "mlflow.utils.autologging_utils"
).setLevel(logging.ERROR)


# ---------------------------------------------------------
# Databricks workspace client
# ---------------------------------------------------------

sp_workspace_client = WorkspaceClient()


# ---------------------------------------------------------
# Create MCP server definitions
# ---------------------------------------------------------

def create_mcp_servers(
    workspace_client: WorkspaceClient,
):
    """
    Create the governed Databricks MCP servers used by SecureMCP.
    """

    host_name = workspace_client.config.host

    logger.info(
        "Databricks workspace host: %s",
        host_name,
    )

    functions_server = DatabricksMCPServer(
        name="securemcp-functions",
        url=(
            f"{host_name}/api/2.0/mcp/functions/"
            "workspace/securemcp"
        ),
        workspace_client=workspace_client,
    )

    ai_search_server = DatabricksMCPServer(
        name="company-knowledge-search",
        url=(
            f"{host_name}/api/2.0/mcp/ai-search/"
            "workspace/securemcp/company_knowledge_index"
        ),
        workspace_client=workspace_client,
    )

    return functions_server, ai_search_server


# ---------------------------------------------------------
# Load one MCP server at a time
# ---------------------------------------------------------

async def load_mcp_server_tools(
    server,
    server_name: str,
):
    """
    Load tools from one MCP server at a time.

    Loading separately makes it much easier to identify
    which MCP server is causing a connection or permission
    problem.
    """

    logger.info(
        "=============================================="
    )

    logger.info(
        "Starting MCP server: %s",
        server_name,
    )

    logger.info(
        "=============================================="
    )

    try:
        client = DatabricksMultiServerMCPClient(
            [server]
        )

        tools = await client.get_tools()

        logger.info(
            "MCP server '%s' loaded successfully.",
            server_name,
        )

        logger.info(
            "MCP server '%s' returned %d tools.",
            server_name,
            len(tools),
        )

        for tool_item in tools:
            logger.info(
                "Loaded tool from '%s': %s",
                server_name,
                getattr(
                    tool_item,
                    "name",
                    "unknown",
                ),
            )

        return tools

    except Exception as exc:
        logger.exception(
            "FAILED to load MCP server '%s'. "
            "The exact error is shown above.",
            server_name,
        )

        raise RuntimeError(
            f"SecureMCP could not load MCP server "
            f"'{server_name}'. "
            f"Original error: {exc}"
        ) from exc


# ---------------------------------------------------------
# Initialize SecureMCP agent
# ---------------------------------------------------------

async def init_agent(
    workspace_client: Optional[WorkspaceClient] = None,
):
    """
    Initialize the SecureMCP LangGraph agent.

    MCP servers are loaded separately so that connection
    and permission problems can be identified clearly.
    """

    workspace_client = (
        workspace_client or sp_workspace_client
    )

    try:

        # -------------------------------------------------
        # Create MCP servers
        # -------------------------------------------------

        (
            functions_server,
            ai_search_server,
        ) = create_mcp_servers(
            workspace_client
        )

        # -------------------------------------------------
        # Load Unity Catalog function tools
        # -------------------------------------------------

        logger.info(
            "Loading Unity Catalog MCP tools..."
        )

        functions_tools = await load_mcp_server_tools(
            functions_server,
            "securemcp-functions",
        )

        # -------------------------------------------------
        # Load AI Search tools
        # -------------------------------------------------

        logger.info(
            "Loading AI Search MCP tools..."
        )

        ai_search_tools = await load_mcp_server_tools(
            ai_search_server,
            "company-knowledge-search",
        )

        # -------------------------------------------------
        # Combine all MCP tools
        # -------------------------------------------------

        tools = (
            functions_tools
            + ai_search_tools
        )

        logger.info(
            "=============================================="
        )

        logger.info(
            "TOTAL SECUREMCP MCP TOOLS LOADED: %d",
            len(tools),
        )

        logger.info(
            "=============================================="
        )

        for tool_item in tools:
            logger.info(
                "Final MCP tool: %s",
                getattr(
                    tool_item,
                    "name",
                    "unknown",
                ),
            )

        # -------------------------------------------------
        # Create LangGraph agent
        # -------------------------------------------------

        agent = create_agent(
            model=ChatDatabricks(
                endpoint=(
                    "databricks-meta-llama-3-3-70b-instruct"
                )
            ),
            tools=tools,
            system_prompt=(
                "You are SecureMCP, a secure company "
                "information AI agent."

                "\n\n"

                "Use the available governed MCP tools "
                "to answer questions about company "
                "sales, inventory, employees, and "
                "company policies."

                "\n\n"

                "Always use actual tool results when "
                "the user asks for company data. "
                "Never invent sales, inventory, "
                "employee, or policy information."

                "\n\n"

                "Use Unity Catalog functions for "
                "structured company data."

                "\n\n"

                "Use the company knowledge AI Search "
                "tool for company policies and "
                "documents."

                "\n\n"

                "When a question requires multiple "
                "pieces of information, call multiple "
                "tools when necessary and combine "
                "their results."

                "\n\n"

                "Respect the access restrictions "
                "enforced by the available tools "
                "and never expose restricted "
                "information."

                "\n\n"

                "If a tool does not return information, "
                "do not invent an answer."
            ),
        )

        logger.info(
            "SecureMCP LangGraph agent initialized "
            "successfully."
        )

        return agent

    except Exception:
        logger.exception(
            "FAILED to initialize SecureMCP agent."
        )
        raise


# ---------------------------------------------------------
# Invoke endpoint
# ---------------------------------------------------------

@invoke()
async def invoke_handler(
    request: ResponsesAgentRequest,
) -> ResponsesAgentResponse:

    outputs = [
        event.item
        async for event in stream_handler(request)
        if event.type == "response.output_item.done"
    ]

    return ResponsesAgentResponse(
        output=outputs
    )


# ---------------------------------------------------------
# Streaming endpoint
# ---------------------------------------------------------

@stream()
async def stream_handler(
    request: ResponsesAgentRequest,
) -> AsyncGenerator[
    ResponsesAgentStreamEvent,
    None,
]:

    # -----------------------------------------------------
    # MLflow session tracking
    # -----------------------------------------------------

    if session_id := get_session_id(request):

        mlflow.update_current_trace(
            metadata={
                "mlflow.trace.session": session_id
            }
        )

    # -----------------------------------------------------
    # Convert incoming request to chat messages
    # -----------------------------------------------------

    messages = {
        "messages": to_chat_completions_input(
            [
                item.model_dump()
                for item in request.input
            ]
        )
    }

    # -----------------------------------------------------
    # Initialize agent
    # -----------------------------------------------------

    agent = await init_agent()

    # -----------------------------------------------------
    # Run LangGraph agent
    # -----------------------------------------------------

    async for event in process_agent_astream_events(
        agent.astream(
            input=messages,
            stream_mode=[
                "updates",
                "messages",
            ],
        )
    ):
        yield event