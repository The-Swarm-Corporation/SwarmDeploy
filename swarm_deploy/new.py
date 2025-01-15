import asyncio
import sqlite3
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, TypeVar, Union
from enum import Enum

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from rich.console import Console
from swarms import Agent, SwarmRouter
from swarms.utils.formatter import formatter

T = TypeVar("T")


class SwarmTypeEnum(str, Enum):
    AGENT_REARRANGE = "AgentRearrange"
    MIXTURE_OF_AGENTS = "MixtureOfAgents"
    SPREADSHEET_SWARM = "SpreadSheetSwarm"
    SEQUENTIAL_WORKFLOW = "SequentialWorkflow"
    CONCURRENT_WORKFLOW = "ConcurrentWorkflow"
    AUTO = "auto"


class AgentConfig(BaseModel):
    agent_name: str
    system_prompt: str
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_loops: int = 1
    model_name: str = "gpt-4o"
    autosave: bool = True
    verbose: bool = True
    dynamic_temperature_enabled: bool = True
    saved_state_path: Optional[str] = None
    user_name: Optional[str] = None
    retry_attempts: int = 1
    context_length: int = 4000
    output_type: str = "string"
    api_base: Optional[str] = None
    api_key: Optional[str] = None


class CreateSwarmRequest(BaseModel):
    name: str
    description: str
    max_loops: int = 1
    agents: List[AgentConfig]
    swarm_type: SwarmTypeEnum
    autosave: bool = True
    flow: Optional[str] = None
    return_json: bool = True
    auto_generate_prompts: bool = False


class SwarmInput(BaseModel):
    task: str = Field(..., description="Task to be executed")
    img: Union[str, None] = Field(
        None, description="Optional image input"
    )
    priority: int = Field(
        default=0, ge=0, le=10, description="Task priority (0-10)"
    )

    class Config:
        extra = "allow"


class SwarmMetadata(BaseModel):
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    version: str = "1.0"
    swarm_type: SwarmTypeEnum
    name: str
    description: str


class SwarmConfig(BaseModel):
    agents: List[AgentConfig]
    output_type: str = Field(
        default="json", description="Output format type"
    )
    name: str
    type: SwarmTypeEnum
    metadata: SwarmMetadata
    flow: Optional[str] = None


class SwarmState(BaseModel):
    config: SwarmConfig
    status: str = Field(default="idle")
    last_activity: float = Field(default_factory=time.time)
    total_tasks_processed: int = Field(default=0)
    active_tasks: int = Field(default=0)


class SwarmOutput(BaseModel):
    id: str = Field(
        ..., description="Unique identifier for the output"
    )
    timestamp: float = Field(default_factory=time.time)
    status: str = Field(
        ..., description="Status of the task execution"
    )
    execution_time: float = Field(
        ..., description="Time taken to execute in seconds"
    )
    result: Any = Field(..., description="Task execution result")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = Field(
        None, description="Error message if task failed"
    )


class SwarmBatchOutput(BaseModel):
    id: str = Field(
        ..., description="Unique identifier for the output"
    )
    timestamp: float = Field(default_factory=time.time)
    status: str = Field(
        ..., description="Status of the task execution"
    )
    execution_time: float = Field(
        ..., description="Time taken to execute in seconds"
    )
    results: List[Any] = Field(
        ..., description="List of batch task results"
    )
    failed_tasks: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SwarmDatabase:
    def __init__(self, db_path: str = "swarm_history.db"):
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        """Initialize database tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Create swarms table with updated schema
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS swarms (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    type TEXT NOT NULL,
                    agents TEXT NOT NULL,
                    output_type TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    flow TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    status TEXT NOT NULL
                )
            """
            )

            # Rest of the tables remain the same
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    swarm_id TEXT NOT NULL,
                    task TEXT NOT NULL,
                    img TEXT,
                    priority INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    result TEXT,
                    error TEXT,
                    execution_time REAL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY (swarm_id) REFERENCES swarms (id)
                )
            """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS swarm_states (
                    swarm_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    last_activity REAL NOT NULL,
                    total_tasks_processed INTEGER NOT NULL,
                    active_tasks INTEGER NOT NULL,
                    FOREIGN KEY (swarm_id) REFERENCES swarms (id)
                )
            """
            )

            conn.commit()

    # Database methods (create_swarm, get_swarm, etc.) remain the same
    # but need to be updated to handle the new schema


class SwarmDeploy:
    def __init__(self):
        self.formatter = formatter
        self.console = Console()

        # Initialize database
        self.db = SwarmDatabase()

        # Dictionary to store active swarm routers
        self.active_swarms: Dict[str, SwarmRouter] = {}

        # Initialize FastAPI
        self.app = FastAPI(title="SwarmDeploy API", debug=True)
        self._setup_routes()

    def _create_agent(self, config: AgentConfig) -> Agent:
        """Create an Agent instance from config"""

        # Create and return agent
        return Agent(
            agent_name=config.agent_name,
            system_prompt=config.system_prompt,
            model_name=config.model_name,
            max_loops=config.max_loops,
            autosave=config.autosave,
            verbose=config.verbose,
            dynamic_temperature_enabled=config.dynamic_temperature_enabled,
            saved_state_path=config.saved_state_path,
            user_name=config.user_name,
            retry_attempts=config.retry_attempts,
            context_length=config.context_length,
            output_type=config.output_type,
        )

    def _setup_routes(self):
        @self.app.post("/v1/swarms/create")
        async def create_swarm(request: CreateSwarmRequest):
            """Create a new swarm instance"""
            swarm_id = str(uuid.uuid4())

            try:
                # Create agents
                agents = [
                    self._create_agent(agent_config)
                    for agent_config in request.agents
                ]

                # Create SwarmRouter
                router = SwarmRouter(
                    name=request.name,
                    description=request.description,
                    max_loops=request.max_loops,
                    agents=agents,
                    swarm_type=request.swarm_type.value,
                    auto_generate_prompts=request.auto_generate_prompts,
                )

                # Store in active swarms
                self.active_swarms[swarm_id] = router

                # Create config and store in database
                metadata = SwarmMetadata(
                    swarm_type=request.swarm_type,
                    name=request.name,
                    description=request.description,
                )

                config = SwarmConfig(
                    agents=request.agents,
                    name=request.name,
                    type=request.swarm_type,
                    metadata=metadata,
                    flow=request.flow,
                )

                self.db.create_swarm(config, swarm_id)

                return {
                    "swarm_id": swarm_id,
                    "name": request.name,
                    "type": request.swarm_type,
                    "status": "created",
                }

            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to create swarm: {str(e)}",
                )

        @self.app.post("/v1/swarms/{swarm_id}/run")
        async def run_swarm_task(
            swarm_id: str, task_input: SwarmInput
        ):
            """Run a task on a specific swarm"""
            if swarm_id not in self.active_swarms:
                raise HTTPException(
                    status_code=404, detail="Swarm not found"
                )

            start_time = time.time()
            router = self.active_swarms[swarm_id]

            try:
                # Update state
                state = SwarmState(
                    config=self.db.get_swarm(swarm_id)["config"],
                    status="processing",
                    active_tasks=1,
                )
                self.db.update_swarm_state(swarm_id, state)

                # Run task
                result = await asyncio.create_task(
                    router.async_run(task_input.task)
                )

                # Create output
                output = SwarmOutput(
                    id=str(uuid.uuid4()),
                    status="completed",
                    execution_time=time.time() - start_time,
                    result=result,
                    metadata={
                        "swarm_type": router.swarm_type,
                        "priority": task_input.priority,
                    },
                )

                # Record in database
                self.db.record_task(swarm_id, task_input, output)

                return output

            except Exception as e:
                error_output = SwarmOutput(
                    id=str(uuid.uuid4()),
                    status="error",
                    execution_time=time.time() - start_time,
                    result=None,
                    error=str(e),
                    metadata={
                        "swarm_type": router.swarm_type,
                        "error_type": type(e).__name__,
                    },
                )

                self.db.record_task(
                    swarm_id, task_input, error_output
                )
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": str(e),
                        "task_id": error_output.id,
                    },
                )
            finally:
                # Update state
                state.status = "idle"
                state.active_tasks = 0
                state.last_activity = time.time()
                self.db.update_swarm_state(swarm_id, state)

        # Add other CRUD endpoints
        @self.app.get("/v1/swarms")
        async def list_swarms():
            return self.db.get_all_swarms()

        @self.app.get("/v1/swarms/{swarm_id}")
        async def get_swarm(swarm_id: str):
            swarm = self.db.get_swarm(swarm_id)
            if not swarm:
                raise HTTPException(
                    status_code=404, detail="Swarm not found"
                )
            return swarm

        @self.app.get("/v1/swarms/{swarm_id}/history")
        async def get_swarm_history(swarm_id: str, limit: int = 100):
            swarm = self.db.get_swarm(swarm_id)
            if not swarm:
                raise HTTPException(
                    status_code=404, detail="Swarm not found"
                )
            return self.db.get_swarm_history(swarm_id, limit)

        @self.app.delete("/v1/swarms/{swarm_id}")
        async def delete_swarm(swarm_id: str):
            if swarm_id in self.active_swarms:
                del self.active_swarms[swarm_id]
            if not self.db.delete_swarm(swarm_id):
                raise HTTPException(
                    status_code=404, detail="Swarm not found"
                )
            return {"status": "deleted"}

    def start(self, host: str = "0.0.0.0", port: int = 8000):
        """Start the FastAPI server"""
        import uvicorn

        self.formatter.print_panel(
            "\nStarting SwarmDeploy API server\n"
            f"Host: {host}\n"
            f"Port: {port}\n",
            title="Server Startup",
            style="bold green",
        )

        uvicorn.run(self.app, host=host, port=port)


if __name__ == "__main__":
    deploy = SwarmDeploy()
    deploy.start()
