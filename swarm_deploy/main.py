
from typing import List, Union, Dict, Any, TypeVar, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import asyncio
import uuid
import time
from rich.console import Console
from swarms.utils.formatter import formatter
from swarm_deploy.callable_name import NameResolver

T = TypeVar("T")


class SwarmInput(BaseModel):
    task: str = Field(..., description="Task to be executed")
    img: Union[str, None] = Field(
        None, description="Optional image input"
    )
    priority: int = Field(
        default=0, ge=0, le=10, description="Task priority (0-10)"
    )

    class Config:
        extra = "allow"  # Allow extra fields without raising validation errors


class SwarmMetadata(BaseModel):
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    version: str = "1.0"
    callable_name: str


class SwarmConfig(BaseModel):
    agents: int = Field(
        gt=0, description="Number of agents in the swarm"
    )
    output_type: str = Field(
        default="json", description="Output format type"
    )
    name: str
    type: str
    metadata: SwarmMetadata


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


class SwarmDeploy:
    def __init__(self, callable_obj: Any):
        self.id = str(uuid.uuid4())
        self.callable = callable_obj
        self.formatter = formatter
        self.console = Console()
        self.resolver = NameResolver()
        self.callable_name = self.resolver.get_name(self.callable)

        # Count agents
        agent_number = len(callable_obj.agents)
        self.config = self._create_config(agent_number)

        # Initialize state and history
        self.state = SwarmState(config=self.config)
        self.task_history: Dict[str, Any] = {}

        # Initialize FastAPI
        self.app = FastAPI(title="SwarmDeploy API", debug=True)
        self._setup_routes()

    def _create_config(self, agents: int) -> SwarmConfig:
        metadata = SwarmMetadata(
            callable_name=self.callable_name,
        )

        return SwarmConfig(
            agents=agents,
            output_type="json",
            name=f"{self.callable_name}",
            type=self.callable_name,
            metadata=metadata,
        )

    def _setup_routes(self):
        @self.app.post(
            f"/v1/swarms/completions/{self.callable_name}",
            response_model=Union[SwarmOutput, SwarmBatchOutput],
        )
        async def create_completion(task_input: SwarmInput):
            start_time = time.time()

            try:
                self.state.active_tasks += 1
                self.state.status = "processing"

                # Add logging to help debug
                self.formatter.print_panel(
                    f"Received task: {task_input.task}\n"
                    f"Priority: {task_input.priority}",
                    title="Task Receipt",
                    style="bold blue",
                )

                try:
                    result = await self.run(
                        task_input.task, task_input.img
                    )

                    if result is None:
                        raise ValueError(
                            "Task execution returned None"
                        )

                    output = SwarmOutput(
                        id=str(uuid.uuid4()),
                        status="completed",
                        execution_time=time.time() - start_time,
                        result=result,
                        metadata={
                            "type": self.config.type,
                            "priority": task_input.priority,
                        },
                    )

                    self.task_history[output.id] = output
                    self.state.total_tasks_processed += 1
                    return output

                except Exception as e:
                    self.formatter.print_panel(
                        f"Task execution error: {str(e)}\n"
                        f"Task: {task_input.task}",
                        title="Execution Error",
                        style="bold red",
                    )

                    error_output = SwarmOutput(
                        id=str(uuid.uuid4()),
                        status="error",
                        execution_time=time.time() - start_time,
                        result=None,
                        error=str(e),
                        metadata={
                            "type": self.config.type,
                            "error_type": type(e).__name__,
                        },
                    )

                    self.task_history[error_output.id] = error_output
                    raise HTTPException(
                        status_code=500,
                        detail={
                            "error": str(e),
                            "task_id": error_output.id,
                        },
                    )

            except HTTPException:
                raise
            except Exception as e:
                self.formatter.print_panel(
                    f"Error processing task: {str(e)}",
                    title="Error",
                    style="bold red",
                )
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                )
            finally:
                self.state.active_tasks -= 1
                self.state.status = (
                    "idle"
                    if self.state.active_tasks == 0
                    else "processing"
                )
                self.state.last_activity = time.time()

    async def run(self, task: str, img: str = None) -> Any:
        """Main entry point for running the callable"""
        try:
            self.formatter.print_panel(
                f"Executing {self.callable_name} with task: {task}"
                + (f" and image: {img}" if img else ""),
                title=f"SwarmDeploy Task - {self.config.type}",
            )

            if asyncio.iscoroutinefunction(self.callable.run):
                result = (
                    await self.callable.run(task)
                    if img is None
                    else await self.callable.run(task, img)
                )
            else:
                result = (
                    await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: (
                            self.callable.run(task)
                            if img is None
                            else self.callable.run(task, img)
                        ),
                    )
                )

            if result is None:
                raise ValueError("Callable returned None")

            return result

        except Exception as e:
            self.formatter.print_panel(
                f"Error in run method: {str(e)}",
                title="Run Error",
                style="bold red",
            )
            raise

    def start(self, host: str = "0.0.0.0", port: int = 8000):
        """Start the FastAPI server"""
        import uvicorn

        self.formatter.print_panel(
            f"Starting SwarmDeploy API server on {host}:{port} for {self.callable_name}\n"
            f"Endpoint: /v1/swarms/completions/{self.callable_name}",
            title="Server Startup",
            style="bold green",
        )

        uvicorn.run(self.app, host=host, port=port)
