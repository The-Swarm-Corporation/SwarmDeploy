import asyncio
import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, TypeVar, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from rich.console import Console
from swarms.utils.formatter import formatter

T = TypeVar("T")


# ===== Models =====
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


# ===== Database Layer =====
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

            # Create swarms table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS swarms (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    agents INTEGER NOT NULL,
                    output_type TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    status TEXT NOT NULL
                )
            """
            )

            # Create tasks table
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

            # Create swarm_states table
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

    def create_swarm(
        self, swarm_config: SwarmConfig, swarm_id: str
    ) -> None:
        """Create a new swarm entry"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            now = time.time()

            cursor.execute(
                """
                INSERT INTO swarms (
                    id, name, type, agents, output_type, metadata,
                    created_at, updated_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    swarm_id,
                    swarm_config.name,
                    swarm_config.type,
                    swarm_config.agents,
                    swarm_config.output_type,
                    json.dumps(swarm_config.metadata.dict()),
                    now,
                    now,
                    "idle",
                ),
            )

            # Initialize swarm state
            cursor.execute(
                """
                INSERT INTO swarm_states (
                    swarm_id, status, last_activity,
                    total_tasks_processed, active_tasks
                ) VALUES (?, ?, ?, ?, ?)
            """,
                (swarm_id, "idle", now, 0, 0),
            )

            conn.commit()

    def get_swarm(self, swarm_id: str) -> Optional[Dict[str, Any]]:
        """Get swarm details by ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM swarms WHERE id = ?", (swarm_id,)
            )
            row = cursor.fetchone()

            if row:
                return dict(row)
            return None

    def update_swarm_state(
        self, swarm_id: str, state: SwarmState
    ) -> None:
        """Update swarm state"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE swarm_states
                SET status = ?, last_activity = ?,
                    total_tasks_processed = ?, active_tasks = ?
                WHERE swarm_id = ?
            """,
                (
                    state.status,
                    state.last_activity,
                    state.total_tasks_processed,
                    state.active_tasks,
                    swarm_id,
                ),
            )
            conn.commit()

    def record_task(
        self,
        swarm_id: str,
        task_input: SwarmInput,
        output: SwarmOutput,
    ) -> None:
        """Record task execution details"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO tasks (
                    id, swarm_id, task, img, priority, status,
                    result, error, execution_time, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    output.id,
                    swarm_id,
                    task_input.task,
                    task_input.img,
                    task_input.priority,
                    output.status,
                    (
                        json.dumps(output.result)
                        if output.result
                        else None
                    ),
                    output.error,
                    output.execution_time,
                    output.timestamp,
                ),
            )
            conn.commit()

    def get_swarm_history(
        self, swarm_id: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get task history for a swarm"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM tasks
                WHERE swarm_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            """,
                (swarm_id, limit),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_all_swarms(self) -> List[Dict[str, Any]]:
        """Get all registered swarms"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM swarms ORDER BY created_at DESC"
            )
            return [dict(row) for row in cursor.fetchall()]

    def delete_swarm(self, swarm_id: str) -> bool:
        """Delete a swarm and its associated data"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                # Delete associated tasks first
                cursor.execute(
                    "DELETE FROM tasks WHERE swarm_id = ?",
                    (swarm_id,),
                )
                # Delete swarm state
                cursor.execute(
                    "DELETE FROM swarm_states WHERE swarm_id = ?",
                    (swarm_id,),
                )
                # Delete swarm
                cursor.execute(
                    "DELETE FROM swarms WHERE id = ?", (swarm_id,)
                )
                conn.commit()
                return True
            except sqlite3.Error:
                return False


# ===== Main SwarmDeploy Class =====
class SwarmDeploy:
    def __init__(self, callable_obj: Any):
        self.id = str(uuid.uuid4())
        self.callable = callable_obj
        self.formatter = formatter
        self.console = Console()
        self.callable_name = callable_obj.__class__.__name__

        # Initialize database
        self.db = SwarmDatabase()

        # Count agents
        agent_number = len(callable_obj.agents)
        self.config = self._create_config(agent_number)

        # Initialize state
        self.state = SwarmState(config=self.config)

        # Register swarm in database
        self.db.create_swarm(self.config, self.id)

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
        # Main task execution endpoint
        @self.app.post(
            f"/v1/swarms/completions/{self.callable_name}/{self.id}",
            response_model=Union[SwarmOutput, SwarmBatchOutput],
        )
        async def create_completion(task_input: SwarmInput):
            start_time = time.time()

            try:
                self.state.active_tasks += 1
                self.state.status = "processing"
                self.db.update_swarm_state(self.id, self.state)

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

                    # Record task in database
                    self.db.record_task(self.id, task_input, output)

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

                    # Record error in database
                    self.db.record_task(
                        self.id, task_input, error_output
                    )

                    raise HTTPException(
                        status_code=500,
                        detail={
                            "error": str(e),
                            "task_id": error_output.id,
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
                self.db.update_swarm_state(self.id, self.state)

        # CRUD endpoints for swarm management
        @self.app.get("/v1/swarms")
        async def list_swarms():
            """Get all registered swarms"""
            return self.db.get_all_swarms()

        @self.app.get("/v1/swarms/{swarm_id}")
        async def get_swarm(swarm_id: str):
            """Get details of a specific swarm"""
            swarm = self.db.get_swarm(swarm_id)
            if not swarm:
                raise HTTPException(
                    status_code=404, detail="Swarm not found"
                )
            return swarm

        @self.app.get("/v1/swarms/{swarm_id}/history")
        async def get_swarm_history(swarm_id: str, limit: int = 100):
            """Get task history for a swarm"""
            swarm = self.db.get_swarm(swarm_id)
            if not swarm:
                raise HTTPException(
                    status_code=404, detail="Swarm not found"
                )
            return self.db.get_swarm_history(swarm_id, limit)

        @self.app.delete("/v1/swarms/{swarm_id}")
        async def delete_swarm(swarm_id: str):
            """Delete a swarm and its history"""
            if not self.db.delete_swarm(swarm_id):
                raise HTTPException(
                    status_code=404, detail="Swarm not found"
                )
            return {"status": "deleted"}

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
            "\nStarting SwarmDeploy API server\n"
            f"Host: {host}\n"
            f"Port: {port}\n"
            f"Swarm Name: {self.callable_name}\n"
            f"Swarm ID: {self.id}\n"
            f"Endpoint: /v1/swarms/completions/{self.callable_name}/{self.id}",
            title="Server Startup",
            style="bold green",
        )

        uvicorn.run(self.app, host=host, port=port)
