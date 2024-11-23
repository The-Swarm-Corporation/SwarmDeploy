import asyncio
import signal
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, TypeVar, Union

import orjson
import structlog
from aiocache import Cache
from aiocache.serializers import PickleSerializer
from fastapi import (
    BackgroundTasks,
    FastAPI,
    HTTPException,
    Query,
    Request,
    Response,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from prometheus_client import Counter, Histogram, generate_latest
from pydantic import BaseModel, Field, validator
from rich.console import Console

from swarms.utils.callable_name import NameResolver
from swarms.utils.formatter import formatter

T = TypeVar("T")

# Enhanced logging setup
logger = structlog.get_logger()

# Metrics
TASK_COUNTER = Counter(
    "swarm_tasks_total", "Total number of tasks processed"
)
TASK_DURATION = Histogram(
    "swarm_task_duration_seconds", "Task execution time"
)
ERROR_COUNTER = Counter(
    "swarm_errors_total", "Total number of errors"
)


class SwarmInput(BaseModel):
    task: str = Field(..., description="Task to be executed")
    img: Optional[str] = Field(
        None, description="Optional image input"
    )
    priority: int = Field(
        default=0, ge=0, le=10, description="Task priority (0-10)"
    )
    timeout: Optional[float] = Field(
        None, gt=0, description="Task timeout in seconds"
    )

    class Config:
        json_loads = orjson.loads
        json_dumps = orjson.dumps


class SwarmMetadata(BaseModel):
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    version: str = "2.0"
    callable_name: str
    environment: str = Field(default="production")

    @validator("updated_at")
    def update_timestamp(cls, v):
        return datetime.now()


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
    max_concurrent_tasks: int = Field(default=100, gt=0)
    task_timeout: float = Field(default=30.0, gt=0)
    retry_attempts: int = Field(default=3, ge=0)
    cache_ttl: int = Field(default=3600, ge=0)  # Cache TTL in seconds


class SwarmState(BaseModel):
    config: SwarmConfig
    status: str = Field(default="idle")
    last_activity: datetime = Field(default_factory=datetime.now)
    total_tasks_processed: int = Field(default=0)
    active_tasks: int = Field(default=0)
    error_count: int = Field(default=0)
    uptime: float = Field(default_factory=lambda: time.time())


class SwarmOutput(BaseModel):
    id: str = Field(
        ..., description="Unique identifier for the output"
    )
    timestamp: datetime = Field(default_factory=datetime.now)
    status: str = Field(
        ..., description="Status of the task execution"
    )
    execution_time: float = Field(
        ..., description="Time taken to execute in seconds"
    )
    result: Any = Field(..., description="Task execution result")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    cached: bool = Field(default=False)

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            # Add any other custom encoders needed
        }

    def dict(self, *args, **kwargs):
        d = super().dict(*args, **kwargs)
        # Ensure result is JSON serializable
        if isinstance(
            d["result"],
            (dict, list, str, int, float, bool, None.__class__),
        ):
            return d
        # Convert non-JSON serializable results to string representation
        d["result"] = str(d["result"])
        return d


class TaskQueue:
    def __init__(self, maxsize: int = 1000):
        self.queue: asyncio.PriorityQueue = asyncio.PriorityQueue(
            maxsize=maxsize
        )
        self._unfinished_tasks = 0
        self._lock = asyncio.Lock()

    async def put(self, priority: int, item: Any):
        await self.queue.put(
            (-priority, item)
        )  # Negative priority for highest-first
        async with self._lock:
            self._unfinished_tasks += 1

    async def get(self):
        priority, item = await self.queue.get()
        return item

    async def task_done(self):
        async with self._lock:
            self._unfinished_tasks -= 1
            if self._unfinished_tasks == 0:
                self.queue._finished.set()


class SwarmDeploy:
    def __init__(
        self,
        callable_obj: Any,
        max_workers: int = None,
        cache_backend: str = "memory",
    ):
        self.id = str(uuid.uuid4())
        self.callable = callable_obj
        self.formatter = formatter
        self.console = Console()
        self.resolver = NameResolver()
        self.callable_name = self.resolver.get_name(self.callable)

        # Enhanced thread pool
        self.thread_pool = ThreadPoolExecutor(
            max_workers=max_workers
            or (
                asyncio.get_event_loop()
                .get_default_executor()
                ._max_workers
                * 2
            ),
            thread_name_prefix=f"swarm_{self.callable_name}",
        )

        # Task queue for priority handling
        self.task_queue = TaskQueue()

        # Setup caching
        self.cache = Cache(
            (
                Cache.MEMORY
                if cache_backend == "memory"
                else Cache.REDIS
            ),
            serializer=PickleSerializer(),
            namespace=f"swarm_{self.callable_name}",
            ttl=3600,
        )

        # Count agents
        agent_number = len(callable_obj.agents)
        self.config = self._create_config(agent_number)

        # Initialize state and history
        self.state = SwarmState(config=self.config)
        self.task_history: Dict[str, Any] = {}

        # Semaphore for controlling concurrent tasks
        self.semaphore = asyncio.Semaphore(
            self.config.max_concurrent_tasks
        )

        # Initialize FastAPI with lifecycle management
        self.app = FastAPI(
            title="SwarmDeploy API",
            debug=False,
            lifespan=self._lifespan,
            default_response_class=Response,
        )

        # Add middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self.app.add_middleware(GZipMiddleware, minimum_size=1000)

        # Setup routes
        self._setup_routes()

        # Initialize background task worker
        self.background_worker_task = None

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        # Startup
        logger.info(
            "Starting SwarmDeploy service", service=self.callable_name
        )
        self.background_worker_task = asyncio.create_task(
            self._background_worker()
        )

        try:
            yield
        finally:
            # Shutdown
            logger.info(
                "Shutting down SwarmDeploy service",
                service=self.callable_name,
            )
            if self.background_worker_task:
                self.background_worker_task.cancel()
                try:
                    await self.background_worker_task
                except asyncio.CancelledError:
                    pass

            await self.cache.close()
            self.thread_pool.shutdown(wait=True)

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
            response_model=Union[SwarmOutput, None],
            response_model_exclude_none=True,
        )
        async def create_completion(
            task_input: SwarmInput,
            background_tasks: BackgroundTasks,
            request: Request,
            response: Response,
        ):
            task_id = str(uuid.uuid4())

            # Check cache first
            cache_key = (
                f"{self.callable_name}:{hash(task_input.task)}"
            )
            if cached_result := await self.cache.get(cache_key):
                logger.info("Cache hit", task_id=task_id)
                return SwarmOutput(
                    id=task_id,
                    status="completed",
                    execution_time=0,
                    result=cached_result,
                    cached=True,
                    metadata={"source": "cache"},
                )

            # Add task to queue
            await self.task_queue.put(
                task_input.priority,
                (
                    task_id,
                    task_input,
                    request.headers.get("X-Request-ID"),
                ),
            )

            # Return immediately if async execution is requested
            if (
                request.headers.get("X-Async-Execution", "").lower()
                == "true"
            ):
                response.status_code = 202
                return {"task_id": task_id, "status": "accepted"}

            # Wait for result with timeout
            try:
                async with asyncio.timeout(
                    task_input.timeout or self.config.task_timeout
                ):
                    while True:
                        if task_id in self.task_history:
                            return self.task_history[task_id]
                        await asyncio.sleep(0.1)
            except asyncio.TimeoutError:
                logger.warning("Task timeout", task_id=task_id)
                response.status_code = 408
                return {"error": "Task timeout"}

        @self.app.get("/metrics")
        async def metrics():
            return Response(
                generate_latest(), media_type="text/plain"
            )

        @self.app.get(
            "/v1/swarms/{swarm_id}", response_model=SwarmState
        )
        async def get_swarm_state(swarm_id: str):
            if swarm_id != self.id:
                raise HTTPException(
                    status_code=404, detail="Swarm not found"
                )
            return self.state

        @self.app.get(
            "/v1/swarms/{swarm_id}/tasks/{task_id}",
            response_model=Union[SwarmOutput, Dict[str, Any]],
        )
        async def get_task_result(
            swarm_id: str,
            task_id: str,
            wait: bool = Query(
                False, description="Wait for task completion"
            ),
        ):
            if swarm_id != self.id:
                raise HTTPException(
                    status_code=404, detail="Swarm not found"
                )

            if task_id in self.task_history:
                return self.task_history[task_id]

            if not wait:
                raise HTTPException(
                    status_code=404, detail="Task not found"
                )

            # Wait for task completion
            try:
                async with asyncio.timeout(
                    30
                ):  # 30-second timeout for waiting
                    while True:
                        if task_id in self.task_history:
                            return self.task_history[task_id]
                        await asyncio.sleep(0.1)
            except asyncio.TimeoutError:
                raise HTTPException(
                    status_code=408,
                    detail="Timeout waiting for task completion",
                )

    async def _background_worker(self):
        """Background worker to process queued tasks"""
        while True:
            try:
                task_id, task_input, request_id = (
                    await self.task_queue.get()
                )
                asyncio.create_task(
                    self._process_task(
                        task_id, task_input, request_id
                    )
                )
            except Exception as e:
                logger.error("Background worker error", error=str(e))
                await asyncio.sleep(1)

    async def _process_task(
        self,
        task_id: str,
        task_input: SwarmInput,
        request_id: Optional[str],
    ):
        """Process a single task with enhanced error handling"""
        start_time = time.time()

        try:
            async with self.semaphore:
                self.state.active_tasks += 1
                self.state.status = "processing"

                logger.info(
                    "Starting task processing",
                    task_id=task_id,
                    task=task_input.task[:100],
                )

                try:
                    result = await self._execute_task(task_input)

                    # Ensure result is JSON serializable
                    if not isinstance(
                        result,
                        (
                            dict,
                            list,
                            str,
                            int,
                            float,
                            bool,
                            None.__class__,
                        ),
                    ):
                        result = str(result)

                    execution_time = time.time() - start_time

                    # Create output
                    output = SwarmOutput(
                        id=task_id,
                        status="completed",
                        execution_time=execution_time,
                        result=result,
                        metadata={
                            "type": self.config.type,
                            "request_id": request_id,
                        },
                    )

                    # Update metrics
                    TASK_COUNTER.inc()
                    TASK_DURATION.observe(execution_time)

                    self.task_history[task_id] = output
                    self.state.total_tasks_processed += 1

                    # Cache successful results
                    cache_key = f"{self.callable_name}:{hash(task_input.task)}"
                    await self.cache.set(
                        cache_key, result, ttl=self.config.cache_ttl
                    )

                    logger.info(
                        "Task completed successfully",
                        task_id=task_id,
                        execution_time=execution_time,
                    )

                    return output

                except Exception as e:
                    logger.error(
                        "Task execution error",
                        task_id=task_id,
                        error=str(e),
                        exc_info=True,
                    )
                    raise

        except Exception as e:
            error_msg = str(e)
            logger.error(
                "Task processing error",
                task_id=task_id,
                error=error_msg,
                exc_info=True,
            )
            ERROR_COUNTER.inc()
            self.state.error_count += 1

            output = SwarmOutput(
                id=task_id,
                status="error",
                execution_time=time.time() - start_time,
                result={"error": error_msg},
                metadata={
                    "request_id": request_id,
                    "error_type": type(e).__name__,
                },
            )
            self.task_history[task_id] = output
            return output

        finally:
            self.state.active_tasks -= 1
            self.state.status = (
                "idle"
                if self.state.active_tasks == 0
                else "processing"
            )
            self.state.last_activity = datetime.now()
            await self.task_queue.task_done()

    async def _execute_task(self, task_input: SwarmInput) -> Any:
        """Execute the task with the callable"""
        try:
            self.formatter.print_panel(
                f"Executing {self.callable_name} with task: {task_input.task}"
                + (
                    f" and image: {task_input.img}"
                    if task_input.img
                    else ""
                ),
                title=f"SwarmDeploy Task - {self.config.type}",
            )

            if asyncio.iscoroutinefunction(self.callable.run):
                try:
                    # Run async function with proper error handling
                    result = await self.callable.run(task_input.task)
                    if result is None:
                        raise ValueError(
                            f"Callable {self.callable_name} returned None"
                        )
                    return result
                except Exception as e:
                    logger.error(
                        f"Error in async execution: {str(e)}",
                        exc_info=True,
                    )
                    raise RuntimeError(
                        f"Task execution failed: {str(e)}"
                    )
            else:
                # Run CPU-bound tasks in thread pool with proper error handling
                try:
                    result = await asyncio.get_event_loop().run_in_executor(
                        self.thread_pool,
                        lambda: (
                            self.callable.run(task_input.task)
                            if task_input.img is None
                            else self.callable.run(
                                task_input.task, task_input.img
                            )
                        ),
                    )
                    if result is None:
                        raise ValueError(
                            f"Callable {self.callable_name} returned None"
                        )
                    return result
                except Exception as e:
                    logger.error(
                        f"Error in sync execution: {str(e)}",
                        exc_info=True,
                    )
                    raise RuntimeError(
                        f"Task execution failed: {str(e)}"
                    )

        except Exception as e:
            logger.error(
                f"Error in _execute_task: {str(e)}", exc_info=True
            )
            raise  # Re-raise the exception to be handled by _process_task

    def _setup_signal_handlers(self):
        """Setup graceful shutdown handlers"""
        for sig in (signal.SIGTERM, signal.SIGINT):
            asyncio.get_event_loop().add_signal_handler(
                sig,
                lambda s=sig: asyncio.create_task(
                    self._shutdown(sig=s)
                ),
            )

    async def _shutdown(self, sig: signal.Signals):
        """Graceful shutdown procedure"""
        logger.info(f"Received exit signal {sig.name}...")

        # Stop accepting new tasks
        self.app.state.accepting_requests = False

        # Wait for active tasks to complete (with timeout)
        try:
            async with asyncio.timeout(
                30
            ):  # 30-second shutdown timeout
                while self.state.active_tasks > 0:
                    logger.info(
                        f"Waiting for {self.state.active_tasks} active tasks to complete..."
                    )
                    await asyncio.sleep(1)
        except asyncio.TimeoutError:
            logger.warning(
                "Shutdown timeout reached with active tasks remaining"
            )

        # Cleanup resources
        await self.cache.close()
        self.thread_pool.shutdown(wait=False)

        # Let event loop handle remaining callbacks
        await asyncio.sleep(1)

    def start(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        workers: int = None,
        ssl_keyfile: str = None,
        ssl_certfile: str = None,
    ):
        """Start the FastAPI server with enhanced configuration"""
        try:

            import platform

            import uvicorn

            # Configure logging
            log_config = uvicorn.config.LOGGING_CONFIG
            log_config["formatters"]["access"][
                "fmt"
            ] = "%(asctime)s - %(levelname)s - %(message)s"

            self.formatter.print_panel(
                f"Starting SwarmDeploy API server on {host}:{port} for {self.callable_name}\n"
                f"Endpoint: /v1/swarms/completions/{self.callable_name}\n"
                f"Workers: {workers or 'auto'}\n"
                f"SSL: {'enabled' if ssl_certfile else 'disabled'}",
                title="Server Startup",
                style="bold green",
            )

            # Different configuration for MacOS vs other platforms
            if platform.system() == "Darwin":  # MacOS
                config = uvicorn.Config(
                    app=self.app,
                    host=host,
                    port=port,
                    workers=1,
                    loop="asyncio",  # Use default asyncio instead of uvloop on MacOS
                    limit_concurrency=1000,
                    limit_max_requests=50000,
                    timeout_keep_alive=30,
                    ssl_keyfile=ssl_keyfile,
                    ssl_certfile=ssl_certfile,
                    log_config=log_config,
                    proxy_headers=True,
                    server_header=False,
                )
            else:  # Linux/Windows
                config = uvicorn.Config(
                    app=self.app,
                    host=host,
                    port=port,
                    workers=workers,
                    loop="uvloop",
                    limit_concurrency=1000,
                    limit_max_requests=50000,
                    timeout_keep_alive=30,
                    ssl_keyfile=ssl_keyfile,
                    ssl_certfile=ssl_certfile,
                    log_config=log_config,
                    proxy_headers=True,
                    server_header=False,
                )

            # Modified signal handling setup
            def handle_signal(sig, frame):
                import sys

                logger.info(
                    f"Received signal {sig}, shutting down..."
                )
                sys.exit(0)

            # Only set up signal handlers if we're not on MacOS or we're the main process
            if platform.system() != "Darwin" or (workers or 1) <= 1:
                import signal

                signal.signal(signal.SIGINT, handle_signal)
                signal.signal(signal.SIGTERM, handle_signal)

            # Start the server
            server = uvicorn.Server(config)
            try:
                server.run()
            except Exception as e:
                logger.error(f"Server error: {str(e)}")
                raise
        except Exception as e:
            logger.info(f"Error: {e}")

    @classmethod
    def create_cluster(
        cls,
        callable_obj: Any,
        num_instances: int,
        start_port: int = 8000,
        **kwargs,
    ) -> List["SwarmDeploy"]:
        """Create a cluster of SwarmDeploy instances"""
        instances = []
        for i in range(num_instances):
            instance = cls(callable_obj, **kwargs)
            port = start_port + i

            # Start each instance in a separate process
            import multiprocessing

            process = multiprocessing.Process(
                target=instance.start, kwargs={"port": port}
            )
            process.start()

            instances.append(
                {
                    "instance": instance,
                    "process": process,
                    "port": port,
                }
            )

        return instances

    async def health_check(self) -> Dict[str, Any]:
        """Return health check information"""
        return {
            "status": (
                "healthy"
                if self.state.status != "error"
                else "unhealthy"
            ),
            "uptime": time.time() - self.state.uptime,
            "active_tasks": self.state.active_tasks,
            "total_tasks": self.state.total_tasks_processed,
            "error_count": self.state.error_count,
            "last_activity": self.state.last_activity.isoformat(),
            "version": self.config.metadata.version,
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Return current metrics"""
        return {
            "tasks_processed": self.state.total_tasks_processed,
            "active_tasks": self.state.active_tasks,
            "error_count": self.state.error_count,
            "cache_hits": int(self.cache.stats().get("hits", 0)),
            "cache_misses": int(self.cache.stats().get("misses", 0)),
            "average_execution_time": (
                float(TASK_DURATION.collect()[0].samples[0].value)
                if TASK_DURATION.collect()
                else 0.0
            ),
        }

    async def clear_cache(self):
        """Clear the cache"""
        await self.cache.clear()
        logger.info("Cache cleared")

    async def cleanup_task_history(self, max_age_hours: int = 24):
        """Clean up old task history entries"""
        current_time = datetime.now()
        keys_to_remove = [
            k
            for k, v in self.task_history.items()
            if (current_time - v.timestamp).total_seconds()
            > max_age_hours * 3600
        ]

        for key in keys_to_remove:
            del self.task_history[key]

        logger.info(
            f"Cleaned up {len(keys_to_remove)} old task history entries"
        )
