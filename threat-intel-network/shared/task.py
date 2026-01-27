"""
A2A Task Lifecycle Management

Implements the A2A task state machine for tracking skill invocations:
- submitted: Task received, queued for processing
- working: Task is being executed
- completed: Task finished successfully
- failed: Task execution failed
"""
import uuid
import asyncio
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


# ============ Task State Machine ============

class TaskState(str, Enum):
    """A2A task states"""
    SUBMITTED = "submitted"
    WORKING = "working"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskArtifact(BaseModel):
    """
    An artifact produced by a task.

    Artifacts can be intermediate or final outputs like:
    - Classifications
    - Proofs
    - Reports
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    mimeType: str = "application/json"
    data: Any
    createdAt: datetime = Field(default_factory=datetime.utcnow)


class TaskError(BaseModel):
    """Task error information"""
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


class Task(BaseModel):
    """
    A2A Task representing a skill invocation.

    The task tracks:
    - State transitions (submitted -> working -> completed/failed)
    - Input parameters
    - Output results
    - Artifacts produced during execution
    - Payment tracking for x402
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    contextId: Optional[str] = None  # Groups related tasks
    skillId: str
    state: TaskState = TaskState.SUBMITTED
    input: Dict[str, Any] = Field(default_factory=dict)
    output: Optional[Dict[str, Any]] = None
    artifacts: List[TaskArtifact] = Field(default_factory=list)
    error: Optional[TaskError] = None

    # Timestamps
    createdAt: datetime = Field(default_factory=datetime.utcnow)
    updatedAt: datetime = Field(default_factory=datetime.utcnow)
    completedAt: Optional[datetime] = None

    # Payment tracking (x402 extension)
    paymentRequired: bool = False
    paymentAmount: Optional[str] = None
    paymentCurrency: str = "USDC"
    paymentChain: str = "eip155:8453"
    paymentReceipt: Optional[str] = None
    paymentVerified: bool = False

    def transition_to(self, new_state: TaskState):
        """Transition task to a new state"""
        valid_transitions = {
            TaskState.SUBMITTED: [TaskState.WORKING, TaskState.FAILED],
            TaskState.WORKING: [TaskState.COMPLETED, TaskState.FAILED],
            TaskState.COMPLETED: [],  # Terminal state
            TaskState.FAILED: [],     # Terminal state
        }

        if new_state not in valid_transitions.get(self.state, []):
            raise ValueError(
                f"Invalid state transition: {self.state} -> {new_state}"
            )

        self.state = new_state
        self.updatedAt = datetime.utcnow()

        if new_state in [TaskState.COMPLETED, TaskState.FAILED]:
            self.completedAt = datetime.utcnow()

    def add_artifact(self, name: str, data: Any, mime_type: str = "application/json"):
        """Add an artifact to the task"""
        artifact = TaskArtifact(name=name, mimeType=mime_type, data=data)
        self.artifacts.append(artifact)
        self.updatedAt = datetime.utcnow()
        return artifact

    def set_output(self, output: Dict[str, Any]):
        """Set the task output"""
        self.output = output
        self.updatedAt = datetime.utcnow()

    def set_error(self, code: str, message: str, details: Optional[Dict[str, Any]] = None):
        """Set task error and transition to failed state"""
        self.error = TaskError(code=code, message=message, details=details)
        self.transition_to(TaskState.FAILED)

    def to_response(self) -> Dict[str, Any]:
        """Convert task to A2A response format"""
        response = {
            "id": self.id,
            "contextId": self.contextId,
            "skillId": self.skillId,
            "state": self.state.value,
            "createdAt": self.createdAt.isoformat(),
            "updatedAt": self.updatedAt.isoformat(),
        }

        if self.output:
            response["output"] = self.output

        if self.artifacts:
            response["artifacts"] = [
                {
                    "id": a.id,
                    "name": a.name,
                    "mimeType": a.mimeType,
                    "data": a.data,
                    "createdAt": a.createdAt.isoformat()
                }
                for a in self.artifacts
            ]

        if self.error:
            response["error"] = {
                "code": self.error.code,
                "message": self.error.message,
                "details": self.error.details
            }

        if self.completedAt:
            response["completedAt"] = self.completedAt.isoformat()

        return response


# ============ Task Store ============

class TaskStore:
    """
    In-memory task storage.

    In production, this should be backed by Redis or a database
    for persistence and distributed access.
    """

    def __init__(self, max_tasks: int = 10000):
        self._tasks: Dict[str, Task] = {}
        self._max_tasks = max_tasks
        self._lock = asyncio.Lock()

    async def create(
        self,
        skill_id: str,
        input_data: Dict[str, Any],
        context_id: Optional[str] = None,
        payment_required: bool = False,
        payment_amount: Optional[str] = None,
        payment_currency: str = "USDC",
        payment_chain: str = "eip155:8453"
    ) -> Task:
        """Create a new task"""
        async with self._lock:
            # Cleanup old tasks if at capacity
            if len(self._tasks) >= self._max_tasks:
                await self._cleanup_old_tasks()

            task = Task(
                skillId=skill_id,
                input=input_data,
                contextId=context_id,
                paymentRequired=payment_required,
                paymentAmount=payment_amount,
                paymentCurrency=payment_currency,
                paymentChain=payment_chain
            )
            self._tasks[task.id] = task
            return task

    async def get(self, task_id: str) -> Optional[Task]:
        """Get a task by ID"""
        return self._tasks.get(task_id)

    async def update(self, task: Task):
        """Update a task"""
        async with self._lock:
            if task.id in self._tasks:
                self._tasks[task.id] = task

    async def delete(self, task_id: str) -> bool:
        """Delete a task"""
        async with self._lock:
            if task_id in self._tasks:
                del self._tasks[task_id]
                return True
            return False

    async def get_by_context(self, context_id: str) -> List[Task]:
        """Get all tasks in a context"""
        return [
            task for task in self._tasks.values()
            if task.contextId == context_id
        ]

    async def get_by_state(self, state: TaskState) -> List[Task]:
        """Get all tasks in a given state"""
        return [
            task for task in self._tasks.values()
            if task.state == state
        ]

    async def get_recent(self, limit: int = 100) -> List[Task]:
        """Get most recent tasks"""
        tasks = sorted(
            self._tasks.values(),
            key=lambda t: t.createdAt,
            reverse=True
        )
        return tasks[:limit]

    async def _cleanup_old_tasks(self):
        """Remove oldest completed/failed tasks"""
        completed = [
            t for t in self._tasks.values()
            if t.state in [TaskState.COMPLETED, TaskState.FAILED]
        ]
        completed.sort(key=lambda t: t.completedAt or t.createdAt)

        # Remove oldest 10%
        to_remove = completed[:max(1, len(completed) // 10)]
        for task in to_remove:
            del self._tasks[task.id]

    def count(self) -> int:
        """Get total task count"""
        return len(self._tasks)

    def count_by_state(self) -> Dict[str, int]:
        """Get task count by state"""
        counts = {state.value: 0 for state in TaskState}
        for task in self._tasks.values():
            counts[task.state.value] += 1
        return counts


# ============ Task Execution Helper ============

async def execute_task(
    task: Task,
    store: TaskStore,
    handler: callable
) -> Task:
    """
    Execute a task using the provided handler.

    The handler should be an async function that takes
    the task input and returns the output.
    """
    # Transition to working
    task.transition_to(TaskState.WORKING)
    await store.update(task)

    try:
        # Execute the handler
        result = await handler(task.input)

        # Set output and transition to completed
        task.set_output(result)
        task.transition_to(TaskState.COMPLETED)
        await store.update(task)

    except Exception as e:
        # Set error and transition to failed
        task.set_error(
            code="EXECUTION_ERROR",
            message=str(e),
            details={"type": type(e).__name__}
        )
        await store.update(task)

    return task


# ============ Global Task Store ============

# Shared task store instance
task_store = TaskStore()
