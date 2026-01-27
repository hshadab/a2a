"""
Server-Sent Events (SSE) Support for A2A Task Streaming

Implements SSE streaming for real-time task progress updates.
https://html.spec.whatwg.org/multipage/server-sent-events.html
"""
import asyncio
import json
from typing import AsyncGenerator, Optional, Dict, Any
from datetime import datetime

from fastapi import Request
from fastapi.responses import StreamingResponse

from .task import TaskStore, Task, TaskState


# ============ SSE Event Types ============

class SSEEventType:
    """Standard A2A SSE event types"""
    TASK_STATUS = "task/status"
    TASK_ARTIFACT = "task/artifact"
    TASK_COMPLETE = "task/complete"
    TASK_ERROR = "task/error"
    HEARTBEAT = "heartbeat"


# ============ SSE Event Formatting ============

def format_sse_event(
    event_type: str,
    data: Dict[str, Any],
    event_id: Optional[str] = None
) -> str:
    """
    Format data as an SSE event.

    SSE format:
        event: <event_type>
        id: <event_id>
        data: <json_data>

    """
    lines = []

    if event_type:
        lines.append(f"event: {event_type}")

    if event_id:
        lines.append(f"id: {event_id}")

    # JSON encode the data
    json_data = json.dumps(data)
    lines.append(f"data: {json_data}")

    # Events are terminated by double newline
    return "\n".join(lines) + "\n\n"


def format_heartbeat() -> str:
    """Format a heartbeat comment (keeps connection alive)"""
    return f": heartbeat {datetime.utcnow().isoformat()}\n\n"


# ============ SSE Streaming ============

async def task_event_stream(
    task_id: str,
    task_store: TaskStore,
    poll_interval: float = 0.5,
    timeout: float = 300.0,
    heartbeat_interval: float = 15.0
) -> AsyncGenerator[str, None]:
    """
    Generate SSE events for task progress.

    Yields events:
    - task/status: When task state changes
    - task/artifact: When new artifacts are added
    - task/complete: When task completes successfully
    - task/error: When task fails

    Args:
        task_id: The task ID to stream
        task_store: The task store to query
        poll_interval: How often to check for updates (seconds)
        timeout: Maximum stream duration (seconds)
        heartbeat_interval: How often to send heartbeats (seconds)
    """
    start_time = asyncio.get_event_loop().time()
    last_heartbeat = start_time
    last_state = None
    last_artifact_count = 0
    event_counter = 0

    while True:
        current_time = asyncio.get_event_loop().time()
        elapsed = current_time - start_time

        # Check timeout
        if elapsed > timeout:
            yield format_sse_event(
                SSEEventType.TASK_ERROR,
                {"error": "Stream timeout", "taskId": task_id},
                event_id=str(event_counter)
            )
            break

        # Send heartbeat if needed
        if current_time - last_heartbeat > heartbeat_interval:
            yield format_heartbeat()
            last_heartbeat = current_time

        # Get task
        task = await task_store.get(task_id)

        if task is None:
            yield format_sse_event(
                SSEEventType.TASK_ERROR,
                {"error": "Task not found", "taskId": task_id},
                event_id=str(event_counter)
            )
            break

        # Check for state change
        if task.state != last_state:
            event_counter += 1
            yield format_sse_event(
                SSEEventType.TASK_STATUS,
                {
                    "taskId": task.id,
                    "state": task.state.value,
                    "updatedAt": task.updatedAt.isoformat()
                },
                event_id=str(event_counter)
            )
            last_state = task.state

        # Check for new artifacts
        if len(task.artifacts) > last_artifact_count:
            for artifact in task.artifacts[last_artifact_count:]:
                event_counter += 1
                yield format_sse_event(
                    SSEEventType.TASK_ARTIFACT,
                    {
                        "taskId": task.id,
                        "artifact": {
                            "id": artifact.id,
                            "name": artifact.name,
                            "mimeType": artifact.mimeType,
                            "createdAt": artifact.createdAt.isoformat()
                        }
                    },
                    event_id=str(event_counter)
                )
            last_artifact_count = len(task.artifacts)

        # Check for completion
        if task.state == TaskState.COMPLETED:
            event_counter += 1
            yield format_sse_event(
                SSEEventType.TASK_COMPLETE,
                task.to_response(),
                event_id=str(event_counter)
            )
            break

        # Check for failure
        if task.state == TaskState.FAILED:
            event_counter += 1
            error_data = {
                "taskId": task.id,
                "error": task.error.model_dump() if task.error else {"message": "Unknown error"}
            }
            yield format_sse_event(
                SSEEventType.TASK_ERROR,
                error_data,
                event_id=str(event_counter)
            )
            break

        # Wait before next poll
        await asyncio.sleep(poll_interval)


def create_sse_response(
    task_id: str,
    task_store: TaskStore,
    poll_interval: float = 0.5,
    timeout: float = 300.0
) -> StreamingResponse:
    """
    Create a FastAPI StreamingResponse for task SSE.

    Usage in endpoint:
        @app.get("/tasks/{task_id}/stream")
        async def stream_task(task_id: str):
            return create_sse_response(task_id, task_store)
    """
    return StreamingResponse(
        task_event_stream(task_id, task_store, poll_interval, timeout),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


# ============ Multi-Task Streaming ============

async def multi_task_event_stream(
    task_ids: list,
    task_store: TaskStore,
    poll_interval: float = 0.5,
    timeout: float = 300.0,
    heartbeat_interval: float = 15.0
) -> AsyncGenerator[str, None]:
    """
    Generate SSE events for multiple tasks.

    Useful for monitoring batch operations.
    """
    start_time = asyncio.get_event_loop().time()
    last_heartbeat = start_time
    task_states: Dict[str, TaskState] = {}
    completed_tasks: set = set()
    event_counter = 0

    while True:
        current_time = asyncio.get_event_loop().time()
        elapsed = current_time - start_time

        # Check timeout
        if elapsed > timeout:
            yield format_sse_event(
                SSEEventType.TASK_ERROR,
                {"error": "Stream timeout"},
                event_id=str(event_counter)
            )
            break

        # Send heartbeat if needed
        if current_time - last_heartbeat > heartbeat_interval:
            yield format_heartbeat()
            last_heartbeat = current_time

        # Check all tasks
        for task_id in task_ids:
            if task_id in completed_tasks:
                continue

            task = await task_store.get(task_id)
            if task is None:
                continue

            # Check for state change
            if task.state != task_states.get(task_id):
                event_counter += 1
                yield format_sse_event(
                    SSEEventType.TASK_STATUS,
                    {
                        "taskId": task.id,
                        "state": task.state.value,
                        "updatedAt": task.updatedAt.isoformat()
                    },
                    event_id=str(event_counter)
                )
                task_states[task_id] = task.state

            # Check for completion
            if task.state in [TaskState.COMPLETED, TaskState.FAILED]:
                completed_tasks.add(task_id)
                event_counter += 1
                event_type = SSEEventType.TASK_COMPLETE if task.state == TaskState.COMPLETED else SSEEventType.TASK_ERROR
                yield format_sse_event(
                    event_type,
                    task.to_response(),
                    event_id=str(event_counter)
                )

        # Check if all tasks complete
        if len(completed_tasks) == len(task_ids):
            break

        await asyncio.sleep(poll_interval)


def create_multi_sse_response(
    task_ids: list,
    task_store: TaskStore,
    poll_interval: float = 0.5,
    timeout: float = 300.0
) -> StreamingResponse:
    """Create SSE response for multiple tasks"""
    return StreamingResponse(
        multi_task_event_stream(task_ids, task_store, poll_interval, timeout),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )
