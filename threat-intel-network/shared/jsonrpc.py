"""
JSON-RPC 2.0 Transport Layer for A2A Protocol

Implements the JSON-RPC 2.0 specification for A2A agent communication.
https://www.jsonrpc.org/specification
"""
import asyncio
import uuid
from typing import Optional, Dict, Any, Callable, Awaitable, List, Union
from functools import wraps
from pydantic import BaseModel, Field
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse


# ============ JSON-RPC 2.0 Models ============

class JSONRPCError(BaseModel):
    """JSON-RPC 2.0 error object"""
    code: int
    message: str
    data: Optional[Any] = None


class JSONRPCRequest(BaseModel):
    """JSON-RPC 2.0 request object"""
    jsonrpc: str = "2.0"
    method: str
    params: Optional[Dict[str, Any]] = None
    id: Optional[Union[str, int]] = None


class JSONRPCResponse(BaseModel):
    """JSON-RPC 2.0 response object"""
    jsonrpc: str = "2.0"
    result: Optional[Any] = None
    error: Optional[JSONRPCError] = None
    id: Optional[Union[str, int]] = None


# ============ Standard Error Codes ============

class JSONRPCErrorCodes:
    """Standard JSON-RPC 2.0 error codes"""
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    # Server errors: -32000 to -32099
    SERVER_ERROR = -32000
    PAYMENT_REQUIRED = -32001
    TASK_NOT_FOUND = -32002
    TASK_FAILED = -32003


def create_error_response(
    code: int,
    message: str,
    data: Optional[Any] = None,
    request_id: Optional[Union[str, int]] = None
) -> JSONRPCResponse:
    """Create a JSON-RPC error response"""
    return JSONRPCResponse(
        error=JSONRPCError(code=code, message=message, data=data),
        id=request_id
    )


def create_success_response(
    result: Any,
    request_id: Optional[Union[str, int]] = None
) -> JSONRPCResponse:
    """Create a JSON-RPC success response"""
    return JSONRPCResponse(result=result, id=request_id)


# ============ JSON-RPC Router ============

# Type alias for method handlers
MethodHandler = Callable[[Dict[str, Any]], Awaitable[Any]]


class JSONRPCRouter:
    """
    Router for JSON-RPC methods.

    Usage:
        router = JSONRPCRouter()

        @router.method("task/send")
        async def task_send(params: dict) -> dict:
            # Handle task/send method
            return {"taskId": "123", "state": "submitted"}

        # In FastAPI
        app.post("/a2a")(router.create_endpoint())
    """

    def __init__(self):
        self._methods: Dict[str, MethodHandler] = {}

    def method(self, name: str):
        """Decorator to register a method handler"""
        def decorator(func: MethodHandler):
            self._methods[name] = func
            @wraps(func)
            async def wrapper(*args, **kwargs):
                return await func(*args, **kwargs)
            return wrapper
        return decorator

    def register(self, name: str, handler: MethodHandler):
        """Register a method handler directly"""
        self._methods[name] = handler

    def get_methods(self) -> List[str]:
        """Get list of registered method names"""
        return list(self._methods.keys())

    async def handle_request(self, request: JSONRPCRequest) -> JSONRPCResponse:
        """Handle a single JSON-RPC request"""
        # Validate jsonrpc version
        if request.jsonrpc != "2.0":
            return create_error_response(
                JSONRPCErrorCodes.INVALID_REQUEST,
                "Invalid JSON-RPC version",
                request_id=request.id
            )

        # Find method handler
        handler = self._methods.get(request.method)
        if handler is None:
            return create_error_response(
                JSONRPCErrorCodes.METHOD_NOT_FOUND,
                f"Method not found: {request.method}",
                request_id=request.id
            )

        # Execute method
        try:
            params = request.params or {}
            result = await handler(params)
            return create_success_response(result, request_id=request.id)
        except PaymentRequiredError as e:
            return create_error_response(
                JSONRPCErrorCodes.PAYMENT_REQUIRED,
                "Payment required",
                data=e.payment_info,
                request_id=request.id
            )
        except TaskNotFoundError as e:
            return create_error_response(
                JSONRPCErrorCodes.TASK_NOT_FOUND,
                str(e),
                request_id=request.id
            )
        except ValueError as e:
            return create_error_response(
                JSONRPCErrorCodes.INVALID_PARAMS,
                str(e),
                request_id=request.id
            )
        except Exception as e:
            return create_error_response(
                JSONRPCErrorCodes.INTERNAL_ERROR,
                f"Internal error: {str(e)}",
                request_id=request.id
            )

    async def handle_batch(self, requests: List[JSONRPCRequest]) -> List[JSONRPCResponse]:
        """Handle a batch of JSON-RPC requests"""
        # Process all requests concurrently
        tasks = [self.handle_request(req) for req in requests]
        responses = await asyncio.gather(*tasks)
        # Filter out notifications (requests without id)
        return [r for r in responses if r.id is not None]

    def create_endpoint(self):
        """Create a FastAPI endpoint handler"""
        router = self

        async def endpoint(request: Request):
            try:
                body = await request.json()
            except Exception:
                return JSONResponse(
                    content=create_error_response(
                        JSONRPCErrorCodes.PARSE_ERROR,
                        "Parse error"
                    ).model_dump(),
                    status_code=200
                )

            # Handle batch request
            if isinstance(body, list):
                if not body:
                    return JSONResponse(
                        content=create_error_response(
                            JSONRPCErrorCodes.INVALID_REQUEST,
                            "Empty batch"
                        ).model_dump(),
                        status_code=200
                    )
                requests = [JSONRPCRequest(**req) for req in body]
                responses = await router.handle_batch(requests)
                return JSONResponse(
                    content=[r.model_dump() for r in responses],
                    status_code=200
                )

            # Handle single request
            try:
                rpc_request = JSONRPCRequest(**body)
            except Exception as e:
                return JSONResponse(
                    content=create_error_response(
                        JSONRPCErrorCodes.INVALID_REQUEST,
                        f"Invalid request: {str(e)}"
                    ).model_dump(),
                    status_code=200
                )

            response = await router.handle_request(rpc_request)

            # Notifications don't return a response
            if rpc_request.id is None:
                return JSONResponse(content=None, status_code=204)

            return JSONResponse(
                content=response.model_dump(),
                status_code=200
            )

        return endpoint


# ============ Custom Exceptions ============

class PaymentRequiredError(Exception):
    """Raised when payment is required for a method"""
    def __init__(self, payment_info: Dict[str, Any]):
        self.payment_info = payment_info
        super().__init__("Payment required")


class TaskNotFoundError(Exception):
    """Raised when a task is not found"""
    def __init__(self, task_id: str):
        self.task_id = task_id
        super().__init__(f"Task not found: {task_id}")


# ============ Helper Functions ============

def create_jsonrpc_endpoint(router: JSONRPCRouter):
    """Create a FastAPI endpoint from a JSONRPCRouter"""
    return router.create_endpoint()


def generate_request_id() -> str:
    """Generate a unique request ID"""
    return str(uuid.uuid4())
