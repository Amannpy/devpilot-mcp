"""
Utility functions and response helpers for API routes.
Ensures consistent error handling, standardized responses, and logging support.
"""

from fastapi import HTTPException
from backend.core.utils.logger import get_logger

logger = get_logger(__name__)


# ----------------------------
# Response Helpers
# ----------------------------

def success_response(data=None, message: str = "Request processed successfully"):
    """
    Standard format for all successful API responses.
    """
    response = {
        "status": "success",
        "message": message,
    }
    if data is not None:
        response["data"] = data
    return response


def error_response(detail: str, status_code: int = 400, log_error: bool = True):
    """
    Standard format for all API errors.
    """
    if log_error:
        logger.error(f"❌ API Error ({status_code}): {detail}")

    raise HTTPException(
        status_code=status_code,
        detail={
            "status": "error",
            "message": detail,
            "code": status_code,
        },
    )


# ----------------------------
# Async Utility Wrappers
# ----------------------------

async def safe_execute(coro, context: str = "operation"):
    """
    Safely execute an async function with centralized error handling and logging.
    Useful for async service calls inside routes.
    """
    try:
        return await coro
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"❌ Exception during {context}: {e}")
        raise HTTPException(
            status_code=500,
            detail={"status": "error", "message": f"Internal server error during {context}"},
        )


# ----------------------------
# Example Usage (for reference)
# ----------------------------
# return success_response(data=result, message="PR analyzed successfully")
# or
# await safe_execute(pr_service.analyze_pr(...), "PR Analysis")
# or
# error_response("Invalid PR ID", 404)
