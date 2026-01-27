"""
Activity Store for Redis persistence.

Handles storing and retrieving activity records from Redis.
"""
import json
from datetime import datetime
from typing import List, Optional
import redis.asyncio as redis

from .config import config
from .activity import Activity, ActivityCategory


class ActivityStore:
    """
    Redis-backed activity store.

    Redis key structure:
    - `activities:all` - LIST of all activities (max 500)
    - `activities:payments` - LIST of payment activities only

    All activities are stored as JSON strings.
    """

    MAX_ACTIVITIES = 500
    MAX_PAYMENT_ACTIVITIES = 200

    def __init__(self):
        self._redis: Optional[redis.Redis] = None

    async def _get_redis(self) -> Optional[redis.Redis]:
        """Get or create Redis connection."""
        if self._redis is None:
            try:
                self._redis = await redis.from_url(
                    config.redis_url,
                    decode_responses=True
                )
                await self._redis.ping()
            except Exception as e:
                print(f"ActivityStore: Redis connection failed: {e}")
                self._redis = None
        return self._redis

    async def save(self, activity: Activity) -> bool:
        """
        Save an activity to Redis.

        Args:
            activity: Activity record to save

        Returns:
            True if saved successfully, False otherwise
        """
        r = await self._get_redis()
        if not r:
            return False

        try:
            # Serialize activity to JSON
            activity_json = activity.model_dump_json()

            # Add to main activity list (prepend, keep max)
            await r.lpush("activities:all", activity_json)
            await r.ltrim("activities:all", 0, self.MAX_ACTIVITIES - 1)

            # If payment activity, also add to payments list
            if activity.category == ActivityCategory.PAYMENT:
                await r.lpush("activities:payments", activity_json)
                await r.ltrim("activities:payments", 0, self.MAX_PAYMENT_ACTIVITIES - 1)

            return True

        except Exception as e:
            print(f"ActivityStore: Failed to save activity: {e}")
            return False

    async def list(
        self,
        limit: int = 50,
        category: Optional[str] = None,
        offset: int = 0
    ) -> List[Activity]:
        """
        Get activities with optional category filter.

        Args:
            limit: Maximum number of activities to return
            category: Optional category filter (discovery, authorization, etc.)
            offset: Number of activities to skip

        Returns:
            List of Activity records (most recent first)
        """
        r = await self._get_redis()
        if not r:
            return []

        try:
            # Get activities from Redis
            activities_json = await r.lrange("activities:all", offset, offset + limit * 2 - 1)

            activities = []
            for json_str in activities_json:
                try:
                    activity = Activity.model_validate_json(json_str)

                    # Apply category filter if specified
                    if category:
                        if activity.category.value != category:
                            continue

                    activities.append(activity)

                    # Stop if we have enough
                    if len(activities) >= limit:
                        break
                except Exception:
                    continue

            return activities

        except Exception as e:
            print(f"ActivityStore: Failed to list activities: {e}")
            return []

    async def get_payments(self, limit: int = 50) -> List[Activity]:
        """
        Get payment activities with blockchain explorer links.

        Args:
            limit: Maximum number of payment activities to return

        Returns:
            List of payment Activity records (most recent first)
        """
        r = await self._get_redis()
        if not r:
            return []

        try:
            payments_json = await r.lrange("activities:payments", 0, limit - 1)

            payments = []
            for json_str in payments_json:
                try:
                    activity = Activity.model_validate_json(json_str)
                    payments.append(activity)
                except Exception:
                    continue

            return payments

        except Exception as e:
            print(f"ActivityStore: Failed to get payments: {e}")
            return []

    async def count(self) -> int:
        """Get total number of stored activities."""
        r = await self._get_redis()
        if not r:
            return 0

        try:
            return await r.llen("activities:all")
        except Exception:
            return 0

    async def count_payments(self) -> int:
        """Get total number of payment activities."""
        r = await self._get_redis()
        if not r:
            return 0

        try:
            return await r.llen("activities:payments")
        except Exception:
            return 0

    async def clear(self) -> bool:
        """Clear all activities (for testing)."""
        r = await self._get_redis()
        if not r:
            return False

        try:
            await r.delete("activities:all", "activities:payments")
            return True
        except Exception:
            return False


# Global activity store instance
activity_store = ActivityStore()
