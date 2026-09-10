from collections import defaultdict
from typing import Any


class FeatureEngineer:
    """
    Converts a user's complete submission history
    into a numerical skill profile.

    The profile contains:
    - Lifetime statistics
    - Last 100 submission statistics
    - Last 20 submission statistics
    - Topic-wise statistics
    - Difficulty-wise statistics

    The generated features will later be used by
    the recommendation scoring / ML model.
    """

    def __init__(
        self,
        medium_term_window: int = 100,
        recent_window: int = 20,
    ):
        self.medium_term_window = medium_term_window
        self.recent_window = recent_window

    # ========================================================
    # Main Feature Engineering Pipeline
    # ========================================================

    def build_user_features(
        self,
        submissions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Build a complete user skill profile.

        Expected submission format:

        {
            "problem_id": "...",
            "difficulty": "Easy",
            "topics": ["Array", "Hash Table"],
            "accepted": True,
            "attempts": 1,
            "time_taken": 15,
            "hints_used": 0,
            "editorial_viewed": False
        }

        The submissions should be ordered from
        oldest -> newest.
        """

        if not submissions:
            return self._empty_features()

        # ----------------------------------------------------
        # Keep ALL submissions
        # ----------------------------------------------------

        lifetime = submissions

        # Last 100 submissions
        medium_term = submissions[
            -self.medium_term_window:
        ]

        # Last 20 submissions
        recent = submissions[
            -self.recent_window:
        ]

        return {
            "total_submissions": len(lifetime),

            # ------------------------------------------------
            # Lifetime performance
            # ------------------------------------------------

            "lifetime": (
                self._calculate_overall_features(
                    lifetime
                )
            ),

            # ------------------------------------------------
            # Medium-term performance
            # ------------------------------------------------

            "medium_term": (
                self._calculate_overall_features(
                    medium_term
                )
            ),

            # ------------------------------------------------
            # Recent performance
            # ------------------------------------------------

            "recent": (
                self._calculate_overall_features(
                    recent
                )
            ),

            # ------------------------------------------------
            # Difficulty performance
            # ------------------------------------------------

            "difficulty": (
                self._calculate_difficulty_features(
                    lifetime
                )
            ),

            # ------------------------------------------------
            # Topic performance
            # ------------------------------------------------

            "topics": (
                self._calculate_topic_features(
                    lifetime
                )
            ),

            # ------------------------------------------------
            # Recent topic performance
            # ------------------------------------------------

            "recent_topics": (
                self._calculate_topic_features(
                    recent
                )
            ),
        }

    # ========================================================
    # Overall Features
    # ========================================================

    def _calculate_overall_features(
        self,
        submissions: list[dict[str, Any]],
    ) -> dict[str, float]:
        """
        Calculate overall performance for a
        specific submission window.
        """

        total = len(submissions)

        if total == 0:
            return {
                "submissions": 0,
                "acceptance_rate": 0.0,
                "failure_rate": 0.0,
                "average_attempts": 0.0,
                "average_time_taken": 0.0,
            }

        accepted = sum(
            1
            for submission in submissions
            if submission.get(
                "accepted",
                False,
            )
        )

        total_attempts = sum(
            submission.get(
                "attempts",
                1,
            )
            for submission in submissions
        )

        total_time = sum(
            submission.get(
                "time_taken",
                0,
            )
            for submission in submissions
        )

        return {
            "submissions": total,

            "acceptance_rate": (
                self._safe_ratio(
                    accepted,
                    total,
                )
            ),

            "failure_rate": (
                self._safe_ratio(
                    total - accepted,
                    total,
                )
            ),

            "average_attempts": (
                total_attempts / total
            ),

            "average_time_taken": (
                total_time / total
            ),
        }

    # ========================================================
    # Difficulty Features
    # ========================================================

    def _calculate_difficulty_features(
        self,
        submissions: list[dict[str, Any]],
    ) -> dict[str, dict[str, float]]:
        """
        Calculate performance separately for
        Easy, Medium and Hard problems.
        """

        difficulties = [
            "Easy",
            "Medium",
            "Hard",
        ]

        result = {}

        for difficulty in difficulties:

            difficulty_submissions = [
                submission
                for submission in submissions
                if str(
                    submission.get(
                        "difficulty",
                        ""
                    )
                ).lower()
                == difficulty.lower()
            ]

            total = len(
                difficulty_submissions
            )

            if total == 0:
                result[
                    difficulty.lower()
                ] = {
                    "attempted": 0,
                    "acceptance_rate": 0.0,
                    "failure_rate": 0.0,
                }

                continue

            accepted = sum(
                1
                for submission
                in difficulty_submissions
                if submission.get(
                    "accepted",
                    False,
                )
            )

            result[
                difficulty.lower()
            ] = {
                "attempted": total,

                "acceptance_rate": (
                    self._safe_ratio(
                        accepted,
                        total,
                    )
                ),

                "failure_rate": (
                    self._safe_ratio(
                        total - accepted,
                        total,
                    )
                ),
            }

        return result

    # ========================================================
    # Topic Features
    # ========================================================

    def _calculate_topic_features(
        self,
        submissions: list[dict[str, Any]],
    ) -> dict[str, dict[str, float]]:
        """
        Calculate performance for every topic
        encountered in the submission history.
        """

        topic_stats = defaultdict(
            lambda: {
                "attempted": 0,
                "accepted": 0,
            }
        )

        for submission in submissions:

            accepted = submission.get(
                "accepted",
                False,
            )

            topics = submission.get(
                "topics",
                [],
            )

            for topic in topics:

                topic = str(
                    topic
                ).strip()

                if not topic:
                    continue

                topic_stats[
                    topic
                ]["attempted"] += 1

                if accepted:
                    topic_stats[
                        topic
                    ]["accepted"] += 1

        result = {}

        for topic, stats in topic_stats.items():

            attempted = stats[
                "attempted"
            ]

            accepted = stats[
                "accepted"
            ]

            acceptance_rate = (
                self._safe_ratio(
                    accepted,
                    attempted,
                )
            )

            result[topic] = {
                "attempted": attempted,

                "accepted": accepted,

                "acceptance_rate": (
                    acceptance_rate
                ),

                "failure_rate": (
                    1.0 - acceptance_rate
                ),
            }

        return result

    # ========================================================
    # Weakness Detection
    # ========================================================

    def get_weak_topics(
        self,
        user_features: dict[str, Any],
        threshold: float = 0.5,
        minimum_attempts: int = 2,
    ) -> list[str]:
        """
        Return topics where the user's
        lifetime acceptance rate is below
        the specified threshold.
        """

        topics = user_features.get(
            "topics",
            {},
        )

        weak_topics = []

        for topic, stats in topics.items():

            attempted = stats.get(
                "attempted",
                0,
            )

            acceptance_rate = stats.get(
                "acceptance_rate",
                0.0,
            )

            if (
                attempted >= minimum_attempts
                and acceptance_rate < threshold
            ):
                weak_topics.append(topic)

        return weak_topics

    # ========================================================
    # Strength Detection
    # ========================================================

    def get_strong_topics(
        self,
        user_features: dict[str, Any],
        threshold: float = 0.75,
        minimum_attempts: int = 2,
    ) -> list[str]:
        """
        Return topics where the user's
        lifetime acceptance rate is above
        the specified threshold.
        """

        topics = user_features.get(
            "topics",
            {},
        )

        strong_topics = []

        for topic, stats in topics.items():

            attempted = stats.get(
                "attempted",
                0,
            )

            acceptance_rate = stats.get(
                "acceptance_rate",
                0.0,
            )

            if (
                attempted >= minimum_attempts
                and acceptance_rate >= threshold
            ):
                strong_topics.append(topic)

        return strong_topics

    # ========================================================
    # Recent Weakness Detection
    # ========================================================

    def get_recent_weak_topics(
        self,
        user_features: dict[str, Any],
        threshold: float = 0.5,
        minimum_attempts: int = 2,
    ) -> list[str]:
        """
        Detect topics the user is currently
        struggling with based on recent submissions.
        """

        topics = user_features.get(
            "recent_topics",
            {},
        )

        weak_topics = []

        for topic, stats in topics.items():

            attempted = stats.get(
                "attempted",
                0,
            )

            acceptance_rate = stats.get(
                "acceptance_rate",
                0.0,
            )

            if (
                attempted >= minimum_attempts
                and acceptance_rate < threshold
            ):
                weak_topics.append(topic)

        return weak_topics

    # ========================================================
    # Utility
    # ========================================================

    @staticmethod
    def _safe_ratio(
        numerator: int,
        denominator: int,
    ) -> float:
        """
        Prevent division by zero.
        """

        if denominator == 0:
            return 0.0

        return numerator / denominator

    # ========================================================
    # Empty Profile
    # ========================================================

    @staticmethod
    def _empty_features() -> dict[str, Any]:
        """
        Return a valid empty feature profile
        for a new user.
        """

        return {
            "total_submissions": 0,

            "lifetime": {
                "submissions": 0,
                "acceptance_rate": 0.0,
                "failure_rate": 0.0,
                "average_attempts": 0.0,
                "average_time_taken": 0.0,
            },

            "medium_term": {
                "submissions": 0,
                "acceptance_rate": 0.0,
                "failure_rate": 0.0,
                "average_attempts": 0.0,
                "average_time_taken": 0.0,
            },

            "recent": {
                "submissions": 0,
                "acceptance_rate": 0.0,
                "failure_rate": 0.0,
                "average_attempts": 0.0,
                "average_time_taken": 0.0,
            },

            "difficulty": {
                "easy": {
                    "attempted": 0,
                    "acceptance_rate": 0.0,
                    "failure_rate": 0.0,
                },
                "medium": {
                    "attempted": 0,
                    "acceptance_rate": 0.0,
                    "failure_rate": 0.0,
                },
                "hard": {
                    "attempted": 0,
                    "acceptance_rate": 0.0,
                    "failure_rate": 0.0,
                },
            },

            "topics": {},

            "recent_topics": {},
        }