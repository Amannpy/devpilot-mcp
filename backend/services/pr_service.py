"""
AI-powered Pull Request (PR) analysis service for MCP backend.
Combines heuristic checks, model-based insights, and optional RAG context.
"""

from typing import Optional, Dict
from backend.core.models.qwen_model import QwenModelWrapper
from backend.core.utils.logger import get_logger
from backend.services.rag_service import RAGService

logger = get_logger(__name__)

class PRService:
    """AI-powered Pull Request analysis service."""

    def __init__(self, use_rag: bool = False):
        self.model = QwenModelWrapper()
        self.use_rag = use_rag
        self.rag_service = RAGService() if use_rag else None

    async def analyze_pr(
        self, title: str, description: Optional[str] = None, diff: Optional[str] = None, code: Optional[str] = None
    ) -> Dict:
        """
        Main PR analysis method.
        Combines heuristic checks and model-based insights, optionally enriched with RAG context.
        """
        try:
            logger.info(f"🔍 Analyzing PR: {title}")

            # Heuristic analysis
            heuristic_suggestions = self._heuristic_analysis(title, description, diff)

            # Prepare context for model
            model_suggestions = {}
            if code:
                context_text = ""
                if self.use_rag and self.rag_service:
                    context_text = await self.rag_service.get_context_for_code(code)

                prompt = f"Analyze the following code in context of PR '{title}': {description or ''}\n{context_text}\n{code}"
                analysis_text = await self.model.generate_text(prompt, task_type="review")
                model_suggestions = {"analysis": analysis_text}

            result = {
                "heuristic": heuristic_suggestions,
                "model": model_suggestions
            }

            logger.info(f"✅ PR analysis completed for: {title}")
            return result

        except Exception as e:
            logger.exception(f"❌ Failed to analyze PR '{title}': {e}")
            return {
                "heuristic": {"suggestions": [], "length_of_diff": len(diff.splitlines()) if diff else 0},
                "model": {"analysis": f"Error analyzing PR: {str(e)}"}
            }

    def _heuristic_analysis(self, title: str, description: Optional[str] = None, diff: Optional[str] = None) -> Dict:
        """
        Simple heuristic checks for PR quality.
        """
        suggestions = []

        if "fix" in title.lower():
            suggestions.append("Ensure this fix includes regression tests.")
        if "refactor" in title.lower():
            suggestions.append("Verify that refactoring does not change public API behavior.")
        if not description or len(description.split()) < 10:
            suggestions.append("PR description seems short; consider adding more details.")
        if diff and diff.count("\n") > 50:
            suggestions.append("This seems like a large PR; consider splitting into smaller parts.")
        if diff and ("TODO" in diff or "FIXME" in diff):
            suggestions.append("Found TODO/FIXME comments in diff; consider resolving before merging.")

        return {
            "suggestions": suggestions or ["Looks good overall!"],
            "length_of_diff": len(diff.splitlines()) if diff else 0
        }
