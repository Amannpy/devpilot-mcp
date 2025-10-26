from typing import Optional, Dict
from backend.core.models.qwen_model import QwenModelWrapper

class PRService:
    """AI-powered Pull Request analysis service."""

    def __init__(self):
        self.model = QwenModelWrapper()

    async def analyze_pr(
        self, title: str, description: str, diff: Optional[str] = None, code: Optional[str] = None
    ) -> Dict:
        """
        Main PR analysis method.
        Combines heuristic checks and model-based insights.
        """
        heuristic_suggestions = self._heuristic_analysis(title, description, diff)
        model_suggestions = {}
        if code:
            prompt = f"Analyze the following code in context of PR '{title}': {description or ''}\n{code}"
            analysis_text = await self.model.generate_text(prompt, task_type="review")
            model_suggestions = {"analysis": analysis_text}

        return {
            "heuristic": heuristic_suggestions,
            "model": model_suggestions
        }

    def _heuristic_analysis(self, title: str, description: str, diff: Optional[str] = None) -> Dict:
        suggestions = []
        if "fix" in title.lower():
            suggestions.append("Ensure this fix includes regression tests.")
        if "refactor" in title.lower():
            suggestions.append("Verify that refactoring does not change public API behavior.")
        if not description or len(description.split()) < 10:
            suggestions.append("PR description seems short; consider adding more details.")
        if diff and diff.count("\n") > 50:
            suggestions.append("This seems like a large PR; consider splitting into smaller parts.")
        if diff and "TODO" in diff or "FIXME" in diff:
            suggestions.append("Found TODO/FIXME comments in diff; consider resolving before merging.")

        return {"suggestions": suggestions or ["Looks good overall!"], "length_of_diff": len(diff.splitlines()) if diff else 0}
