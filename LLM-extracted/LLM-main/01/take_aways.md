### Term: Top-K Retrieval Behavior

#### Static Top-K (Config-Driven)
The search limit is determined strictly by an external environment setting:
* `"size": self.settings.get("search_size")`

* **Pros**: Predictable token costs, strict context limits, and centralized configuration management.
* **Cons**: No flexibility for complex queries and wastes tokens on irrelevant documents.

---

#### Dynamic Top-K (Argument-Driven)
The search limit is determined on-the-fly by the calling script:
* `num_results=5`

* **Pros**: High query adaptability, optimized API costs, and easy A/B testing.
* **Cons**: Risk of context overflow and logic becomes scattered across code.
