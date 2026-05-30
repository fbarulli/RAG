from enum import Enum

class EncodeMode(str, Enum):
    question = "question"
    qa       = "qa"
    answer   = "answer"

    @property
    def suffix(self) -> str:
        return {"qa": "_qa", "answer": "_answer"}.get(self.value, "")

    def encode_text(self, question: str, answer: str) -> str:
        if self == EncodeMode.qa:
            return f"{question} {answer}"
        if self == EncodeMode.answer:
            return answer
        return question
