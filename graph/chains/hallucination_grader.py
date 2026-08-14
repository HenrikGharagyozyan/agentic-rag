from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence


llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0,
)


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: bool = Field(
        description="Answer is grounded in the facts."
    )

structured_llm_grader = llm.with_structured_output(GradeHallucinations)


system = """You are a grader assessing whether an LLM generation is grounded in
and supported by a set of retrieved facts.

Give a binary score.
True means that the answer is grounded in and supported by the facts.
False means that the answer contains information that is not supported by the facts.
"""


hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Set of facts:\n\n{documents}\n\n"
            "LLM generation:\n\n{generation}",
        ),
    ]
)


hallucination_grader: RunnableSequence = (
    hallucination_prompt | structured_llm_grader
)