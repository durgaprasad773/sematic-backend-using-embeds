from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAI
from langchain_groq import ChatGroq
from properties import GROQ_API_KEY, OPEN_AI_API_KEY, GOOGLE_API_KEY
from config import LLM_PROVIDER, Groq_model, OpenAI_model, Google_model


class LLM:
    def __init__(self, provider=LLM_PROVIDER):
        self.provider = provider.lower()
        if self.provider == "groq":
            self.llm = ChatGroq(
                api_key=GROQ_API_KEY,
                model=Groq_model,
            )

        elif self.provider == "openai":
            self.llm = OpenAI(api_key=OPEN_AI_API_KEY, model=OpenAI_model)
        elif self.provider == "google":
            self.llm = ChatGoogleGenerativeAI(
                api_key=GOOGLE_API_KEY, model=Google_model
            )
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def invoke(
        self,
        prompt_text,
    ):
        return self.llm.invoke(prompt_text)


if __name__ == "__main__":
    llm = LLM(provider="groq")
    response = llm.invoke("Hello, how are you?")
    print(response)
