import os

from dotenv import load_dotenv


load_dotenv()


def main():
    print("Hello from agentic-rag!")
    print(os.environ.get("LANGSMITH_API_KEY"))

if __name__ == "__main__":
    main()
