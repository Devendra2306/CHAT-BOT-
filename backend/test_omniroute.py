import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

def test():
    print("Testing connection to OmniRoute...")
    try:
        llm = ChatOpenAI(
            model="gemini-flash-latest",
            temperature=0.2,
            base_url="http://localhost:20128/v1",
            api_key=os.environ.get("OMNIROUTE_API_KEY")
        )
        response = llm.invoke("Hello, who are you?")
        print("Success! Response:")
        print(response.content)
    except Exception as e:
        print("Error during connection test:", e)

if __name__ == "__main__":
    test()
