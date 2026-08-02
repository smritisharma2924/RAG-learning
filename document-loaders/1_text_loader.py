from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

loader = TextLoader("document-loaders/1_cricket.txt")

docs = loader.load()

model = ChatGoogleGenerativeAI(model = 'gemini-3.1-flash-lite')

prompt = PromptTemplate(
    template = 'Summarize the following poem in 50-100 words\n{poem}',
    input_variables = ['poem']
)

parser = StrOutputParser()

chain = prompt | model | parser

print(docs[0].page_content)
print("\n\nSummary:\n")
print(chain.invoke({'poem': docs[0].page_content}))