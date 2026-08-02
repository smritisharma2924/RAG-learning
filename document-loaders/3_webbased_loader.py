from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

url = 'https://www.apple.com/in/shop/product/myqy3zm/a/earpods-usb-c'
loader = WebBaseLoader(url)

docs = loader.load()

model = ChatGoogleGenerativeAI(model = 'gemini-3.1-flash-lite')

prompt = PromptTemplate(
    template='Answer the following question\n{question}\n based on the following text\n{text}',
    input_varibales=['question', 'text']
)

parser = StrOutputParser()

question = input('Enter your question: ')

chain = prompt | model | parser

print(chain.invoke({'question':question, 'text':docs}))