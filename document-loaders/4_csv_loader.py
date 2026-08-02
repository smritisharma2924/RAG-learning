from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path = 'document-loaders/4_abc-company-info.csv')

data = loader.load()

print(len(data))
print(data[0])