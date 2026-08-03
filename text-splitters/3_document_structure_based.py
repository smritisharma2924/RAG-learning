from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

markdown_text = """
# Artificial Intelligence

AI enables machines to perform tasks that typically require human intelligence.

## Machine Learning

Machine Learning allows systems to learn patterns from data.

### Supervised Learning

Uses labeled datasets for prediction.

### Unsupervised Learning

Finds hidden patterns without labels.

## Deep Learning

Deep Learning uses neural networks with many layers.

### Transformers

Transformers power modern Large Language Models.
"""

python_text = """
class Student:
    def __init__(self, name, marks):
        self.name = name
        self.marks = marks

    def grade(self):
        if self.marks >= 90:
            return "A"
        elif self.marks >= 75:
            return "B"
        else:
            return "C"


def average(numbers):
    return sum(numbers) / len(numbers)


students = [
    Student("Alice", 95),
    Student("Bob", 82),
    Student("Charlie", 68)
]

for student in students:
    print(student.name, student.grade())

scores = [90, 80, 70, 60]
print("Average:", average(scores))
"""

md_splitter = RecursiveCharacterTextSplitter.from_language(
    language = Language.MARKDOWN,
    chunk_size = 300,
    chunk_overlap = 0
)

py_splitter = RecursiveCharacterTextSplitter.from_language(
    language = Language.PYTHON,
    chunk_size = 100,
    chunk_overlap = 0
)

md_chunks = md_splitter.split_text(markdown_text)
py_chunks = py_splitter.split_text(python_text)

print("MARKDOWN: \n", md_chunks)
print("\n\n")
print("PYTHON: \n", py_chunks)