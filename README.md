import openai
import chromadb
import re
import os
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# Set OpenAI API Key
openai.api_key = "your-openai-api-key"

# Setup for LangChain
embeddings = OpenAIEmbeddings()

# Initialize ChromaDB client
client = chromadb.Client()

# Create or connect to a Chroma collection
collection_name = "golang_test_files"
if collection_name not in client.list_collections():
    collection = client.create_collection(collection_name)
else:
    collection = client.get_collection(collection_name)

# Create the prompt template for test generation
test_generation_prompt = """
You are a Golang unit test generator. You will be provided with the following:
- The Go code for a file containing multiple functions
- The name of a test file (containing previous tests for similar functions)

You need to generate high-quality, efficient unit tests for **all functions** in the provided Go file.

Unit tests should cover edge cases, typical cases, and failure scenarios for each function.

Here is the code in the Go file:
{go_file_code}

Here are the previous tests (if any) from the test file "{test_file_name}":
{previous_tests}

Your task is to return the unit tests for all functions in the Go file in the Go testing format.
"""

# Initialize the LangChain LLM with OpenAI API
llm = OpenAI(temperature=0.2)

# Create LangChain prompt template
prompt_template = PromptTemplate(input_variables=["go_file_code", "test_file_name", "previous_tests"], template=test_generation_prompt)
llm_chain = LLMChain(prompt=prompt_template, llm=llm)


# Function to store unit tests in ChromaDB
def store_unit_tests(test_file_name, unit_tests):
    # Add the unit tests to the Chroma database with the test file name
    collection.add(
        documents=[unit_tests],
        metadatas=[{"test_file_name": test_file_name}],
        ids=[test_file_name]
    )


# Function to store all Go test files in ChromaDB
def store_test_file(test_file_path):
    with open(test_file_path, 'r') as file:
        test_file_content = file.read()
    
    test_file_name = os.path.basename(test_file_path)
    
    # Add the test file content to ChromaDB
    collection.add(
        documents=[test_file_content],
        metadatas=[{"test_file_name": test_file_name}],
        ids=[test_file_name]
    )
    print(f"Stored test file: {test_file_name}")


# Function to retrieve all unit tests from ChromaDB based on the test file name
def retrieve_all_tests(test_file_name):
    # Query ChromaDB for all relevant unit tests based on the test file name
    results = collection.query(
        query_embeddings=None,
        n_results=3,  # Get top 3 most relevant unit tests
        where={"test_file_name": test_file_name}
    )
    
    if results['documents']:
        return "\n\n".join(results['documents'])
    else:
        return ""  # Return an empty string if no previous tests are found


# Function to automatically identify functions from Go file code
def extract_functions(go_file_code):
    # Use regex to identify functions in the Go file (simple pattern for example)
    function_pattern = re.compile(r'func (\w+)\s?\(.*\)\s?\w+ {')
    functions = function_pattern.findall(go_file_code)
    return functions


# Function to generate new unit tests based on the provided Go file code
def generate_unit_tests(test_file_name, go_file_code):
    # First, retrieve all previous tests from the test file if available
    previous_tests = retrieve_all_tests(test_file_name)

    # Extract all functions from the provided Go file code
    functions = extract_functions(go_file_code)
    
    # If there are no functions, return an error message
    if not functions:
        return "No functions found in the provided Go file."

    # Generate the new unit tests using LangChain
    unit_tests = llm_chain.run({
        "go_file_code": go_file_code, 
        "test_file_name": test_file_name, 
        "previous_tests": previous_tests
    })
    
    # Store the generated unit tests in ChromaDB for future use
    store_unit_tests(test_file_name, unit_tests)
    
    return unit_tests


# Function to scan the repository and store all Go test files in ChromaDB
def store_all_test_files_in_repo(repo_path):
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            if file.endswith('_test.go'):
                test_file_path = os.path.join(root, file)
                store_test_file(test_file_path)


# Example usage:
if __name__ == "__main__":
    # Define the path to the Go repository (or a specific directory containing Go files)
    repo_path = "/path/to/your/repository"
    
    # Store all test files from the repository into ChromaDB
    store_all_test_files_in_repo(repo_path)

    # Example Go file code (you would replace this with actual Go code)
    go_file_code = """
    package main

    import "testing"

    // Function to be tested
    func Add(a, b int) int {
        return a + b
    }

    // Another function to be tested
    func Multiply(a, b int) int {
        return a * b
    }
    """

    # Test file name for which we want to generate unit tests
    test_file_name = "math_operations_test.go"

    # Generate unit tests for all functions in the Go file
    generated_tests = generate_unit_tests(test_file_name, go_file_code)
    print("Generated Unit Tests:")
    print(generated_tests)
