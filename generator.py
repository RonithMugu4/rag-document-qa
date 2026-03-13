from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from retriever import retrieve_relevant_chunks

# Build the grounded prompt template
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant that answers questions based ONLY on the provided context.
If the answer is not contained in the context, say "I don't know based on the provided document."
Do NOT use your general knowledge or make anything up.

Context:
{context}

Question: {question}

Answer:"""
)

def generate_answer(question: str) -> str:
    # Step 1: Retrieve the most relevant chunks for the question
    chunks = retrieve_relevant_chunks(question, k=4)

    # Step 2: Join the chunks into a single context string
    context = "\n\n".join([chunk.page_content for chunk in chunks])

    # Step 3: Format the prompt with the context and question
    prompt = prompt_template.format(context=context, question=question)

    # Step 4: Send the grounded prompt to the LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)  # temperature=0 means no randomness
    response = llm.invoke(prompt)

    return response.content


if __name__ == "__main__":
    # Test with a question about your PDF
    question = "What is social media mining?"
    print(f"Question: {question}\n")
    answer = generate_answer(question)
    print(f"Answer: {answer}")

    # Test the 'I don't know' behaviour
    print("\n" + "="*50 + "\n")
    question2 = "What is the GDP of Australia?"
    print(f"Question: {question2}\n")
    answer2 = generate_answer(question2)
    print(f"Answer: {answer2}")
