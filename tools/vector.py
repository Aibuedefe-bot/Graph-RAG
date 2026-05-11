from llm import llm, embeddings
from graph import graph
from langchain_neo4j import Neo4jVector
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

neo4jvector = Neo4jVector.from_existing_index(
    embeddings,
    graph=graph,
    index_name="studentRightsIndex",
    node_label="Document",
    text_node_property="text",
    embedding_node_property="embedding",
    retrieval_query="""
        RETURN node.text AS text,
               score,
               {source_file: node.source_file, page: node.page} AS metadata
    """,
)

retriever = neo4jvector.as_retriever(search_kwargs={"k": 6})

SYSTEM_PROMPT = """You are the Nigerian Student Rights QA assistant.
Answer questions using ONLY the provided context from the SERVICOM and Student Rights knowledge base.
Be precise, cite specific rights or sections when available, and stay factual.
If the answer is not in the context, say "I don't have that information in the knowledge base."

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rights_retriever_chain = create_retrieval_chain(retriever, question_answer_chain)


def get_student_rights(input: str) -> str:
    result = rights_retriever_chain.invoke({"input": input})
    return result["answer"]