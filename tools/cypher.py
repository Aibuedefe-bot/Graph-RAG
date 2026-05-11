from llm import llm
from graph import graph
from langchain_neo4j import GraphCypherQAChain
from langchain_core.prompts import PromptTemplate

CYPHER_GENERATION_TEMPLATE = """You are a Neo4j Cypher expert for a Nigerian Student Rights knowledge graph.

Graph schema:
{schema}

Write a Cypher query to answer the question below.
- Document nodes have properties: text, source_file, page
- For keyword search use: toLower(d.text) CONTAINS toLower('keyword')
- Return at most 5 results
- Only output the Cypher query — no markdown, no explanation

Question: {question}
Cypher query:"""

CYPHER_QA_TEMPLATE = """You are an expert on Nigerian student rights and SERVICOM policies.
Use the Cypher query results below to answer the question accurately.
If the results are empty, say "I couldn't find that information in the knowledge graph."

Results:
{context}

Question: {question}
Answer:"""

cypher_generation_prompt = PromptTemplate(
    input_variables=["schema", "question"],
    template=CYPHER_GENERATION_TEMPLATE,
)

cypher_qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=CYPHER_QA_TEMPLATE,
)

cypher_qa = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    cypher_prompt=cypher_generation_prompt,
    qa_prompt=cypher_qa_prompt,
    verbose=True,
    allow_dangerous_requests=True,
)


def get_student_rights_from_graph(input: str) -> str:
    result = cypher_qa.invoke({"query": input})
    return result.get("result", "No results found in the knowledge graph.")