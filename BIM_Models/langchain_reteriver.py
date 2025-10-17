"""
GraphRAG Retriever using Neo4j + Ollama + LangChain
----------------------------------------------------
Flexible retriever that can handle any type of query
"""

from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from neo4j import GraphDatabase


class Neo4jGraphHybridRetriever(BaseRetriever):
    """
    Custom retriever combining Chroma vector search and Neo4j graph traversal.
    Handles any query type dynamically.
    """
    
    vectorstore: Chroma
    neo4j_uri: str
    neo4j_auth: tuple
    llm: OllamaLLM
    max_traverse_depth: int = 1
    k: int = 4
    driver: Any = None
    
    class Config:
        """Pydantic config for arbitrary types."""
        arbitrary_types_allowed = True
    
    def __init__(
        self,
        vectorstore: Chroma,
        neo4j_uri: str,
        neo4j_auth: tuple,
        llm_model: str,
        max_traverse_depth: int = 1,
        k: int = 4,
    ):
        """Initialize the hybrid retriever."""
        super().__init__(
            vectorstore=vectorstore,
            neo4j_uri=neo4j_uri,
            neo4j_auth=neo4j_auth,
            llm=OllamaLLM(model=llm_model),
            max_traverse_depth=max_traverse_depth,
            k=k,
            driver=None,
        )
        object.__setattr__(self, 'driver', GraphDatabase.driver(neo4j_uri, auth=neo4j_auth))
    
    def ingest_neo4j_data(self):
        """
        Ingest data from Neo4j into Chroma vector store.
        Creates searchable documents from all graph nodes.
        """
        print("\n📥 Ingesting data from Neo4j into vector store...")
        
        cypher = """
        MATCH (n)
        OPTIONAL MATCH (n)-[r]->(m)
        WITH n, collect({type: type(r), target: m.name, targetType: m.ifcType}) as relationships
        RETURN 
            id(n) AS node_id,
            labels(n) AS labels,
            n.ifcType AS ifcType,
            n.name AS name,
            n.ifcId AS ifcId,
            n.description AS description,
            properties(n) AS properties,
            relationships
        LIMIT 1000
        """
        
        documents = []
        with self.driver.session() as sess:
            result = sess.run(cypher)
            
            for record in result:
                node_data = record.data()
                
                # Build rich text description
                text_parts = []
                
                # Basic info
                if node_data.get('ifcType'):
                    text_parts.append(f"Type: {node_data['ifcType']}")
                if node_data.get('name'):
                    text_parts.append(f"Name: {node_data['name']}")
                if node_data.get('ifcId'):
                    text_parts.append(f"IFC ID: {node_data['ifcId']}")
                if node_data.get('description'):
                    text_parts.append(f"Description: {node_data['description']}")
                
                # Properties
                props = node_data.get('properties', {})
                for key, value in props.items():
                    if key not in ['name', 'ifcType', 'ifcId', 'description'] and value:
                        if isinstance(value, (int, float, str, bool)):
                            text_parts.append(f"{key}: {value}")
                
                # Relationships (add context about what this node connects to)
                relationships = node_data.get('relationships', [])
                if relationships:
                    rel_texts = []
                    for rel in relationships[:5]:  # Limit to 5 relationships
                        if rel.get('target') and rel.get('type'):
                            rel_texts.append(f"{rel['type']} {rel['target']}")
                    if rel_texts:
                        text_parts.append(f"Connected to: {', '.join(rel_texts)}")
                
                page_content = " | ".join(text_parts)
                
                # Create document with metadata
                labels = node_data.get('labels', [])
                doc = Document(
                    page_content=page_content,
                    metadata={
                        "node_id": node_data['node_id'],
                        "labels": ", ".join(labels) if labels else "",
                        "ifcType": node_data.get('ifcType') or "",
                        "name": node_data.get('name') or "",
                        "ifcId": node_data.get('ifcId') or "",
                    }
                )
                documents.append(doc)
        
        if documents:
            print(f"📝 Adding {len(documents)} documents to vector store...")
            self.vectorstore.add_documents(documents)
            print("✅ Data ingestion complete!")
        else:
            print("⚠️ No documents found in Neo4j to ingest.")
        
        return len(documents)  
    
    def traverse_from_nodes(self, seed_node_ids: List[int], depth: int = 1) -> List[Dict]:
        """
        Traverse the Neo4j graph from seed nodes to find related entities.
        Now supports configurable depth and ANY relationship type.
        """
        # Dynamic traversal based on depth
        cypher = f"""
        MATCH path = (n)-[r*1..{depth}]-(m)
        WHERE id(n) IN $ids
        WITH n, m, relationships(path) as rels, length(path) as path_length
        RETURN 
            id(n) AS from_id,
            id(m) AS to_id,
            path_length,
            [rel in rels | type(rel)] AS rel_types,
            labels(n) AS from_labels,
            labels(m) AS to_labels,
            n.name AS from_name,
            n.ifcType AS from_type,
            m.name AS to_name,
            m.ifcType AS to_type,
            properties(n) AS from_props,
            properties(m) AS to_props
        ORDER BY path_length
        LIMIT 100
        """
        
        with self.driver.session() as sess:
            result = sess.run(cypher, {"ids": seed_node_ids})
            return [r.data() for r in result]
    
    def execute_aggregation_query(self, query: str) -> Optional[Dict]:
        """
        Execute aggregation queries like counting, summing, etc.
        This allows the LLM to get exact counts/stats from Neo4j.
        """
        # Detect if this is a counting/aggregation query
        query_lower = query.lower()
        
        if "how many" in query_lower or "count" in query_lower:
            # Extract what to count
            if "door" in query_lower:
                entity_type = "IfcDoor"
            elif "window" in query_lower:
                entity_type = "IfcWindow"
            elif "wall" in query_lower:
                entity_type = "IfcWall"
            elif "room" in query_lower or "space" in query_lower:
                entity_type = "IfcSpace"
            elif "floor" in query_lower and "building" in query_lower:
                entity_type = "IfcBuildingStorey"
            else:
                # Generic count
                cypher = """
                MATCH (n)
                WHERE n.ifcType IS NOT NULL
                RETURN n.ifcType as type, count(n) as count
                ORDER BY count DESC
                """
                with self.driver.session() as sess:
                    result = sess.run(cypher)
                    counts = {r["type"]: r["count"] for r in result}
                    return {"type": "summary", "counts": counts}
            
            # Count specific entity type
            cypher = """
            MATCH (n)
            WHERE n.ifcType = $entity_type
            RETURN count(n) AS total_count, 
                   collect(n.name)[0..10] AS sample_names
            """
            with self.driver.session() as sess:
                result = sess.run(cypher, {"entity_type": entity_type})
                record = result.single()
                if record:
                    return {
                        "type": "count",
                        "entity_type": entity_type,
                        "count": record["total_count"],
                        "samples": record["sample_names"]
                    }
        
        return None
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun = None,
    ) -> List[Document]:
        """
        Retrieve relevant documents via:
        1. Aggregation queries (for counts/stats)
        2. Vector search (for semantic similarity)
        3. Graph traversal (for relationships)
        """
        print(f"\n🔍 Processing query: '{query}'")
        
        all_docs = []
        
        # Step 1: Try aggregation query first (for "how many" questions)
        agg_result = self.execute_aggregation_query(query)
        if agg_result:
            print(f"📊 Aggregation query result: {agg_result}")
            
            # Create a document with the aggregation result
            if agg_result["type"] == "count":
                content = f"There are {agg_result['count']} {agg_result['entity_type']} elements in the database."
                if agg_result.get('samples'):
                    content += f" Examples: {', '.join([s for s in agg_result['samples'] if s])}"
            else:
                # Summary of all types
                content = "Element counts: " + ", ".join([f"{k}: {v}" for k, v in agg_result.get('counts', {}).items()])
            
            doc = Document(
                page_content=content,
                metadata={"source": "neo4j_aggregation", "query_type": "count"}
            )
            all_docs.append(doc)
        
        # Step 2: Vector search
        print(f"🔍 Vector search...")
        # search for k nearest embeddings that have the same meaning
        vector_docs = self.vectorstore.similarity_search(query, k=self.k)
        
        if vector_docs:
            print(f"📄 Found {len(vector_docs)} documents from vector search")
            for i, doc in enumerate(vector_docs[:3], 1):
                print(f"   {i}. {doc.page_content[:80]}...")
            all_docs.extend(vector_docs)
        else:
            print("⚠️ No documents found in vector search")
        
        # Step 3: Extract node IDs and traverse graph
        seed_ids = [doc.metadata.get("node_id") for doc in vector_docs if doc.metadata.get("node_id")]
        
        if seed_ids:
            print(f"🌐 Graph traversal from {len(seed_ids)} seed nodes...")
            traversals = self.traverse_from_nodes(seed_ids, depth=self.max_traverse_depth)
            
            # Convert to documents
            for tr in traversals:
                content = self._format_graph_content(tr)
                
                from_labels = tr.get("from_labels", [])
                to_labels = tr.get("to_labels", [])
                rel_types = tr.get("rel_types", [])
                
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": "neo4j_graph",
                        "from_id": tr["from_id"],
                        "to_id": tr["to_id"],
                        "relationships": ", ".join(rel_types) if rel_types else "",
                        "from_type": tr.get("from_type") or "",
                        "to_type": tr.get("to_type") or "",
                        "path_length": tr.get("path_length", 1),
                    }
                )
                all_docs.append(doc)
            
            print(f"🔗 Found {len(traversals)} graph connections")
        
        print(f"✅ Total: {len(all_docs)} documents retrieved")
        return all_docs
    
    def _format_graph_content(self, traversal: Dict) -> str:
        """Format graph relationship into readable text."""
        from_name = traversal.get("from_name") or f"Node_{traversal['from_id']}"
        to_name = traversal.get("to_name") or f"Node_{traversal['to_id']}"
        
        from_type = traversal.get("from_type", "")
        to_type = traversal.get("to_type", "")
        
        if from_type:
            from_name = f"{from_name} ({from_type})"
        if to_type:
            to_name = f"{to_name} ({to_type})"
        
        rel_types = traversal.get("rel_types", [])
        rel_str = " -> ".join(rel_types) if rel_types else "RELATED_TO"
        
        path_length = traversal.get("path_length", 1)
        depth_str = f" [depth: {path_length}]" if path_length > 1 else ""
        
        return f"{from_name} -[{rel_str}]-> {to_name}{depth_str}"
    
    def close(self):
        """Close Neo4j driver connection."""
        if self.driver:
            self.driver.close()
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except:
            pass


def create_retrieval_chain(retriever: Neo4jGraphHybridRetriever) -> RetrievalQA:
    """Create a RetrievalQA chain."""
    return RetrievalQA.from_chain_type(
        llm=retriever.llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        verbose=True,
    )


if __name__ == "__main__":
    # Configuration
    NEO4J_URI = "neo4j://127.0.0.1:7687"
    NEO4J_AUTH = ("neo4j", "kg_bim_cranehall")
    EMBEDDING_MODEL = "qwen2.5-coder:7b"
    LLM_MODEL = "llama3.2-vision:11b"
    
    print("🚀 Initializing GraphRAG Retriever...")
    
    # 1️⃣ Create embeddings
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    # 2️⃣ Initialize vector store
    vectorstore = Chroma(
        collection_name="graph_docs",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )
    
    # 3️⃣ Create retriever
    retriever = Neo4jGraphHybridRetriever(
        vectorstore=vectorstore,
        neo4j_uri=NEO4J_URI,
        neo4j_auth=NEO4J_AUTH,
        llm_model=LLM_MODEL,
        max_traverse_depth=2,
        k=4,
    )
    
    # 4️⃣ Ingest data (comment out after first run)
    # doc_count = retriever.ingest_neo4j_data()
    # if doc_count == 0:
    #     print("\n⚠️ No data ingested. Check your Neo4j database.")
    #     retriever.close()
    #     exit()
    
    # 5️⃣ Create QA chain
    qa_chain = create_retrieval_chain(retriever)
    
    # 6️⃣ Test with diverse queries
    questions = [
        "what's connected to Doors_Dbl_Glass_4_ARENA2"
    ]
    
    for question in questions:
        print(f"\n{'='*70}")
        print(f"🤖 Question: {question}")
        print('='*70)
        
        try:
            result = qa_chain.invoke({"query": question})
            
            print("\n🧠 Answer:")
            print(result["result"])
            
            if "source_documents" in result and result["source_documents"]:
                print(f"\n📚 Used {len(result['source_documents'])} source documents")
                
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        print()  # Add spacing between questions
    
    # Clean up
    #retriever.close()
    #print("\n✅ All queries complete!")