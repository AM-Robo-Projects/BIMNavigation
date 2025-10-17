"""
GraphRAG Retriever using Neo4j + Ollama + LangChain
----------------------------------------------------
Flexible retriever that can handle any type of query
"""

from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from neo4j import GraphDatabase
import math
# ...existing code... (removed duplicate import)

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

    def _flatten_and_extract_props(self,props: dict) -> Tuple[str, Optional[tuple]]:
        parts = []
        position = None

        def _format_value(v: Any) -> str:
            # allow assignment to outer 'position'
            nonlocal position

            if v is None:
                return ""
            if isinstance(v, (str, int, float, bool)):
                return str(v)
            if isinstance(v, (list, tuple)):
                # if list of 2-3 numeric values -> treat as position
                if all(isinstance(x, (int, float)) for x in v) and 2 <= len(v) <= 3:
                    position = tuple(float(x) for x in v)
                    return f"position:{position}"
                return ", ".join(_format_value(x) for x in v)
            if isinstance(v, dict):
                # check for common keys (case-insensitive)
                lower_keys = {k.lower(): val for k, val in v.items()}
                if any(k in lower_keys for k in ("x", "y")):
                    try:
                        x_raw = lower_keys.get("x")
                        y_raw = lower_keys.get("y")
                        z_raw = lower_keys.get("z", 0)
                        x = float(x_raw) if x_raw is not None else None
                        y = float(y_raw) if y_raw is not None else None
                        z = float(z_raw) if z_raw is not None else 0.0
                        if x is not None and y is not None:
                            pos = (x, y, z)
                            position = pos
                            return f"position:{pos}"
                    except Exception:
                        pass
                # fallback: stringify dict
                return "; ".join(f"{kk}:{_format_value(vv)}" for kk, vv in v.items())
            # fallback
            return str(v)

        for k, v in props.items():
            if v is None:
                continue
            val_str = _format_value(v)
            if val_str:
                parts.append(f"{k}: {val_str}")

        return " | ".join(parts), position
    
    
    def ingest_neo4j_data(self, limit: int = 1000) -> int:
        """
        Ingest nodes into the vector store. Flattens nested properties and
        extracts position if available (stored in metadata['position']).
        """
        cypher = f"""
        MATCH (n)
        OPTIONAL MATCH (n)-[r]->(m)
        WITH n, collect({{type: type(r), target: m.name, targetType: m.ifcType}}) AS relationships
        RETURN 
            id(n) AS node_id,
            labels(n) AS labels,
            n.ifcType AS ifcType,
            n.name AS name,
            n.ifcId AS ifcId,
            n.description AS description,
            properties(n) AS properties,
            relationships
        LIMIT $limit
        """

        documents = []
        with self.driver.session() as sess:
            result = sess.run(cypher, {"limit": limit})
            for record in result:
                node = record.data()
                props = node.get("properties", {}) or {}
                # flatten + position extraction
                flat_props_text, extracted_pos = self._flatten_and_extract_props(props)

                text_parts = []
                if node.get("ifcType"):
                    text_parts.append(f"A {node['ifcType']}")
                if node.get("name"):
                    text_parts.append(f"named {node['name']}")
                if node.get("ifcId"):
                    text_parts.append(f"IFC ID {node['ifcId']}")
                if node.get("description"):
                    text_parts.append(node["description"])
                if flat_props_text:
                    text_parts.append(f"Properties: {flat_props_text}")

                # relationships summary
                rels = node.get("relationships") or []
                if rels:
                    rel_texts = []
                    for rel in rels[:6]:
                        t = rel.get("type") or ""
                        tgt = rel.get("target") or ""
                        if t and tgt:
                            rel_texts.append(f"{t} {tgt}")
                    if rel_texts:
                        text_parts.append("Connected to: " + ", ".join(rel_texts))

                page_content = ". ".join(text_parts)

                # metadata must keep node_id as int for traversal
                node_id = node.get("node_id")
                try:
                    node_id_int = int(node_id)
                except Exception:
                    node_id_int = node_id  # fallback, but we prefer int

                metadata = {
                    "node_id": node_id_int,
                    "ifcType": node.get("ifcType") or "",
                    "name": node.get("name") or "",
                    "ifcId": node.get("ifcId") or "",
                }
                # store position (prefer tuple)
                if extracted_pos:
                    metadata["position"] = extracted_pos
                else:
                    # attempt extraction from properties if not found earlier
                    # some nodes may include e.g. properties['centroid'] etc.
                    # you can extend this fallback if you find the exact key name
                    pass

                documents.append(Document(page_content=page_content, metadata=metadata))

        if documents:
            print(f"Adding {len(documents)} documents to vector store...")
            # ensure using vectorstore.add_documents (or equivalent API)
            self.vectorstore.add_documents(documents)
            print("Ingestion complete")
        else:
            print("No documents to ingest")
        
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
        LIMIT 2000
        """
        
        with self.driver.session() as sess:
            result = sess.run(cypher, {"ids": seed_node_ids})
            return [r.data() for r in result]
    
    def get_available_entity_types(self) -> List[str]:
        """Get all IFC entity types available in the database."""
        cypher = """
        MATCH (n)
        WHERE n.ifcType IS NOT NULL
        RETURN DISTINCT n.ifcType AS type
        ORDER BY type
        """
        with self.driver.session() as sess:
            result = sess.run(cypher)
            return [r["type"] for r in result]
    
    def get_graph_schema(self) -> str:
        """Get complete graph schema without arbitrary limits."""
        schema_parts = []
        
        with self.driver.session() as sess:
            # Get all IFC types with their properties
            query = """
            MATCH (n)
            WHERE n.ifcType IS NOT NULL
            WITH n.ifcType AS type, keys(n) AS props
            WITH type, collect(DISTINCT props) AS all_prop_sets
            UNWIND all_prop_sets AS prop_set
            UNWIND prop_set AS prop
            WITH type, collect(DISTINCT prop) AS unique_props
            RETURN type, unique_props
            ORDER BY type
            """
            
            schema_parts.append("IFC Types and Properties:")
            result = sess.run(query)
            for record in result:
                type_name = record["type"]
                props = sorted(record["unique_props"])
                schema_parts.append(f"  {type_name}: {', '.join(props)}")
            
            # Get all relationship types
            rel_query = "CALL db.relationshipTypes()"
            rels = [r["relationshipType"] for r in sess.run(rel_query)]
            schema_parts.append(f"\nRelationship Types: {', '.join(rels)}")
        
        return "\n".join(schema_parts)
    
    def execute_aggregation_query(self, query: str) -> Optional[Dict]:
        """
        Execute aggregation queries dynamically.
        Now detects entity types automatically from the database.
        """
        query_lower = query.lower()
        
        if "how many" in query_lower or "count" in query_lower:
            # Get all available entity types
            available_types = self.get_available_entity_types()
            
            # Try to match query to an entity type
            matched_type = None
            for ifc_type in available_types:
                # Extract the meaningful part (e.g., "IfcDoor" -> "door")
                type_keyword = ifc_type.replace("Ifc", "").lower()
                if type_keyword in query_lower:
                    matched_type = ifc_type
                    break
            
            if matched_type:
                # Count specific entity type
                cypher = """
                MATCH (n)
                WHERE n.ifcType = $entity_type
                RETURN count(n) AS total_count, 
                       collect(n.name)[0..10] AS sample_names,
                       collect(DISTINCT labels(n)) AS labels
                """
                with self.driver.session() as sess:
                    result = sess.run(cypher, {"entity_type": matched_type})
                    record = result.single()
                    if record:
                        return {
                            "type": "count",
                            "entity_type": matched_type,
                            "count": record["total_count"],
                            "samples": [s for s in record["sample_names"] if s],
                            "labels": record["labels"]
                        }
            else:
                # No specific type matched - return summary of all types
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
        
        return None
    
    def generate_cypher_from_query(self, query: str) -> Optional[str]:
        """
        Use LLM to generate Cypher query from natural language.
        This makes the system fully flexible for any query type.
        """
        schema = self.get_graph_schema()
        
        prompt = f"""You are a Cypher query generator for Neo4j. Given a graph schema and a natural language question, generate a valid Cypher query.

        Graph Schema:
        {schema}

        Important rules:
        1. Use MATCH patterns to find nodes
        2. Use WHERE clauses for filtering
        3. Use RETURN to specify what to return
        4. For counting, use count()
        5. For aggregations, use collect(), sum(), avg(), etc.
        6. Always limit results to avoid huge responses (LIMIT 100)
        7. Return ONLY the Cypher query, no explanations

        Question: {query}

        Cypher query:"""
        
        try:
            cypher_query = self.llm.invoke(prompt).strip()
            # Clean up the response (remove markdown code blocks if present)
            cypher_query = cypher_query.replace("```cypher", "").replace("```", "").strip()
            print(f"🔧 Generated Cypher: {cypher_query}")
            return cypher_query
        except Exception as e:
            print(f"⚠️ Failed to generate Cypher: {e}")
            return None
    
    def execute_generated_cypher(self, query: str) -> Optional[List[Dict]]:
        """
        Generate and execute a Cypher query from natural language.
        This is the most flexible approach.
        """
        cypher = self.generate_cypher_from_query(query)
        if not cypher:
            return None
        
        try:
            with self.driver.session() as sess:
                result = sess.run(cypher)
                records = [r.data() for r in result]
                print(f"✅ Query returned {len(records)} results")
                return records
        except Exception as e:
            print(f"⚠️ Cypher execution failed: {e}")
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
        
        # Step 1: Try to generate and execute a custom Cypher query
        # This is the most flexible approach for complex queries
        cypher_results = self.execute_generated_cypher(query)
        if cypher_results:
            print(f"🔧 Generated Cypher returned {len(cypher_results)} results")
            for result in cypher_results[:5]:  # Limit to first 5
                content = " | ".join([f"{k}: {v}" for k, v in result.items()])
                doc = Document(
                    page_content=content,
                    metadata={"source": "neo4j_generated_cypher", "result": result}
                )
                all_docs.append(doc)
        
        # Step 2: Try simple aggregation query (backup for counting)
        agg_result = self.execute_aggregation_query(query)
        if agg_result:
            print(f"📊 Aggregation query result: {agg_result}")
            
            if agg_result["type"] == "count":
                content = f"There are {agg_result['count']} {agg_result['entity_type']} elements in the database."
                if agg_result.get('samples'):
                    content += f" Examples: {', '.join([s for s in agg_result['samples'] if s])}"
            else:
                content = "Element counts: " + ", ".join([f"{k}: {v}" for k, v in agg_result.get('counts', {}).items()])
            
            doc = Document(
                page_content=content,
                metadata={"source": "neo4j_aggregation", "query_type": "count"}
            )
            all_docs.append(doc)
        
        # Step 3: Vector search
        print(f"🔍 Vector search...")
        vector_docs = self.vectorstore.similarity_search(query, k=self.k)
        
        if vector_docs:
            print(f"📄 Found {len(vector_docs)} documents from vector search")
            for i, doc in enumerate(vector_docs[:3], 1):
                print(f"   {i}. {doc.page_content[:80]}...")
            all_docs.extend(vector_docs)
        else:
            print("⚠️ No documents found in vector search")
        
        # Step 4: Extract node IDs and traverse graph
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
        k=20,
    )
    
    # 4️⃣ Ingest data (comment out after first run)
    doc_count = retriever.ingest_neo4j_data()
    if doc_count == 0:
        print("\n⚠️ No data ingested. Check your Neo4j database.")
        retriever.close()
        exit()
    
    # 5️⃣ Create QA chain
    qa_chain = create_retrieval_chain(retriever)
    
    # 6️⃣ Test with diverse queries
    questions = [
        "can you tell me how many glass doors in the building and their positions?",
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
   # retriever.close()
    print("\n✅ All queries complete!")