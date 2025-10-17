"""
GraphRAG Retriever with Complete Data Preservation
---------------------------------------------------
Preserves ALL Neo4j data and makes it fully searchable
"""

from typing import List, Dict, Any, Optional
import json
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from neo4j import GraphDatabase


class Neo4jGraphHybridRetriever(BaseRetriever):
    """
    Complete retriever that preserves ALL Neo4j data.
    Uses multiple strategies for different query types.
    """
    
    vectorstore: Chroma
    neo4j_uri: str
    neo4j_auth: tuple
    llm: OllamaLLM
    max_traverse_depth: int = 1
    k: int = 20
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
        k: int = 20,
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
    
    def ingest_neo4j_data(self, include_relationships: bool = True):
        """
        Comprehensive data ingestion that preserves ALL information.
        Creates multiple representations for optimal search.
        """
        print("\n📥 Ingesting complete Neo4j data...")
        
        # Strategy: Create multiple documents per node for different search purposes
        documents = []
        
        with self.driver.session() as sess:
            # Get all nodes with ALL properties and relationships
            query = """
            MATCH (n)
            WHERE n.ifcType IS NOT NULL
            OPTIONAL MATCH (n)-[r]->(m)
            WITH n, 
                 collect({
                     rel_type: type(r), 
                     target_id: elementId(m),
                     target_type: m.ifcType,
                     target_name: m.name
                 }) as relationships
            RETURN 
                 elementId(n) AS node_id,
                labels(n) AS labels,
                properties(n) AS properties,
                relationships
            """
            
            result = sess.run(query)
            
            for record in result:
                node_id = record["node_id"]
                labels = record["labels"]
                props = record["properties"]
                relationships = [r for r in record["relationships"] if r['rel_type']]
                
                # Extract key properties
                ifc_type = props.get('ifcType', '')
                name = props.get('name', f'Node_{node_id}')
                
                # ========================================
                # Document 1: Natural Language Description
                # ========================================
                # This is for semantic/conceptual search
                nl_parts = []
                
                if ifc_type:
                    nl_parts.append(f"This is a {ifc_type.replace('Ifc', '')} element")
                if name:
                    nl_parts.append(f"named '{name}'")
                
                # Add descriptive properties
                if props.get('description'):
                    nl_parts.append(f"Description: {props['description']}")
                
                # Spatial info in natural language and metadata
                extracted_pos = None
                try:
                    if all(k in props for k in ['position_x', 'position_y', 'position_z']):
                        extracted_pos = [float(props['position_x']), float(props['position_y']), float(props['position_z'])]
                        nl_parts.append(f"located at position ({extracted_pos[0]}, {extracted_pos[1]}, {extracted_pos[2]})")
                    else:
                        # check for a single 'position' field that might be list/tuple or string
                        pos_field = props.get('position') or props.get('centroid') or props.get('coordinates')
                        if isinstance(pos_field, (list, tuple)) and 2 <= len(pos_field) <= 3:
                            # ensure numeric
                            try:
                                extracted_pos = [float(x) for x in pos_field]
                                nl_parts.append(f"located at position ({', '.join(map(str, extracted_pos))})")
                            except Exception:
                                extracted_pos = None
                        elif isinstance(pos_field, str):
                            # try to pull numbers from the string
                            import re
                            nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+", pos_field)
                            if 2 <= len(nums) <= 3:
                                try:
                                    extracted_pos = [float(x) for x in nums[:3]]
                                    nl_parts.append(f"located at position ({', '.join(map(str, extracted_pos))})")
                                except Exception:
                                    extracted_pos = None
                except Exception:
                    extracted_pos = None
                
                # Dimensional info
                if 'width' in props:
                    nl_parts.append(f"with width {props['width']}")
                if 'height' in props:
                    nl_parts.append(f"with height {props['height']}")
                
                # Relationships in natural language
                if relationships:
                    rel_descriptions = []
                    for rel in relationships[:5]:  # Limit for readability
                        if rel['target_name']:
                            rel_descriptions.append(f"{rel['rel_type']} {rel['target_name']}")
                    if rel_descriptions:
                        nl_parts.append(f"It is connected via: {', '.join(rel_descriptions)}")
                
                nl_content = ". ".join(nl_parts) + "."
                
                # Prepare position-safe metadata for vector stores (no lists)
                pos_meta = {}
                if extracted_pos is not None:
                    try:
                        pos_meta = {
                            "position": json.dumps(extracted_pos),
                            "position_x": float(extracted_pos[0]),
                            "position_y": float(extracted_pos[1]),
                        }
                        if len(extracted_pos) >= 3:
                            pos_meta["position_z"] = float(extracted_pos[2])
                    except Exception:
                        pos_meta = {"position": json.dumps(extracted_pos)}

                doc_nl = Document(
                    page_content=nl_content,
                    metadata={
                        "node_id": node_id,
                        # include explicit position metadata if available (safe types)
                        **pos_meta,
                        "doc_type": "natural_language",
                        "ifcType": ifc_type,
                        "name": name,
                        "labels": ", ".join(labels),
                        # Store ALL properties as JSON for complete preservation
                        "all_properties": json.dumps(props),
                        "has_relationships": len(relationships) > 0,
                    }
                )
                documents.append(doc_nl)
                
                # ========================================
                # Document 2: Structured Property List
                # ========================================
                # This is for property-based search (e.g., "width", "height")
                prop_parts = [f"Type: {ifc_type}", f"Name: {name}"]
                
                for key, value in props.items():
                    if key not in ['ifcType', 'name', 'ifcId']:
                        prop_parts.append(f"{key}: {value}")
                
                prop_content = " | ".join(prop_parts)
                
                doc_props = Document(
                    page_content=prop_content,
                    metadata={
                        "node_id": node_id,
                        # include explicit position metadata if available (safe types)
                        **pos_meta,
                        "doc_type": "properties",
                        "ifcType": ifc_type,
                        "name": name,
                        "labels": ", ".join(labels),
                        "all_properties": json.dumps(props),
                    }
                )
                documents.append(doc_props)
                
                # ========================================
                # Document 3: Relationship-focused
                # ========================================
                # This is for relationship queries
                if relationships:
                    rel_parts = [f"{name} ({ifc_type})"]
                    rel_parts.append("Relationships:")
                    
                    for rel in relationships:
                        if rel['target_name']:
                            rel_parts.append(f"- {rel['rel_type']} → {rel['target_name']} ({rel['target_type']})")
                    
                    rel_content = "\n".join(rel_parts)
                    
                    doc_rel = Document(
                        page_content=rel_content,
                        metadata={
                            "node_id": node_id,
                            # include explicit position metadata if available (safe types)
                            **pos_meta,
                            "doc_type": "relationships",
                            "ifcType": ifc_type,
                            "name": name,
                            "labels": ", ".join(labels),
                            "relationship_count": len(relationships),
                            "all_properties": json.dumps(props),
                        }
                    )
                    documents.append(doc_rel)
        
        if documents:
            print(f"📝 Adding {len(documents)} documents to vector store...")
            print(f"   - Per node: ~2-3 documents (NL + Properties + Relationships)")
            
            # Batch insert for performance
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                self.vectorstore.add_documents(batch)
                print(f"   ✓ Processed {min(i + batch_size, len(documents))}/{len(documents)}")
            
            print("✅ Complete data ingestion finished!")
        else:
            print("⚠️ No documents found in Neo4j.")
        
        return len(documents)
    
    def get_complete_graph_schema(self) -> str:
        """Get complete schema without limits."""
        schema_parts = []
        
        with self.driver.session() as sess:
            # Get all IFC types with ALL their properties
            query = """
            MATCH (n)
            WHERE n.ifcType IS NOT NULL
            WITH n.ifcType AS type, keys(n) AS props
            UNWIND props AS prop
            WITH type, collect(DISTINCT prop) AS unique_props
            RETURN type, unique_props
            ORDER BY type
            """
            
            schema_parts.append("Complete IFC Schema:")
            schema_parts.append("=" * 50)
            result = sess.run(query)
            for record in result:
                type_name = record["type"]
                props = sorted(record["unique_props"])
                schema_parts.append(f"\n{type_name}:")
                schema_parts.append(f"  Properties: {', '.join(props)}")
            
            # Get all relationship types with frequency
            rel_query = """
            MATCH ()-[r]->()
            RETURN type(r) AS rel_type, count(r) AS count
            ORDER BY count DESC
            """
            schema_parts.append("\n" + "=" * 50)
            schema_parts.append("Relationship Types:")
            for record in sess.run(rel_query):
                schema_parts.append(f"  - {record['rel_type']} ({record['count']} instances)")
        
        return "\n".join(schema_parts)
    
    def query_neo4j_direct(self, cypher: str, params: Dict = None) -> List[Dict]:
        """
        Execute arbitrary Cypher queries directly.
        This preserves ALL data without conversion.
        """
        try:
            with self.driver.session() as sess:
                result = sess.run(cypher, params or {})
                records = []
                for record in result:
                    # Convert Neo4j record to dict, preserving all types
                    record_dict = {}
                    for key in record.keys():
                        value = record[key]
                        # Handle Neo4j types
                        if hasattr(value, '__dict__'):
                            record_dict[key] = dict(value)
                        else:
                            record_dict[key] = value
                    records.append(record_dict)
                return records
        except Exception as e:
            print(f"⚠️ Cypher query failed: {e}")
            return []
    
    def generate_cypher_from_query(self, query: str) -> Optional[str]:
        """Generate Cypher query using LLM."""
        schema = self.get_complete_graph_schema()
        
        prompt = f"""  You are an expert in BIM (Building Information Modeling) and Neo4j graph databases.

        I have imported an IFC building model into Neo4j with the following verified schema:

        Nodes represent IFC elements (e.g., IfcDoor, IfcWall, IfcSpace) and have these properties:

        ifcId (string): Unique IFC GlobalId
        ifcType (string): IFC class (e.g., "IfcDoor")
        name (string): Element name (may be "Unnamed")
        materials (list of strings, optional)
        Geometric data (only if geometry was available during import):
        position_x, position_y, position_z (centroid coordinates)
        boundingBox: a list like [minX, minY, minZ, maxX, maxY, maxZ]
        Doors/Windows only: isPassable (boolean), width, height
        Relationships (all directed):

        CONTAINS (e.g., Building → Storey)
        BOUNDED_BY (Space → Wall/Door)
        CONNECTS_TO (Space → Space, with through: ifcId)
        HAS_OPENING (Wall → Door/Window)

        Critical Notes for Query Generation:

        If asked for a natural language answer, reason step-by-step using the schema {schema}.
        If asked for a Cypher query, output only valid, executable Neo4j Cypher that follows the above rules.
        If data might be missing (e.g., no geometry), mention it or use WHERE exists(node.position).

        example queries : 

        # For general querying and exploration

        MATCH p=()-[:CONTAINS]->() RETURN p LIMIT 600;
        MATCH (n:IfcDoor) RETURN n LIMIT 100;

        #To find windows in walls and their positions

        MATCH (container)-[:CONTAINS]->(window)
        WHERE window.ifcType = 'IfcWindow'
        RETURN 
        container.ifcType AS containerType,
        container.name AS containerName,
        window.ifcId AS windowId,
        window.name AS windowName,
        window.position_x,
        window.position_y,
        window.position_z
        
        # To find windows in walls using HAS_OPENING relationship

        MATCH (container)-[:HAS_OPENING]->(window)
        WHERE window.ifcType = 'IfcWindow'
        RETURN 
        container.ifcType AS containerType,
        container.name AS containerName,
        window.ifcId AS windowId,
        window.name AS windowName,
        window.position_x,
        window.position_y,
        window.position_z

        # To find doors in rooms in general

        MATCH (container)-[:CONTAINS]->(door)
        WHERE door.ifcType = 'IfcDoor'
        RETURN 
        container.ifcType AS containerType,
        container.name AS containerName,
        door.ifcId AS doorId,
        door.name AS doorName,
        door.position_x,
        door.position_y,
        door.position_z

        # To find doors in walls using HAS_OPENING relationship
        MATCH (container)-[:HAS_OPENING]->(door)
        WHERE door.ifcType = 'IfcDoor'
        RETURN 
        container.ifcType AS containerType,
        container.name AS containerName,
        door.ifcId AS doorId,
        door.name AS doorName,
        door.position_x,
        door.position_y,
        door.position_z

Question: {query}

Cypher query:"""
        
        try:
            cypher = self.llm.invoke(prompt).strip()
            # Clean up response
            cypher = cypher.replace("```cypher", "").replace("```", "").strip()
            # Remove any text before/after the query
            lines = cypher.split('\n')
            cypher_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
            cypher = '\n'.join(cypher_lines)
            
            print(f"🔧 Generated Cypher:\n{cypher}")
            return cypher
        except Exception as e:
            print(f"⚠️ Cypher generation failed: {e}")
            return None
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun = None,
    ) -> List[Document]:
        """
        Multi-strategy retrieval:
        1. Direct Cypher (for structured queries)
        2. Vector search (for semantic queries)
        3. Graph traversal (for relationships)
        """
        print(f"\n{'='*70}")
        print(f"🔍 Query: '{query}'")
        print('='*70)
        
        all_docs = []
        
        # ========================================
        # Strategy 1: Direct Cypher Execution
        # ========================================
        # Best for: counts, filters, structured queries
        cypher = self.generate_cypher_from_query(query)
        print (cypher)

        if cypher:
            results = self.query_neo4j_direct(cypher)
            if results:
                print(f"✅ Direct Cypher returned {len(results)} results")
                
                # DEBUG: show raw direct results (first few)
                print("[DEBUG] Raw direct Cypher results sample:")
                for rr in results[:5]:
                    try:
                        print(json.dumps(rr, default=str, ensure_ascii=False))
                    except Exception:
                        print(str(rr))
                print("[END DEBUG]")

                for i, result in enumerate(results[:20]):  # Limit display
                    # Create rich document from result
                    content_parts = []
                    metadata = {"source": "neo4j_direct", "result_index": i}

                    # temporary holder for potential position components
                    pos_vals = {}

                    for key, value in result.items():
                        lk = str(key).lower()
                        # Node or relationship object
                        if isinstance(value, dict):
                            if 'ifcType' in value:
                                # prefer name + type
                                content_parts.append(f"{value.get('name', 'Unnamed')} ({value.get('ifcType')})")
                            metadata[key] = json.dumps(value)
                            # also extract any positional props inside node dict
                            if isinstance(value, dict):
                                for pk in ('position_x', 'position_y', 'position_z'):
                                    if pk in value and value[pk] is not None:
                                        try:
                                            metadata[pk] = float(value[pk])
                                            pos_vals[pk] = float(value[pk])
                                        except Exception:
                                            metadata[pk] = str(value[pk])
                        else:
                            # Primitive value: check if it's a position component or position container
                            if any(q in lk for q in ['position_x', 'positiony', 'position_x'.replace('_','')]):
                                # direct position components
                                try:
                                    metadata[lk] = float(value) if value is not None else None
                                    pos_vals[lk] = float(value) if value is not None else None
                                except Exception:
                                    metadata[lk] = str(value)
                                content_parts.append(f"{key}: {value}")
                            elif 'position' in lk and isinstance(value, (list, tuple)):
                                # whole position list
                                try:
                                    coords = [float(x) for x in value]
                                    metadata['position'] = json.dumps(coords)
                                    if len(coords) > 0:
                                        metadata['position_x'] = coords[0]
                                    if len(coords) > 1:
                                        metadata['position_y'] = coords[1]
                                    if len(coords) > 2:
                                        metadata['position_z'] = coords[2]
                                    content_parts.append(f"position: ({', '.join(map(str, coords))})")
                                except Exception:
                                    metadata[key] = str(value)
                            else:
                                # fallback
                                content_parts.append(f"{key}: {value}")
                                metadata[key] = str(value) if value is not None else ""

                    # if we collected separate position components, add consolidated position to content/metadata
                    if pos_vals:
                        try:
                            px = pos_vals.get('position_x') or pos_vals.get('positionx') or pos_vals.get('position_x')
                            py = pos_vals.get('position_y') or pos_vals.get('positiony') or pos_vals.get('position_y')
                            pz = pos_vals.get('position_z') or pos_vals.get('positionz') or None
                            coords = []
                            if px is not None:
                                coords.append(float(px))
                            if py is not None:
                                coords.append(float(py))
                            if pz is not None:
                                coords.append(float(pz))
                            if coords:
                                metadata['position'] = json.dumps(coords)
                                metadata['position_x'] = coords[0]
                                if len(coords) > 1:
                                    metadata['position_y'] = coords[1]
                                if len(coords) > 2:
                                    metadata['position_z'] = coords[2]
                                content_parts.append(f"Position: ({', '.join(map(str, coords))})")
                        except Exception:
                            pass

                    content = " | ".join(content_parts)
                    doc = Document(page_content=content, metadata=metadata)
                    all_docs.append(doc)

                # DEBUG: show created documents from direct Cypher
                print("[DEBUG] Created Documents from direct Cypher (sample):")
                for d in all_docs[:min(10, len(all_docs))]:
                    try:
                        print(f"- page_content: {d.page_content}")
                        print(f"  metadata: {json.dumps(d.metadata, default=str, ensure_ascii=False)}")
                    except Exception:
                        print(f"  metadata: {d.metadata}")
                print("[END DEBUG]")
        
        # ========================================
        # Strategy 2: Vector Search
        # ========================================
        # Best for: semantic/conceptual queries
        print(f"\n🔍 Vector search with k={self.k}...")
        vector_docs = self.vectorstore.similarity_search(query, k=self.k)
        
        if vector_docs:
            print(f"📄 Vector search found {len(vector_docs)} documents")
            # Show diversity
            doc_types = {}
            for doc in vector_docs:
                dt = doc.metadata.get('doc_type', 'unknown')
                doc_types[dt] = doc_types.get(dt, 0) + 1
            print(f"   Document types: {doc_types}")
            
            all_docs.extend(vector_docs)

            # DEBUG: sample vector_docs (metadata + page_content)
            print("[DEBUG] Vector search sample docs:")
            for d in vector_docs[:5]:
                try:
                    print(f"- page_content: {d.page_content}")
                    print(f"  metadata: {json.dumps(d.metadata, default=str, ensure_ascii=False)}")
                except Exception:
                    print(f"  metadata: {d.metadata}")
            print("[END DEBUG]")
        
        # ========================================
        # Strategy 3: Graph Traversal
        # ========================================
        # Best for: exploring relationships
        seed_ids = []
        for doc in vector_docs:
            node_id = doc.metadata.get('node_id')
            if node_id and node_id not in seed_ids:
                seed_ids.append(node_id)
        
        if seed_ids:
            print(f"\n🌐 Graph traversal from {len(seed_ids)} seed nodes...")
            
            # Multi-hop traversal
            traversal_query = f"""
            MATCH path = (n)-[*1..{self.max_traverse_depth}]-(m)
            WHERE elementId(n) IN $ids AND elementId(n) <> elementId(m)
            WITH n, m, relationships(path) as rels, length(path) as dist
            WHERE dist <= {self.max_traverse_depth}
            RETURN DISTINCT
                elementId(n) as from_id,
                elementId(m) as to_id,
                n.name as from_name,
                n.ifcType as from_type,
                m.name as to_name,
                m.ifcType as to_type,
                [r in rels | type(r)] as rel_path,
                dist,
                properties(m) as to_props
            ORDER BY dist
            LIMIT 1000
            """
            
            traversals = self.query_neo4j_direct(traversal_query, {"ids": seed_ids})
            
            if traversals:
                print(f"🔗 Found {len(traversals)} graph connections")
                
                for tr in traversals:
                    from_name = tr.get('from_name', f"Node_{tr['from_id']}")
                    to_name = tr.get('to_name', f"Node_{tr['to_id']}")
                    rel_path = tr.get('rel_path', [])
                    dist = tr.get('dist', 1)
                    # Build content
                    content = f"{from_name} ({tr.get('from_type', 'Unknown')}) "
                    content += f"-[{' -> '.join(rel_path)}]-> " if rel_path else "-[RELATED_TO]-> "
                    content += f"{to_name} ({tr.get('to_type', 'Unknown')}) "
                    content += f"[distance: {dist}]"

                    # Parse target properties (may be dict or JSON string)
                    raw_to_props = tr.get('to_props', {})
                    if isinstance(raw_to_props, str):
                        try:
                            to_props = json.loads(raw_to_props)
                        except Exception:
                            to_props = {}
                    elif isinstance(raw_to_props, dict):
                        to_props = raw_to_props
                    else:
                        to_props = {}

                    # extract numeric position fields if present
                    pos_meta = {}
                    try:
                        if all(k in to_props for k in ('position_x', 'position_y')):
                            px = float(to_props.get('position_x'))
                            py = float(to_props.get('position_y'))
                            pz = float(to_props.get('position_z')) if 'position_z' in to_props else None
                            coords = [px, py] + ([pz] if pz is not None else [])
                            pos_meta = {
                                'position': json.dumps(coords),
                                'position_x': px,
                                'position_y': py,
                            }
                            if pz is not None:
                                pos_meta['position_z'] = pz
                            # append readable position to content
                            content += f" Position: ({', '.join(map(str, coords))})"
                        else:
                            # check nested fields like 'centroid' or 'position'
                            cand = to_props.get('position') or to_props.get('centroid') or to_props.get('coordinates')
                            if isinstance(cand, (list, tuple)) and 2 <= len(cand) <= 3:
                                try:
                                    coords = [float(x) for x in cand]
                                    pos_meta = {
                                        'position': json.dumps(coords),
                                        'position_x': coords[0],
                                        'position_y': coords[1],
                                    }
                                    if len(coords) > 2:
                                        pos_meta['position_z'] = coords[2]
                                    content += f" Position: ({', '.join(map(str, coords))})"
                                except Exception:
                                    pass
                    except Exception:
                        pos_meta = {}

                    metadata = {
                        "source": "neo4j_traversal",
                        "from_id": tr['from_id'],
                        "to_id": tr['to_id'],
                        "distance": dist,
                        # keep raw properties string for debugging
                        "to_properties": json.dumps(to_props),
                    }
                    # merge safe pos_meta
                    metadata.update(pos_meta)

                    doc = Document(page_content=content, metadata=metadata)
                    all_docs.append(doc)

                # DEBUG: sample traversal docs
                print("[DEBUG] Traversal docs sample:")
                for d in all_docs[-5:]:
                    try:
                        print(f"- page_content: {d.page_content}")
                        print(f"  metadata: {json.dumps(d.metadata, default=str, ensure_ascii=False)}")
                    except Exception:
                        print(f"  metadata: {d.metadata}")
                print("[END DEBUG]")
        
        print(f"\n✅ Total retrieved: {len(all_docs)} documents")
        print('='*70)
        return all_docs
    
    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
    
    def __del__(self):
        """Cleanup."""
        try:
            self.close()
        except:
            pass


def create_retrieval_chain(retriever: Neo4jGraphHybridRetriever) -> RetrievalQA:
    """Create QA chain."""
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
    
    print("🚀 Initializing Complete GraphRAG Retriever...")
    
    # Initialize
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(
        collection_name="bim_complete_v2",  # New collection name
        embedding_function=embeddings,
        persist_directory="./chroma_db_complete"
    )
    
    retriever = Neo4jGraphHybridRetriever(
        vectorstore=vectorstore,
        neo4j_uri=NEO4J_URI,
        neo4j_auth=NEO4J_AUTH,
        llm_model=LLM_MODEL,
        max_traverse_depth=2,
        k=50,
    )
    
    #Ingest data (comment out after first run)
    # print("\n" + "="*70)
    # print("INGESTION PHASE")
    # print("="*70)
    # doc_count = retriever.ingest_neo4j_data()
    # print(f"\n✅ Ingested {doc_count} documents")
    
    # if doc_count == 0:
    #     print("\n⚠️ No data found in Neo4j")
    #     retriever.close()
    #     exit()
    
    # Create QA chain
    qa_chain = create_retrieval_chain(retriever)
    
    # Test queries
    print("\n" + "="*70)
    print("QUERY PHASE")
    print("="*70)
    
    test_queries = [
            "can you tell me all the doors and their bounding boxes and positions in the building?", 
    ]
    
    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"❓ Question: {query}")
        print('='*70)
        
        try:
            result = qa_chain.invoke({"query": query})
            print(f"\n💡 Answer:\n{result['result']}")
            
            if result.get('source_documents'):
                print(f"\n📚 Used {len(result['source_documents'])} sources")
                
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    #retriever.close()
    print("\n✅ Complete!")