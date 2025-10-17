qa_chain = RetrievalQA.from_chain_type(llm, retriever)
```

The LLM receives:
```
Context:
- Name: Main_Door | Type: IfcDoor
- Name: Back_Door | Type: IfcDoor  
- Building_Floor_1 -[HAS_OPENING]-> Door_123
- Door_123 -[ADJACENT_TO]-> Wall_456

Question: How many doors?

Answer: Based on the context...
```

---

## 🔑 **Key Concepts**

### **1. Embeddings (Vector Representations)**
```
Text: "red door"     → Vector: [0.2, -0.5, 0.8, ...]  (768 numbers)
Text: "crimson gate" → Vector: [0.19, -0.48, 0.82, ...] (similar!)
Text: "blue window"  → Vector: [-0.3, 0.7, -0.1, ...]  (different)
```

Similar meanings = similar vectors = found by similarity search

### **2. Graph Relationships**
```
(Door)-[:HAS_OPENING]->(Room)
(Door)-[:CONNECTS]->(Corridor)
(Door)-[:PART_OF]->(Building)
```

Traditional search: finds "door"
Graph search: finds "door + room + corridor + building"

### **3. Hybrid Retrieval**
- **Vector alone**: Might miss connected context
- **Graph alone**: Might miss semantically similar nodes
- **Combined**: Best of both worlds!

---

## 🎬 **Real Example**

**Question:** *"Show me emergency exits"*

**Vector Search finds:**
```
- "Name: Exit_Door_1 | Type: IfcDoor | description: Emergency exit"
- "Name: Fire_Exit_B | Type: IfcDoor"
```

**Graph Traversal adds:**
```
- Emergency_Exit_1 -[LEADS_TO]-> Stairwell_A
- Stairwell_A -[CONNECTS_TO]-> Ground_Floor
- Emergency_Exit_1 -[HAS_SIGN]-> Exit_Sign_Red
- Emergency_Exit_1 -[PART_OF]-> Fire_Safety_System
```

**LLM synthesizes:**
*"There are 2 emergency exits: Exit_Door_1 leads to Stairwell A which connects to the ground floor, and Fire_Exit_B is part of the fire safety system..."*

---

## 🔧 **Why This Architecture?**

| Method | Strengths | Weaknesses |
|--------|-----------|------------|
| **Keyword Search** | Fast, exact matches | Misses synonyms, context |
| **Vector Search** | Semantic understanding | No structural relationships |
| **Graph Traversal** | Follows relationships | Needs starting point |
| **GraphRAG** | All of the above! | More complex |

---

## 📊 **Data Flow Diagram**
```
User Question
    ↓
┌───────────────────────────────────────┐
│  1. Embed Question (Ollama)           │
│     "doors" → [0.3, -0.2, 0.9, ...]   │
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│  2. Vector Search (Chroma)            │
│     Find similar embeddings           │
│     Returns: 4 door documents         │
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│  3. Extract node_ids: [42, 87, ...]   │
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│  4. Graph Traversal (Neo4j)           │
│     MATCH (n)-[r]->(m) WHERE id(n)... │
│     Returns: 20 relationship docs     │
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│  5. Combine: 4 + 20 = 24 documents    │
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│  6. LLM Generation (Ollama)           │
│     Read all 24 docs, answer question │
└───────────────────────────────────────┘
    ↓
Answer to User