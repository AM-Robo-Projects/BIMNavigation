from urllib import response
from neo4j import GraphDatabase
import torch
import ollama

# An example cypher query to get context about a room

def get_graph_context(cypher_query, params=None):
    uri = "neo4j://127.0.0.1:7687"
    #auth = ("neo4j", "duplex_only") 
    auth = ("neo4j", "kg_bim123") 

    print ("connection to database established")
    with GraphDatabase.driver(uri, auth=auth) as driver:
        with driver.session() as session:
            result = session.run(cypher_query, params or {})
            lines = []
            for record in result:
                # Format each record as needed
                lines.append(str(record.data()))
            return "\n".join(lines)


def ollama_chat (model_name, prompt,image_path= None):

    #qwen2.5-coder:7b or qwen2.5vl:3b #llama3.2-vision:11b

    try :
        print(f"Model {model_name} is loaded.")
        if image_path is not None :
                response = ollama.chat(
                model= model_name,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [image_path]  # Can be path or base64
                }]
                )

                return response['message']['content']
    
        else:
                response = ollama.chat(
                model= model_name,
                messages=[{
                    'role': 'user',
                    'content': prompt 
                }]
                )

                return response['message']['content']
        
    except Exception as e:

            print(f"ERROR :{e}")
            return None
    

# Example usage:
if __name__ == "__main__":

    
 


    schema = """
    Node Types (All IFC Products)
    Every node corresponds to an IfcProduct (a physical or spatial element in the building). All nodes share a common base structure but may include additional properties depending on their type.

    Common Node Properties:
    ifcId (string): Unique IFC GlobalId of the element.
    ifcType (string): The IFC class name (e.g., IfcWall, IfcSpace, IfcDoor).
    name (string): Human-readable name (defaults to "Unnamed" if missing).
    label (string): Same as ifcType; used for categorization in the graph.
    Optional Node Properties:
    boundingBox (object): Axis-aligned bounding box in world coordinates (min/max points), if geometry exists.
    position (array of 3 numbers): Centroid (x, y, z) of the element’s geometry, if geometry exists.
    materials (array of strings): List of material names associated with the element (extracted from material associations).
    For doors and windows only (IfcDoor / IfcWindow):
    isPassable (boolean): Always True.
    width (number or null): Overall width of the opening.
    height (number or null): Overall height of the opening.
    ✅ Note: Every IFC product becomes a node—this includes walls, slabs, doors, windows, spaces, sites, buildings, etc. 

    Relationship Types
    All relationships are directed (source → target) and labeled with a semantic type.

    1. CONTAINS
    Meaning: A spatial or aggregating container holds another element.
    Sources:
    From IfcRelContainedInSpatialStructure: e.g., a IfcSpace contains a IfcFurniture.
    From IfcRelAggregates: e.g., a IfcSite contains a IfcBuilding, or a IfcBuilding contains IfcBuildingStorey.
    Direction: Container → Contained element.
    2. BOUNDED_BY
    Meaning: A space is bounded (physically enclosed) by a building element (e.g., wall, floor, door).
    Source: IfcSpace
    Target: IfcProduct (e.g., wall, door)
    From: IfcRelSpaceBoundary relationships in IFC.
    3. CONNECTS_TO
    Meaning: Two spaces are connected via a shared passable element (like a door or window).
    Source & Target: Two different IfcSpace nodes.
    Trigger: When a single building element (e.g., a door) appears in the space boundaries of two or more spaces.
    Extra Property:
    through (string): The ifcId of the connecting element (e.g., the door that links the spaces).
    4. HAS_OPENING
    Meaning: A wall (or other host element) has an opening filled by a door or window.
    Source: Host element (e.g., IfcWall)
    Target: Filler element (IfcDoor or IfcWindow)
    From: Chain of IFC relationships:
    IfcRelFillsElement → IfcOpeningElement → IfcRelVoidsElement → Host wall. """

    # prompt = prompt = f"""
    # You are an expert in Neo4j and IFC data modeling.
    
    # Question:
    # Can you provide a Cypher query to give me the door positions in different rooms? Please follow the output rules below.
    # The walls are represented as IfcWall nodes, and doors as Door nodes. The IfcWall nodes have a relationship CONTAINS to the Door nodes. The position of the door is stored in the position property of the  Door node. 


    # Task:
    # Generate a Cypher query that answers the following question based on the provided graph schema {schema} following the guideline below.

    # Graph Schema:
    # {schema}

    # Output Rules:
    # 1. Output only the Cypher query — no explanations, comments, code fences, or markdown formatting.
    # 2. The output must start directly with a Cypher keyword such as MATCH, CREATE, MERGE, or RETURN.
    # 3. Do not wrap the query in quotes or other delimiters.
    # 4. Ensure the query is syntactically correct and consistent with the given schema{schema}.
    # """
    
    prompt = """
        You are an expert in BIM (Building Information Modeling) and Neo4j graph databases.

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

        If asked for a natural language answer, reason step-by-step using the schema.
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
        

        Question: Can you provide me a Cypher query to give me the window positions in different rooms? Please follow the output rules provided and you have some examples please refer to them (Note that doors and Windows are not contained in spaces but in another containers), don't add any explanations, comments, code fences, no cypher in the beginning, or markdown formatting.

    """
    answer = ollama_chat(model_name= "qwen2.5-coder:7b" ,prompt=prompt)
    print("Model answer:", answer)
    
    
        # answer = gemma3nLLM(prompt=prompt)
    # print("Model answer:", answer)

    cypher = f"""{answer}"""

     # Get context from Neo4j""" 

    context = get_graph_context(cypher)
    print("Context from Neo4j:", context)

    user_position = [8.0, 5.6, 1.3]  # Example user position

    prompt = f""" I am currently in the construction site and I want to know where is the nearest window to this position {user_position} according to the BIM model information below : {context},note that the positions provided from the context are in mm and the user positon is in m. Please answer in a concise manner with all the details from the provided context. If you don't have enough information, just say "I don't know" ."""
    
    
    answer = ollama_chat(model_name="llama3.2-vision:11b",prompt=prompt)
    print("Final answer:", answer)

    
    


