#!/usr/bin/env python3
import os
import time
import rclpy
from rclpy.node import Node

import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


from neo4j import GraphDatabase
import ollama
import base64
import threading
import queue # Import the queue module
from neo4j.exceptions import CypherSyntaxError

class MyNode(Node):

    def __init__(self):
        super().__init__('my_node')
        self.get_logger().info('Node has been started!')

        self.create_subscription(Image, "/rgb", self.image_callback, 10) #isaac_sim 

        # self.create_subscription(Image, "/camera/image_raw", self.image_callback, 10) #gazebo

        self.bridge = CvBridge()

        # OPTIMIZATION 1: Create a queue to hold the latest frame for processing.
        # maxsize=1 means we only ever store the most recent frame, automatically
        # discarding older ones if the VLM can't keep up.
        self.vlm_queue = queue.Queue(maxsize=1)

        # OPTIMIZATION 2: Start a single, long-lived worker thread for VLM processing.
        # This is far more efficient than creating a new thread for every frame.
        # daemon=True ensures the thread will exit when the main program does.
        self.processing_thread = threading.Thread(target=self.vlm_worker, daemon=True)
        self.processing_thread.start()
        
    def image_callback(self, msg):
        """
        This function is now very fast. It just converts the image and puts it in a queue.
        """
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # OPTIMIZATION 3: Instead of writing to disk, encode image to JPEG in memory.
            _, buffer = cv2.imencode('.jpg', cv_image)
            image_bytes = buffer.tobytes()

            # Try to put the latest frame into the queue. If the queue is full
            # (meaning the VLM is still processing the last one), it will drop the old
            # frame and add the new one, ensuring we always process the most recent data.
            if not self.vlm_queue.empty():
                try:
                    self.vlm_queue.get_nowait()  # Discard the old frame
                except queue.Empty:
                    pass
            self.vlm_queue.put(image_bytes)

        except Exception as e:
            self.get_logger().error(f"Image callback failed: {e}")

    def vlm_worker(self):
        """
        The worker loop that runs in a separate thread.
        It waits for an image to appear in the queue and processes it.
        """
        self.get_logger().info('VLM worker thread started.')

        while rclpy.ok():
            try:
                # This call will block until an item is available in the queue.
                image_bytes = self.vlm_queue.get(timeout=1.0) # Timeout to allow checking rclpy.ok()

                LLM_prompt="""
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
                

                Question: Can you provide me a Cypher query to give me the door positions in different rooms using one of the queries provided?
                Please follow the output rules provided and you have some examples please refer to them, don't add any explanations, comments, code fences, no cypher in the beginning, or markdown formatting.

            """
                answer  = self.process_with_llm(LLM_prompt)   
                cypher_query= answer
                
                context = self.get_graph_context(cypher_query)
                print (context)
                
                robot_position = [-1.4,-0.4843,-0.497]
                #self.process_with_vlm(image_bytes,prompt="I am standing in this office and I want to know if there is any problem with the current wall, please answer in a concise manner with all the details from the provided context. If you don't have enough information, just say 'I don't know' .")
                self.process_with_vlm(image_bytes,prompt="you're the eye of an inspection robot with the position "+str(robot_position)+" in a building construction site, and you want to know where is the nearest door to this position according to the BIM model information below : "+context+",note that the positions provided from the context are in mm and the robot positon is in m. Please answer in a concise manner with all the details from the provided context. If you don't have enough information, just say 'I don't know' .")
        

            except CypherSyntaxError as e:
                self.get_logger().error(f"Cypher syntax error: {e}")    
        
            except queue.Empty:
                continue # No image, just loop again
            except Exception as e:
                self.get_logger().error(f"Error in VLM worker: {e}")

    def process_with_vlm(self, image_bytes,prompt):
        """
        Call the VLM model with image bytes from memory.
        """
        try:
           
            #qwen2.5vl:3b : Slow but high quality
            #moondream2:1.8 : Fast but lower quality
            #llama3.2-vision:11b
            

            response = ollama.chat(
                model='llama3.2-vision:11b', # Using a standard model name, change if needed
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [image_bytes] # The library handles base64 encoding for bytes
                }]
            )
            
            self.get_logger().info(f"VLM response:\n{response['message']['content']}")
        except Exception as e:
            self.get_logger().error(f"Ollama call failed: {e}")

    def process_with_llm (self, prompt):
        try:
            response = ollama.chat(
                model='qwen2.5-coder:7b', # Using a standard model name, change if needed
                messages=[{
                    'role': 'user',
                    'content': prompt
                }]
            )
            self.get_logger().info(f"LLM response:\n{response['message']['content']}")
        except Exception as e:
            self.get_logger().error(f"Ollama call failed: {e}")


        return response['message']['content']

   

 

    def get_graph_context(self, cypher_query, params=None):

        uri = "neo4j://127.0.0.1:7687"
        #auth = ("neo4j", "duplex_only") # local MACHINE
        #auth = ("neo4j", "kg_bim123")  # RTX 4090 Machine
        auth = ("neo4j", "kg_bim_cranehall") # local MACHINE

        with GraphDatabase.driver(uri, auth=auth) as driver:
            with driver.session() as session:
                result = session.run(cypher_query, params or {})
                lines = []
                count = 0
                for record in result:
                    lines.append(str(record.data()))
                    count += 1

                if  count == 0:
                    lines.append("No results found.")
                    self.get_logger().warning("No results found for Neo4j query.")
                return "\n".join(lines)

def main(args=None):
    rclpy.init(args=args)
    node = MyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()