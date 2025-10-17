import ifcopenshell

class BIMDataExtractor:
    def __init__(self, ifc_path, element_type="doors"):
        self.ifc_file = ifcopenshell.open(ifc_path)
        self.element_type = element_type.lower()

        # Select elements by type
        if self.element_type == "doors":
            self.elements = self.ifc_file.by_type("IfcDoor")
        elif self.element_type == "windows":
            self.elements = self.ifc_file.by_type("IfcWindow")
        elif self.element_type == "walls":
            self.elements = self.ifc_file.by_type("IfcWall")
        else:
            raise ValueError(f"Unsupported element type: {element_type}")

    def get_element_name(self, element):
        # Try Name, ObjectType, PredefinedType, LongName
        if element.Name:
            return element.Name
        elif hasattr(element, "ObjectType") and element.ObjectType:
            return element.ObjectType
        elif hasattr(element, "PredefinedType") and element.PredefinedType:
            return element.PredefinedType
        elif hasattr(element, "LongName") and element.LongName:
            return element.LongName
        else:
            return f"Element_{element.GlobalId}"

    def get_element_type(self, element):
        if hasattr(element, "PredefinedType") and element.PredefinedType:
            return element.PredefinedType
        else:
            return element.is_a()

    def get_element_material(self, element):
        # Attempt to get material from IFC relationships
        try:
            rels = [rel for rel in self.ifc_file.by_type("IfcRelAssociatesMaterial") if rel.RelatedObjects and element in rel.RelatedObjects]
            if rels:
                mat = rels[0].RelatingMaterial
                return getattr(mat, "Name", "Unknown")
        except:
            return "Unknown"
        return "Unknown"

    def get_global_location(self, placement):
        """Recursively compute global XYZ from IfcLocalPlacement"""
        try:
            coords = placement.RelativePlacement.Location.Coordinates
        except:
            return [0,0,0]
        
        parent = getattr(placement, "PlacementRelTo", None)
        if parent:
            parent_coords = self.get_global_location(parent)
            coords = [coords[0]+parent_coords[0], coords[1]+parent_coords[1], coords[2]+parent_coords[2]]
        return coords

    def get_room_info(self, element):
        # Attempt to get room assignment
        try:
            rels = [rel for rel in self.ifc_file.by_type("IfcRelContainedInSpatialStructure") if element in rel.RelatedElements]
            if rels:
                return [rel.RelatingStructure.Name for rel in rels if hasattr(rel.RelatingStructure, "Name")]
        except:
            return []
        return []

    def to_text_chunk(self, elem_data):
        #z=y in robot coordinates , y=x in robot coordinates
     
        x, y, z = elem_data["Location"]
        rooms = ", ".join(elem_data["Rooms"]) if elem_data["Rooms"] else "Unknown"
        return (
            f"Element ID: {elem_data['GlobalId']}\n"
            f"Name: {elem_data['Name']}\n"
            f"Type: {elem_data['Type']}\n"
            f"Material: {elem_data['Material']}\n"
            f"Location: x={x/1000}, y={y/1000}, z={z/1000}\n"
            f"Rooms: {rooms}\n"
            f"Description: {elem_data['Type']} named {elem_data['Name']} located in {rooms}.\n"
        )

    def run(self):
        text_chunks = []
        for elem in self.elements:
            elem_data = {
                "Name": self.get_element_name(elem),
                "Type": self.get_element_type(elem),
                "GlobalId": elem.GlobalId,
                "Material": self.get_element_material(elem),
                "Location": [0,0,0],
                "Rooms": self.get_room_info(elem)
            }

            if elem.ObjectPlacement:
                elem_data["Location"] = self.get_global_location(elem.ObjectPlacement)

            text_chunks.append(self.to_text_chunk(elem_data))
        
        return text_chunks

if __name__ == "__main__":
    # Example usage
    extractor = BIMDataExtractor("crane_hall_v10.ifc", "doors")
    doors_text_chunks = extractor.run()

    # Save to file
    with open("doors_rag_chunks.txt", "w") as f:
        for chunk in doors_text_chunks:
            f.write(chunk + "\n")
