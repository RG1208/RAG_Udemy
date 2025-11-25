from typing import List
from langchain_core.documents import Document
import json

print("Method 2: Custom JSON Processing - Extract Project Names for Each Employee")

def process_json_inteligently(file_path: str) -> List[Document]:
    with open(file_path, "r") as f:
        data = json.load(f)
    documents = []
   
    for emp in data.get("employees", []):
        content= f"""Employee Profile:
        name:{emp['name']}
        role:{emp['role']}
        """
        
        doc=Document(
           page_content=content,
           metadata={
               'source':file_path,
               'employee_id':emp['id'],
               'data_type':'employee_profile',
               'emp_name':emp['name'],
               'role':emp['role'],
               'employee_id':emp['id']
           } 
        )
    documents.append(doc)
    
    return documents