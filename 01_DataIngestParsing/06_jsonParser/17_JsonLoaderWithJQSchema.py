from langchain_community.document_loaders import JSONLoader
import json

#method 1: jsonLoader with jq_schema

print("Method 1: JSONLoader -Extract Details")
employee_loader=JSONLoader(
    file_path='../data/json_files/company_data.json',
    jq_schema='.employees[]',
    text_content=False
)

employee_docs=employee_loader.load()
for doc in employee_docs:
    print(doc.page_content)
    print(doc.metadata)
print("----------------")
print(employee_docs)