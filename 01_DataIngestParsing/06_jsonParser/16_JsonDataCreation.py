import json
import os

os.makedirs("data/json_files", exist_ok=True)

json_data={
    "company":"Company 1",
    "employees":[
        {
            "id":1,
            "name":"John Doe",
            "department":"Sales",
            "role":"Manager",
            "projects":[
                {
                    "name":"Project A",
                    "duration_months":6
                },
                {
                    "name":"Project B",
                    "duration_months":12
                }
            ]
        },
        {
            "id":2,
            "name":"Jane Smith",
            "department":"Development",
            "role":"Developer",
            "projects":[
                {
                    "name":"Project C",
                    "duration_months":9
                }
            ]
        },
        {
            "id":3,
            "name":"Emily Johnson",
            "department":"Marketing",
            "role":"Coordinator",
            "projects":[
                {
                    "name":"Project D",
                    "duration_months":4
                },
                {
                    "name":"Project E",
                    "duration_months":8
                }
            ]
        }
    ],
    "departments":{
        "engineering":{
            "head":"Alice Brown",
            "budget":500000,
            "team_size":25
        },
        "sales":{
            "head":"Bob White",
            "budget":300000,
            "team_size":15
        },
    }
}

# print(json_data)

with open("data/json_files/company_data.json", "w") as json_file:
    json.dump(json_data, json_file, indent=2)

jsonl_data = [
    {"timestamp": "2024-01-01", "event": "user_login", "user_id": 123},
    {"timestamp": "2024-01-01", "event": "page_view", "user_id": 123, "page": "/home"},
    {"timestamp": "2024-01-01", "event": "purchase", "user_id": 123, "amount": 99.99}
]

with open('data/json_files/events.jsonl', 'w') as f:
    for item in jsonl_data:
        f.write(json.dumps(item) + '\n')