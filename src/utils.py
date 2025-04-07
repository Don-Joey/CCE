import json
import ast
def write_output(filename, list_string):
    with open(filename, 'w') as file:
        json.dump(list_string,file,indent=4)
        #for string in list_string:
        #    file.write(string + '\n')

def read_json(filename):
    with open(filename, "rb") as f:
        return json.load(f)

def read_jsonl(filename):
    data_list = []
    # Open the JSONL file
    with open(filename, 'r', encoding = 'utf-8') as file:
        for line in file:
            # Parse the JSON object from the line and append it to the list
            data_list.append(json.loads(line.strip()))
    return data_list

def prompt_to_chatml(prompt: str, start_token: str = "<|im_start|>", end_token: str = "<|im_end|>"):
        """Convert a text prompt to ChatML formal

        Examples
        --------
        >>> prompt = "<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n<|im_start|>system
        name=example_user\nKnock knock.\n<|im_end|>\n<|im_start|>system name=example_assistant\nWho's
        there?\n<|im_end|>\n<|im_start|>user\nOrange.\n<|im_end|>"
        >>> print(prompt)
        <|im_start|>system
        You are a helpful assistant.
        <|im_end|>
        <|im_start|>system name=example_user
        Knock knock.
        <|im_end|>
        <|im_start|>system name=example_assistant
        Who's there?
        <|im_end|>
        <|im_start|>user
        Orange.
        <|im_end|>
        >>> prompt_to_chatml(prompt)
        [{'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'Knock knock.'},
        {'role': 'assistant', 'content': "Who's there?"},
        {'role': 'user', 'content': 'Orange.'}]
        """
        prompt = prompt.strip()
        assert prompt.startswith(start_token)
        assert prompt.endswith(end_token)

        def string_to_dict(to_convert):
            """Converts a string with equal signs to dictionary. E.g.
            >>> string_to_dict(" name=user university=stanford")
            {'name': 'user', 'university': 'stanford'}
            """
            return {s.split("=", 1)[0]: s.split("=", 1)[1] for s in to_convert.split(" ") if len(s) > 0}

        message = []
        for p in prompt.split("<|im_start|>")[1:]:
            newline_splitted = p.split("\n", 1)
            role = newline_splitted[0].strip()
            content = newline_splitted[1].split(end_token, 1)[0].strip()

            if role.startswith("system") and role != "system":
                # based on https://github.com/openai/openai-cookbook/blob/main/examples
                # /How_to_format_inputs_to_ChatGPT_models.ipynb
                # and https://github.com/openai/openai-python/blob/main/chatml.md it seems that system can specify a
                # dictionary of other args
                other_params = string_to_dict(role.split("system", 1)[-1])
                role = "system"
            else:
                other_params = dict()

            message.append(dict(content=content, role=role, **other_params))

        return message

def parse_formatted_text(text):
    # Remove the markdown-like code block delimiters
    cleaned_text = text.strip().strip('```python\n').strip('\n```')
    
    # Use ast.literal_eval to safely evaluate the string into a Python object
    parsed_data = ast.literal_eval(cleaned_text)
    
    return parsed_data

def extract_generation(case):
    '''
    {
        "instruction": "explain master slave replication nsql",
        "multiturn": false,
        "judge_prompt": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "explain master slave replication nsql\n"
            },
            "Master-slave replication is a database replication model commonly used in SQL databases to ensure data redundancy and increase availability and scalability. Here\u2019s a breakdown of how it works and its key features:\n\n### Overview\n\n1. **Master Node**: \n   - The master database is the primary node where all write operations (inserts, updates, deletes) are performed. It serves as the authoritative data source.\n   - The master is responsible for processing changes and then propagating these changes to the slave nodes.\n\n2. **Slave Nodes**: \n   - Slave databases are read-only replicas of the master database. They receive updates from the master to stay synchronized.\n   - The slaves can be used to distribute read traffic, thereby offloading work from the master and improving performance.\n\n### Key Features\n\n- **Data Consistency**: The master node ensures consistency, while the slave nodes are meant to reflect the current state of the master as accurately as possible. However, there might be a slight delay (latency) due to the time it takes to propagate changes.\n\n- **Scalability**: By distributing read operations across multiple slave nodes, the system can handle more requests, thus improving read scalability.\n\n- **Fault Tolerance**: In case the master node fails, one of the slave nodes can potentially be promoted to become the new master. This setup can provide some level of high availability, though additional mechanisms are needed for automated failover.\n\n- **Load Balancing**: Read operations can be load-balanced across the slave nodes to enhance performance and reduce load on the master node.\n\n- **Backup and Recovery**: Since slave nodes contain copies of the data, they can be used for backup purposes without affecting the performance of the master node.\n\n### Considerations\n\n- **Latency**: There can be a lag between when a change is made on the master and when it is reflected on the slaves. This is often referred to as \"replication lag.\"\n\n- **Consistency**: Applications must be designed to handle eventual consistency. Not all reads from slaves may reflect the most recent writes until the replication process catches up.\n\n- **Complexity**: Managing a master-slave setup involves additional complexity, including handling replication lag, configuring failover mechanisms, and maintaining consistency.\n\n- **Writes Bottleneck**: Since all write operations must go through the master, the master can become a bottleneck for write-heavy workloads.\n\nThis model is widely used in scenarios where the read load significantly exceeds the write load, and eventual consistency is acceptable. However, for applications requiring strict consistency or high write throughput, other architectures such as multi-master replication or sharded databases might be more appropriate."
        ],
        "answer_a": [
            {
                "content": {
                    "role": "user",
                    "content": "explain master slave replication nsql\n"
                }
            },
            {
                "content": "Master-slave replication is a database replication model commonly used in SQL databases to ensure data redundancy and increase availability and scalability. Here\u2019s a breakdown of how it works and its key features:\n\n### Overview\n\n1. **Master Node**: \n   - The master database is the primary node where all write operations (inserts, updates, deletes) are performed. It serves as the authoritative data source.\n   - The master is responsible for processing changes and then propagating these changes to the slave nodes.\n\n2. **Slave Nodes**: \n   - Slave databases are read-only replicas of the master database. They receive updates from the master to stay synchronized.\n   - The slaves can be used to distribute read traffic, thereby offloading work from the master and improving performance.\n\n### Key Features\n\n- **Data Consistency**: The master node ensures consistency, while the slave nodes are meant to reflect the current state of the master as accurately as possible. However, there might be a slight delay (latency) due to the time it takes to propagate changes.\n\n- **Scalability**: By distributing read operations across multiple slave nodes, the system can handle more requests, thus improving read scalability.\n\n- **Fault Tolerance**: In case the master node fails, one of the slave nodes can potentially be promoted to become the new master. This setup can provide some level of high availability, though additional mechanisms are needed for automated failover.\n\n- **Load Balancing**: Read operations can be load-balanced across the slave nodes to enhance performance and reduce load on the master node.\n\n- **Backup and Recovery**: Since slave nodes contain copies of the data, they can be used for backup purposes without affecting the performance of the master node.\n\n### Considerations\n\n- **Latency**: There can be a lag between when a change is made on the master and when it is reflected on the slaves. This is often referred to as \"replication lag.\"\n\n- **Consistency**: Applications must be designed to handle eventual consistency. Not all reads from slaves may reflect the most recent writes until the replication process catches up.\n\n- **Complexity**: Managing a master-slave setup involves additional complexity, including handling replication lag, configuring failover mechanisms, and maintaining consistency.\n\n- **Writes Bottleneck**: Since all write operations must go through the master, the master can become a bottleneck for write-heavy workloads.\n\nThis model is widely used in scenarios where the read load significantly exceeds the write load, and eventual consistency is acceptable. However, for applications requiring strict consistency or high write throughput, other architectures such as multi-master replication or sharded databases might be more appropriate."
            }
        ],
        "answer_b": [
            {
                "content": {
                    "role": "user",
                    "content": "explain master slave replication nsql\n"
                }
            },
            {
                "content": "Master-slave replication is a data replication technique used in databases, including NoSQL databases. It involves a primary database server (the \"master\") that handles all write operations, and one or more secondary database servers (the \"slaves\") that replicate the data from the master and handle read operations. This architecture is commonly used to improve scalability, reliability, and performance.\n\n### Key Concepts:\n\n1. **Master Node**: \n   - This is the primary server where all write operations occur. It processes data modifications like inserts, updates, and deletes.\n   - The master node is responsible for ensuring data consistency and integrity.\n\n2. **Slave Nodes**: \n   - These are secondary servers that replicate the data from the master. They can handle read requests, which helps distribute the load and improve read performance.\n   - Slaves can be used for backup purposes, fault tolerance, and load balancing.\n\n3. **Replication Process**:\n   - The master node sends changes (data modifications) to the slave nodes. This can happen in real-time or at scheduled intervals, depending on the system's configuration.\n   - The slaves apply these changes to their databases, effectively mirroring the master\u2019s data state.\n\n4. **Read and Write Operations**:\n   - Write operations are directed to the master, ensuring that all updates go through a single point to maintain consistency.\n   - Read operations can be distributed among the slave nodes, allowing multiple clients to access data simultaneously without overloading the master.\n\n5. **Failover and Recovery**:\n   - If the master fails, a slave can be promoted to master to ensure continuity of service. Some systems may require manual intervention, while others can automate this process.\n\n### Advantages:\n\n- **Scalability**: By distributing read requests across multiple slave nodes, the system can handle more read operations as demand grows.\n- **Performance**: Offloading read operations to slaves can reduce the load on the master and improve response times for end-users.\n- **Redundancy**: Having multiple copies of the data across different nodes improves data availability and fault tolerance.\n\n### Disadvantages:\n\n- **Consistency**: Since slaves replicate data asynchronously, there may be a slight delay (replication lag) before they reflect the latest changes made to the master. This can lead to stale reads.\n- **Single Point of Failure**: The master node is critical; if it goes down and there are no available slaves to promote, write operations will halt until the master is restored.\n- **Complexity**: Managing a master-slave setup can add complexity to the system, especially in terms of configuration, monitoring, and handling failovers.\n\n### Use Cases:\n\n- **Read-heavy Applications**: Applications that require high read throughput, such as content delivery networks or reporting systems, benefit from this architecture.\n- **Data Warehousing**: Systems that require data aggregation and analysis can use slave nodes for reporting while keeping the master focused on transactions.\n- **High Availability Requirements**: Systems that cannot afford downtime can leverage master-slave replication to ensure that data is always accessible.\n\n### Conclusion:\n\nMaster-slave replication is a powerful technique used in NoSQL databases to improve the performance and scalability of applications. It\u2019s important to understand both its benefits and limitations to effectively implement it in a database architecture."
            }
        ]
    },

    {
        "instruction": "How do I detail a car?",
        "multiturn": false,
        "judge_prompt": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "How do I detail a car?\n"
            },
            "Detailing a car involves thoroughly cleaning, restoring, and finishing both the interior and exterior to achieve a high-quality cleanliness and polish. Here\u2019s a step-by-step guide to help you detail your car:\n\n### Exterior Detailing\n\n1. **Gather Supplies:**\n   - Car wash soap\n   - Buckets (at least two, one for soapy water and one for rinsing)\n   - Microfiber wash mitts\n   - Hose or pressure washer\n   - Clay bar\n   - Car polish\n   - Car wax or sealant\n   - Microfiber towels\n   - Wheel cleaner and brushes\n   - Tire shine product\n\n2. **Wash the Car:**\n   - Rinse the car thoroughly with water to remove loose dirt.\n   - Use the two-bucket method: one bucket with soapy water and one with clean water for rinsing the mitt.\n   - Start washing from the top down, using a microfiber wash mitt.\n   - Rinse the mitt frequently in the clean water bucket.\n\n3. **Clean the Wheels:**\n   - Use a wheel cleaner and appropriate brushes to scrub the wheels.\n   - Rinse thoroughly to remove all cleaner and brake dust.\n\n4. **Clay Bar Treatment:**\n   - Use a clay bar to remove embedded contaminants from the paint surface.\n   - Lubricate the surface with a clay lubricant or soapy water.\n   - Gently glide the clay bar over the paint until it feels smooth.\n\n5. **Polish:**\n   - Use a car polish to remove minor scratches and swirl marks.\n   - Apply with an applicator pad or a dual-action polisher.\n   - Work in small sections and buff off with a microfiber towel.\n\n6. **Wax or Sealant:**\n   - Apply a layer of car wax or sealant for protection and shine.\n   - Use an applicator pad and follow the product instructions.\n   - Buff off with a clean microfiber towel.\n\n7. **Tire Dressing:**\n   - Apply tire shine product to the tires for a glossy look.\n\n### Interior Detailing\n\n1. **Gather Supplies:**\n   - Vacuum cleaner\n   - Interior cleaner or all-purpose cleaner\n   - Microfiber cloths\n   - Soft brushes\n   - Glass cleaner\n   - Leather conditioner (if applicable)\n\n2. **Vacuum:**\n   - Remove floor mats and vacuum the seats, carpets, and mats thoroughly.\n   - Use attachments to reach crevices and under seats.\n\n3. **Clean Surfaces:**\n   - Use an interior cleaner with a microfiber cloth to clean the dashboard, console, and other surfaces.\n   - Use soft brushes for vents and tight spots.\n\n4. **Clean the Glass:**\n   - Use a glass cleaner and microfiber cloth to clean the inside of the windows.\n\n5. **Condition Leather:**\n   - If you have leather seats, apply a leather conditioner to keep them supple and prevent cracking.\n\n6. **Final Touches:**\n   - Replace floor mats and ensure everything is dry and spotless.\n\n### Tips:\n- Work in the shade or a cool area to prevent water spots.\n- Always read product instructions before use.\n- Take your time to ensure every area is thoroughly cleaned.\n\nThis guide should help you achieve a professional-level detail for your vehicle."
        ],
        "response_output": "Detailing a car involves thoroughly cleaning, restoring, and finishing both the interior and exterior to achieve a high-quality cleanliness and polish. Here\u2019s a step-by-step guide to help you detail your car:\n\n### Exterior Detailing\n\n1. **Gather Supplies:**\n   - Car wash soap\n   - Buckets (at least two, one for soapy water and one for rinsing)\n   - Microfiber wash mitts\n   - Hose or pressure washer\n   - Clay bar\n   - Car polish\n   - Car wax or sealant\n   - Microfiber towels\n   - Wheel cleaner and brushes\n   - Tire shine product\n\n2. **Wash the Car:**\n   - Rinse the car thoroughly with water to remove loose dirt.\n   - Use the two-bucket method: one bucket with soapy water and one with clean water for rinsing the mitt.\n   - Start washing from the top down, using a microfiber wash mitt.\n   - Rinse the mitt frequently in the clean water bucket.\n\n3. **Clean the Wheels:**\n   - Use a wheel cleaner and appropriate brushes to scrub the wheels.\n   - Rinse thoroughly to remove all cleaner and brake dust.\n\n4. **Clay Bar Treatment:**\n   - Use a clay bar to remove embedded contaminants from the paint surface.\n   - Lubricate the surface with a clay lubricant or soapy water.\n   - Gently glide the clay bar over the paint until it feels smooth.\n\n5. **Polish:**\n   - Use a car polish to remove minor scratches and swirl marks.\n   - Apply with an applicator pad or a dual-action polisher.\n   - Work in small sections and buff off with a microfiber towel.\n\n6. **Wax or Sealant:**\n   - Apply a layer of car wax or sealant for protection and shine.\n   - Use an applicator pad and follow the product instructions.\n   - Buff off with a clean microfiber towel.\n\n7. **Tire Dressing:**\n   - Apply tire shine product to the tires for a glossy look.\n\n### Interior Detailing\n\n1. **Gather Supplies:**\n   - Vacuum cleaner\n   - Interior cleaner or all-purpose cleaner\n   - Microfiber cloths\n   - Soft brushes\n   - Glass cleaner\n   - Leather conditioner (if applicable)\n\n2. **Vacuum:**\n   - Remove floor mats and vacuum the seats, carpets, and mats thoroughly.\n   - Use attachments to reach crevices and under seats.\n\n3. **Clean Surfaces:**\n   - Use an interior cleaner with a microfiber cloth to clean the dashboard, console, and other surfaces.\n   - Use soft brushes for vents and tight spots.\n\n4. **Clean the Glass:**\n   - Use a glass cleaner and microfiber cloth to clean the inside of the windows.\n\n5. **Condition Leather:**\n   - If you have leather seats, apply a leather conditioner to keep them supple and prevent cracking.\n\n6. **Final Touches:**\n   - Replace floor mats and ensure everything is dry and spotless.\n\n### Tips:\n- Work in the shade or a cool area to prevent water spots.\n- Always read product instructions before use.\n- Take your time to ensure every area is thoroughly cleaned.\n\nThis guide should help you achieve a professional-level detail for your vehicle."
    },

    '''
    new_case_a, new_case_b = {}, {}
    new_case_a["instruction"] = case["instruction"]
    new_case_a["multiturn"] = case["multiturn"]
    new_case_a["judge_prompt"] = case["judge_prompt"]
    new_case_a["response_output"] = case["answer_a"][1]["content"]
    new_case_b["instruction"] = case["instruction"]
    new_case_b["multiturn"] = case["multiturn"]
    new_case_b["judge_prompt"] = case["judge_prompt"]
    new_case_b["response_output"] = case["answer_b"][1]["content"]
    return new_case_a, new_case_b
