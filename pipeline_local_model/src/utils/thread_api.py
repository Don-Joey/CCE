import glob

import requests, random
from queue import Queue
import threading
import json
import openai, time, os

from openai import OpenAI
from .io import read_jsonl
import httpx
from fastchat.conversation import get_conv_template

from .io import write_output

class Generation1_Api_configuration():
    def __init__(self, data_root, config):
        model_name = config["model_name"]
        self.client=OpenAI(api_key='<KEY>')
        self.client.api_key=config['api_key']
        self.client.base_url=config['base_url']#'http://rerverseapi.workergpt.cn/v1'
        self.target = data_root+"/thread_output_buffer/"+"default_generate_"+model_name+"_OUTPUT*"
        if os.path.isdir(data_root+"/thread_output_buffer"):
            pass
        else:
            os.mkdir(data_root+"/thread_output_buffer")
        self.model_name = model_name
        self.temperature = config["temperature"]





def thread_evaluate_api(prompts, api_config):
    
    
    class Crawl_thread(threading.Thread):
        def __init__(self, thread_id, queue, api_config):
            threading.Thread.__init__(self)
            self.thread_id = thread_id
            #self.filename = target.replace("*", "_thread" + str(self.thread_id) + ".jsonl")
            self.queue = queue
            self.api = api_config
        def openai_template(self, system_prompt, user_prompt):
            template = "chatgpt"
            conv = get_conv_template(template)

            conv.append_message(conv.roles[0], user_prompt)
            conv.append_message(conv.roles[1], None)
            conv.set_system_message(system_prompt)
            messages = conv.to_openai_api_messages()
            return messages
        def run(self):
            print("start thread:", self.thread_id)
            self.crawl_spider()
            print("quit thread:", self.thread_id)
        
        def crawl_spider(self):
            global all_get_data3, candidate_key
            while True:
                if self.queue.empty():
                    break
                else:
                    row= self.queue.get()
                    #msgs=row["prompt_new"]
                    msgs = row #self.openai_template(row[0], row[1])
                    try:
                        success = False
                        for attempt in range(5):
                            try:
                                #client.api_key = random.choice(all_keys)
                                response = self.api.client.chat.completions.create(
                                    model = self.api.model_name,
                                    messages=msgs, 
                                    temperature=self.api.temperature
                                )
                            except Exception as e:
                                time.sleep(random.randint(1, 30))
                                print(f'{e}')
                            except openai.error.APIerror:
                                time.sleep(random.randint(1, 30))
                                print(f"API GG")
                            else:
                                success=True
                                break
                        if success:
                            res = response.choices[0].message.content
                            row.append(res)
                            filename = self.api.target.replace("*", "_thread" + str(self.thread_id) + ".jsonl")
                            with open(filename, "a+", encoding = 'utf-8') as f_a:
                                f_a.write(json.dumps(row, ensure_ascii=False)+"\n")
                    except:
                        pass
    
    all_data = prompts
    pageQueue = Queue(len(all_data))
    for p in all_data:
        pageQueue.put(p)
    print(pageQueue.qsize())

    crawl_threads = []
    crawl_name_list = range(50)
    for thread_id in crawl_name_list:
        thread = Crawl_thread(thread_id, pageQueue, api_config)
        time.sleep(0.5)
        thread.start()
        crawl_threads.append(thread)
    for thread in crawl_threads:
        thread.join()  # Wait for each thread to finish
    print(api_config.target)
    return api_config.target




def thread_generate_api(prompts, api_config):
    
    
    class Crawl_thread(threading.Thread):
        def __init__(self, thread_id, queue, api_config):
            threading.Thread.__init__(self)
            self.thread_id = thread_id
            #self.filename = target.replace("*", "_thread" + str(self.thread_id) + ".jsonl")
            self.queue = queue
            self.api = api_config
        def openai_template(self, system_prompt, user_prompt):
            template = "chatgpt"
            conv = get_conv_template(template)

            conv.append_message(conv.roles[0], user_prompt)
            conv.append_message(conv.roles[1], None)
            conv.set_system_message(system_prompt)
            messages = conv.to_openai_api_messages()
            return messages
        def run(self):
            print("start thread:", self.thread_id)
            self.crawl_spider()
            print("quit thread:", self.thread_id)
        
        def crawl_spider(self):
            global all_get_data3, candidate_key
            while True:
                if self.queue.empty():
                    break
                else:
                    row= self.queue.get()
                    #msgs=row["prompt_new"]
                    msgs = row #self.openai_template(row[0], row[1])
                    print(msgs)
                    try:
                        success = False
                        for attempt in range(5):
                            try:
                                #client.api_key = random.choice(all_keys)
                                response = self.api.client.chat.completions.create(
                                    model = self.api.model_name,
                                    messages=msgs, 
                                    temperature=self.api.temperature
                                )
                            except Exception as e:
                                time.sleep(random.randint(1, 30))
                                print(f'{e}')
                            except openai.error.APIerror:
                                time.sleep(random.randint(1, 30))
                                print(f"API GG")
                            else:
                                success=True
                                break
                        if success:
                            res = response.choices[0].message.content
                            row.append(res)
                            filename = self.api.target.replace("*", "_thread" + str(self.thread_id) + ".jsonl")
                            with open(filename, "a+", encoding = 'utf-8') as f_a:
                                f_a.write(json.dumps(row, ensure_ascii=False)+"\n")
                    except:
                        pass
    
    all_data = prompts
    pageQueue = Queue(len(all_data))
    for p in all_data:
        pageQueue.put(p)
    print(pageQueue.qsize())

    crawl_threads = []
    crawl_name_list = range(2)
    for thread_id in crawl_name_list:
        thread = Crawl_thread(thread_id, pageQueue, api_config)
        time.sleep(0.5)
        thread.start()
        crawl_threads.append(thread)
    for thread in crawl_threads:
        thread.join()  # Wait for each thread to finish
    print(api_config.target)
    return api_config.target


