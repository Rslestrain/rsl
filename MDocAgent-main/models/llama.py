# models/llama.py
from models.base_model import BaseModel
from openai import OpenAI
import os

class Llama3(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = "qwen2.5-7b-instruct"
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY", config.get("api_key", "")),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        
        self.create_ask_message = lambda question: {
            "role": "user",
            "content": question,
        }
        self.create_ans_message = lambda ans: {
            "role": "assistant",
            "content": ans,
        }
    
    def create_text_message(self, texts, question): 
        prompt = ""
        for text in texts:
            prompt = prompt + text + '\n'
        message = {
            "role": "user",
            "content": f"{prompt}\n{question}",
        }
        return message
    
    def predict(self, question, texts=None, images=None, history=None):
        messages = self.process_message(question, texts, images, history)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=getattr(self.config, 'temperature', 0),
            max_tokens=getattr(self.config, 'max_new_tokens', 256),
        )
        result = response.choices[0].message.content
        messages.append(self.create_ans_message(result))
        return result, messages
        
    def is_valid_history(self, history):
        if not isinstance(history, list):
            return False
        for item in history:
            if not isinstance(item, dict):
                return False
            if "role" not in item or "content" not in item:
                return False
            if not isinstance(item["role"], str) or not isinstance(item["content"], str):
                return False
        return True