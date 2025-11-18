# models/qwen.py
from models.base_model import BaseModel
from openai import OpenAI
import base64
import os

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

class Qwen2VL(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = "qwen-vl-max-latest"
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY", config.get("api_key", "")),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.create_ask_message = lambda question: {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
            ],
        }
        self.create_ans_message = lambda ans: {
            "role": "assistant",
            "content": [
                {"type": "text", "text": ans},
            ],
        }
        
    def create_text_message(self, texts, question):
        content = []
        for text in texts:
            content.append({"type": "text", "text": text})
        content.append({"type": "text", "text": question})
        message = {
            "role": "user",
            "content": content
        }
        return message
        
    def create_image_message(self, images, question):
        content = []
        for image_path in images:
            content.append({
                "type": "image_url", 
                "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image_path)}"}
            })
        content.append({"type": "text", "text": question})
        message = {
            "role": "user",
            "content": content
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
            if not isinstance(item["role"], str):
                return False
            # content can be string or list
            if not isinstance(item["content"], (str, list)):
                return False
        return True

class Qwen2_5VL(Qwen2VL):
    def __init__(self, config):
        super().__init__(config)
        # 使用相同的模型和配置