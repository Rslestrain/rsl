# app.py
import os
import sys
import json
import shutil
import uuid
import subprocess
from pathlib import Path
from datetime import datetime
import tempfile
import threading
import time

from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename
import yaml

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Global session storage (in production, use Redis or database)
sessions = {}

class DocumentProcessor:
    def __init__(self, session_id):
        self.session_id = session_id
        self.session_dir = os.path.join('sessions', session_id)
        self.data_dir = os.path.join(self.session_dir, 'data')
        self.config_dir = os.path.join(self.session_dir, 'config')
        self.results_dir = os.path.join(self.session_dir, 'results')
        
        # Create session directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.config_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, 'documents'), exist_ok=True)
        
    def save_file(self, file, filename):
        """Save uploaded file and create sample.json"""
        doc_path = os.path.join(self.data_dir, 'documents', filename)
        file.save(doc_path)
        
        # Create samples.json
        sample_data = [{
            "doc_id": filename,
            "question": "",
            "answer": ""
        }]
        
        samples_path = os.path.join(self.data_dir, 'samples.json')
        with open(samples_path, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, indent=2, ensure_ascii=False)
            
        return doc_path
    
    def create_config_files(self, retrieval_method):
        """Create configuration files for the session"""
        os.makedirs(self.config_dir, exist_ok=True)
        
        # 复制所有必要的配置文件到会话目录
        project_config_dir = os.path.join(project_root, 'config')
        
        # 复制 agent 配置
        agent_dir = os.path.join(self.config_dir, 'agent')
        os.makedirs(agent_dir, exist_ok=True)
        for agent_file in ['image_agent.yaml', 'text_agent.yaml', 'general_agent.yaml', 'sum_agent.yaml', 'base.yaml']:
            src = os.path.join(project_config_dir, 'agent', agent_file)
            dst = os.path.join(agent_dir, agent_file)
            if os.path.exists(src):
                shutil.copy2(src, dst)
        
        # 复制 model 配置
        model_dir = os.path.join(self.config_dir, 'model')
        os.makedirs(model_dir, exist_ok=True)
        for model_file in ['qwen2vl.yaml', 'openai.yaml', 'llama31.yaml', 'base.yaml']:
            src = os.path.join(project_config_dir, 'model', model_file)
            dst = os.path.join(model_dir, model_file)
            if os.path.exists(src):
                shutil.copy2(src, dst)
        
        # 复制 retrieval 配置
        retrieval_dir = os.path.join(self.config_dir, 'retrieval')
        os.makedirs(retrieval_dir, exist_ok=True)
        for retrieval_file in ['text.yaml', 'image.yaml', 'base.yaml']:
            src = os.path.join(project_config_dir, 'retrieval', retrieval_file)
            dst = os.path.join(retrieval_dir, retrieval_file)
            if os.path.exists(src):
                shutil.copy2(src, dst)
        
        # 定义检索相关的键（根据 config/retrieval/base.yaml）
        retrieval_top_k = 10
        text_question_key = 'question'
        image_question_key = 'question'
        
        # 创建主配置文件
        config = {
            'dataset': {
                'name': f'session_{self.session_id}',
                'top_k': 1,
                'question_key': 'question',
                'gt_key': 'answer',
                'page_id_key': 'page_ids',
                'truncate_len': None,
                'max_page': 1000,
                'max_character_per_page': 100000,
                'use_mix': False,
                'data_dir': os.path.abspath(self.data_dir),
                'result_dir': os.path.abspath(self.results_dir),
                'extract_path': os.path.abspath(os.path.join(self.session_dir, 'tmp')),
                'document_path': os.path.abspath(os.path.join(self.data_dir, 'documents')),
                'sample_path': os.path.abspath(os.path.join(self.data_dir, 'samples.json')),
                'sample_with_retrieval_path': os.path.abspath(os.path.join(self.data_dir, 'sample-with-retrieval-results.json')),
                # 添加检索相关的键
                'r_text_key': f'text-top-{retrieval_top_k}-{text_question_key}',
                'r_image_key': f'image-top-{retrieval_top_k}-{image_question_key}',
                'r_mix_key': f'mix-top-{retrieval_top_k}-{text_question_key}',
                'r_text_index_key': f'text-index-path-{text_question_key}',
            },
            
            'retrieval': {
                'model_type': 'text' if retrieval_method != 'image' else 'image',
                'top_k': retrieval_top_k,
                'doc_key': 'doc_id',
                'text_question_key': text_question_key,
                'image_question_key': image_question_key,
                'r_text_key': f'text-top-{retrieval_top_k}-{text_question_key}',
                'r_image_key': f'image-top-{retrieval_top_k}-{image_question_key}',
                'r_mix_key': f'mix-top-{retrieval_top_k}-{text_question_key}',
                'r_text_index_key': f'text-index-path-{text_question_key}',
                'cuda_visible_devices': '0'
            },
            
            'mdoc_agent': {
                'cuda_visible_devices': '0',
                'truncate_len': None,
                'save_freq': 10,
                'ans_key': f'ans_session_{self.session_id}',
                'save_message': False,
                'agents': [
                    {
                        'agent': 'image_agent',
                        'model': 'qwen2vl'
                    },
                    {
                        'agent': 'text_agent',
                        'model': 'openai'
                    },
                    {
                        'agent': 'general_agent',
                        'model': 'qwen2vl'
                    }
                ],
                'sum_agent': {
                    'agent': 'sum_agent',
                    'model': 'qwen2vl'
                }
            },
            
            'run-name': f'session_{self.session_id}',
        }
        
        # 根据检索方式调整配置
        if retrieval_method == 'text':
            config['retrieval']['model_name'] = 'ColbertRetrieval'
            config['retrieval']['class_path'] = 'retrieval.text_retrieval.ColbertRetrieval'
        elif retrieval_method == 'image':
            config['retrieval']['model_name'] = 'ColpaliRetrieval'
            config['retrieval']['class_path'] = 'retrieval.image_retrieval.ColpaliRetrieval'
            config['retrieval']['embed_dir'] = os.path.abspath(os.path.join(self.session_dir, 'tmp', 'ColpaliRetrieval', 'question'))
            config['retrieval']['batch_size'] = 2
        else:  # both
            config['retrieval']['model_name'] = 'ColbertRetrieval'
            config['retrieval']['class_path'] = 'retrieval.text_retrieval.ColbertRetrieval'
        
        # 保存主配置文件
        main_config_path = os.path.join(self.config_dir, 'custom.yaml')
        with open(main_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
        
        return main_config_path
    
    def run_extraction(self):
        """Run document extraction"""
        try:
            original_cwd = os.getcwd()
            os.chdir(project_root)
            
            env = os.environ.copy()
            # 使用绝对路径
            config_path = os.path.abspath(self.config_dir)
            env['HYDRA_CONFIG_PATH'] = config_path
            
            cmd = [
                sys.executable, 'scripts/extract.py',
                f'--config-path={config_path}',  # 使用绝对路径
                '--config-name=custom'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
            
            if result.returncode != 0:
                print(f"Extraction stderr: {result.stderr}")
                raise Exception(f"Extraction failed: {result.stderr}")
                
            return True
            
        except subprocess.TimeoutExpired:
            return False
        except Exception as e:
            print(f"Extraction error: {e}")
            return False
        finally:
            os.chdir(original_cwd)
    
    def run_retrieval(self, retrieval_method):
        """Run retrieval setup"""
        try:
            original_cwd = os.getcwd()
            os.chdir(project_root)
            
            env = os.environ.copy()
            # 使用绝对路径
            config_path = os.path.abspath(self.config_dir)
            env['HYDRA_CONFIG_PATH'] = config_path
            env['CUDA_VISIBLE_DEVICES'] = '0'
            
            # Run text retrieval
            if retrieval_method in ['text', 'both']:
                # 创建文本检索配置
                text_config = self._create_text_retrieval_config()
                text_config_path = os.path.join(self.config_dir, 'text_custom.yaml')
                with open(text_config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(text_config, f, allow_unicode=True, default_flow_style=False)
                
                cmd = [
                    sys.executable, 'scripts/retrieve.py',
                    f'--config-path={config_path}',
                    '--config-name=text_custom'
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
                if result.returncode != 0:
                    print(f"Text retrieval stderr: {result.stderr}")
                    raise Exception(f"Text retrieval failed: {result.stderr}")
            
            # Run image retrieval
            if retrieval_method in ['image', 'both']:
                # 创建图像检索配置
                image_config = self._create_image_retrieval_config()
                image_config_path = os.path.join(self.config_dir, 'image_custom.yaml')
                with open(image_config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(image_config, f, allow_unicode=True, default_flow_style=False)
                
                cmd = [
                    sys.executable, 'scripts/retrieve.py',
                    f'--config-path={config_path}',
                    '--config-name=image_custom'
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
                if result.returncode != 0:
                    print(f"Image retrieval stderr: {result.stderr}")
                    raise Exception(f"Image retrieval failed: {result.stderr}")
                    
            return True
            
        except subprocess.TimeoutExpired:
            return False
        except Exception as e:
            print(f"Retrieval error: {e}")
            return False
        finally:
            os.chdir(original_cwd)

    def _create_text_retrieval_config(self):
        """Create text retrieval configuration"""
        return {
            'dataset': {
                'name': f'session_{self.session_id}',
                'top_k': 1,
                'question_key': 'question',
                'gt_key': 'answer',
                'page_id_key': 'page_ids',
                'truncate_len': None,
                'max_page': 1000,
                'max_character_per_page': 100000,
                'use_mix': False,
                'data_dir': os.path.abspath(self.data_dir),
                'result_dir': os.path.abspath(self.results_dir),
                'extract_path': os.path.abspath(os.path.join(self.session_dir, 'tmp')),
                'document_path': os.path.abspath(os.path.join(self.data_dir, 'documents')),
                'sample_path': os.path.abspath(os.path.join(self.data_dir, 'samples.json')),
                'sample_with_retrieval_path': os.path.abspath(os.path.join(self.data_dir, 'sample-with-retrieval-results.json'))
            },
            'retrieval': {
                'model_type': 'text',
                'model_name': 'ColbertRetrieval',
                'class_path': 'retrieval.text_retrieval.ColbertRetrieval',
                'top_k': 10,
                'doc_key': 'doc_id',
                'text_question_key': 'question',
                'image_question_key': 'question',
                'r_text_key': 'text-top-10-question',
                'r_image_key': 'image-top-10-question',
                'r_mix_key': 'mix-top-10-question',
                'r_text_index_key': 'text-index-path-question',
                'cuda_visible_devices': '0'
            }
        }

    def _create_image_retrieval_config(self):
        """Create image retrieval configuration"""
        return {
            'dataset': {
                'name': f'session_{self.session_id}',
                'top_k': 1,
                'question_key': 'question',
                'gt_key': 'answer',
                'page_id_key': 'page_ids',
                'truncate_len': None,
                'max_page': 1000,
                'max_character_per_page': 100000,
                'use_mix': False,
                'data_dir': os.path.abspath(self.data_dir),
                'result_dir': os.path.abspath(self.results_dir),
                'extract_path': os.path.abspath(os.path.join(self.session_dir, 'tmp')),
                'document_path': os.path.abspath(os.path.join(self.data_dir, 'documents')),
                'sample_path': os.path.abspath(os.path.join(self.data_dir, 'samples.json')),
                'sample_with_retrieval_path': os.path.abspath(os.path.join(self.data_dir, 'sample-with-retrieval-results.json'))
            },
            'retrieval': {
                'model_type': 'image',
                'model_name': 'ColpaliRetrieval',
                'class_path': 'retrieval.image_retrieval.ColpaliRetrieval',
                'embed_dir': os.path.abspath(os.path.join(self.session_dir, 'tmp', 'ColpaliRetrieval', 'question')),
                'batch_size': 2,
                'top_k': 10,
                'doc_key': 'doc_id',
                'text_question_key': 'question',
                'image_question_key': 'question',
                'r_text_key': 'text-top-10-question',
                'r_image_key': 'image-top-10-question',
                'r_mix_key': 'mix-top-10-question',
                'r_text_index_key': 'text-index-path-question',
                'cuda_visible_devices': '0'
            }
        }
    
    def run_inference(self, question, retrieval_method):
        """Run inference with question"""
        try:
            # Update samples.json with question
            samples_path = os.path.join(self.data_dir, 'sample-with-retrieval-results.json')
            if not os.path.exists(samples_path):
                samples_path = os.path.join(self.data_dir, 'samples.json')
                
            with open(samples_path, 'r', encoding='utf-8') as f:
                samples = json.load(f)
            
            if samples:
                samples[0]['question'] = question
                
            with open(samples_path, 'w', encoding='utf-8') as f:
                json.dump(samples, f, indent=2, ensure_ascii=False)
            
            # Run inference
            original_cwd = os.getcwd()
            os.chdir(project_root)
            
            env = os.environ.copy()
            # 使用绝对路径
            config_path = os.path.abspath(self.config_dir)
            env['HYDRA_CONFIG_PATH'] = config_path
            env['CUDA_VISIBLE_DEVICES'] = '0'
            
            cmd = [
                sys.executable, 'scripts/predict.py',
                f'--config-path={config_path}',  # 使用绝对路径
                '--config-name=custom',
                f'run-name=session_{self.session_id}'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
            
            if result.returncode != 0:
                print(f"Inference stderr: {result.stderr}")
                raise Exception(f"Inference failed: {result.stderr}")
            
            # Read result
            result_files = list(Path(self.results_dir).glob('*.json'))
            if not result_files:
                raise Exception("No result file found")
                
            latest_result = max(result_files, key=os.path.getctime)
            
            with open(latest_result, 'r', encoding='utf-8') as f:
                results = json.load(f)
                
            if results and len(results) > 0:
                ans_key = f'ans_session_{self.session_id}'
                answer = results[0].get(ans_key, '抱歉，未能获取到回答。')
                return answer
            else:
                return '抱歉，未能获取到回答。'
                
        except subprocess.TimeoutExpired:
            return '推理超时，请稍后重试。'
        except Exception as e:
            print(f"Inference error: {e}")
            return f'推理过程中出现错误：{str(e)}'
        finally:
            os.chdir(original_cwd)


@app.route('/')
def index():
    """Serve the main page"""
    # Read the HTML content from the artifact
    with open('mdoc_frontend.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    return html_content


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': '没有文件'})
        
        file = request.files['file']
        retrieval_method = request.form.get('retrieval_method', 'text')
        
        if file.filename == '':
            return jsonify({'success': False, 'message': '没有选择文件'})
        
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'success': False, 'message': '只支持PDF文件'})
        
        # Create session
        session_id = str(uuid.uuid4())
        processor = DocumentProcessor(session_id)
        
        # Save file
        filename = secure_filename(file.filename)
        processor.save_file(file, filename)
        
        # Create config files
        processor.create_config_files(retrieval_method)
        
        # Store session
        sessions[session_id] = {
            'processor': processor,
            'filename': filename,
            'retrieval_method': retrieval_method,
            'status': 'uploaded',
            'created_at': datetime.now()
        }
        
        return jsonify({
            'success': True,
            'message': '文件上传成功',
            'doc_id': session_id
        })
        
    except Exception as e:
        print(f"Upload error: {e}")
        return jsonify({'success': False, 'message': f'上传失败：{str(e)}'})


@app.route('/extract', methods=['POST'])
def extract_document():
    """Handle document extraction"""
    try:
        data = request.get_json()
        doc_id = data.get('doc_id')
        
        if doc_id not in sessions:
            return jsonify({'success': False, 'message': '会话不存在'})
        
        session = sessions[doc_id]
        processor = session['processor']
        
        # Run extraction in background thread
        def extract():
            success = processor.run_extraction()
            session['status'] = 'extracted' if success else 'extract_failed'
        
        thread = threading.Thread(target=extract)
        thread.start()
        thread.join(timeout=300)  # 5 minute timeout
        
        if session['status'] == 'extracted':
            return jsonify({'success': True, 'message': '提取完成'})
        else:
            return jsonify({'success': False, 'message': '提取失败'})
            
    except Exception as e:
        print(f"Extract error: {e}")
        return jsonify({'success': False, 'message': f'提取失败：{str(e)}'})


@app.route('/setup_retrieval', methods=['POST'])
def setup_retrieval():
    """Handle retrieval setup"""
    try:
        data = request.get_json()
        doc_id = data.get('doc_id')
        retrieval_method = data.get('retrieval_method', 'text')
        
        if doc_id not in sessions:
            return jsonify({'success': False, 'message': '会话不存在'})
        
        session = sessions[doc_id]
        processor = session['processor']
        
        # Run retrieval setup
        def setup():
            success = processor.run_retrieval(retrieval_method)
            session['status'] = 'ready' if success else 'retrieval_failed'
        
        thread = threading.Thread(target=setup)
        thread.start()
        thread.join(timeout=600)  # 10 minute timeout
        
        if session['status'] == 'ready':
            return jsonify({'success': True, 'message': '检索索引建立完成'})
        else:
            return jsonify({'success': False, 'message': '索引建立失败'})
            
    except Exception as e:
        print(f"Retrieval setup error: {e}")
        return jsonify({'success': False, 'message': f'索引建立失败：{str(e)}'})


@app.route('/ask', methods=['POST'])
def ask_question():
    """Handle question answering"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        retrieval_method = data.get('retrieval_method', 'text')
        
        if not question:
            return jsonify({'success': False, 'message': '问题不能为空'})
        
        # Find the most recent ready session (in production, should track per user)
        ready_sessions = [
            (k, v) for k, v in sessions.items() 
            if v['status'] == 'ready'
        ]
        
        if not ready_sessions:
            return jsonify({'success': False, 'message': '没有准备好的文档，请先上传并处理文档'})
        
        # Use the most recent session
        session_id, session = max(ready_sessions, key=lambda x: x[1]['created_at'])
        processor = session['processor']
        
        # Run inference
        answer = processor.run_inference(question, retrieval_method)
        
        return jsonify({
            'success': True,
            'answer': answer,
            'message': '问答完成'
        })
        
    except Exception as e:
        print(f"Ask error: {e}")
        return jsonify({'success': False, 'message': f'问答失败：{str(e)}'})


@app.route('/status/<doc_id>')
def get_status(doc_id):
    """Get processing status"""
    if doc_id in sessions:
        return jsonify({
            'success': True,
            'status': sessions[doc_id]['status']
        })
    else:
        return jsonify({'success': False, 'message': '会话不存在'})


def cleanup_old_sessions():
    """Clean up old sessions (run periodically)"""
    cutoff = datetime.now()
    cutoff = cutoff.replace(hour=cutoff.hour-1)  # 1 hour ago
    
    to_remove = []
    for session_id, session in sessions.items():
        if session['created_at'] < cutoff:
            # Clean up files
            session_dir = session['processor'].session_dir
            if os.path.exists(session_dir):
                shutil.rmtree(session_dir)
            to_remove.append(session_id)
    
    for session_id in to_remove:
        del sessions[session_id]


if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('sessions', exist_ok=True)
    
    # Create the HTML file from artifact
    html_content = '''<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MDocAgent - 多模态文档问答系统</title>
    <!-- Add your CSS here -->
</head>
<body>
    <!-- Add your HTML here -->
</body>
</html>'''
    
    print("Starting MDocAgent Web Server...")
    print("Please make sure you have:")
    print("1. Set up your API keys (DASHSCOPE_API_KEY)")
    print("2. Installed all required dependencies")
    print("3. GPU available for image processing")
    
    # Start cleanup timer
    import threading
    def periodic_cleanup():
        while True:
            time.sleep(3600)  # Every hour
            cleanup_old_sessions()
    
    cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
    cleanup_thread.start()
    
    app.run(host='0.0.0.0', port=5000, debug=True)