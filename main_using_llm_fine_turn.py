from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import time
import json
from typing import List, Dict, Any
import uuid
from datetime import datetime

# Khởi tạo model và tokenizer (bình thường sẽ làm trong startup event)
def initialize_local_model():
    """Khởi tạo model GPT-2 đã fine-tuning"""
    try:
        # Đường dẫn tới model đã fine-tuning
        model_path = "./models/gpt2_cinema_finetuned"
        
        # Tải model và tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        model = GPT2LMHeadModel.from_pretrained(model_path)
        
        # Đưa model lên GPU nếu có
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        logger.info(f"Local GPT-2 model loaded successfully on {device}")
        return model, tokenizer, device
    except Exception as e:
        logger.error(f"Error loading local GPT-2 model: {e}")
        return None, None, None

# Tạo cache cho model
local_model_cache = {
    "model": None,
    "tokenizer": None,
    "device": None,
    "initialized": False
}

def get_local_model():
    """Lấy model từ cache hoặc khởi tạo mới"""
    if not local_model_cache["initialized"]:
        model, tokenizer, device = initialize_local_model()
        if model and tokenizer:
            local_model_cache["model"] = model
            local_model_cache["tokenizer"] = tokenizer
            local_model_cache["device"] = device
            local_model_cache["initialized"] = True
    
    return local_model_cache["model"], local_model_cache["tokenizer"], local_model_cache["device"]

def format_perplexity_response(generated_text, model_name="gpt2-cinema"):
    """Format output giống như Perplexity API để frontend không cần thay đổi"""
    # Sử dụng cấu trúc JSON giống Perplexity
    response = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "model": model_name,
        "created": int(time.time()),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generated_text.strip()
                },
                "finish_reason": "stop"
            }
        ]
    }
    return response

async def generate_with_local_model(messages: List[Dict[str, str]], temperature=0.7, max_tokens=1000):
    """Tạo phản hồi sử dụng model GPT-2 đã fine-tuning kết hợp với RAG"""
    # Lấy model từ cache
    model, tokenizer, device = get_local_model()
    
    if not model or not tokenizer:
        # Nếu không thể tải model, trả về thông báo lỗi
        return {
            "error": "Không thể tải model GPT-2, vui lòng kiểm tra lại cài đặt."
        }
    
    try:
        # Trích xuất system message và user message
        system_message = ""
        user_message = ""
        
        for msg in messages:
            if msg.get("role") == "system":
                system_message = msg.get("content", "")
            elif msg.get("role") == "user":
                user_message = msg.get("content", "")
        
        # Lấy ngữ cảnh từ RAG nếu user_message tồn tại
        if user_message:
            context = get_relevant_context(user_message)
        else:
            context = ""
        
        # Kết hợp system message, context và user message
        if context:
            prompt = f"{system_message}\n\nTHÔNG TIN THAM KHẢO:\n{context}\n\nNgười dùng: {user_message}\nTrợ lý:"
        else:
            prompt = f"{system_message}\n\nNgười dùng: {user_message}\nTrợ lý:"
        
        # Mã hóa prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Tạo trạng thái tạo câu trả lời
        gen_params = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "top_k": 50,
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
            "no_repeat_ngram_size": 3,
        }
        
        # Ghi log cho debug
        logger.info(f"Generating response for prompt: {prompt[:100]}...")
        
        # Tạo câu trả lời
        with torch.no_grad():
            output = model.generate(**gen_params)
        
        # Giải mã câu trả lời
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Tách phần trả lời từ prompt
        assistant_reply = generated_text[len(prompt):].strip()
        
        # Ghi log câu trả lời
        logger.info(f"Generated response: {assistant_reply[:100]}...")
        
        # Format response theo định dạng Perplexity API
        response = format_perplexity_response(assistant_reply)
        
        return response
    
    except Exception as e:
        logger.error(f"Error generating response with local model: {e}")
        return {
            "error": f"Lỗi khi tạo câu trả lời: {str(e)}"
        }

# Endpoint để sử dụng model local
@app.post("/local/chat/completions")
async def local_model_completions(request: Request):
    """Endpoint tương tự như /chat/completions nhưng sử dụng model local"""
    try:
        # Đọc nội dung từ request
        body = await request.json()
        
        # Log request (nếu cần)
        logger.info(f"Received request to local model: {body}")
        
        # Lấy thông số cần thiết
        messages = body.get("messages", [])
        temperature = body.get("temperature", 0.7)
        max_tokens = body.get("max_tokens", 1000)
        
        # Tạo câu trả lời
        response = await generate_with_local_model(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Trả về câu trả lời
        return Response(
            content=json.dumps(response),
            media_type="application/json",
            status_code=200
        )
    
    except Exception as e:
        logger.error(f"Error in local model endpoint: {str(e)}")
        return {"error": str(e)}

# Mở rộng endpoint proxy_perplexity để có thể chuyển đổi giữa local và API
@app.post("/chat/completions")
async def proxy_perplexity(request: Request):
    """Endpoint xử lý request và quyết định sử dụng local model hoặc Perplexity API"""
    try:
        # Đọc nội dung từ request
        body = await request.json()
        
        # Log request (không log trong môi trường production)
        logger.info(f"Received request: {body}")
        
        # Cấu hình để kiểm soát việc sử dụng model local
        use_local_model = False  # Mặc định là False, có thể thay đổi sau
        
        # Kiểm tra header hoặc param để quyết định sử dụng local model
        # Ví dụ: body.get("use_local_model", False)
        
        if use_local_model:
            # Sử dụng model local
            return await local_model_completions(request)
        else:
            # Sử dụng Perplexity API - giữ nguyên code như đã có
            # ...code hiện tại của proxy_perplexity...
            
          # Tạo headers với API key
          headers = {
              "Authorization": PERPLEXITY_API_KEY,
              "Content-Type": "application/json"
          }
        
        # Gửi request đến Perplexity API
        async with httpx.AsyncClient() as client:
            perplexity_response = await client.post(
                f"{PERPLEXITY_API_URL}/chat/completions",
                json=body,
                headers=headers
            )
            
            # Lấy response data
            response_data = perplexity_response.json()
            
            # Log response (không log trong môi trường production)
            logger.info(f"Perplexity response: {response_data}")
            
            # Lưu vào cache
            save_to_cache(str(messages), model, response_data)
            
            # Trả về response với cùng status code
            return Response(
                content=json.dumps(response_data),
                media_type="application/json",
                status_code=perplexity_response.status_code
            )
            
    except Exception as e:
        logger.error(f"Error in proxy: {str(e)}")
        return {"error": str(e)}

# Thêm trường thông tin model vào endpoint status
@app.get("/status")
async def status():
    db_status = "connected" if connection_pool else "disconnected"
    vector_status = "initialized" if vector_store else "not initialized"
    local_model_status = "loaded" if local_model_cache["initialized"] else "not loaded"
    
    return {
        "status": "running",
        "database": db_status,
        "vector_store": vector_status,
        "local_model": local_model_status,
        "cache_items": len(prompt_response_cache)
    }