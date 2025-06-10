from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
import httpx
import json
import logging
import os
from typing import List, Dict, Any, Optional
import mysql.connector
from mysql.connector import pooling
import time
from datetime import datetime
import hashlib
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import pandas as pd

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("perplexity_proxy.log")
    ]
)
logger = logging.getLogger("perplexity-cinema-proxy")

app = FastAPI(title="Perplexity API Proxy for Cinema")

# Thêm CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API key của Perplexity
PERPLEXITY_API_KEY = "Bearer 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"
PERPLEXITY_API_URL = "https://api.perplexity.ai"

# Cấu hình kết nối DB
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "dat123456",  # Thay đổi mật khẩu thực tế
    "database": "rap_phim_online",
    "port": 3306
}

# Tạo connection pool
try:
    connection_pool = pooling.MySQLConnectionPool(
        pool_name="rap_phim_online",
        pool_size=5,
        **DB_CONFIG
    )
    logger.info("Database connection pool created successfully")
except Exception as e:
    logger.error(f"Error creating database connection pool: {e}")
    connection_pool = None

# Cache cho các query
query_cache = {}
embeddings_cache = {}
vector_store = None

# Model embeddings
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # Hỗ trợ tiếng Việt
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


def initialize_vector_store():
    """Khởi tạo vector store với dữ liệu từ database"""
    global vector_store
    
    if vector_store is not None:
        return vector_store
    
    try:
        # Tạo kết nối đến DB
        connection = connection_pool.get_connection()
        cursor = connection.cursor(dictionary=True)
        
        # Lấy dữ liệu từ các bảng quan trọng
        data_chunks = []
        
        # 1. Phim
        cursor.execute("SELECT * FROM phim")
        phim_list = cursor.fetchall()
        for phim in phim_list:
            content = f"""
            THÔNG TIN PHIM:
            ID: {phim['id']}
            Tên phim: {phim['ten']}
            Thể loại: {phim['the_loai']}
            Đạo diễn: {phim['dao_dien']}
            Diễn viên: {phim['dien_vien']}
            Độ dài: {phim['do_dai']} phút
            Độ tuổi: {phim['do_tuoi']}+
            Mô tả: {phim['mo_ta']}
            Năm sản xuất: {phim['nam_sx']}
            Hãng sản xuất: {phim['hang_sx']}
            Ngôn ngữ: {phim['ngon_ngu']}
            Trạng thái: {phim['trang_thai']}
            Đánh giá: {phim['danh_gia']}
            """
            data_chunks.append(content)
        
        # 2. Phòng
        cursor.execute("SELECT * FROM phong")
        phong_list = cursor.fetchall()
        for phong in phong_list:
            content = f"""
            THÔNG TIN PHÒNG:
            ID: {phong['id']}
            Tên phòng: {phong['ten']}
            Loại phòng: {phong['loai_phong']}
            Sức chứa: {phong['suc_chua']} ghế
            Vị trí: {phong['vi_tri']}
            Mô tả: {phong['mo_ta']}
            Trạng thái: {phong['trang_thai']}
            """
            data_chunks.append(content)
        
        # 3. Lịch chiếu
        cursor.execute("""
            SELECT lc.id, lc.bat_dau, lc.ket_thuc, lc.gia_ve, lc.trang_thai, 
                   p.ten as ten_phim, ph.ten as ten_phong, ph.loai_phong
            FROM lich_chieu lc
            JOIN phim p ON lc.phim_id = p.id
            JOIN phong ph ON lc.phong_id = ph.id
        """)
        lich_chieu_list = cursor.fetchall()
        for lich_chieu in lich_chieu_list:
            content = f"""
            THÔNG TIN LỊCH CHIẾU:
            ID: {lich_chieu['id']}
            Phim: {lich_chieu['ten_phim']}
            Phòng: {lich_chieu['ten_phong']} (Loại: {lich_chieu['loai_phong']})
            Thời gian bắt đầu: {lich_chieu['bat_dau']}
            Thời gian kết thúc: {lich_chieu['ket_thuc']}
            Giá vé: {lich_chieu['gia_ve']} VND
            Trạng thái: {lich_chieu['trang_thai']}
            """
            data_chunks.append(content)
        
        # 4. Thanh toán
        cursor.execute("SELECT * FROM thanh_toan")
        thanh_toan_list = cursor.fetchall()
        payment_info = "PHƯƠNG THỨC THANH TOÁN ĐƯỢC HỖ TRỢ:\n"
        for thanh_toan in thanh_toan_list:
            payment_info += f"- {thanh_toan['ten']}\n"
        data_chunks.append(payment_info)
        
        # 5. Tổng hợp thông tin chung
        summary = """
        THÔNG TIN RẠP PHIM:
        Tên rạp: Cinema Booking
        Địa chỉ: Hà Nội, Việt Nam
        Các loại phòng: 2D, 3D, IMAX
        Chính sách giá:
        - Phim 2D: 80,000 - 100,000 VND
        - Phim 3D: 90,000 - 120,000 VND
        - Phim IMAX: 120,000 - 150,000 VND
        - Ghế VIP: cộng thêm 20,000 VND
        Ưu đãi:
        - Khách hàng thân thiết được giảm 5% giá vé
        - Khách hàng VIP được giảm 10% giá vé
        - Ngày thứ 3 hàng tuần: giảm 20% giá vé
        Các tiện ích:
        - Đồ ăn, nước uống 
        - Bãi đỗ xe
        - Phòng chờ VIP
        """
        data_chunks.append(summary)
        
        # Đóng kết nối
        cursor.close()
        connection.close()
        
        # Tạo text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Chia nhỏ các đoạn văn bản
        texts = text_splitter.create_documents([chunk for chunk in data_chunks])
        
        # Tạo vector store
        vector_store = FAISS.from_documents(texts, embeddings)
        logger.info(f"Vector store initialized with {len(texts)} text chunks")
        
        return vector_store
        
    except Exception as e:
        logger.error(f"Error initializing vector store: {e}")
        return None


def query_database(query, params=None):
    """Truy vấn cơ sở dữ liệu"""
    if connection_pool is None:
        return None
    
    # Tạo key để cache
    cache_key = f"{query}_{str(params)}"
    
    # Kiểm tra cache
    if cache_key in query_cache:
        result, timestamp = query_cache[cache_key]
        # Cache có hiệu lực trong 5 phút
        if time.time() - timestamp < 300:
            return result
    
    try:
        # Tạo kết nối từ pool
        connection = connection_pool.get_connection()
        cursor = connection.cursor(dictionary=True)
        
        # Thực hiện truy vấn
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        
        # Lấy kết quả
        result = cursor.fetchall()
        
        # Lưu vào cache
        query_cache[cache_key] = (result, time.time())
        
        # Đóng kết nối
        cursor.close()
        connection.close()
        
        return result
    except Exception as e:
        logger.error(f"Database query error: {e}")
        return None


def get_relevant_context(query_text):
    """Lấy ngữ cảnh liên quan từ vector store dựa trên câu hỏi"""
    global vector_store
    
    # Khởi tạo vector store nếu chưa có
    if vector_store is None:
        vector_store = initialize_vector_store()
    
    if vector_store is None:
        return ""
    
    try:
        # Tìm kiếm các đoạn văn bản liên quan
        similar_docs = vector_store.similarity_search(query_text, k=3)
        
        # Tạo ngữ cảnh
        context = "\n\n".join([doc.page_content for doc in similar_docs])
        
        # Tìm kiếm thêm thông tin từ database dựa trên các từ khóa
        keywords = ["phim", "movie", "film", "lịch chiếu", "showtimes", "phòng", "room", "ghế", "seat", 
                   "vé", "ticket", "giá", "price", "thanh toán", "payment"]
        
        for keyword in keywords:
            if keyword.lower() in query_text.lower():
                # Thực hiện các truy vấn bổ sung dựa trên từ khóa
                if keyword in ["phim", "movie", "film"]:
                    # Tìm kiếm phim theo tên
                    movie_results = query_database(
                        "SELECT * FROM phim WHERE ten LIKE %s OR mo_ta LIKE %s LIMIT 3", 
                        (f"%{query_text}%", f"%{query_text}%")
                    )
                    if movie_results:
                        context += "\n\nKẾT QUẢ TÌM KIẾM PHIM:\n"
                        for movie in movie_results:
                            context += f"- {movie['ten']} ({movie['trang_thai']}): {movie['mo_ta'][:100]}...\n"
                
                elif keyword in ["lịch chiếu", "showtimes"]:
                    # Tìm kiếm lịch chiếu trong 7 ngày tới
                    schedule_results = query_database(
                        """
                        SELECT lc.bat_dau, p.ten as ten_phim, ph.ten as ten_phong, lc.gia_ve
                        FROM lich_chieu lc
                        JOIN phim p ON lc.phim_id = p.id
                        JOIN phong ph ON lc.phong_id = ph.id
                        WHERE lc.bat_dau > NOW() AND lc.bat_dau < DATE_ADD(NOW(), INTERVAL 7 DAY)
                        ORDER BY lc.bat_dau
                        LIMIT 5
                        """
                    )
                    if schedule_results:
                        context += "\n\nLỊCH CHIẾU SẮP TỚI:\n"
                        for schedule in schedule_results:
                            context += f"- {schedule['ten_phim']} - {schedule['bat_dau']} - Phòng {schedule['ten_phong']} - {schedule['gia_ve']} VND\n"
                
                elif keyword in ["giá", "price"]:
                    context += "\n\nBẢNG GIÁ VÉ XEM PHIM:\n"
                    context += "- Phim 2D: 80,000 - 100,000 VND\n"
                    context += "- Phim 3D: 90,000 - 120,000 VND\n"
                    context += "- Phim IMAX: 120,000 - 150,000 VND\n"
                    context += "- Ghế VIP: cộng thêm 20,000 VND\n"
                
                break
        
        return context
    
    except Exception as e:
        logger.error(f"Error getting relevant context: {e}")
        return ""


def enhance_prompt_with_context(user_message, system_message=""):
    """Nâng cao prompt với ngữ cảnh từ vector store và database"""
    
    # Lấy ngữ cảnh liên quan
    context = get_relevant_context(user_message)
    
    # Nếu có ngữ cảnh, thêm vào system message
    if context:
        enhanced_system = f"{system_message}\n\nDƯỚI ĐÂY LÀ THÔNG TIN THAM KHẢO VỀ RẠP PHIM:\n{context}\n\nHãy sử dụng thông tin trên để trả lời câu hỏi của người dùng một cách chính xác và đầy đủ nếu có liên quan. Nếu thông tin không có trong dữ liệu tham khảo, hãy trả lời dựa trên kiến thức của bạn."
        return enhanced_system
    
    return system_message


# Caching kết quả trả về
prompt_response_cache = {}

def get_cached_response(prompt, model):
    """Lấy kết quả từ cache nếu có"""
    cache_key = hashlib.md5(f"{prompt}_{model}".encode()).hexdigest()
    
    if cache_key in prompt_response_cache:
        response, timestamp = prompt_response_cache[cache_key]
        # Cache có hiệu lực trong 1 giờ
        if time.time() - timestamp < 3600:
            return response
    
    return None


def save_to_cache(prompt, model, response):
    """Lưu kết quả vào cache"""
    cache_key = hashlib.md5(f"{prompt}_{model}".encode()).hexdigest()
    prompt_response_cache[cache_key] = (response, time.time())


# Cập nhật cho hàm proxy_perplexity
@app.post("/chat/completions")
async def proxy_perplexity(request: Request):
    try:
        # Đọc nội dung từ request
        body = await request.json()
        
        # Log request (không log trong môi trường production)
        logger.info(f"Received request: {body}")
        
        # Lấy model
        model = body.get("model", "sonar-pro")
        
        # Lấy messages
        messages = body.get("messages", [])
        
        # Tìm system message và user message
        system_message = ""
        user_message = ""
        
        for msg in messages:
            if msg.get("role") == "system":
                system_message = msg.get("content", "")
            elif msg.get("role") == "user":
                user_message = msg.get("content", "")
        
        # Nếu có system message và user message, nâng cao system message với ngữ cảnh
        if system_message and user_message:
            enhanced_system = enhance_prompt_with_context(user_message, system_message)
            
            # Cập nhật system message trong messages
            for i, msg in enumerate(messages):
                if msg.get("role") == "system":
                    messages[i]["content"] = enhanced_system
                    break
            
            # Cập nhật messages trong body
            body["messages"] = messages
        
        # Kiểm tra cache
        cache_key = hashlib.md5(f"{str(messages)}_{model}".encode()).hexdigest()
        cached_response = get_cached_response(str(messages), model)
        
        if cached_response:
            logger.info("Returning cached response")
            return Response(
                content=json.dumps(cached_response),
                media_type="application/json",
                status_code=200
            )
        
        # Tạo headers với API key
        headers = {
            "Authorization": PERPLEXITY_API_KEY,
            "Content-Type": "application/json"
        }
        
        # Sử dụng backup response trong trường hợp lỗi
        backup_response = {
            "id": f"backup-{int(time.time())}",
            "model": model,
            "created": int(time.time()),
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Xin chào! Tôi là trợ lý AI của Bài tập lớn Android Cô Vân Anh Cinema. Tôi có thể giúp bạn với thông tin về lịch chiếu, phim đang chiếu, giá vé và các chương trình khuyến mãi. Rất vui được phục vụ bạn!"
                },
                "finish_reason": "stop"
            }]
        }
        
        # Gửi request đến Perplexity API với timeout
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                logger.info(f"Sending request to Perplexity API: {PERPLEXITY_API_URL}/chat/completions")
                perplexity_response = await client.post(
                    f"{PERPLEXITY_API_URL}/chat/completions",
                    json=body,
                    headers=headers
                )
                
                # Kiểm tra status code
                if perplexity_response.status_code != 200:
                    logger.error(f"Perplexity API returned status code: {perplexity_response.status_code}")
                    logger.error(f"Response: {perplexity_response.text}")
                    # Trả về response với backup trong trường hợp lỗi
                    return Response(
                        content=json.dumps(backup_response),
                        media_type="application/json",
                        status_code=200
                    )
                
                # Lấy response data
                try:
                    response_data = perplexity_response.json()
                    # Log response (không log trong môi trường production)
                    logger.info(f"Perplexity response received successfully")
                    
                    # Validate response structure
                    if "choices" not in response_data or not response_data["choices"]:
                        logger.error(f"Invalid response structure: {response_data}")
                        return Response(
                            content=json.dumps(backup_response),
                            media_type="application/json",
                            status_code=200
                        )
                    
                    # Lưu vào cache
                    save_to_cache(str(messages), model, response_data)
                    
                    # Trả về response với cùng status code
                    return Response(
                        content=json.dumps(response_data),
                        media_type="application/json",
                        status_code=200
                    )
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding Perplexity response: {e}")
                    logger.error(f"Raw response: {perplexity_response.text[:500]}")
                    return Response(
                        content=json.dumps(backup_response),
                        media_type="application/json",
                        status_code=200
                    )
                
        except httpx.RequestError as e:
            logger.error(f"Request error to Perplexity API: {e}")
            return Response(
                content=json.dumps(backup_response),
                media_type="application/json",
                status_code=200
            )
            
    except Exception as e:
        logger.error(f"Error in proxy: {str(e)}", exc_info=True)  # Thêm exc_info=True để log stack trace
        
        # Trả về response dự phòng để frontend không bị lỗi
        backup_response = {
            "id": f"error-{int(time.time())}",
            "model": "error-fallback",
            "created": int(time.time()),
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Xin lỗi, đã có lỗi xảy ra khi xử lý yêu cầu của bạn. Vui lòng thử lại sau hoặc liên hệ với rạp chiếu phim để được hỗ trợ."
                },
                "finish_reason": "stop"
            }]
        }
        
        return Response(
            content=json.dumps(backup_response),
            media_type="application/json",
            status_code=200
        )

# Endpoint kiểm tra trạng thái
@app.get("/status")
async def status():
    db_status = "connected" if connection_pool else "disconnected"
    vector_status = "initialized" if vector_store else "not initialized"
    
    return {
        "status": "running",
        "database": db_status,
        "vector_store": vector_status,
        "cache_items": len(prompt_response_cache)
    }


# Khởi tạo vector store khi khởi động ứng dụng
@app.on_event("startup")
async def startup_event():
    global vector_store
    vector_store = initialize_vector_store()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)