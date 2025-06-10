# 🎬 Cinema Booking API Proxy

Hệ thống proxy API cho rạp phim với tích hợp AI chatbot thông minh, hỗ trợ tìm kiếm và đặt vé phim trực tuyến.

## 🌟 Tính năng chính

- **API Proxy**: Tích hợp với Perplexity AI API
- **Local AI Model**: Hỗ trợ model GPT-2 fine-tuned cho domain phim ảnh
- **RAG System**: Retrieval-Augmented Generation với FAISS vector store
- **Database Integration**: Kết nối MySQL với connection pooling
- **Multi-language Support**: Hỗ trợ tiếng Việt và tiếng Anh
- **Caching System**: Cache thông minh cho queries và embeddings
- **RESTful API**: FastAPI với documentation tự động

## 🛠️ Công nghệ sử dụng

- **Backend**: FastAPI, Python 3.8+
- **AI/ML**: 
  - Transformers (GPT-2)
  - LangChain
  - FAISS
  - HuggingFace Embeddings
- **Database**: MySQL với mysql-connector-python
- **HTTP Client**: httpx
- **Vector Search**: sentence-transformers

## 📋 Yêu cầu hệ thống

- Python 3.8 hoặc cao hơn
- MySQL 5.7+ hoặc MariaDB 10.2+
- RAM: Tối thiểu 4GB (khuyến nghị 8GB+ cho model GPT-2)
- GPU: Tùy chọn (để tăng tốc inference)

## 🚀 Cài đặt

### 1. Clone repository

```bash
git clone <repository-url>
cd python_android
```

### 2. Tạo virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate     # Windows
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 4. Cấu hình database

```sql
-- Tạo database
CREATE DATABASE rap_phim_online CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- Tạo user (tùy chọn)
CREATE USER 'cinema_user'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON rap_phim_online.* TO 'cinema_user'@'localhost';
FLUSH PRIVILEGES;
```

### 5. Cấu hình môi trường

Tạo file `.env` trong thư mục gốc:

```env
# Database Configuration
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=dat123456
DB_NAME=rap_phim_online
DB_PORT=3306

# API Keys
PERPLEXITY_API_KEY=your_perplexity_api_key_here

# Model Configuration
MODEL_PATH=./model/
EMBEDDINGS_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=False
```

### 6. Khởi tạo database schema

```bash
# Import schema SQL file (nếu có)
mysql -u root -p rap_phim_online < schema.sql
```

## 🎯 Cấu trúc dự án

```
python_android/
├── main.py                          # API proxy chính với Perplexity
├── main_using_llm_fine_turn.py     # API với local GPT-2 model
├── model/
│   └── best.pt                      # Pre-trained model weights
├── requirements.txt                 # Python dependencies
├── .env                            # Environment variables
├── .gitignore                      # Git ignore rules
├── README.md                       # Documentation
├── perplexity_proxy.log            # Log file
└── workspace.code-workspace        # VS Code workspace
```

## 🔧 Sử dụng

### Khởi động server

```bash
# Sử dụng Perplexity API proxy
python main.py

# Hoặc sử dụng local model
python main_using_llm_fine_turn.py

# Với uvicorn (khuyến nghị cho production)
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### API Endpoints

#### 1. Chat Completions (Perplexity Proxy)

```bash
POST /chat/completions
Content-Type: application/json

{
  "messages": [
    {"role": "system", "content": "Bạn là trợ lý AI cho rạp phim"},
    {"role": "user", "content": "Có phim gì hay chiếu hôm nay?"}
  ],
  "model": "llama-3.1-sonar-small-128k-online",
  "temperature": 0.7,
  "max_tokens": 1000
}
```

#### 2. Local Model Completions

```bash
POST /local/chat/completions
Content-Type: application/json

{
  "messages": [
    {"role": "user", "content": "Tìm phim hành động hay nhất"}
  ],
  "temperature": 0.7,
  "max_tokens": 1000
}
```

#### 3. Health Check

```bash
GET /status
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "database": "connected",
  "vector_store": "initialized",
  "models": {
    "perplexity": "available",
    "local_gpt2": "loaded"
  }
}
```

## 💾 Database Schema

Hệ thống sử dụng các bảng chính:

- **phim**: Thông tin phim (tên, thể loại, đạo diễn, diễn viên, etc.)
- **phong**: Thông tin phòng chiếu (tên, loại, sức chứa, etc.)
- **lich_chieu**: Lịch chiếu phim (thời gian, giá vé, etc.)
- **thanh_toan**: Phương thức thanh toán
- **khach_hang**: Thông tin khách hàng
- **dat_ve**: Thông tin đặt vé

## 🤖 AI Features

### RAG (Retrieval-Augmented Generation)

Hệ thống sử dụng RAG để cung cấp thông tin chính xác về:
- Thông tin phim đang chiếu
- Lịch chiếu và giá vé
- Thông tin phòng chiếu
- Chính sách và ưu đãi

### Vector Search

- **Embeddings Model**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Vector Store**: FAISS
- **Chunk Size**: 1000 characters
- **Overlap**: 200 characters

## 📊 Monitoring & Logging

Logs được ghi vào:
- Console output
- File: `perplexity_proxy.log`

Log levels:
- INFO: Thông tin hoạt động bình thường
- ERROR: Lỗi hệ thống
- DEBUG: Chi tiết debug (chỉ khi DEBUG=True)

## 🔒 Bảo mật

- API key được bảo vệ qua environment variables
- Database credentials không được commit
- CORS được cấu hình cho cross-origin requests
- Input validation và sanitization

## 🚀 Deployment

### Development

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Production

```bash
# Với Gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8000

# Với Docker (tùy chọn)
docker build -t cinema-api .
docker run -p 8000:8000 cinema-api
```

## 🧪 Testing

```bash
# Test API endpoints
curl -X POST "http://localhost:8000/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Xin chào"}],
    "model": "llama-3.1-sonar-small-128k-online"
  }'

# Test health check
curl -X GET "http://localhost:8000/status"
```

## 🤝 Đóng góp

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Tạo Pull Request

## 📝 TODO

- [ ] Thêm authentication và authorization
- [ ] Implement rate limiting
- [ ] Thêm unit tests
- [ ] Docker containerization
- [ ] CI/CD pipeline
- [ ] API documentation với Swagger UI
- [ ] Monitoring với Prometheus/Grafana
- [ ] Caching với Redis

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 📞 Liên hệ

- **Developer**: quydat09
- **Email**: [your-email@example.com]
- **Project Link**: [https://github.com/quydat09/python_android]

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [Perplexity AI](https://perplexity.ai/) - AI API service
- [HuggingFace](https://huggingface.co/) - Transformers và embeddings
- [LangChain](https://langchain.com/) - RAG framework
- [FAISS](https://faiss.ai/) - Vector similarity search
