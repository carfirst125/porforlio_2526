import os
from dotenv import load_dotenv

# Load .env từ thư mục gốc của app
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path)

import logging

logging.basicConfig(
    level=logging.INFO,  # hoặc DEBUG nếu bạn cần chi tiết hơn
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]  # In ra terminal
)
logger = logging.getLogger(__name__)