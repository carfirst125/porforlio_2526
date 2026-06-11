"""
GPU & Ollama Diagnostic Script
Chạy từ thư mục version_2/:  python scripts/check_gpu.py
"""
import json
import subprocess
import sys
import urllib.request
import urllib.error

OLLAMA_BASE = "http://localhost:11434"
SEP = "─" * 60


def section(title):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print('═'*60)


def ok(msg):   print(f"  ✅ {msg}")
def warn(msg): print(f"  ⚠️  {msg}")
def err(msg):  print(f"  ❌ {msg}")
def info(msg): print(f"  ℹ️  {msg}")


# ── 1. NVIDIA GPU (nvidia-smi) ───────────────────────────────────────────────
section("1. NVIDIA GPU — nvidia-smi")
try:
    result = subprocess.run(
        ["nvidia-smi",
         "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10
    )
    if result.returncode == 0:
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                name, mem_total, mem_used, mem_free, util, temp = parts[:6]
                ok(f"GPU: {name}")
                info(f"VRAM total : {mem_total} MB")
                info(f"VRAM used  : {mem_used} MB")
                info(f"VRAM free  : {mem_free} MB")
                info(f"GPU util   : {util}%")
                info(f"Temperature: {temp}°C")

                # Check if VRAM sufficient for deepseek-r1:8b (~5-6GB)
                total = int(mem_total)
                free  = int(mem_free)
                if free >= 5000:
                    ok(f"VRAM free ({free} MB) đủ cho deepseek-r1:8b (~5-6 GB)")
                elif free >= 3000:
                    warn(f"VRAM free ({free} MB) có thể không đủ cho deepseek-r1:8b — một số layers sẽ chạy trên CPU")
                else:
                    err(f"VRAM free ({free} MB) quá ít — model sẽ chủ yếu chạy trên CPU (chậm)")
    else:
        err(f"nvidia-smi error: {result.stderr.strip()}")
except FileNotFoundError:
    warn("nvidia-smi không tìm thấy — không có NVIDIA GPU hoặc driver chưa cài")
except Exception as e:
    err(f"Lỗi: {e}")


# ── 2. Ollama process: model đang chạy ──────────────────────────────────────
section("2. Ollama — Models đang chạy (/api/ps)")
try:
    req = urllib.request.Request(f"{OLLAMA_BASE}/api/ps")
    with urllib.request.urlopen(req, timeout=5) as resp:
        data = json.loads(resp.read())
    models = data.get("models", [])
    if not models:
        warn("Không có model nào đang load trong Ollama (chưa có request nào gần đây)")
    else:
        for m in models:
            name   = m.get("name", "?")
            size   = m.get("size", 0) / 1e9
            size_v = m.get("size_vram", 0) / 1e9
            detail = m.get("details", {})
            info(f"Model : {name}")
            info(f"Size  : {size:.2f} GB")
            if size_v > 0:
                ok(f"VRAM  : {size_v:.2f} GB  ← model đang dùng GPU VRAM!")
                cpu_portion = size - size_v
                if cpu_portion > 0.1:
                    warn(f"CPU RAM: {cpu_portion:.2f} GB (một phần chạy trên CPU)")
                else:
                    ok("Toàn bộ model nằm trên GPU!")
            else:
                err("VRAM = 0 → model đang chạy hoàn toàn trên CPU (nguyên nhân chậm!)")
            info(f"Params: {detail.get('parameter_size','?')}, quant: {detail.get('quantization_level','?')}")
except urllib.error.URLError:
    err("Không kết nối được Ollama tại http://localhost:11434 — hãy chạy 'ollama serve' trước")
except Exception as e:
    err(f"Lỗi: {e}")


# ── 3. Ollama tags — model đã pull ──────────────────────────────────────────
section("3. Ollama — Models đã pull (/api/tags)")
try:
    req = urllib.request.Request(f"{OLLAMA_BASE}/api/tags")
    with urllib.request.urlopen(req, timeout=5) as resp:
        data = json.loads(resp.read())
    for m in data.get("models", []):
        name = m.get("name", "?")
        size = m.get("size", 0) / 1e9
        info(f"{name:40s} {size:.1f} GB")
except Exception as e:
    err(f"Lỗi: {e}")


# ── 4. .env settings ─────────────────────────────────────────────────────────
section("4. Settings (.env)")
try:
    with open(".env", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and any(
                k in line for k in ["OLLAMA_NUM_GPU", "LLM_MODEL", "OLLAMA_BASE", "EMBEDDING_MODEL"]
            ):
                key, _, val = line.partition("=")
                if "OLLAMA_NUM_GPU" in key:
                    v = val.strip()
                    if v == "-1":
                        ok(f"{key}={v}  (−1 = offload ALL layers to GPU ✓)")
                    elif v == "0":
                        err(f"{key}={v}  (0 = CPU only! Đây là lý do chạy chậm!)")
                    else:
                        warn(f"{key}={v}  (chỉ offload {v} layers lên GPU — có thể tăng lên -1)")
                else:
                    info(f"{key}={val.strip()}")
except FileNotFoundError:
    warn(".env không tìm thấy — dùng .env.example làm template")
except Exception as e:
    err(f"Lỗi đọc .env: {e}")


# ── 5. Quick embed test (kiểm tra Ollama response time) ─────────────────────
section("5. Ollama Latency Test")
import time
try:
    payload = json.dumps({
        "model": "bge-m3:latest",
        "prompt": "thẻ tín dụng VIB",
        "stream": False
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read())
    elapsed = time.time() - t0
    dim = len(result.get("embedding", []))
    if elapsed < 1.0:
        ok(f"bge-m3 embed: {elapsed:.2f}s  dim={dim} (NHANH - GPU đang hoạt động)")
    elif elapsed < 5.0:
        warn(f"bge-m3 embed: {elapsed:.2f}s  dim={dim} (trung bình - có thể CPU)")
    else:
        err(f"bge-m3 embed: {elapsed:.2f}s  dim={dim} (CHẬM - chạy trên CPU)")
except Exception as e:
    warn(f"Embed test failed: {e}")


# ── 6. Tóm tắt & khuyến nghị ────────────────────────────────────────────────
section("6. Khuyến nghị")
print("""
  Nếu GPU không được dùng (size_vram=0 hoặc OLLAMA_NUM_GPU=0):

  A) Kiểm tra CUDA driver:
     nvidia-smi               # phải hiện GPU
     nvcc --version           # CUDA toolkit

  B) Kiểm tra Ollama build có CUDA không:
     ollama --version         # phải là phiên bản hỗ trợ CUDA
     # Tải lại từ https://ollama.com nếu cần

  C) Đảm bảo .env có:
     OLLAMA_NUM_GPU=-1        # -1 = tất cả layers lên GPU

  D) Nếu VRAM không đủ cho deepseek-r1:8b (~5-6GB):
     → Dùng model nhỏ hơn: LLM_MODEL=llama3.2:3b  (~2GB VRAM)
     → Hoặc: LLM_MODEL=deepseek-r1:1.5b            (~1GB VRAM)
     → Hoặc tăng VRAM bằng cách đóng ứng dụng khác

  E) Kiểm tra nhanh GPU đang dùng trong khi chat:
     Mở Task Manager → Performance → GPU
     (VRAM usage tăng khi chatbot đang xử lý = GPU đang dùng)
""")
print(SEP)
