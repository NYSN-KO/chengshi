# main.py
import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import json

# 导入您的两个推理脚本
from scripts.cnn_dnn_inference import run_cnn_dnn_inference
from scripts.oct_inference import run_oct_inference

# --- 配置 ---
UPLOAD_DIR = Path("uploaded_images")
TEMP_RESULT_DIR = Path("temp_results")
UPLOAD_DIR.mkdir(exist_ok=True)
TEMP_RESULT_DIR.mkdir(exist_ok=True)
MODEL_DIR = Path("models")

app = FastAPI(title="Fly.io 图像处理服务")

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """根路径返回前端页面。"""
    html_path = Path("static") / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=500, detail="前端 HTML 文件未找到。")
    return html_path.read_text()

@app.post("/upload-and-process/")
async def upload_and_process(file: UploadFile = File(...)):
    """
    处理图片上传，并将其输入到两个模型推理脚本中。
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="文件类型错误，请上传图片。")

    # 1. 保存上传的图片
    file_location = UPLOAD_DIR / file.filename
    try:
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件保存失败: {e}")

    # 清理本次运行的临时目录
    current_run_dir = TEMP_RESULT_DIR / file.filename.split('.')[0]
    if current_run_dir.exists():
        shutil.rmtree(current_run_dir)
    current_run_dir.mkdir(parents=True, exist_ok=True)

    oct_output = {}
    cnn_dnn_output = {}

    try:
        # --- 2. 调用第一个模型 (OCT 分割脚本) ---
        oct_model_path = MODEL_DIR / "best_model.pth"
        if not oct_model_path.exists():
             oct_result_summary = f"错误：模型文件 {oct_model_path.name} 不存在。"
       else:
             oct_result_summary, id_path, ov_path = run_oct_inference(
                image_path=file_location,
                model_path=oct_model_path,
                temp_dir=current_run_dir 
             )
             oct_output["summary"] = oct_result_summary
             # 仅返回文件名，实际部署时可能需要额外的服务来提供这些文件
             oct_output["id_mask_file"] = id_path.name 
             oct_output["overlay_file"] = ov_path.name

        # --- 3. 调用第二个模型 (CNN + DNN 融合预测模型) ---
        cnn_dnn_model_path = MODEL_DIR / "cnn_dnn_best_703.pt"
        if not cnn_dnn_model_path.exists():
            cnn_dnn_output["summary"] = f"错误：模型文件 {cnn_dnn_model_path.name} 不存在。"
        else:
            cnn_dnn_result_str = run_cnn_dnn_inference(
                image_path=file_location,
                model_path=cnn_dnn_model_path
            )
            cnn_dnn_output["summary"] = cnn_dnn_result_str

        return {
            "filename": file.filename,
            "status": "success",
            "model_1_oct_output": oct_output,
            "model_2_cnn_dnn_output": cnn_dnn_output
        }
        
    except Exception as e:
        print(f"推理失败: {e}")
        raise HTTPException(status_code=500, detail=f"模型推理失败: {type(e).__name__}: {e}")
        
    finally:
        # 清理上传的图片
        os.remove(file_location)
        # 保持临时结果文件，方便调试。

if __name__ == "__main__":
    import uvicorn
