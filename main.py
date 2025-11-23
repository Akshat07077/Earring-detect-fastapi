from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
from pydantic import BaseModel
import os
import cv2
import uuid
import pandas as pd
from pathlib import Path
import shutil
from typing import List, Optional, Dict, Any

app = FastAPI(title="Earring Detection API", version="1.0.0")

# Pydantic models for API responses
class Prediction(BaseModel):
    id: str
    class_name: str
    confidence: float
    image_url: str

class PipeGroup(BaseModel):
    pipe_id: str
    item_count: int
    image_url: str

class FrontResult(BaseModel):
    original_filename: str
    result_image_url: str
    predictions: List[Prediction]

class TopResult(BaseModel):
    original_filename: str
    result_image_url: str
    pipe_groups: List[PipeGroup]
    total_count: int

class MappedResult(BaseModel):
    mapped_id: str
    detected_class: str
    count: int
    stand_name: str

class ProcessResponse(BaseModel):
    success: bool
    front: FrontResult
    top: TopResult
    mapped: List[MappedResult]
    excel_url: str
    message: Optional[str] = None

class ErrorResponse(BaseModel):
    success: bool
    error: str

# Configuration
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
CROPPED_FOLDER = 'static/cropped'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

# Create folders if they don't exist
for folder in [UPLOAD_FOLDER, RESULT_FOLDER, CROPPED_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Roboflow API clients
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="0lfsVUdCmiTvLOVlDejI"
)


def secure_filename(filename: str) -> str:
    """Secure filename similar to werkzeug's secure_filename"""
    import re
    filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
    return filename


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def draw_label(img, text, x, y, color=(0, 255, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    cv2.rectangle(img, (x, y - 20), (x + text_size[0], y), color, -1)
    cv2.putText(img, text, (x, y - 5), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)


def group_predictions_by_columns(predictions, x_threshold=50, fixed_width=100, fixed_height=200):
    predictions = sorted(predictions, key=lambda p: p['x'])
    columns = []
    for pred in predictions:
        added_to_column = False
        for col in columns:
            avg_x = sum(p['x'] for p in col) / len(col)
            if abs(pred['x'] - avg_x) <= x_threshold:
                col.append(pred)
                added_to_column = True
                break
        if not added_to_column:
            columns.append([pred])

    for col in columns:
        col.sort(key=lambda p: p['y'])

    merged_columns = []
    for col in columns:
        current_group = []
        groups = []
        col.sort(key=lambda p: p['y'])
        for pred in col:
            if not current_group:
                current_group.append(pred)
            else:
                first_y = current_group[0]['y']
                if abs(pred['y'] - first_y) <= fixed_height:
                    current_group.append(pred)
                else:
                    groups.append(current_group)
                    current_group = [pred]
        if current_group:
            groups.append(current_group)
        merged_columns.extend(groups)

    merged_columns.sort(key=lambda col: sum(p['y'] for p in col) / len(col))
    return merged_columns, fixed_width, fixed_height


def group_detections_by_rows(predictions, y_threshold=100):
    """Group detections into rows by Y proximity and sort each row right to left."""
    rows = []
    predictions.sort(key=lambda p: p['y'])  # Top to bottom

    for pred in predictions:
        added = False
        for row in rows:
            avg_y = sum(p['y'] for p in row) / len(row)
            if abs(pred['y'] - avg_y) <= y_threshold:
                row.append(pred)
                added = True
                break
        if not added:
            rows.append([pred])

    # Sort each row right to left (x descending)
    for row in rows:
        row.sort(key=lambda p: p['x'], reverse=True)

    # Flatten the list row by row
    return [pred for row in rows for pred in row]


async def process_front_image(file: UploadFile, confidence=0.40):
    original_filename = secure_filename(file.filename)
    save_name = str(uuid.uuid4()) + "_" + original_filename
    save_path = os.path.join(UPLOAD_FOLDER, save_name)
    
    # Save uploaded file (reset file pointer to beginning)
    await file.seek(0)
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    custom_configuration = InferenceConfiguration(confidence_threshold=confidence, iou_threshold=0.4)

    # Detection with sku-110k model
    print("Sending image to sku-110K...")
    with CLIENT.use_configuration(custom_configuration):
        detection_result = CLIENT.infer(save_path, model_id="earring-crop-ugfno/1")

    img = cv2.imread(save_path)
    height, width = img.shape[:2]

    predictions = []
    id_counter = 1
    min_conf_threshold = 0.40
    print("detection_result -> ", detection_result, "\n")
    if detection_result.get('predictions'):
        sorted_detections = group_detections_by_rows(detection_result['predictions'], y_threshold=100)
        for pred in sorted_detections:
            x, y = int(pred['x']), int(pred['y'])
            w, h = int(pred['width']), int(pred['height'])
            start_x, start_y = max(x - w // 2, 0), max(y - h // 2, 0)
            end_x, end_y = min(x + w // 2, width), min(y + h // 2, height)

            # Draw bounding box
            cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
            draw_label(img, f"Earring-ID-{id_counter}", start_x, start_y)

            # Crop and classify
            crop = img[start_y:end_y, start_x:end_x]
            crop_filename = f"crop_{original_filename}_{id_counter}.jpg"
            crop_path = os.path.join(CROPPED_FOLDER, crop_filename)
            cv2.imwrite(crop_path, crop)

            classification_result = CLIENT.infer(crop_path, model_id="main-model-xscmn/2")

            if classification_result.get('predictions'):
                best_pred = max(classification_result['predictions'], key=lambda x: x['confidence'])
                if best_pred['confidence'] >= min_conf_threshold:
                    predictions.append({
                        'id': f'Earring-ID-{id_counter}',
                        'class': best_pred['class'],
                        'confidence': best_pred['confidence'],
                        'image': f'/static/cropped/{crop_filename}'
                    })
                else:
                    predictions.append({
                        'id': f'Earring-ID-{id_counter}',
                        'class': 'Unknown',
                        'confidence': best_pred['confidence'],
                        'image': f'/static/cropped/{crop_filename}'
                    })
            else:
                predictions.append({
                    'id': f'Earring-ID-{id_counter}',
                    'class': 'Unknown',
                    'confidence': 0.0,
                    'image': f'/static/cropped/{crop_filename}'
                })

            id_counter += 1

    result_img_name = "result_" + save_name
    result_img_path = os.path.join(RESULT_FOLDER, result_img_name)
    cv2.imwrite(result_img_path, img)

    print("Front completed......")
    return {
        'original_filename': original_filename,
        'result_image': f'/static/results/{result_img_name}',
        'predictions': predictions
    }


async def process_top_image(file: UploadFile, confidence=0.50, x_threshold=50):
    original_filename = secure_filename(file.filename)
    save_name = str(uuid.uuid4()) + "_" + original_filename
    save_path = os.path.join(UPLOAD_FOLDER, save_name)
    
    # Save uploaded file (reset file pointer to beginning)
    await file.seek(0)
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    custom_configuration = InferenceConfiguration(confidence_threshold=confidence, iou_threshold=0.5)

    # Detection with sku-110k model
    print("Sending image to earring-crop-ugfno/1..")
    with CLIENT.use_configuration(custom_configuration):
        crop_result = CLIENT.infer(save_path, model_id="earring-crop-ugfno/1")

    img = cv2.imread(save_path)
    height, width = img.shape[:2]

    id_counter = 1
    min_conf_threshold = 0.50
    pipe_count = []
    if crop_result.get('predictions'):
        sorted_detections = group_detections_by_rows(crop_result['predictions'], y_threshold=100)
        for pred in sorted_detections:
            x, y = int(pred['x']), int(pred['y'])
            w, h = int(pred['width']), int(pred['height'])
            start_x, start_y = max(x - w // 2, 0), max(y - h // 2, 0)
            end_x, end_y = min(x + w // 2, width), min(y + h // 2, height)

            draw_label(img, f"Pipe-ID-{id_counter}", start_x, start_y)

            # save cropped images
            crop = img[start_y:end_y, start_x:end_x]
            crop_filename = f"crop_{original_filename}_{id_counter}.jpg"
            crop_path = os.path.join(CROPPED_FOLDER, crop_filename)
            cv2.imwrite(crop_path, crop)

            with CLIENT.use_configuration(custom_configuration):
                crop_detection_result = CLIENT.infer(crop_path, model_id="earring-box-count-0cf1h/1")

            if crop_detection_result.get('predictions'):
                for i, pred in enumerate(crop_detection_result['predictions']):
                    x, y = int(pred['x']), int(pred['y'])
                    w, h = int(pred['width']), int(pred['height'])
                    start_x, start_y = max(x - w // 2, 0), max(y - h // 2, 0)
                    end_x, end_y = min(x + w // 2, width), min(y + h // 2, height)

                    cv2.rectangle(crop, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
                    draw_label(crop, f"item", start_x, start_y, (0, 255, 0))

            pipe_count.append({
                'pipe_id': f'Pipe-{id_counter}',
                'item_count': len(crop_detection_result['predictions']),
                'image': f'/static/cropped/{crop_filename}'
            })

            id_counter += 1

    result_img_name = "result_" + save_name
    result_img_path = os.path.join(RESULT_FOLDER, result_img_name)
    cv2.imwrite(result_img_path, img)

    return {
        'original_filename': original_filename,
        'result_image': f'/static/results/{result_img_name}',
        'pipe_groups': pipe_count,
        'total_count': len(pipe_count)
    }


def map_ids(file_name: str, front_predictions: List[dict], top_pipes: List[dict]):
    print("front predictions : ", front_predictions)
    print("top pipes : ", top_pipes)
    print("file name : ", file_name)

    mapped_results = []
    for i, front_pred in enumerate(front_predictions):
        if i < len(top_pipes):
            mapped_results.append({
                'mapped_id': front_pred['id'],  # Earring-ID-X
                'detected_class': front_pred['class'],
                'count': top_pipes[i]['item_count'],
                'stand_name': file_name
            })
        else:
            mapped_results.append({
                'mapped_id': front_pred['id'],
                'detected_class': front_pred['class'],
                'count': 0,
                'stand_name': file_name
            })
    return mapped_results


def generate_excel(mapped_results: List[dict]):
    df = pd.DataFrame(mapped_results)
    excel_path = os.path.join(RESULT_FOLDER, 'earring_results.xlsx')
    df.to_excel(excel_path, index=False, engine='openpyxl')
    return excel_path


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main page with upload form"""
    return templates.TemplateResponse("index2.html", {
        "request": request,
        "error": None,
        "results": []
    })


@app.post("/", response_class=HTMLResponse)
async def process_images(
    request: Request,
    images: List[UploadFile] = File(...),
    confidence: Optional[float] = Form(0.50)
):
    """Process uploaded images"""
    error = None
    results = []

    if not images or len(images) != 2:
        error = "⚠️ Please upload exactly two images: one with 'front' and one with 'top' in the filename."
        return templates.TemplateResponse("index2.html", {
            "request": request,
            "error": error,
            "results": []
        })

    # Validate files
    front_file = None
    top_file = None
    for file in images:
        if not file.filename or not allowed_file(file.filename):
            error = f"⚠️ Invalid file: {file.filename}"
            return templates.TemplateResponse("index2.html", {
                "request": request,
                "error": error,
                "results": []
            })
        filename_lower = file.filename.lower()
        if 'front' in filename_lower:
            front_file = file
        elif 'top' in filename_lower:
            top_file = file

    if not front_file or not top_file:
        error = "⚠️ Please ensure one image has 'front' and the other has 'top' in the filename."
        return templates.TemplateResponse("index2.html", {
            "request": request,
            "error": error,
            "results": []
        })

    try:
        # Process files (they will be saved internally in the processing functions)
        front_result = await process_front_image(front_file, confidence=0.40)
        top_result = await process_top_image(top_file, confidence=confidence)
        
        mapped_results = map_ids(front_file.filename, front_result['predictions'], top_result['pipe_groups'])
        generate_excel(mapped_results)

        results.append({
            'front': front_result,
            'top': top_result,
            'mapped': mapped_results
        })
    except Exception as e:
        error = f"⚠️ Error processing images: {str(e)}"
        return templates.TemplateResponse("index2.html", {
            "request": request,
            "error": error,
            "results": []
        })

    return templates.TemplateResponse("index2.html", {
        "request": request,
        "error": error,
        "results": results
    })


@app.get("/download_excel")
async def download_excel():
    """Download the generated Excel file"""
    excel_path = os.path.join(RESULT_FOLDER, 'earring_results.xlsx')
    if os.path.exists(excel_path):
        return FileResponse(
            excel_path,
            filename="earring_results.xlsx",
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    raise HTTPException(status_code=404, detail="Excel file not found")


# ========== JSON API ENDPOINTS FOR FRONTEND ==========

@app.post("/api/process", response_model=ProcessResponse)
async def api_process_images(
    images: List[UploadFile] = File(...),
    confidence: Optional[float] = Form(0.50)
):
    """
    Process front and top images and return JSON response.
    
    - **images**: List of 2 image files (one with 'front' in filename, one with 'top')
    - **confidence**: Confidence threshold (default: 0.50)
    
    Returns JSON with detection results, classifications, and mapped data.
    """
    if not images or len(images) != 2:
        raise HTTPException(
            status_code=400,
            detail="Please upload exactly two images: one with 'front' and one with 'top' in the filename."
        )

    # Validate and identify files
    front_file = None
    top_file = None
    for file in images:
        if not file.filename or not allowed_file(file.filename):
            raise HTTPException(status_code=400, detail=f"Invalid file: {file.filename}")
        filename_lower = file.filename.lower()
        if 'front' in filename_lower:
            front_file = file
        elif 'top' in filename_lower:
            top_file = file

    if not front_file or not top_file:
        raise HTTPException(
            status_code=400,
            detail="Please ensure one image has 'front' and the other has 'top' in the filename."
        )

    try:
        # Process images
        front_result = await process_front_image(front_file, confidence=0.40)
        top_result = await process_top_image(top_file, confidence=confidence)
        
        # Map IDs
        mapped_results = map_ids(front_file.filename, front_result['predictions'], top_result['pipe_groups'])
        
        # Generate Excel
        excel_path = generate_excel(mapped_results)
        excel_url = "/download_excel"

        # Convert to Pydantic models
        front_predictions = [
            Prediction(
                id=pred['id'],
                class_name=pred['class'],
                confidence=pred['confidence'],
                image_url=pred['image']
            )
            for pred in front_result['predictions']
        ]

        top_pipes = [
            PipeGroup(
                pipe_id=pipe['pipe_id'],
                item_count=pipe['item_count'],
                image_url=pipe['image']
            )
            for pipe in top_result['pipe_groups']
        ]

        mapped = [
            MappedResult(
                mapped_id=m['mapped_id'],
                detected_class=m['detected_class'],
                count=m['count'],
                stand_name=m['stand_name']
            )
            for m in mapped_results
        ]

        response = ProcessResponse(
            success=True,
            front=FrontResult(
                original_filename=front_result['original_filename'],
                result_image_url=front_result['result_image'],
                predictions=front_predictions
            ),
            top=TopResult(
                original_filename=top_result['original_filename'],
                result_image_url=top_result['result_image'],
                pipe_groups=top_pipes,
                total_count=top_result['total_count']
            ),
            mapped=mapped,
            excel_url=excel_url
        )

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing images: {str(e)}")


@app.post("/api/classify")
async def api_classify_image(
    image: UploadFile = File(...),
    confidence: Optional[float] = Form(0.40)
):
    """
    Classify a single earring image.
    
    - **image**: Single image file to classify
    - **confidence**: Confidence threshold (default: 0.40)
    
    Returns classification results.
    """
    if not image.filename or not allowed_file(image.filename):
        raise HTTPException(status_code=400, detail="Invalid file format")

    try:
        # Save uploaded file
        original_filename = secure_filename(image.filename)
        save_name = str(uuid.uuid4()) + "_" + original_filename
        save_path = os.path.join(UPLOAD_FOLDER, save_name)
        
        await image.seek(0)
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # Classify using the classification model
        classification_result = CLIENT.infer(save_path, model_id="earring-id-new/3")

        if classification_result.get('predictions'):
            best_pred = max(classification_result['predictions'], key=lambda x: x['confidence'])
            return {
                "success": True,
                "original_filename": original_filename,
                "class": best_pred['class'],
                "confidence": best_pred['confidence'],
                "all_predictions": [
                    {
                        "class": p['class'],
                        "confidence": p['confidence']
                    }
                    for p in classification_result['predictions']
                ]
            }
        else:
            return {
                "success": True,
                "original_filename": original_filename,
                "class": "Unknown",
                "confidence": 0.0,
                "all_predictions": []
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error classifying image: {str(e)}")


@app.post("/api/process/front")
async def api_process_front_only(
    image: UploadFile = File(...),
    confidence: Optional[float] = Form(0.40)
):
    """
    Process only front image - detect and classify earrings.
    
    - **image**: Image file with 'front' in filename (optional but recommended)
    - **confidence**: Confidence threshold (default: 0.40)
    
    Returns JSON with detection and classification results.
    """
    if not image.filename or not allowed_file(image.filename):
        raise HTTPException(status_code=400, detail="Invalid file format")

    try:
        # Process front image
        front_result = await process_front_image(image, confidence=confidence)
        
        # Convert to response format
        front_predictions = [
            {
                "id": pred['id'],
                "class_name": pred['class'],
                "confidence": pred['confidence'],
                "image_url": pred['image']
            }
            for pred in front_result['predictions']
        ]

        return {
            "success": True,
            "original_filename": front_result['original_filename'],
            "result_image_url": front_result['result_image'],
            "predictions": front_predictions,
            "total_detected": len(front_predictions)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing front image: {str(e)}")


@app.post("/api/process/top")
async def api_process_top_only(
    image: UploadFile = File(...),
    confidence: Optional[float] = Form(0.50)
):
    """
    Process only top image - detect pipes and count items.
    
    - **image**: Image file with 'top' in filename (optional but recommended)
    - **confidence**: Confidence threshold (default: 0.50)
    
    Returns JSON with pipe detection and item counts.
    """
    if not image.filename or not allowed_file(image.filename):
        raise HTTPException(status_code=400, detail="Invalid file format")

    try:
        # Process top image
        top_result = await process_top_image(image, confidence=confidence)
        
        # Convert to response format
        pipe_groups = [
            {
                "pipe_id": pipe['pipe_id'],
                "item_count": pipe['item_count'],
                "image_url": pipe['image']
            }
            for pipe in top_result['pipe_groups']
        ]

        return {
            "success": True,
            "original_filename": top_result['original_filename'],
            "result_image_url": top_result['result_image'],
            "pipe_groups": pipe_groups,
            "total_pipes": top_result['total_count'],
            "total_items": sum(pipe['item_count'] for pipe in pipe_groups)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing top image: {str(e)}")


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Earring Detection API"}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
s
