from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import io
from PIL import Image
from PIL.ExifTags import TAGS
import time
import logging
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
import traceback
import numpy as np
import cv2
import os
from io import BytesIO

# Ek kütüphaneler (imagehash, torchvision, pillow_heif gibi) importları
try:
    import imagehash  # pip install imagehash

    IMAGEHASH_AVAILABLE = True
except ImportError:
    IMAGEHASH_AVAILABLE = False

try:
    import torchvision
    import torch
    from torchvision import transforms

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
    HEIC_SUPPORT_AVAILABLE = True
except ImportError:
    HEIC_SUPPORT_AVAILABLE = False


# API modelleri
class AnalysisResult(BaseModel):
    is_ai_generated: bool
    confidence: float
    indicators: List[str]
    metadata: Dict[str, Any]
    dct_analysis: Optional[Dict[str, Any]] = None
    noise_analysis: Optional[Dict[str, Any]] = None
    frequency_analysis: Optional[Dict[str, Any]] = None


class RawAnalysisResult(BaseModel):
    metadata: Dict[str, Any]
    dct_analysis: Dict[str, Any]
    noise_analysis: Dict[str, Any]
    frequency_analysis: Dict[str, Any]


class BatchAnalysisResult(BaseModel):
    results: List[AnalysisResult]


class CompareResult(BaseModel):
    image1: AnalysisResult
    image2: AnalysisResult
    summary: str


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: float


# Logger yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("api.log"),
    ]
)
logger = logging.getLogger(__name__)

# FastAPI uygulaması
app = FastAPI(
    title="Gelişmiş AI Görüntü Dedektörü API",
    description="Detaylı EXIF, C2PA ve görüntü analizi ile AI üretimi tespit eder ve görüntüleri karşılaştırır",
    version="0.5.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Yardımcı Fonksiyonlar
def make_serializable(value):
    """EXIF değerlerini JSON serileştirilebilir hale getirir."""
    if hasattr(value, 'numerator') and hasattr(value, 'denominator'):
        if value.denominator == 0:
            return str(value)
        return float(value.numerator) / float(value.denominator)
    elif isinstance(value, bytes):
        try:
            return value.decode('utf-8', errors='replace')
        except Exception:
            return str(value)
    elif isinstance(value, (list, tuple)):
        return [make_serializable(item) for item in value]
    elif isinstance(value, dict):
        return {k: make_serializable(v) for k, v in value.items()}
    else:
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        return str(value)


def get_image_metadata(image_data: bytes) -> Dict:
    """Detaylı EXIF ve temel görüntü verilerini çıkarır."""
    metadata = {}
    try:
        img = Image.open(io.BytesIO(image_data))
        metadata.update({
            "format": img.format,
            "mode": img.mode,
            "size": list(img.size)
        })
        # Genişletilmiş EXIF kontrolü: GPS, çekim tarihi, yazılım bilgisi vs.
        if hasattr(img, '_getexif') and img._getexif() is not None:
            exif_data = img._getexif()
            if exif_data:
                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, tag_id)
                    metadata[str(tag)] = make_serializable(value)
    except Exception as e:
        logger.error(f"Metadata çıkarma hatası: {str(e)}")
        logger.debug(traceback.format_exc())
        metadata["error"] = str(e)
    return metadata


def analyze_dct_coefficients(image_data: bytes) -> Dict:
    """DCT katsayıları analizi (JPEG artifaktlarını tespit için). Herhangi bir hata durumunda 'error' alanı eklenir."""
    try:
        img_np = np.asarray(bytearray(BytesIO(image_data).read()), dtype=np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        block_size = 8
        dct_coefs = []
        for i in range(0, h - block_size + 1, block_size):
            for j in range(0, w - block_size + 1, block_size):
                block = gray[i:i + block_size, j:j + block_size].astype(np.float32)
                dct = cv2.dct(block)
                dct_coefs.append(dct)
        dc_coefs = [dct[0, 0] for dct in dct_coefs]
        ac_coefs = []
        for dct in dct_coefs:
            ac = dct.flatten()[1:]
            ac_coefs.extend(ac)
        dc_mean = np.mean(dc_coefs) if dc_coefs else 0
        dc_std = np.std(dc_coefs) if dc_coefs else 0
        ac_mean = np.mean(ac_coefs) if ac_coefs else 0
        ac_std = np.std(ac_coefs) if ac_coefs else 0
        hist, _ = np.histogram(ac_coefs, bins=50) if ac_coefs else ([], None)
        hist_norm = (hist / np.sum(hist)).tolist() if len(hist) > 0 and np.sum(hist) > 0 else []
        return {
            "dc_mean": float(dc_mean),
            "dc_std": float(dc_std),
            "ac_mean": float(ac_mean),
            "ac_std": float(ac_std),
            "histogram": hist_norm,
            "block_count": len(dct_coefs)
        }
    except Exception as e:
        logger.error(f"DCT analizi hatası: {str(e)}")
        logger.debug(traceback.format_exc())
        return {"error": str(e)}


def analyze_noise_patterns(image_data: bytes) -> Dict:
    """Gürültü paternlerini analiz eder. Herhangi bir hata durumunda 'error' alanı eklenir."""
    try:
        img_np = np.asarray(bytearray(BytesIO(image_data).read()), dtype=np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        noise = cv2.absdiff(img, blur)
        b_noise, g_noise, r_noise = cv2.split(noise)
        b_mean, b_std = np.mean(b_noise), np.std(b_noise)
        g_mean, g_std = np.mean(g_noise), np.std(g_noise)
        r_mean, r_std = np.mean(r_noise), np.std(r_noise)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        lap_mean, lap_std = np.mean(np.abs(laplacian)), np.std(np.abs(laplacian))
        return {
            "blue_channel": {"mean": float(b_mean), "std": float(b_std)},
            "green_channel": {"mean": float(g_mean), "std": float(g_std)},
            "red_channel": {"mean": float(r_mean), "std": float(r_std)},
            "laplacian": {"mean": float(lap_mean), "std": float(lap_std)},
            "noise_level": float((b_std + g_std + r_std) / 3.0)
        }
    except Exception as e:
        logger.error(f"Gürültü analizi hatası: {str(e)}")
        logger.debug(traceback.format_exc())
        return {"error": str(e)}


def analyze_frequency_domain(image_data: bytes) -> Dict:
    """Fourier dönüşümü ile frekans analizi yapar. Herhangi bir hata durumunda 'error' alanı eklenir."""
    try:
        img_np = np.asarray(bytearray(BytesIO(image_data).read()), dtype=np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
        h, w = magnitude_spectrum.shape
        center_x, center_y = w // 2, h // 2
        r_low = min(center_x, center_y) // 3
        r_mid = min(center_x, center_y) * 2 // 3
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        mask_low = dist_from_center <= r_low
        mask_mid = (dist_from_center > r_low) & (dist_from_center <= r_mid)
        mask_high = dist_from_center > r_mid
        energy_low = np.sum(magnitude_spectrum[mask_low])
        energy_mid = np.sum(magnitude_spectrum[mask_mid])
        energy_high = np.sum(magnitude_spectrum[mask_high])
        total_energy = energy_low + energy_mid + energy_high
        if total_energy > 0:
            energy_low_norm = energy_low / total_energy
            energy_mid_norm = energy_mid / total_energy
            energy_high_norm = energy_high / total_energy
        else:
            energy_low_norm = energy_mid_norm = energy_high_norm = 0
        return {
            "energy_low_freq": float(energy_low_norm),
            "energy_mid_freq": float(energy_mid_norm),
            "energy_high_freq": float(energy_high_norm),
            "energy_ratio_high_low": float(energy_high / energy_low) if energy_low > 0 else 0
        }
    except Exception as e:
        logger.error(f"Frekans analizi hatası: {str(e)}")
        logger.debug(traceback.format_exc())
        return {"error": str(e)}


def combine_analysis_results(metadata: Dict, dct_analysis: Dict, noise_analysis: Dict,
                             frequency_analysis: Dict) -> Dict:
    """
    Tüm analiz sonuçlarını birleştirerek AI üretimi olasılığını hesaplar.
    AI imzası için C2PA, CAI ve OpenAI'ya özgü belirteçleri de kontrol eder.
    """
    indicators = []
    confidence = 0.0

    # 1. EXIF analizi: Daha fazla alan kontrolü
    exif_expected_fields = ["Make", "Model", "ExposureTime", "FNumber", "ISOSpeedRatings", "DateTimeOriginal"]
    missing_fields = [field for field in exif_expected_fields if field not in metadata]
    if len(missing_fields) >= 4:
        indicators.append(f"Önemli EXIF verileri eksik: {', '.join(missing_fields)}")
        confidence += min(0.2, 0.04 * len(missing_fields))

    # 2. DCT katsayıları analizi
    if "error" not in dct_analysis:
        if dct_analysis.get("ac_std", 0) < 10:
            indicators.append(f"Düşük JPEG artifaktları (AC std: {dct_analysis.get('ac_std', 0):.2f})")
            confidence += 0.15
        dc_ac_ratio = dct_analysis.get("dc_mean", 0) / dct_analysis.get("ac_mean", 1)
        if dc_ac_ratio > 5000:
            indicators.append(f"Olağandışı DC/AC oranı: {dc_ac_ratio:.2f}")
            confidence += 0.15

    # 3. Gürültü analizi
    if "error" not in noise_analysis:
        if noise_analysis.get("noise_level", 0) < 2:
            indicators.append(f"Düşük doğal gürültü seviyesi: {noise_analysis.get('noise_level', 0):.2f}")
            confidence += 0.2
        if noise_analysis.get("laplacian", {}).get("std", 0) < 5:
            indicators.append(f"Düşük kenar detayı: {noise_analysis.get('laplacian', {}).get('std', 0):.2f}")
            confidence += 0.15

    # 4. Frekans analizi
    if "error" not in frequency_analysis:
        if frequency_analysis.get("energy_ratio_high_low", 0) < 0.02:
            indicators.append(f"Düşük yüksek frekans oranı: {frequency_analysis.get('energy_ratio_high_low', 0):.4f}")
            confidence += 0.15
        if frequency_analysis.get("energy_mid_freq", 0) > 0.75:
            indicators.append(f"Yüksek orta frekans enerjisi: {frequency_analysis.get('energy_mid_freq', 0):.2f}")
            confidence += 0.15

    # 5. Format ve C2PA/CAI kontrolü: C2PA, CAI imzaları ve benzeri belirteçler
    metadata_format = metadata.get("format", "").lower()
    if metadata_format == "heic":
        indicators.append("HEIC formatı genellikle gerçek cihazlardan kaydedilir")
        confidence -= 0.2

    # AI imzası anahtar kelimeler listesi (C2PA, CAI ve OpenAI belirteçleri)
    ai_keywords = [
        "dalle", "midjourney", "stable diffusion", "ai generated", "artificial intelligence", "gan",
        "chatgpt", "openai", "gpt-4o", "c2pa", "cai", "claim generator", "actionsoftwareagentname"
    ]
    # Metadata içerisinde, özellikle yazılım bilgileri ve ilgili alanlarda arama yapılıyor
    for key, value in metadata.items():
        if isinstance(value, str):
            for keyword in ai_keywords:
                if keyword.lower() in value.lower():
                    indicators.append(f"AI yazılım imzası: '{keyword}' bulundu ({key})")
                    confidence += 0.8
                    break

    confidence = min(1.0, max(0.0, confidence))
    if not indicators:
        indicators.append("Görüntüde AI üretimi belirtisi bulunamadı")

    return {
        "is_ai_generated": confidence > 0.6,
        "confidence": confidence,
        "indicators": indicators
    }


# API Endpoint’leri

@app.post("/api/detect", response_model=AnalysisResult)
async def detect_ai_image(image_file: UploadFile = File(...)):
    """
    Yüklenen görüntüyü analiz edip, AI üretimi olup olmadığını belirler.
    """
    try:
        contents = await image_file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Boş dosya gönderildi")

        try:
            img = Image.open(io.BytesIO(contents))
            img_format = img.format.lower() if img.format else "unknown"
            if img_format not in ["jpeg", "jpg", "png", "heic", "heif"]:
                logger.warning(f"Desteklenmeyen format: {img_format}")
        except Exception as e:
            logger.error(f"Görüntü açılırken hata: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Geçersiz görüntü formatı: {str(e)}")

        metadata = get_image_metadata(contents)
        dct_analysis = analyze_dct_coefficients(contents)
        noise_analysis = analyze_noise_patterns(contents)
        frequency_analysis = analyze_frequency_domain(contents)

        combined_result = combine_analysis_results(metadata, dct_analysis, noise_analysis, frequency_analysis)

        result = {
            "is_ai_generated": combined_result["is_ai_generated"],
            "confidence": combined_result["confidence"],
            "indicators": combined_result["indicators"],
            "metadata": metadata,
            "dct_analysis": dct_analysis,
            "noise_analysis": noise_analysis,
            "frequency_analysis": frequency_analysis
        }
        try:
            JSONResponse(content=result)
            return result
        except TypeError as e:
            logger.error(f"JSON serileştirme hatası: {str(e)}")
            result["metadata"] = {"info": "Metadata serileştirilemedi, hata: " + str(e)}
            return result
    except Exception as e:
        logger.error(f"API çağrısı hatası: {str(e)}")
        logger.debug(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/detect/batch", response_model=BatchAnalysisResult)
async def batch_detect_ai_images(image_files: List[UploadFile] = File(...)):
    """
    Aynı anda birden fazla görüntüyü analiz eder.
    """
    results = []
    for image_file in image_files:
        try:
            contents = await image_file.read()
            if not contents:
                continue
            metadata = get_image_metadata(contents)
            dct_analysis = analyze_dct_coefficients(contents)
            noise_analysis = analyze_noise_patterns(contents)
            frequency_analysis = analyze_frequency_domain(contents)
            combined_result = combine_analysis_results(metadata, dct_analysis, noise_analysis, frequency_analysis)
            results.append({
                "is_ai_generated": combined_result["is_ai_generated"],
                "confidence": combined_result["confidence"],
                "indicators": combined_result["indicators"],
                "metadata": metadata,
                "dct_analysis": dct_analysis,
                "noise_analysis": noise_analysis,
                "frequency_analysis": frequency_analysis
            })
        except Exception as e:
            logger.error(f"Batch analizinde hata: {str(e)}")
    return {"results": results}


@app.post("/api/analysis_details", response_model=RawAnalysisResult)
async def analysis_details(image_file: UploadFile = File(...)):
    """
    Detaylı ara analiz sonuçlarını (metadata, DCT, gürültü, frekans) döner.
    Bu endpoint, ham analiz verilerini incelemek isteyenler içindir.
    """
    try:
        contents = await image_file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Boş dosya gönderildi")
        metadata = get_image_metadata(contents)
        dct_analysis = analyze_dct_coefficients(contents)
        noise_analysis = analyze_noise_patterns(contents)
        frequency_analysis = analyze_frequency_domain(contents)
        return {
            "metadata": metadata,
            "dct_analysis": dct_analysis,
            "noise_analysis": noise_analysis,
            "frequency_analysis": frequency_analysis
        }
    except Exception as e:
        logger.error(f"Detaylı analiz hatası: {str(e)}")
        logger.debug(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/compare", response_model=CompareResult)
async def compare_images(
        image_file1: UploadFile = File(...),
        image_file2: UploadFile = File(...)
):
    """
    İki görüntünün metadata ve analiz sonuçlarını karşılaştırarak hangisinin AI tarafından üretilmiş olabileceğini tespit eder.
    """
    try:
        contents1 = await image_file1.read()
        contents2 = await image_file2.read()
        if not contents1 or not contents2:
            raise HTTPException(status_code=400, detail="Her iki dosya da gönderilmelidir.")

        # Her iki görüntü için analizleri çalıştırıyoruz; her modülde hata olsa bile diğerleri devam eder.
        metadata1 = get_image_metadata(contents1)
        dct_analysis1 = analyze_dct_coefficients(contents1)
        noise_analysis1 = analyze_noise_patterns(contents1)
        frequency_analysis1 = analyze_frequency_domain(contents1)
        combined_result1 = combine_analysis_results(metadata1, dct_analysis1, noise_analysis1, frequency_analysis1)
        analysis_result1 = {
            "is_ai_generated": combined_result1["is_ai_generated"],
            "confidence": combined_result1["confidence"],
            "indicators": combined_result1["indicators"],
            "metadata": metadata1,
            "dct_analysis": dct_analysis1,
            "noise_analysis": noise_analysis1,
            "frequency_analysis": frequency_analysis1
        }

        metadata2 = get_image_metadata(contents2)
        dct_analysis2 = analyze_dct_coefficients(contents2)
        noise_analysis2 = analyze_noise_patterns(contents2)
        frequency_analysis2 = analyze_frequency_domain(contents2)
        combined_result2 = combine_analysis_results(metadata2, dct_analysis2, noise_analysis2, frequency_analysis2)
        analysis_result2 = {
            "is_ai_generated": combined_result2["is_ai_generated"],
            "confidence": combined_result2["confidence"],
            "indicators": combined_result2["indicators"],
            "metadata": metadata2,
            "dct_analysis": dct_analysis2,
            "noise_analysis": noise_analysis2,
            "frequency_analysis": frequency_analysis2
        }

        # Özet karşılaştırma: Hangi görüntüde daha yüksek AI confidence skoru var?
        if combined_result1["confidence"] > combined_result2["confidence"]:
            summary = f"{image_file1.filename} AI üretimi olma olasılığı daha yüksek (Confidence: {combined_result1['confidence']:.2f})."
        elif combined_result2["confidence"] > combined_result1["confidence"]:
            summary = f"{image_file2.filename} AI üretimi olma olasılığı daha yüksek (Confidence: {combined_result2['confidence']:.2f})."
        else:
            summary = "Her iki görüntüde benzer AI üretim olasılığı tespit edildi."

        return {
            "image1": analysis_result1,
            "image2": analysis_result2,
            "summary": summary
        }
    except Exception as e:
        logger.error(f"Karşılaştırma analizinde hata: {str(e)}")
        logger.debug(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """
    API sağlık kontrolü
    """
    return {
        "status": "ok",
        "version": "0.5.0",
        "timestamp": time.time()
    }


if __name__ == "__main__":
    import uvicorn

    logger.info("Gelişmiş AI Görüntü Dedektörü API başlatılıyor...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
