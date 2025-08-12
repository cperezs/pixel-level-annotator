import asyncio
import threading
import logging
import json
import base64
import io
import requests
import cv2
import numpy as np
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from PIL import Image

logger = logging.getLogger("WebService")

class AnnotationRequest(BaseModel):
    image: str  # Base64 encoded image
    layers: List[Dict[str, str]]  # List of {"name": str, "color": str}
    callback_url: str

class WebService:
    def __init__(self, annotation_app):
        self.app = FastAPI()
        self.annotation_app = annotation_app
        self.server = None
        self.server_thread = None
        self.current_request = None
        self.is_running = False
        
        # Setup FastAPI routes
        self.app.post("/annotate")(self.handle_annotation_request)
        
    async def handle_annotation_request(self, request: AnnotationRequest):
        """Handle incoming annotation requests"""
        try:
            logger.info(f"Received annotation request with {len(request.layers)} layers")
            
            # Check if service is busy
            if self.current_request is not None:
                raise HTTPException(status_code=503, detail="Service is currently busy with another annotation")
            
            # Store the current request
            self.current_request = request
            
            # Decode the base64 image
            image_data = base64.b64decode(request.image)
            image = Image.open(io.BytesIO(image_data))
            
            # Save the image temporarily
            temp_image_path = "/tmp/temp_annotation_image.png"
            image.save(temp_image_path)
            
            # Update layers configuration
            layers_content = "\n".join([f"{layer['name']} {layer['color']}" for layer in request.layers])
            with open("layers.txt", "w") as f:
                f.write(layers_content + "\n")
            
            # Signal the main app to load the image and enter annotation mode
            self.annotation_app.load_web_service_image(temp_image_path, request)
            
            return {"status": "accepted", "message": "Annotation request received"}
            
        except HTTPException:
            # Re-raise HTTP exceptions as-is
            raise
        except Exception as e:
            logger.error(f"Error handling annotation request: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def start_server(self):
        """Start the FastAPI server in a background thread"""
        if self.is_running:
            return
            
        def run_server():
            config = uvicorn.Config(self.app, host="localhost", port=8000, log_level="info")
            self.server = uvicorn.Server(config)
            asyncio.run(self.server.serve())
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        self.is_running = True
        logger.info("Web service started on http://localhost:8000")
    
    def stop_server(self):
        """Stop the FastAPI server"""
        if not self.is_running:
            return
            
        if self.server:
            asyncio.run(self.server.shutdown())
        self.is_running = False
        self.current_request = None
        logger.info("Web service stopped")
    
    def submit_annotations(self, annotation_images: Dict[str, np.ndarray]):
        """Submit completed annotations to the callback URL"""
        if not self.current_request:
            logger.error("No current request to submit annotations for")
            return False
        
        try:
            # Prepare the response data
            response_data = {
                "status": "completed",
                "annotations": {}
            }
            
            # Convert annotation images to base64
            for layer_name, annotation_image in annotation_images.items():
                # Convert numpy array to PNG bytes
                success, buffer = cv2.imencode('.png', annotation_image)
                if success:
                    # Convert to base64
                    img_base64 = base64.b64encode(buffer).decode('utf-8')
                    response_data["annotations"][layer_name] = img_base64
                else:
                    logger.error(f"Failed to encode annotation image for layer {layer_name}")
            
            # Send to callback URL
            response = requests.post(
                self.current_request.callback_url,
                json=response_data,
                timeout=30
            )
            
            if response.status_code == 200:
                logger.info("Annotations submitted successfully")
                self.current_request = None
                return True
            else:
                logger.error(f"Failed to submit annotations: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error submitting annotations: {e}")
            return False
    
    def cancel_current_request(self):
        """Cancel the current annotation request"""
        if not self.current_request:
            return
            
        try:
            # Send cancellation to callback URL
            response_data = {
                "status": "cancelled",
                "annotations": {}
            }
            
            response = requests.post(
                self.current_request.callback_url,
                json=response_data,
                timeout=30
            )
            
            logger.info("Annotation request cancelled")
            
        except Exception as e:
            logger.error(f"Error sending cancellation: {e}")
        finally:
            self.current_request = None