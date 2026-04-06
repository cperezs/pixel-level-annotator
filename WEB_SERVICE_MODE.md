# Web Service Mode

The pixel-level annotator now includes a web service mode that allows external applications to send images for annotation via HTTP API.

## Features

- **Toggle Web Service Mode**: Enable/disable via checkbox in the left toolbar
- **FastAPI Integration**: RESTful API endpoint at `http://localhost:8000/annotate`
- **Image List Disabled**: When active, the local image list is disabled  
- **Submit/Cancel Controls**: Dedicated buttons appear when processing web requests
- **Progress-based Submit**: Submit button only enabled at 100% completion
- **Callback Support**: Results are automatically sent to provided callback URL

## Usage

### 1. Enable Web Service Mode

1. Start the application: `python main.py`
2. Check the "Web Service Mode" checkbox in the left toolbar
3. The service will start listening on `http://localhost:8000`
4. Image list will be disabled (grayed out)

### 2. Send Annotation Request

Send a POST request to `http://localhost:8000/annotate` with the following JSON payload:

```json
{
    "image": "base64_encoded_image_data",
    "layers": [
        {"name": "background", "color": "#2596be"},
        {"name": "objects", "color": "#9925be"},
        {"name": "text", "color": "#be4d25"}
    ],
    "callback_url": "http://your-server.com/callback"
}
```

### 3. Annotate the Image

- The image loads automatically in the annotation tool
- Layer buttons update to match the request
- Use normal annotation tools (pen, selector, fill)
- Progress bar shows completion percentage

### 4. Submit or Cancel

- **Submit**: Only enabled at 100% completion, sends annotations to callback URL
- **Cancel**: Immediately cancels the request and notifies callback URL

### 5. Callback Response

Completed annotations are sent as JSON to your callback URL:

```json
{
    "status": "completed",
    "annotations": {
        "background": "base64_encoded_png",
        "objects": "base64_encoded_png", 
        "text": "base64_encoded_png"
    }
}
```

Cancelled requests receive:

```json
{
    "status": "cancelled",
    "annotations": {}
}
```

## API Reference

### POST /annotate

**Request Body:**
- `image` (string): Base64 encoded image (PNG/JPEG)
- `layers` (array): Layer definitions with name and hex color
- `callback_url` (string): URL to receive results

**Response:**
- `200 OK`: Request accepted
- `503 Service Unavailable`: Service busy with another annotation

## Testing

The `frontend` folder contains a demo application to test the web service functionality from an HTML page.
