# Pixel Annotator Frontend

Demo frontend web application for selecting image areas and sending them to the Pixel Level Annotator service.

## Features

- **Image Selection**: Click and drag to select rectangular areas on the image
- **Visual Feedback**: Dashed border and corner handles for the selected area
- **Layer Configuration**: Add/remove annotation layers with custom names and colors
- **Callback Configuration**: Set custom callback URL for receiving completed annotations
- **Real-time Info**: Display selection dimensions and area

## Setup

1. Install dependencies:
```bash
cd frontend
npm install
```

2. Make sure you have an image at `./fullpage.jpg` (relative to the frontend root).

3. Start the server:
```bash
npm start
```

4. Open your browser at `http://localhost:3000`

## Usage

1. The image will be displayed on the left panel
2. Click and drag on the image to select a rectangular area
3. Configure layers in the right panel (add/remove/customize)
4. Set the callback URL (default: `http://localhost:3000/callback`)
5. Click "Send to Annotator" to send the selection to the annotation service

## Callback Endpoint

The server includes a `/callback` endpoint that receives completed annotations:

```javascript
POST http://localhost:3000/callback

Request body (from annotation service):
{
  "status": "completed",
  "annotations": {
    "layer_name": "base64_encoded_image_data",
    ...
  }
}
```

## API Integration

The frontend sends requests to `http://localhost:8000/annotate` with the following format:

```javascript
{
  "image": "base64_encoded_selected_area",
  "layers": [
    { "name": "layer_name", "color": "#RRGGBB" },
    ...
  ],
  "callback_url": "http://localhost:3000/callback"
}
```

## Development

For auto-reload during development:
```bash
npm run dev
```

## Requirements

- Node.js >= 14
- Running annotation service at `http://localhost:8000`
- Image file at `./fullpage.jpg`

## Attributions

The project includes a demo image, available in the public domain at [WikimediaCommons](https://commons.wikimedia.org/wiki/File:DwtkII-as-dur-fuga.jpg).
