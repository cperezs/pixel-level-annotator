// Canvas and image setup
const canvas = document.getElementById('imageCanvas');
const ctx = canvas.getContext('2d');
const image = new Image();
const statusDiv = document.getElementById('status');
const selectionInfo = document.getElementById('selectionInfo');
const resultsDiv = document.getElementById('results');

// Selection state
let isSelecting = false;
let startX = 0;
let startY = 0;
let currentX = 0;
let currentY = 0;
let hasSelection = false;
let selectionRect = null;

// Default layers configuration (layers logic removed from UI)
const DEFAULT_LAYERS = [
    { name: 'selection', color: '#FF0000' }
];

// Load image
image.src = './fullpage.jpg';
image.onload = () => {
    canvas.width = image.width;
    canvas.height = image.height;
    drawImage();
    showStatus('Image loaded successfully', 'success');
};

image.onerror = () => {
    showStatus('Error loading image. Make sure data/fullpage.jpg exists.', 'error');
};

// Draw image and selection
function drawImage() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(image, 0, 0);
    
    if (hasSelection && selectionRect) {
        drawSelection(selectionRect.x, selectionRect.y, selectionRect.width, selectionRect.height);
    } else if (isSelecting) {
        const width = currentX - startX;
        const height = currentY - startY;
        drawSelection(startX, startY, width, height);
    }
}

// Draw selection rectangle with dashed border
function drawSelection(x, y, width, height) {
    ctx.save();
    
    // Semi-transparent overlay on non-selected areas
    ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Clear the selected area
    ctx.clearRect(x, y, width, height);
    ctx.drawImage(image, x, y, width, height, x, y, width, height);
    
    // Draw dashed border
    ctx.strokeStyle = '#3498db';
    ctx.lineWidth = 2;
    ctx.setLineDash([10, 5]);
    ctx.strokeRect(x, y, width, height);
    
    // Draw corner handles
    const handleSize = 8;
    ctx.fillStyle = '#3498db';
    ctx.setLineDash([]);
    
    // Top-left
    ctx.fillRect(x - handleSize/2, y - handleSize/2, handleSize, handleSize);
    // Top-right
    ctx.fillRect(x + width - handleSize/2, y - handleSize/2, handleSize, handleSize);
    // Bottom-left
    ctx.fillRect(x - handleSize/2, y + height - handleSize/2, handleSize, handleSize);
    // Bottom-right
    ctx.fillRect(x + width - handleSize/2, y + height - handleSize/2, handleSize, handleSize);
    
    ctx.restore();
}

// Mouse event handlers
canvas.addEventListener('mousedown', (e) => {
    const rect = canvas.getBoundingClientRect();
    startX = e.clientX - rect.left;
    startY = e.clientY - rect.top;
    isSelecting = true;
    hasSelection = false;
    selectionRect = null;
    updateButtons();
});

canvas.addEventListener('mousemove', (e) => {
    if (!isSelecting) return;
    
    const rect = canvas.getBoundingClientRect();
    currentX = e.clientX - rect.left;
    currentY = e.clientY - rect.top;
    drawImage();
});

canvas.addEventListener('mouseup', (e) => {
    if (!isSelecting) return;
    
    isSelecting = false;
    const rect = canvas.getBoundingClientRect();
    currentX = e.clientX - rect.left;
    currentY = e.clientY - rect.top;
    
    const width = currentX - startX;
    const height = currentY - startY;
    
    // Only create selection if it has some size
    if (Math.abs(width) > 5 && Math.abs(height) > 5) {
        // Normalize coordinates (handle dragging in any direction)
        const x = Math.min(startX, currentX);
        const y = Math.min(startY, currentY);
        const w = Math.abs(width);
        const h = Math.abs(height);
        
        selectionRect = { x, y, width: w, height: h };
        hasSelection = true;
        // Clear previous results when a new valid selection is created
        clearResults();
        drawImage();
        updateSelectionInfo();
        updateButtons();
        showStatus(`Selection created: ${w}x${h} pixels`, 'info');
    } else {
        drawImage();
    }
});

// Update selection info display
function updateSelectionInfo() {
    if (!hasSelection || !selectionRect) {
        selectionInfo.innerHTML = '<p>No selection</p>';
        return;
    }
    
    const { x, y, width, height } = selectionRect;
    selectionInfo.innerHTML = `
        <p><strong>Position:</strong> (${Math.round(x)}, ${Math.round(y)})</p>
        <p><strong>Size:</strong> ${Math.round(width)} × ${Math.round(height)} px</p>
        <p><strong>Area:</strong> ${Math.round(width * height).toLocaleString()} px²</p>
    `;
}

// Button controls
const clearBtn = document.getElementById('clearSelection');
const sendBtn = document.getElementById('sendSelection');

clearBtn.addEventListener('click', () => {
    hasSelection = false;
    selectionRect = null;
    drawImage();
    updateSelectionInfo();
    updateButtons();
    showStatus('Selection cleared', 'info');
});

sendBtn.addEventListener('click', async () => {
    if (!hasSelection || !selectionRect) return;
    
    await sendToAnnotator();
});

function updateButtons() {
    clearBtn.disabled = !hasSelection;
    sendBtn.disabled = !hasSelection;
}

// Show status message
function showStatus(message, type = 'info') {
    statusDiv.textContent = message;
    statusDiv.className = `status ${type}`;
    
    if (type === 'success' || type === 'info') {
        setTimeout(() => {
            statusDiv.className = 'status';
        }, 5000);
    }
}

// Layers management removed from UI; using DEFAULT_LAYERS

// Extract selected area and convert to base64
async function getSelectedImageBase64() {
    if (!hasSelection || !selectionRect) return null;
    
    const { x, y, width, height } = selectionRect;
    
    // Create a temporary canvas for the cropped image
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = width;
    tempCanvas.height = height;
    const tempCtx = tempCanvas.getContext('2d');
    
    // Draw the selected portion
    tempCtx.drawImage(image, x, y, width, height, 0, 0, width, height);
    
    // Convert to base64
    return tempCanvas.toDataURL('image/png').split(',')[1];
}

// Send to annotator endpoint
async function sendToAnnotator() {
    // Reset previous results preview before sending a new request
    clearResults();
    // Inform the frontend server to clear last annotations to avoid stale polling
    try { await fetch('/annotations/reset', { method: 'POST' }); } catch (_) {}
    const layers = DEFAULT_LAYERS;
    const callbackUrl = document.getElementById('callbackUrl').value.trim();
    
    if (!callbackUrl) {
        showStatus('Please enter a callback URL', 'error');
        return;
    }
    
    try {
        showStatus('Extracting image data...', 'info');
        const imageBase64 = await getSelectedImageBase64();
        
        if (!imageBase64) {
            showStatus('Failed to extract image data', 'error');
            return;
        }
        
        const payload = {
            image: imageBase64,
            callback_url: callbackUrl
        };
        
        showStatus('Sending to annotation service...', 'info');
        
        const response = await fetch('http://localhost:8000/annotate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload)
        });
        
        const result = await response.json();
        
        if (response.ok) {
            showStatus(`Success: ${result.message || 'Request accepted'}`, 'success');
            console.log('Response:', result);
            const reqId = ++currentRequestId;
            await waitForAnnotations(reqId);
        } else {
            showStatus(`Error: ${result.detail || 'Request failed'}`, 'error');
            console.error('Error response:', result);
        }
    } catch (error) {
        showStatus(`Error: ${error.message}`, 'error');
        console.error('Error sending to annotator:', error);
    }
}

// Initialize
updateButtons();

// ---- Results handling ----
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// Global request id to cancel previous waits when a new request starts
let currentRequestId = 0;

async function waitForAnnotations(requestId, intervalMs = 2000) {
    clearResults();
    showStatus('Waiting for annotation results...', 'info');
    while (requestId === currentRequestId) {
        try {
            const res = await fetch('/annotations', { cache: 'no-store' });
            if (!res.ok) throw new Error('Failed to fetch annotations');
            const data = await res.json();
            if (data && data.annotations && Object.keys(data.annotations).length > 0) {
                // Only render if still the latest request
                if (requestId === currentRequestId) {
                    renderAnnotations(data.annotations);
                    showStatus('Annotations received', 'success');
                }
                return;
            }
        } catch (e) {
            console.error('Polling error:', e);
        }
        await sleep(intervalMs);
    }
    // If we exit the loop, a newer request started; silently stop.
}

function clearResults() {
    if (resultsDiv) resultsDiv.innerHTML = '';
}

function renderAnnotations(annotations) {
    if (!resultsDiv) return;
    resultsDiv.innerHTML = '';
    Object.entries(annotations).forEach(([layerName, base64]) => {
        const card = document.createElement('div');
        card.className = 'result-card';
        const title = document.createElement('h4');
        title.textContent = layerName;
        const img = document.createElement('img');
        img.src = `data:image/png;base64,${base64}`;
        img.alt = `${layerName} annotation`;
        card.appendChild(title);
        card.appendChild(img);
        resultsDiv.appendChild(card);
    });
}
