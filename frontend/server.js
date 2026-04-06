const express = require('express');
const cors = require('cors');
const path = require('path');

const app = express();
const PORT = 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.static(__dirname));

// Serve static files from parent directory for data folder
app.use('/data', express.static(path.join(__dirname, '../data')));

// Root route - serve index.html
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

// In-memory store for last annotations received
let lastAnnotations = null;
let lastAnnotationsVersion = 0;

// Callback endpoint to receive completed annotations
app.post('/callback', (req, res) => {
    console.log('\n=== Annotation Callback Received ===');
    console.log('Status:', req.body.status);
    
    if (req.body.status === 'completed') {
        const annotations = req.body.annotations;
        console.log('Annotations received for layers:', Object.keys(annotations));
        lastAnnotations = annotations;
        lastAnnotationsVersion += 1;
        
        // Log annotation details
        Object.entries(annotations).forEach(([layerName, base64Data]) => {
            console.log(`  - ${layerName}: ${base64Data.length} bytes (base64)`);
        });
        
        // Here you can process the annotations:
        // - Save to disk
        // - Send to another service
        // - Store in database
        // etc.
        
        res.set('Cache-Control', 'no-store');
        res.json({
            success: true,
            message: 'Annotations received successfully',
            layers_processed: Object.keys(annotations).length,
            version: lastAnnotationsVersion
        });
    } else if (req.body.status === 'cancelled') {
        console.log('Annotation was cancelled');
        res.set('Cache-Control', 'no-store');
        res.json({
            success: true,
            message: 'Cancellation acknowledged'
        });
    } else {
        console.log('Unknown status:', req.body.status);
        res.set('Cache-Control', 'no-store');
        res.status(400).json({
            success: false,
            message: 'Unknown status'
        });
    }
});

// Endpoint for the frontend to fetch the last received annotations
app.get('/annotations', (req, res) => {
    res.set('Cache-Control', 'no-store');
    res.json({ annotations: lastAnnotations, version: lastAnnotationsVersion });
});

// Endpoint to reset last annotations (avoid serving stale results)
app.post('/annotations/reset', (req, res) => {
    lastAnnotations = null;
    lastAnnotationsVersion += 1; // bump version to indicate state change
    res.set('Cache-Control', 'no-store');
    res.json({ success: true, message: 'Annotations reset', version: lastAnnotationsVersion });
});

// Health check endpoint
app.get('/health', (req, res) => {
    res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Start server
app.listen(PORT, () => {
    console.log(`\n=== Pixel Annotator Frontend Server ===`);
    console.log(`Server running at http://localhost:${PORT}`);
    console.log(`Callback endpoint: http://localhost:${PORT}/callback`);
    console.log(`\nMake sure:`);
    console.log(`  1. The annotation service is running on http://localhost:8000`);
    console.log(`  2. The image exists at data/fullpage.jpg`);
    console.log(`\nPress Ctrl+C to stop\n`);
});

// Error handling
app.use((err, req, res, next) => {
    console.error('Error:', err.message);
    res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: err.message
    });
});
