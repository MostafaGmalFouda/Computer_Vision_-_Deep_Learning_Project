document.addEventListener('DOMContentLoaded', () => {
    
    // Constants
    const ASL_CLASSES = ['A', 'B', 'Blank', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                         'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'];
    const IMAGES_PER_PAGE = 20;

    const API_URL_PREDICT = 'http://127.0.0.1:5000/predict'; 
    const API_URL_GET_PATHS = 'http://127.0.0.1:5000/get_all_image_paths'; 
    const API_URL_GET_BATCH = 'http://127.0.0.1:5000/get_image_batch';
    const API_URL_TEST_RANDOM = 'http://127.0.0.1:5000/test_random_image';

    // Elements
    const navButtons = document.querySelectorAll('.nav-button');
    const pages = document.querySelectorAll('.page-content');
    
    // Data Page Elements
    const classSelect = document.getElementById('class-select');
    const imageDisplay = document.querySelector('.image-display');
    const beforeProcessBtn = document.getElementById('before-process-btn');
    const afterProcessBtn = document.getElementById('after-process-btn');
    const dataToggleButtons = document.querySelectorAll('.data-toggle');
    const prevButton = document.getElementById('prev-page-btn');
    const nextButton = document.getElementById('next-page-btn');
    const pageInfo = document.getElementById('page-info');

    // Camera Elements
    const cameraToggleBtn = document.getElementById('camera-toggle-btn');
    const cameraIcon = document.getElementById('camera-icon');
    const cameraStatusText = document.getElementById('camera-status-text');
    const liveCameraFeed = document.getElementById('live-camera-feed');
    const predictedClassElement = document.getElementById('predicted-class');
    const confidenceValueElement = document.getElementById('confidence-value');
    const emojiElement = document.getElementById('confidence-emoji');
    const croppedHandDisplay = document.getElementById('cropped-hand-display'); 
    const canvas = document.createElement('canvas'); 
    const context = canvas.getContext('2d');

    // New Testing Page Elements
    const randomTestBtn = document.getElementById('random-test-btn');
    const testResultArea = document.getElementById('test-result-area');
    const testImageDisplay = document.getElementById('test-image-display');
    const testActual = document.getElementById('test-actual');
    const testPredicted = document.getElementById('test-predicted');
    const testStatus = document.getElementById('test-status');

    // State Variables
    let totalImageCount = 0; let currentPage = 0; let currentClass = 'A'; let currentProcessType = 'before'; 
    let stream = null; let isProcessing = false;

    // --- Navigation Logic ---
    function showPage(pid) {
        pages.forEach(p => p.classList.remove('active-page'));
        document.getElementById(pid).classList.add('active-page');
        if(pid === 'data-page') fetchAllImagePaths();
    }
    navButtons.forEach(btn => btn.addEventListener('click', () => {
        navButtons.forEach(b => b.classList.remove('active')); btn.classList.add('active');
        showPage(btn.getAttribute('data-page'));
    }));

    // --- Random Testing Logic ---
    if(randomTestBtn) {
        randomTestBtn.addEventListener('click', async () => {
            randomTestBtn.disabled = true;
            randomTestBtn.textContent = "Loading...";
            
            try {
                const res = await fetch(API_URL_TEST_RANDOM);
                const data = await res.json();
                
                if(!res.ok) throw new Error(data.error);

                // Show Results
                testResultArea.style.display = 'block';
                testImageDisplay.src = data.image;
                testActual.textContent = data.actual;
                testPredicted.textContent = data.predicted;
                testStatus.textContent = data.status;

                // Color Coding
                if(data.status === 'CORRECT') {
                    testStatus.style.backgroundColor = '#4CAF50'; 
                } else {
                    testStatus.style.backgroundColor = '#F44336'; 
                }

            } catch(e) {
                alert(`Error: ${e.message}`);
            } finally {
                randomTestBtn.disabled = false;
                randomTestBtn.textContent = "🎲 Test Random Image";
            }
        });
    }

    // --- Data Pagination Logic ---
    function updatePaginationDisplay() {
        const totalPages = Math.ceil(totalImageCount / IMAGES_PER_PAGE);
        if (totalImageCount === 0) { pageInfo.textContent = 'Page 0 of 0'; prevButton.disabled = true; nextButton.disabled = true; return; }
        pageInfo.textContent = `Page ${currentPage} of ${totalPages}`;
        prevButton.disabled = currentPage <= 1; nextButton.disabled = currentPage >= totalPages;
    }

    async function fetchImageBatch(className, pageNumber, processType) {
        imageDisplay.innerHTML = `<p>Loading...</p>`;
        try {
            const res = await fetch(`${API_URL_GET_BATCH}/${className}/${pageNumber}/${processType}`);
            const data = await res.json();
            if(!res.ok) throw new Error(data.error);
            const images = data.images;
            if (images.length === 0) { imageDisplay.innerHTML = `<p>Empty.</p>`; return; }
            let html = ''; images.forEach(b64 => html += `<img src="${b64}">`);
            imageDisplay.innerHTML = html;
        } catch (e) { imageDisplay.innerHTML = `<p style="color:red">${e.message}</p>`; }
        updatePaginationDisplay();
    }
    
    async function fetchAllImagePaths() {
        currentClass = classSelect.value;
        currentProcessType = beforeProcessBtn.classList.contains('active-toggle') ? 'before' : 'after';
        try {
            const res = await fetch(`${API_URL_GET_PATHS}/${currentClass}/${currentProcessType}`);
            const data = await res.json();
            if(!res.ok) throw new Error(data.error);
            totalImageCount = data.image_count; currentPage = 1;
            if (totalImageCount > 0) fetchImageBatch(currentClass, currentPage, currentProcessType);
            else { imageDisplay.innerHTML = `<p>Empty.</p>`; currentPage = 0; }
        } catch (e) { totalImageCount = 0; currentPage = 0; imageDisplay.innerHTML = `<p style="color:red">${e.message}</p>`; }
        updatePaginationDisplay();
    }

    // --- Data Page Event Listeners ---
    function updateDataToggle(btn) {
        dataToggleButtons.forEach(b => { b.classList.remove('active-toggle'); b.textContent = b.id.includes('before')?'Before Processing':'After Processing'; });
        btn.classList.add('active-toggle'); btn.textContent = '✔ ' + btn.textContent; fetchAllImagePaths();
    }
    dataToggleButtons.forEach(btn => btn.addEventListener('click', (e) => updateDataToggle(e.target)));
    if(beforeProcessBtn) updateDataToggle(beforeProcessBtn);

    if (classSelect) { ASL_CLASSES.forEach(cls => { let opt = document.createElement('option'); opt.value = cls; opt.textContent = cls; classSelect.appendChild(opt); }); classSelect.addEventListener('change', fetchAllImagePaths); }
    if(prevButton) prevButton.addEventListener('click', () => { if(currentPage > 1) { currentPage--; fetchImageBatch(currentClass, currentPage, currentProcessType); }});
    if(nextButton) nextButton.addEventListener('click', () => { if(currentPage < Math.ceil(totalImageCount/IMAGES_PER_PAGE)) { currentPage++; fetchImageBatch(currentClass, currentPage, currentProcessType); }});

    // --- Camera Logic ---
    function updateConfidenceEmoji(conf) {
        let e = ''; 
        if (conf > 0) { 
            if (conf >= 95) e = '😁';
            else if (conf >= 80) e = '🙂'; 
            else if (conf >= 65) e = '😐';
            else if (conf >= 50) e = '🙁'; 
            else if (conf >= 1) e = '😭'; 
        }
        emojiElement.textContent = e;
    }

    async function startCamera() {
        cameraStatusText.textContent = 'Starting...';
        try {
            // Constraints for the video feed
            const constraints = { video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } } };
            stream = await navigator.mediaDevices.getUserMedia(constraints);
            liveCameraFeed.srcObject = stream;
            
            liveCameraFeed.onloadedmetadata = () => {
                liveCameraFeed.play();
                cameraIcon.textContent = '📹'; 
                cameraStatusText.style.display = 'none'; 
                liveCameraFeed.style.display = 'block';
                startPredictionLoop(); 
            };
        } catch (err) { cameraStatusText.textContent = `Error: ${err.name}`; stopCamera(); }
    }

    function stopCamera() {
        if (stream) stream.getTracks().forEach(t => t.stop()); 
        liveCameraFeed.srcObject = null;
        cameraIcon.textContent = '📸'; 
        cameraStatusText.textContent = 'Camera Off'; 
        cameraStatusText.style.display = 'block'; 
        liveCameraFeed.style.display = 'none';
        croppedHandDisplay.src = ""; // Clear cropped image
        croppedHandDisplay.style.display = 'none';
        predictedClassElement.textContent = '-'; 
        confidenceValueElement.textContent = '0';
        emojiElement.textContent = '';
    }

    function startPredictionLoop() {
        // Set canvas dimensions to match video size
        canvas.width = liveCameraFeed.videoWidth || 640; 
        canvas.height = liveCameraFeed.videoHeight || 480;

        const loop = async () => {
            if (!liveCameraFeed.srcObject || !liveCameraFeed.srcObject.active) return;
            if (isProcessing) { requestAnimationFrame(loop); return; }
            
            isProcessing = true;
            
            // Capture and send the full image to the server
            context.drawImage(liveCameraFeed, 0, 0, canvas.width, canvas.height);
            const imageData = canvas.toDataURL('image/jpeg', 0.8).split(',')[1]; 
            
            try {
                const res = await fetch(API_URL_PREDICT, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: imageData })
                });
                const result = await res.json();
                
                // Update Prediction results
                predictedClassElement.textContent = result.predicted_class || '...';
                confidenceValueElement.textContent = result.confidence;
                updateConfidenceEmoji(result.confidence);

                // Display the cropped hand image received from API
                if (result.hand_detected && result.hand_image) {
                    croppedHandDisplay.src = result.hand_image;
                    croppedHandDisplay.style.display = 'block'; 
                } else {
                    croppedHandDisplay.src = "";
                    croppedHandDisplay.style.display = 'none';
                }

            } catch (e) {
                // API Error handling
                predictedClassElement.textContent = 'Err';
            } finally { 
                isProcessing = false; 
                // Control framerate to prevent overloading the API
                setTimeout(() => requestAnimationFrame(loop), 100); 
            }
        };
        loop();
    }
    if(cameraToggleBtn) cameraToggleBtn.addEventListener('click', () => { if (stream && stream.active) stopCamera(); else startCamera(); });
});