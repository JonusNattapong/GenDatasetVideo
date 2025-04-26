// Placeholder for JavaScript functionality
// Will handle form submission, API calls, status updates, etc.

document.addEventListener('DOMContentLoaded', () => {
    console.log('GenDatasetVideo Frontend Initialized');

    const generateForm = document.getElementById('generate-form');
    const statusDisplay = document.getElementById('generation-status');
    const progressDisplay = document.getElementById('generation-progress');
    const videoContainer = document.getElementById('video-container');
    const metadataDisplay = document.getElementById('metadata-display');

    if (generateForm) {
        generateForm.addEventListener('submit', async (event) => {
            event.preventDefault(); // Prevent default form submission
            console.log('Generate form submitted');
            statusDisplay.textContent = 'Submitting...';
            progressDisplay.style.display = 'block';
            progressDisplay.value = 0;
            videoContainer.innerHTML = ''; // Clear previous video
            metadataDisplay.textContent = ''; // Clear previous metadata

            const formData = new FormData(generateForm);
            const data = Object.fromEntries(formData.entries());

            // Convert numeric fields from form (which are strings)
            try {
                data.seed = parseInt(data.seed, 10);
                data.motion_bucket_id = parseInt(data.motion_bucket_id, 10);
                data.fps = parseInt(data.fps, 10);
                data.noise_aug_strength = parseFloat(data.noise_aug_strength);
            } catch (error) {
                console.error("Error parsing form data:", error);
                statusDisplay.textContent = 'Error: Invalid input parameters.';
                progressDisplay.style.display = 'none';
                return; // Stop submission if parsing fails
            }


            console.log('Sending data:', data);

            // --- Call Backend API ---
            try {
                statusDisplay.textContent = 'Sending request to backend...';
                progressDisplay.value = 10; // Indicate sending

                const response = await fetch('/api/generate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(data), // Send prompt and parameters
                });

                progressDisplay.value = 50; // Indicate response received
                statusDisplay.textContent = 'Processing response...';

                if (!response.ok) {
                    let errorDetail = "Unknown error";
                    try {
                        const errorJson = await response.json();
                        errorDetail = errorJson.detail || JSON.stringify(errorJson);
                    } catch (e) {
                        errorDetail = await response.text();
                    }
                    throw new Error(`HTTP error! status: ${response.status}, detail: ${errorDetail}`);
                }

                const result = await response.json();
                console.log('Generation result:', result);
                statusDisplay.textContent = 'Generation Complete! (Backend Simulated)'; // Update based on backend message
                progressDisplay.value = 100;


                // Display video and metadata
                if (result.video_url) {
                    // Construct the full URL if needed, or use relative path if served from same origin
                    const videoElement = document.createElement('video');
                    videoElement.src = result.video_url;
                    videoElement.controls = true;
                    videoElement.style.maxWidth = '100%'; // Ensure video fits container
                    videoContainer.innerHTML = ''; // Clear placeholder
                    videoContainer.appendChild(videoElement);
                } else {
                     videoContainer.innerHTML = '<p>Video generation failed or URL not provided.</p>';
                }
                metadataDisplay.textContent = JSON.stringify(result.metadata || {}, null, 2);

            } catch (error) {
                console.error('Error during generation request:', error);
                statusDisplay.textContent = `Error: ${error.message}`;
            } finally {
                 // Hide progress bar after completion or error
                 setTimeout(() => { progressDisplay.style.display = 'none'; }, 1000);
            }
            /* // Keep the old example commented out for reference if needed
            try {
                const response = await fetch('/api/generate', { // Assuming backend endpoint is /api/generate
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(data),
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const result = await response.json();
                console.log('Generation result:', result);
                statusDisplay.textContent = 'Generation Complete!';
                progressDisplay.style.display = 'none';

                // Assuming result contains video_url and metadata
                if (result.video_url) {
                    videoContainer.innerHTML = `<video controls src="${result.video_url}"></video>`;
                } else {
                     videoContainer.innerHTML = '<p>Video generation failed or URL not provided.</p>';
                }
                metadataDisplay.textContent = JSON.stringify(result.metadata || {}, null, 2);

            } catch (error) {
                console.error('Error during generation request:', error);
                statusDisplay.textContent = `Error: ${error.message}`;
                progressDisplay.style.display = 'none';
            }
            */
        });
    } else {
        console.error("Generate form not found!");
    }

    // --- TODO: Add logic for dataset management section ---
});
