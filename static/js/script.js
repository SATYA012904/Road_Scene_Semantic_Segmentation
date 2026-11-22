
document.addEventListener("DOMContentLoaded", () => {

    const fileInput = document.getElementById("fileInput");
    const uploadButton = document.getElementById("uploadButton");
    const uploadArea = document.getElementById("uploadArea");
    const fileName = document.getElementById("fileName");
    const analyzeButton = document.getElementById("analyzeButton");

    const modelSelect = document.getElementById("modelSelect");
    const yoloToggle = document.getElementById("yoloToggle");

    const originalPreview = document.getElementById("originalPreview");

    let selectedFile = null;
    let previewURL = null;

    // Open file picker
    uploadButton.addEventListener("click", () => {
        fileInput.click();
    });

    // Handle file selection
    fileInput.addEventListener("change", (event) => {
        const file = event.target.files[0];

        if (file) {
            handleFile(file);
        }
    });

    // Validate selected file
    function handleFile(file) {

        const allowedTypes = [
    "image/jpeg",
    "image/png",
    "video/mp4",
    "video/avi",
    "video/x-msvideo",
    "video/quicktime"
];

        if (!allowedTypes.includes(file.type)) {
            alert("Please select a JPG, PNG, MP4, AVI, or MOV file.");

            fileInput.value = "";
            selectedFile = null;
            analyzeButton.disabled = true;
            fileName.textContent = "No file selected";

            return;
        }

        selectedFile = file;

        fileName.textContent = file.name;
        analyzeButton.disabled = false;

        if (previewURL) {
            URL.revokeObjectURL(previewURL);
        }

        previewURL = URL.createObjectURL(file);

        originalPreview.innerHTML = "<h4>Original Input</h4>";

        if (file.type.startsWith("image/")) {

    const image = document.createElement("img");
    image.src = previewURL;
    image.alt = "Original uploaded image";

    originalPreview.appendChild(image);

} else if (file.type.startsWith("video/")) {

    const video = document.createElement("video");
    video.src = previewURL;
    video.controls = true;
    video.style.width = "100%";

    originalPreview.appendChild(video);
}
    }

    // Drag and drop
    uploadArea.addEventListener("dragover", (event) => {
        event.preventDefault();
        uploadArea.classList.add("dragging");
    });

    uploadArea.addEventListener("dragleave", () => {
        uploadArea.classList.remove("dragging");
    });

    uploadArea.addEventListener("drop", (event) => {
        event.preventDefault();

        uploadArea.classList.remove("dragging");

        const file = event.dataTransfer.files[0];

        if (file) {
            handleFile(file);
            fileInput.files = event.dataTransfer.files;
        }
    });

    // Send image to Flask
    analyzeButton.addEventListener("click", async () => {

        if (!selectedFile) {
            alert("Please select an image first.");
            return;
        }

        const formData = new FormData();

        formData.append("file", selectedFile);
        formData.append("model", modelSelect.value);
        formData.append("use_yolo", yoloToggle.checked);

        analyzeButton.disabled = true;
        analyzeButton.textContent = "Analyzing...";

        try {

            const endpoint = selectedFile.type.startsWith("video/")
    ? "/api/analyze-video"
    : "/api/analyze";

const response = await fetch(endpoint, {
    method: "POST",
    body: formData
});

            const responseText = await response.text();

console.log("HTTP Status:", response.status);
console.log("Server Response:", responseText);

let data;

try {
    data = JSON.parse(responseText);
} catch (error) {
    throw new Error("Server returned an invalid response. Check the Flask terminal.");
}

if (!response.ok) {
    throw new Error(data.error || "Analysis failed.");
}

            if (!response.ok) {
                throw new Error(data.error || "Analysis failed.");
            }

            console.log("Analysis result:", data);

            // Update original image
            originalPreview.innerHTML = `
                <h4>Original Input</h4>
                <img src="${data.original}" alt="Original image">
            `;

            // Update segmentation result
            const segmentationPreview =
                document.getElementById("segmentationPreview");

            segmentationPreview.innerHTML = `
                <h4>Segmentation Output</h4>
                <img src="${data.segmentation}" alt="Segmentation result">
            `;

            // Update detection result
            const detectionPreview =
                document.getElementById("detectionPreview");

            if (data.detection) {
                detectionPreview.innerHTML = `
                    <h4>YOLO Detection</h4>
                    <img src="${data.detection}" alt="YOLO detection result">
                `;
            } else {
                detectionPreview.innerHTML = `
                    <h4>YOLO Detection</h4>
                    <p>Object detection disabled.</p>
                `;
            }

            // Update analytics
            document.getElementById("totalObjects").textContent =
                data.total_objects;

            document.getElementById("carCount").textContent =
                data.object_counts.Car || 0;

            document.getElementById("personCount").textContent =
                data.object_counts.person || 0;

            document.getElementById("processingTime").textContent =
                data.processing_time + " sec";

            alert("Road scene analysis completed successfully!");

        } catch (error) {

            console.error("Analysis error:", error);

            alert("Analysis failed: " + error.message);

        } finally {

            analyzeButton.disabled = false;
            analyzeButton.textContent = "Analyze Road Scene";

        }

    });

});
