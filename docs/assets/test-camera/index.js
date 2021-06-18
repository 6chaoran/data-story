const webcam = document.getElementById("webcam");
const constraints = { audio: false, video: { width: 360, height: 360, facingMode: "user" } }
const startCamBtn = document.getElementById("startCamBtn");
const stopCamBtn = document.getElementById("stopCamBtn");

startCamBtn.addEventListener("click", startCam);
stopCamBtn.addEventListener("click", stopCam);


navigator.getUserMedia = navigator.getUserMedia ||
    navigator.webkitGetUserMedia ||
    navigator.mozGetUserMedia;



function startCam() {
    if (navigator.getUserMedia) {
        navigator.getUserMedia(constraints,
            function (stream) {
                webcam.srcObject = stream;
                webcam.onloadedmetadata = function (e) {
                    webcam.play();
                };
            },
            function (err) {
                console.log("The following error occurred: " + err.name);
            }
        );
    } else {
        console.log("getUserMedia not supported");
        const text = document.createElement('p');
        text.innerText = "getUserMedia not supported";
        document.body.append(text);
    }
}

function stopCam() {
    webcam.srcObject.getTracks().forEach(track => {
        track.stop();
    })
    webcam.srcObject = null;
}