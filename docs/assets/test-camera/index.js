const webcam = document.getElementById("webcam");
const constraints = { audio: false, video: { width: 360, height: 360, facingMode: "user" } }
const startCamBtn = document.getElementById("startCamBtn");
const stopCamBtn = document.getElementById("stopCamBtn");

startCamBtn.addEventListener("click", startCam);
stopCamBtn.addEventListener("click", stopCam);

function startCam() {

    if (navigator.mediaDevices.getUserMedia) {
        const text = document.createElement('p');
        text.innerText = "enabling";
        document.body.append(text);
        navigator.mediaDevices.getUserMedia(constraints)
            .then(
                function (stream) {
                    webcam.srcObject = stream;
                    webcam.onloadedmetadata = function (e) {
                        webcam.play();
                    };
                }
            ).catch(function (err) {
                console.log("The following error occurred: " + err.name);
                const text = document.createElement('p');
                text.innerText = "The following error occurred: " + err.name;
                document.body.append(text);
            })
    } else {
        console.log("getUserMedia not supported");
        const text = document.createElement('p');
        text.innerText = "getUserMedia not supported";
        document.body.append(text);
    }
}

function stopCam() {
    webcam.pause();
    webcam.srcObject.getTracks().forEach(track => {
        track.stop();
    })
    webcam.srcObject = null;
}