// buttons
const enableWebcamButton = document.getElementById('webcamButton');
const stopWebcamButton = document.getElementById('stopCamButton');
const predictButton = document.getElementById('predictButton');
const uploadButton = document.getElementById('uploadButton');
predictButton.addEventListener('click', predictImage);
uploadButton.addEventListener('click', upload);
predictButton.classList.add("removed");
stopWebcamButton.classList.add("removed");

// input elements
const webcam = document.getElementById('webcam');  // webcam element
const canvas = document.getElementById("canvas"); // red box
const image = document.getElementById('image');  // image element
const results = document.getElementById("results"); // predicted results
webcam.classList.add("removed");
webcam.addEventListener('loadeddata', predictWebcam);
const files = document.querySelector("#myfile"); //uploaded file;
files.addEventListener('change', previewImage);

// models
const IMAGE_SIZE = 224;  // mobilenet input size
const model = loadModel(); // load trained mobilenet
const boxes = drawBox(); // red box for facial focusing

// prediction results
const BMI = document.querySelector('#BMI');
const AGE = document.querySelector('#AGE');
const SEX = document.querySelector('#SEX');
const clock = document.querySelector('#clock');
const fps = 4;
const timeOut = 5; // stop capturing after 5 seconds;

// moving average window (for smoonthing prediction results)
const maWindow = 12;
const BMIPool = [];
const AGEPool = [];
const SEXPool = [];
let maCount = 0;

// disable webcam for mobile view
if (window.screen.width <= 812) {
    enableWebcamButton.classList.add("removed");
    document.getElementById("webcam-guides").classList.add("removed");
    const elmToShow = document.getElementsByClassName("show-on-mobile");
    elmToShow.forEach(i => i.classList.remove("removed"));
}

// load trained mobilenet
async function loadModel() {
    const modelPath = "/data-story/assets/webcam_bmi/models/mobileNet/model.json"
    const model = tf.loadLayersModel(modelPath);
    console.log("model is loaded");
    // warm up the model
    model.then(m => {
        const x = tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3]);
        m.predict(x);
    })
    return model;
}

// read image from HTMLElement and return processed tf.Tensor
function loadImage(elm, withBox) {
    //elm is the HTMLElement (image / webcam)
    const x = tf.tidy(() => {
        // read tf.Tensor from elm and convert to float
        let img = tf.cast(tf.browser.fromPixels(elm), 'float32');
        img = img.expandDims(0);
        if (withBox) {
            // crop the tf.Tensor inside the red box
            img = tf.image.cropAndResize(img, boxes, [0], [IMAGE_SIZE, IMAGE_SIZE]);
        } else {
            img = tf.image.resizeBilinear(img, [IMAGE_SIZE, IMAGE_SIZE]);
        }
        const offset = tf.scalar(127.5);
        // convert to range [-1,1] for mobilenet input
        const normalized = img.sub(offset).div(offset);
        return normalized;
    });
    return x;
}

// Check if webcam access is supported.
function getUserMediaSupported() {
    return !!(navigator.mediaDevices &&
        navigator.mediaDevices.getUserMedia);
}

// If webcam supported, add event listener to button for when user
// wants to activate it to call enableCam function which we will 
// define in the next step.
if (getUserMediaSupported()) {
    enableWebcamButton.addEventListener('click', enableCam);
    stopWebcamButton.addEventListener('click', stopCam);
} else {
    console.warn('getUserMedia() is not supported by your browser');
}

// Enable the live webcam view and start prediction.
function enableCam(event) {
    webcam.srcObject = null;
    // Only continue if the model has finished loading.
    if (!model) {
        return;
    }
    resetClock();
    // getUsermedia parameters to force webcam but not audio.
    const constraints = {
        video: true,
        audio: false
    };
    // Activate the webcam stream.
    navigator.mediaDevices.getUserMedia(constraints)
        .then(function (stream) {
            webcam.srcObject = stream;
            webcam.play();
        });
    //show red box
    canvas.classList.remove("removed");
    // show results
    results.classList.remove("removed");
    // Hide/Show Webcam buttons
    enableWebcamButton.classList.add("removed");
    stopWebcamButton.classList.remove("removed");
    webcam.classList.remove("removed");
    predictButton.classList.add("removed");
    image.src = "";
}

// Stop the webcam
function stopCam(event) {
    if (webcam.srcObject != null) {
        webcam.pause();
        webcam.srcObject.getTracks().forEach(track => {
            track.stop();
        });
        // webcam.srcObject = null;
    }
    enableWebcamButton.classList.remove("removed");
    stopWebcamButton.classList.add("removed");
    webcam.removeEventListener('loadeddata', predictWebcam);

}

// predict function
async function predict() {
    // stop webcam after timeout
    if (maCount > timeOut * fps) {
        stopCam();
    }
    if (!webcam.paused && webcam.srcObject != null) {
        const x = loadImage(webcam, withBox = true);
        model.then(loadedModel => loadedModel.predict(x))
            .then((preds) => {
                let [a, b, c] = preds;
                [bmi, age, sex] = [a.dataSync()[0], b.dataSync()[0], c.dataSync()[0]];
                a.dispose();
                b.dispose();
                c.dispose();
                // compute MA to smoothen the predictions
                [bmi, age, sex] = computeMA(bmi, age, sex, size = maWindow);
                bmi = bmi == null ? "detecting" : bmi.toFixed(2);
                age = age == null ? "detecting" : age.toFixed(0);
                sex = sex == null ? "detecting" : `${sex > 0.5 ? 'Male' : 'Female'}(${sex.toFixed(2)})`;
                BMI.innerHTML = `BMI: ${bmi}`;
                AGE.innerHTML = `AGE: ${age}`;
                SEX.innerHTML = `SEX: ${sex}`;
            });
    }

}

// Loop over the predict function
// frequency is controlled by fps
function predictWebcam() {
    return limitLoop(predict, fps);
}

// reset the clock and MA pools
async function resetClock() {
    maCount = 0;
    BMIPool.length = 0;
    AGEPool.length = 0;
    SEXPool.length = 0;
    
}

// draw a box on webcam
function drawBox() {
    [w, h] = [canvas.width, canvas.height] = [webcam.width, webcam.height];
    const ctx = canvas.getContext("2d");
    ctx.beginPath();
    ctx.lineWidth = "4";
    ctx.strokeStyle = "red";
    leftTop = [x, y] = [200, 50];
    ctx.rect(x, y, 224, 224);
    ctx.stroke();
    canvas.classList.add("removed");
    // return the boxes to crop the image
    const boxes = [[y / h, x / w, (y + 224) / h, (x + 224) / w]];
    return boxes;
}

// crop the image inside the box
function captureBox(elm) {
    const cropSize = [224, 224];
    boxInd = [0];
    let x = tf.cast(tf.browser.fromPixels(elm), 'float32');
    x = x.expandDims(0);
    y = tf.image.cropAndResize(x, boxes, boxInd, cropSize);
    y = tf.cast(y, 'int32');
    return y.gather(0);
}


// upload image for prediction
async function upload() {
    files.click();
}

function previewImage() {
    resetClock();
    webcam.srcObject = null;
    canvas.classList.add("removed");
    webcam.classList.add("removed");
    image.src = URL.createObjectURL(files.files[0]);
    if (image.src != "") {
        predictButton.classList.remove("removed");
    } else {
        predictButton.classList.add("removed");
    }
}

function predictImage() {
    const x = loadImage(image, withBox = false);
    model.then(loadedModel => loadedModel.predict(x))
        .then((preds) => {
            let [a, b, c] = preds;
            [bmi, age, sex] = [a.dataSync()[0], b.dataSync()[0], c.dataSync()[0]];
            a.dispose();
            b.dispose();
            c.dispose();
            bmi = bmi == null ? "detecting" : bmi.toFixed(2);
            age = age == null ? "detecting" : age.toFixed(0);
            sex = sex == null ? "detecting" : `${sex > 0.5 ? 'Male' : 'Female'}(${sex.toFixed(2)})`;
            BMI.innerHTML = `BMI: ${bmi}`;
            AGE.innerHTML = `AGE: ${age}`;
            SEX.innerHTML = `SEX: ${sex}`;
        })
}

// wrap up function on top of requestAninmationFrame
// to allow controlled by fps
function limitLoop(fn, fps) {
    // Use var then = Date.now(); if you
    // don't care about targetting < IE9
    var then = new Date().getTime();

    // custom fps, otherwise fallback to 60
    fps = fps || 60;
    var interval = 1000 / fps;

    return (function loop() {
        requestAnimationFrame(loop);

        // again, Date.now() if it's available
        var now = new Date().getTime();
        var delta = now - then;

        if (delta > interval) {
            // Update time
            // now - (delta % interval) is an improvement over just
            // using then = now, which can end up lowering overall fps
            then = now - (delta % interval);

            // call the fn
            fn();
        }
    })(0);
}


// for MA computation
function computeMA(bmi, age, sex, size) {
    // moving average
    let BMIAvg = null;
    let AGEAvg = null;
    let SEXAvg = null;
    if (BMIPool.length >= size) {
        BMIAvg = BMIPool.reduce((a, b) => a + b, 0) / size;
        AGEAvg = AGEPool.reduce((a, b) => a + b, 0) / size;
        SEXAvg = SEXPool.reduce((a, b) => a + b, 0) / size;
        BMIPool.shift();
        AGEPool.shift();
        SEXPool.shift();
    }
    BMIPool.push(bmi);
    AGEPool.push(age);
    SEXPool.push(sex);
    maCount++;
    clock.innerHTML = `elapsed ${parseInt(maCount / fps)} seconds`;
    return [BMIAvg, AGEAvg, SEXAvg];
}