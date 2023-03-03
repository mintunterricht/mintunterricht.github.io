/* Coding and Layer Area Code */

let layerTypes = new Sortable(document.querySelector('#layer-types'), {
    group: {
        name: 'shared',
        pull: 'clone',
        put: false,
    },
    animation: 100,
    sort: false,
});

let modelLayers = new Sortable(document.querySelector('#model-layers'), {
    group: {
        name: 'shared',
    },
    animation: 100,
    removeOnSpill: true,
});

const trainBtn = document.querySelector('#train-btn');

trainBtn.addEventListener('click', runTraining);

const layerAreaToggleBtn = document.querySelector('#layer-area-toggle-btn');
const arrowSVGs = layerAreaToggleBtn.querySelectorAll('svg');
arrowSVGs[1].style.display = 'none';

const layerArea = document.querySelector('#layer-area');

layerAreaToggleBtn.addEventListener('click', () => {
    if (layerArea.style.display === 'none') {
        layerArea.style.display = 'flex';
        arrowSVGs[0].style.display = 'block';
        arrowSVGs[1].style.display = 'none';
    } else {
        layerArea.style.display = 'none';
        arrowSVGs[0].style.display = 'none';
        arrowSVGs[1].style.display = 'block';
    }
});

const dataScreenBtn = document.querySelector('#data-screen-btn');
const modelScreenBtn = document.querySelector('#model-screen-btn');
const trainingScreenBtn = document.querySelector('#training-screen-btn');

const dataScreen = document.querySelector('#data-screen');
const modelScreen = document.querySelector('#model-screen');
const trainingScreen = document.querySelector('#training-screen');


modelScreen.style.display = 'none';
trainingScreen.style.display = 'none';

function changeToDataScreen() {
    dataScreenBtn.classList.add('active');
    modelScreenBtn.classList.remove('active');
    trainingScreenBtn.classList.remove('active');
    dataScreen.style.display = 'flex';
    modelScreen.style.display = 'none';
    trainingScreen.style.display = 'none';
}

dataScreenBtn.addEventListener('click', changeToDataScreen);

function changeToModelScreen() {
    dataScreenBtn.classList.remove('active');
    modelScreenBtn.classList.add('active');
    trainingScreenBtn.classList.remove('active');
    dataScreen.style.display = 'none';
    modelScreen.style.display = 'flex';
    trainingScreen.style.display = 'none';
}

modelScreenBtn.addEventListener('click', changeToModelScreen);

function changeToTrainingScreen() {
    dataScreenBtn.classList.remove('active');
    modelScreenBtn.classList.remove('active');
    trainingScreenBtn.classList.add('active');
    dataScreen.style.display = 'none';
    modelScreen.style.display = 'none';
    trainingScreen.style.display = 'flex';
}

trainingScreenBtn.addEventListener('click', changeToTrainingScreen);

/* TensorFlow.js Model */

async function getData() {
    const carsDataResponse = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json');
    const carsData = await carsDataResponse.json();
    const cleaned = carsData.map(car => ({
      mpg: car.Miles_per_Gallon,
      horsepower: car.Horsepower,
    }))
    .filter(car => (car.mpg != null && car.horsepower != null));
  
    return cleaned;
}

let dataset = '';

async function render_chart(data) {
    let ctx = document.getElementById("data-viz").getContext('2d');
    let options = {
        scales: {
            y: {
                title: {
                display: true,
                text: 'Gefahrene Meilen pro Gallone'
                }
            },
            x: {
                title: {
                display: true,
                text: 'Pferdestärke'
                }
            }
        } 
    }
    let dataViz = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets:
            [{
                label: 'PS vs. gefahrene Meilen pro Gallone', // Name the series
                data: data, // Specify the data values array
                borderColor: '#FF595E',
                backgroundColor: '#FF595E', // Add custom color background (Points and Fill)           
            }]
        },
        options: options
    });
}

async function load_data() {
    if (problemSelection.value === 'regression') {
        // Load and plot the original input data that we are going to train on.
        const data = await getData();
        const values = data.map(d => ({
            x: d.horsepower,
            y: d.mpg,
        }));

        dataset = data;
        await render_chart(values);
    } else {
        alert('Kein Datensatz ausgewählt!');
    }
    
}

const dataLoadBtn = document.querySelector('#data-load-btn');

dataLoadBtn.addEventListener('click', load_data)

const problemSelection = document.querySelector('#problem-selection');
const regressionDatasets = document.querySelector('#regression-datasets');
const classificationDatasets = document.querySelector('#classification-datasets');

if (problemSelection.value === 'regression') {
    regressionDatasets.style.display = 'block';
    classificationDatasets.style.display = 'none';
} else {
    regressionDatasets.style.display = 'none';
    classificationDatasets.style.display = 'block';
}

problemSelection.addEventListener('change', () => {
    if (problemSelection.value === 'regression') {
        regressionDatasets.style.display = 'block';
        classificationDatasets.style.display = 'none';
    } else {
        regressionDatasets.style.display = 'none';
        classificationDatasets.style.display = 'block';
    }
});

async function createModelSummary(layers) {
    const modelSummaryDiv = document.querySelector('#model-summary');
    for (let layer of layers) {
        const layerInfo = document.createElement('div');
        layerInfo.classList.add('layer-info');
        const layerName = document.createElement('span');
        layerName.classList.add('layer-name');
        layerName.innerText = 'Name: ' + layer.name;
        layerInfo.append(layerName);
        const layerUnits = document.createElement('span');
        layerUnits.classList.add('layer-units');
        layerUnits.innerText = 'Neuronen: ' + layer.units;
        layerInfo.append(layerUnits);
        modelSummaryDiv.append(layerInfo);
    }
}

async function runTraining() {
    const trainingProcessIndicator = document.querySelector('#training-process-indicator');
    trainingProcessIndicator.innerText = 'Training läuft...';
    const layerNodes = document.querySelector('#model-layers').querySelectorAll('.layer');
    const layers = [];
    for (let layerNode of layerNodes) {
        layers.push({layerName: layerNode.querySelector('.layer-type').innerText, numNeurons: layerNode.querySelector('#num-neurons').value, activationFunction: layerNode.querySelector('#activation-function').value}) 
    }
    const model = createModel(layers);

    await createModelSummary(model.layers);
    // Convert the data to a form we can use for training.
    const tensorData = convertToTensor(dataset);
    const {inputs, labels} = tensorData;
    // Train the model
    let history = await trainModel(model, inputs, labels);
    trainingProcessIndicator.innerText = 'Training erfolgreich beendet!';
    console.log(history);
    // Make some predictions using the model and compare them to the
    // original data
    testModel(model, dataset, tensorData);

}

// More code will be added below
function createModel(layers) {
    // Create a sequential model
    const model = tf.sequential();

    // Add a single input layer
    model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}));

    for (let layer of layers) {
        if (layer.layerName === 'Dense Layer') {
            model.add(tf.layers.dense({units: parseInt(layer.numNeurons), useBias: true}));
        }
    }

    // Add an output layer
    model.add(tf.layers.dense({units: 1, useBias: true}));

    return model;
}

/**
 * Convert the input data to tensors that we can use for machine
 * learning. We will also do the important best practices of _shuffling_
 * the data and _normalizing_ the data
 * MPG on the y-axis.
 */
function convertToTensor(data) {
    // Wrapping these calculations in a tidy will dispose any
    // intermediate tensors.

    return tf.tidy(() => {
        // Step 1. Shuffle the data
        tf.util.shuffle(data);

        // Step 2. Convert data to Tensor
        const inputs = data.map(d => d.horsepower)
        const labels = data.map(d => d.mpg);

        const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
        const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

        //Step 3. Normalize the data to the range 0 - 1 using min-max scaling
        const inputMax = inputTensor.max();
        const inputMin = inputTensor.min();
        const labelMax = labelTensor.max();
        const labelMin = labelTensor.min();

        const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
        const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

        return {
        inputs: normalizedInputs,
        labels: normalizedLabels,
        // Return the min/max bounds so we can use them later.
        inputMax,
        inputMin,
        labelMax,
        labelMin,
        }
    });
}

async function trainModel(model, inputs, labels) {
    // Prepare the model for training.
    model.compile({
        optimizer: tf.train.adam(),
        loss: tf.losses.meanSquaredError,
        metrics: ['mse'],
    });

    const batchSize = 32;
    const epochs = 10;

    return await model.fit(inputs, labels, {
        batchSize,
        epochs,
        shuffle: true,
    });
}

function testModel(model, inputData, normalizationData) {
    const {inputMax, inputMin, labelMin, labelMax} = normalizationData;

    // Generate predictions for a uniform range of numbers between 0 and 1;
    // We un-normalize the data by doing the inverse of the min-max scaling
    // that we did earlier.
    const [xs, preds] = tf.tidy(() => {

        const xsNorm = tf.linspace(0, 1, 100);
        const predictions = model.predict(xsNorm.reshape([100, 1]));

        const unNormXs = xsNorm
        .mul(inputMax.sub(inputMin))
        .add(inputMin);

        const unNormPreds = predictions
        .mul(labelMax.sub(labelMin))
        .add(labelMin);

        // Un-normalize the data
        return [unNormXs.dataSync(), unNormPreds.dataSync()];
    });


    const predictedPoints = Array.from(xs).map((val, i) => {
        return {x: val, y: preds[i]}
    });

    const originalPoints = inputData.map(d => ({
        x: d.horsepower, y: d.mpg,
    }));

    render_output_chart(originalPoints, predictedPoints);
}

async function render_output_chart(originalData, predictedData) {
    let ctx = document.getElementById("output-viz").getContext('2d');
    let options = {
        scales: {
            y: {
                title: {
                display: true,
                text: 'Gefahrene Meilen pro Gallone'
                }
            },
            x: {
                title: {
                display: true,
                text: 'Pferdestärke'
                }
            }
        } 
    }
    let dataViz = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets:
            [
                {
                    label: 'Vorhersage', // Name the series
                    data: predictedData, // Specify the data values array
                    borderColor: '#2374AB', // Add custom color border 
                    backgroundColor: '#2374AB', // Add custom color background (Points and Fill)           
                },
                {
                    label: 'Originaldaten', // Name the series
                    data: originalData, // Specify the data values array
                    borderColor: '#FF595E', // Add custom color border 
                    backgroundColor: '#FF595E', // Add custom color background (Points and Fill)           
                },
            ]
        },
        options: options
    });
}

// document.addEventListener('DOMContentLoaded', run);