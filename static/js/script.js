// =========================
// DOM Elements
// =========================

const uploadBtn = document.getElementById("uploadBtn");
const trainBtn = document.getElementById("trainBtn");

const fileInput = document.getElementById("csvFile");

const status = document.getElementById("status");

const preview = document.getElementById("preview");

const rows = document.getElementById("rows");
const cols = document.getElementById("cols");
const missing = document.getElementById("missing");

const statistics = document.getElementById("statistics");

const headerStatus = document.getElementById("headerStatus");

const targetSelect = document.getElementById("targetSelect");
const taskSelect = document.getElementById("taskSelect");
const modelSelect = document.getElementById("modelSelect");

const results = document.getElementById("results");

const plotImage = document.getElementById("plotImage");
const copilotText =
document.getElementById("copilotText");

// =========================
// Upload Dataset
// =========================

uploadBtn.addEventListener("click", async () => {

    const file = fileInput.files[0];

    if (!file) {

        status.innerHTML = "Please choose a CSV file.";

        return;

    }


    const headerToggle = document.getElementById("headerToggle");


    const formData = new FormData();


    formData.append(
        "file",
        file
    );


    formData.append(
        "has_header",
        headerToggle.checked
    );


    try {


        status.innerHTML = "Uploading...";


        const response = await fetch("/upload", {

            method:"POST",

            body:formData

        });


        const data = await response.json();

        if(data.copilot){
            copilotText.innerHTML =
            data.copilot.replace(/\n/g,"<br>");
        }



       if (!response.ok || data.error) {

    console.error("UPLOAD ERROR:", data);

    status.innerHTML = `
        <span style="color:#ef4444;">
            Upload failed: ${data.error || "Unknown server error"}
        </span>
    `;

    return;
}



        status.innerHTML="Upload Successful ✅";



        createTable(data);



        headerStatus.innerHTML =
        data.header_detected
        ?
        "Detected ✅"
        :
        "Not Detected ❌";



        targetSelect.innerHTML="";


        data.columns.forEach(col=>{


            targetSelect.innerHTML +=

            `<option value="${col}">
            ${col}
            </option>`;


        });



        targetSelect.value=data.target;



        taskSelect.value=data.task;


        updateModels(data.task);



        rows.innerHTML=data.summary.rows;

        cols.innerHTML=data.summary.columns;

        missing.innerHTML=data.summary.missing;



        let html=`

        <table border="1" cellpadding="8">

        <tr>

        <th>Column</th>
        <th>Mean</th>
        <th>Std</th>
        <th>Min</th>
        <th>Max</th>

        </tr>

        `;



        const stats=data.summary.statistics;



        for(let column in stats){


            html+=`

            <tr>

            <td>${column}</td>

            <td>${stats[column].mean ?? "-"}</td>

            <td>${stats[column].std ?? "-"}</td>

            <td>${stats[column].min ?? "-"}</td>

            <td>${stats[column].max ?? "-"}</td>


            </tr>

            `;


        }


        html+="</table>";

        statistics.innerHTML=html;



    }

    catch(err){


        console.error(err);

        status.innerHTML=
        "Cannot connect to Flask server.";

    }

    function updateModels(task) {

    modelSelect.innerHTML = "";

    let models = [];

    if (task === "Classification") {

        models = [
            "Logistic Regression",
            "Random Forest",
            "SVM"
        ];

    } else {

        models = [
            "Linear Regression",
            "Random Forest Regressor",
            "SVR"
        ];

    }

    models.forEach(model => {

        modelSelect.innerHTML += `
            <option value="${model}">
                ${model}
            </option>
        `;

    });

    // Automatically select the first model
    modelSelect.selectedIndex = 0;
}


});

// =========================
// Task Type Changed
// =========================

taskSelect.addEventListener("change", () => {

    // Rebuild the model list for the selected task
    updateModels(taskSelect.value);

});

// =========================
// Preview Table
// =========================

function createTable(data) {

    let html = "<table border='1' cellpadding='8'>";

    html += "<tr>";

    data.columns.forEach(col => {

        html += `<th>${col}</th>`;

    });

    html += "</tr>";

    data.preview.forEach(row => {

        html += "<tr>";

        data.columns.forEach(col => {

            html += `<td>${row[col]}</td>`;

        });

        html += "</tr>";

    });

    html += "</table>";

    preview.innerHTML = html;

}


// =========================
// Model Dropdown
// =========================

function updateModels(task) {

    modelSelect.innerHTML = "";

    if (task === "Classification") {

        const models = [

            "Logistic Regression",

            "Random Forest",

            "SVM"

        ];

        models.forEach(model => {

            modelSelect.innerHTML += `
                <option value="${model}">
                    ${model}
                </option>
            `;

        });

    }

    else {

        const models = [

            "Linear Regression",

            "Random Forest Regressor",

            "SVR"

        ];

        models.forEach(model => {

            modelSelect.innerHTML += `
                <option value="${model}">
                    ${model}
                </option>
            `;

        });

    }

}

// ============================================================
// TASK TYPE CHANGE
// ============================================================

taskSelect.addEventListener("change", () => {

    const selectedTask = taskSelect.value;

    updateModels(selectedTask);

});

// =========================
// Train Model
// =========================

trainBtn.addEventListener("click", async () => {

    results.innerHTML = `
        <h3>🤖 Training Model...</h3>
        <p>Please wait...</p>
    `;

    plotImage.style.display = "none";

    try {

        const response = await fetch("/train", {

            method: "POST",

            headers: {

                "Content-Type": "application/json"

            },

            body: JSON.stringify({

                target: targetSelect.value,

                task: taskSelect.value,

                model: modelSelect.value

            })

        });


        const responseText = await response.text();

let data;

try {

    data = JSON.parse(responseText);

} catch (error) {

    console.error("SERVER RESPONSE:", responseText);

    status.innerHTML = `
        <span style="color:#ef4444;">
            Server returned an invalid response.
        </span>
    `;

    return;
}

        //--------------------------------------------------
        // Backend Error
        //--------------------------------------------------

        if (!response.ok) {

            results.innerHTML = `
                <h3 style="color:red;">
                    ❌ Training Failed
                </h3>

                <p>${data.error}</p>
            `;

            return;

        }


        //--------------------------------------------------
        // AI Copilot
        //--------------------------------------------------

        if (data.copilot) {

            copilotText.innerHTML = `
                <strong>🤖 AI Recommendation</strong>

                <br><br>

                ${data.copilot.replace(/\n/g, "<br>")}
            `;

        }

        else {

            copilotText.innerHTML = `
                No recommendation available.
            `;

        }


        //--------------------------------------------------
        // Metrics
        //--------------------------------------------------

        let html = `

        <h3>📈 Evaluation Metrics</h3>

        <table border="1" cellpadding="8">

            <tr>

                <th>Metric</th>

                <th>Value</th>

            </tr>

        `;


        for (let key in data.metrics) {

            if (key === "Confusion Matrix") continue;

            if (key === "Labels") continue;

            html += `

                <tr>

                    <td>${key}</td>

                    <td>${JSON.stringify(data.metrics[key])}</td>

                </tr>

            `;

        }

        html += "</table><br>";


        //--------------------------------------------------
        // Confusion Matrix
        //--------------------------------------------------

        if (data.plots.confusion_matrix) {

            html += `

                <h3>📊 Confusion Matrix</h3>

                <img

                    src="${data.plots.confusion_matrix}?t=${Date.now()}"

                    style="max-width:700px;border-radius:10px;border:1px solid #ccc;">

                <br><br>

            `;

        }


        //--------------------------------------------------
        // Actual vs Predicted
        //--------------------------------------------------

        if (data.plots.actual_vs_predicted) {

            html += `

                <h3>📈 Actual vs Predicted</h3>

                <img

                    src="${data.plots.actual_vs_predicted}?t=${Date.now()}"

                    style="max-width:700px;border-radius:10px;border:1px solid #ccc;">

                <br><br>

            `;

        }


        //--------------------------------------------------
        // Feature Importance
        //--------------------------------------------------

        if (data.plots.feature_importance) {

            html += `

                <h3>⭐ Feature Importance</h3>

                <img

                    src="${data.plots.feature_importance}?t=${Date.now()}"

                    style="max-width:700px;border-radius:10px;border:1px solid #ccc;">

            `;

        }


        //--------------------------------------------------
        // Finished
        //--------------------------------------------------

        results.innerHTML = html;

    }

    catch (err) {

        console.error(err);

        results.innerHTML = `

            <h3 style="color:red;">

                Server Error

            </h3>

            <p>

                ${err.message}

            </p>

        `;

    }

});