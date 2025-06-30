const uploadContainer = document.getElementById('uploadContainer');
const fileInput = document.getElementById('fileInput');
const documentViewer = document.getElementById('documentViewer');
const paraphraseCountElement = document.getElementById('paraphraseCount');
const generateButton = document.getElementById('generateParaphrases');
const toneSelect = document.getElementById('toneSelect');
const paraphraseResults = document.getElementById('paraphraseResults');
const similarityModal = document.getElementById('similarityModal');
const similarityDetailsContent = document.getElementById('similarityDetailsContent');
const closeButton = document.querySelector('.close-button');
const exportDocxBtn = document.getElementById('exportDocxBtn');

let selectedText = "";

const API_URL = "http://127.0.0.1:5000";

let currentAnalysisResults = null;
let currentUploadId = null;

uploadContainer.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (e) => handleFile(e.target.files[0]));

function handleFile(file) {
  if (!file) return;

  let formData = new FormData();
  formData.append("file", file);

  documentViewer.innerHTML = "<p>Uploading and analyzing...</p>";
  paraphraseCountElement.innerText = "Analyzing...";
  paraphraseResults.innerHTML = "";
  selectedText = "";
  currentAnalysisResults = null;
  currentUploadId = null;

  fetch(`${API_URL}/upload`, {
      method: "POST",
      body: formData
  })
  .then(response => response.json())
  .then(data => {
      if (data.error) {
          documentViewer.innerHTML = `<p style="color: red;">Upload and analysis failed: ${data.error}</p>`;
          paraphraseCountElement.innerText = "Analysis failed.";
          alert("Upload and analysis failed: " + data.error);
      } else {
          currentAnalysisResults = data;
          currentUploadId = data.upload_id;

          displayHighlightedText(data.original_text, data.highlights);

          paraphraseCountElement.innerText = `Total Paraphrase Groups: ${data.paraphrase_count}`;

          console.log("Analysis Results:", data);
      }
  })
  .catch(error => {
      documentViewer.innerHTML = `<p style="color: red;">An error occurred during upload or analysis.</p>`;
      paraphraseCountElement.innerText = "Analysis failed.";
      console.error("Upload or analysis failed:", error);
      alert("An error occurred during upload or analysis.");
  });
}

function displayHighlightedText(text, highlights) {
    const highlightedTextDiv = document.getElementById('documentViewer');
    highlightedTextDiv.innerHTML = '';

    let html = '';
    let lastIndex = 0;

    highlights.sort((a, b) => a.start - b.start);

    highlights.forEach(highlight => {
        html += escapeHTML(text.substring(lastIndex, highlight.start));

        const highlightedSegment = text.substring(highlight.start, highlight.end);
        html += `<span class="highlight group-${highlight.group_id}"
                       data-start="${highlight.start}"
                       data-end="${highlight.end}"
                       data-group-id="${highlight.group_id}">`
             + escapeHTML(highlightedSegment)
             + `</span>`;

        lastIndex = highlight.end;
    });

    html += escapeHTML(text.substring(lastIndex));

    highlightedTextDiv.innerHTML = html;

    addHighlightClickListeners();
}

function escapeHTML(str) {
    const div = document.createElement('div');
    div.appendChild(document.createTextNode(str));
    return div.innerHTML;
}


function addHighlightClickListeners() {
    documentViewer.querySelectorAll('.highlight').forEach(span => {
        span.addEventListener('click', (event) => {
            event.preventDefault();
            event.stopPropagation();

            const start = parseInt(event.target.dataset.start);
            const end = parseInt(event.target.dataset.end);
            const groupId = parseInt(event.target.dataset.groupId);

            const clickedSegmentText = event.target.innerText;

            displaySimilarityDetails(groupId, clickedSegmentText);
        });
    });
}

function displaySimilarityDetails(groupId, clickedSegmentText) {
    if (!currentAnalysisResults || !currentAnalysisResults.paraphrase_groups_details) {
        console.error("Analysis results not available.");
        return;
    }

    const groupDetails = currentAnalysisResults.paraphrase_groups_details.find(group => group.group_id === groupId);

    if (!groupDetails) {
        console.error(`Group details not found for group ID: ${groupId}`);
        return;
    }

    let detailsHtml = `<h4>Segments in this group (Group ${groupId}):</h4>`;

    groupDetails.segments.forEach(segment => {
        const segmentText = escapeHTML(segment.text);
        const displayClass = (segment.text.trim() === clickedSegmentText.trim()) ? 'similarity-pair clicked-segment' : 'similarity-pair';

        detailsHtml += `<p class="${displayClass}"><strong>Segment:</strong> ${segmentText}</p>`;
    });

     detailsHtml += `<h4>Pairwise Similarity Scores:</h4>`;

    groupDetails.pairwise_scores.forEach(pair => {
        const segment1 = groupDetails.segments.find(seg => seg.index === pair.segment1_index);
        const segment2 = groupDetails.segments.find(seg => seg.index === pair.segment2_index);

        if (segment1 && segment2) {
             detailsHtml += `<p><strong>Score:</strong> ${pair.score.toFixed(4)}<br>`
                         + `"${escapeHTML(segment1.text)}" vs. "${escapeHTML(segment2.text)}"`
                         + `</p>`;
        }
    });


    similarityDetailsContent.innerHTML = detailsHtml;
    similarityModal.style.display = 'flex';
}

closeButton.addEventListener('click', () => {
    similarityModal.style.display = 'none';
});

window.addEventListener('click', (event) => {
    if (event.target === similarityModal) {
        similarityModal.style.display = 'none';
    }
});


documentViewer.addEventListener("mouseup", function () {
    const selection = window.getSelection().toString().trim();
    if (selection && selection.length > 0) {
        selectedText = selection;
        paraphraseCountElement.innerText = `Selected Text for Rewrite: "${selectedText.substring(0, Math.min(selectedText.length, 50))}..."`;
    } else {
        selectedText = "";
         if (currentAnalysisResults) {
             paraphraseCountElement.innerText = `Total Paraphrase Groups: ${currentAnalysisResults.paraphrase_count}`;
         } else {
             paraphraseCountElement.innerText = "Upload a document to see paraphrase analysis.";
         }
    }
});

generateButton.addEventListener("click", async function () {
    if (!selectedText) {
        alert("Please select a sentence or click a highlighted segment first to select text for rewriting.");
        return;
    }

    let tone = toneSelect.value;
    let response = await fetch(`${API_URL}/generate_paraphrases`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sentence: selectedText, tone: tone })
    });
    let result = await response.json();

    paraphraseResults.innerHTML = `<strong>Generated Paraphrases for: "${selectedText.substring(0, Math.min(selectedText.length, 100))}..."</strong><br>`;

    if (result.error) {
        paraphraseResults.innerHTML += `<p style="color: red;">Error generating paraphrases: ${result.error}</p>`;
    } else if (result.paraphrases && result.paraphrases.length > 0) {
        result.paraphrases.forEach((para, index) => {
            let paraDiv = document.createElement("div");
            paraDiv.innerHTML = `<p>${index + 1}. ${escapeHTML(para)} <button class="apply-rewrite" data-new-text="${escapeHTML(para).replace(/"/g, '&quot;')}">Apply</button></p>`;
            paraphraseResults.appendChild(paraDiv);
        });
         addApplyRewriteListeners();
    } else {
         paraphraseResults.innerHTML += "<p>No paraphrases generated.</p>";
    }
});

function addApplyRewriteListeners() {
    paraphraseResults.querySelectorAll('.apply-rewrite').forEach(button => {
        button.addEventListener('click', async (event) => {
            const newText = event.target.dataset.newText;
            const originalText = selectedText;

            if (!currentUploadId || !originalText || !newText) {
                alert("Cannot apply rewrite. Missing document ID, original text, or new text.");
                return;
            }

            let response = await fetch(`${API_URL}/replace_sentence`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    upload_id: currentUploadId,
                    original_sentence: originalText,
                    new_sentence: newText
                })
            });
            let result = await response.json();

            if (result.error) {
                alert("Failed to apply rewrite: " + result.error);
            } else {
                alert("Rewrite applied!");

                // Update the displayed text live
                const currentText = documentViewer.innerText;
                const updatedText = currentText.replace(originalText, newText);

                documentViewer.innerText = "";  // Clear existing
                currentAnalysisResults.original_text = updatedText;
                displayHighlightedText(updatedText, currentAnalysisResults.highlights);
                selectedText = "";

                paraphraseCountElement.innerText = `Total Paraphrase Groups: ${currentAnalysisResults.paraphrase_count}`;
            }
        });
    });
}

similarityModal.style.display = 'none';

documentViewer.innerHTML = escapeHTML(documentViewer.innerText);
function exportDocument(fileType) {
    if (!currentUploadId) {
        alert("Please upload and analyze a document before exporting.");
        return;
    }

    const exportUrl = `${API_URL}/export/${currentUploadId}/${fileType}`;
    window.open(exportUrl, "_blank");
}

exportDocxBtn.addEventListener("click", () => exportDocument("docx"));
