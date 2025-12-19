// Tab switching
const tabBtns = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');

tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        const tabName = btn.dataset.tab;

        tabBtns.forEach(b => b.classList.remove('active'));
        tabContents.forEach(c => c.classList.remove('active'));

        btn.classList.add('active');
        document.getElementById(`${tabName}-tab`).classList.add('active');
    });
});

// Max results slider
const maxResultsSlider = document.getElementById('max-results');
const maxResultsValue = document.getElementById('max-results-value');

maxResultsSlider.addEventListener('input', (e) => {
    maxResultsValue.textContent = e.target.value;
});

// Sections
const inputSection = document.querySelector('.input-section');
const loadingSection = document.getElementById('loading-section');
const resultsSection = document.getElementById('results-section');
const errorSection = document.getElementById('error-section');

function showSection(section) {
    [inputSection, loadingSection, resultsSection, errorSection].forEach(s => {
        s.style.display = 'none';
    });
    section.style.display = 'block';
    section.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Check button
const checkBtn = document.getElementById('check-btn');
const claimInput = document.getElementById('claim-input');
const urlInput = document.getElementById('url-input');

checkBtn.addEventListener('click', async () => {
    const activeTab = document.querySelector('.tab-btn.active').dataset.tab;
    const claim = activeTab === 'text' ? claimInput.value.trim() : '';
    const url = activeTab === 'url' ? urlInput.value.trim() : '';
    const maxResults = parseInt(maxResultsSlider.value);

    if (!claim && !url) {
        alert('Please enter a claim or URL to fact-check');
        return;
    }

    await checkClaim(claim, url, maxResults);
});

// Allow Enter key in textarea
claimInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && e.ctrlKey) {
        checkBtn.click();
    }
});

// API call
async function checkClaim(claim, url, maxResults) {
    showSection(loadingSection);

    const requestBody = {
        max_results: maxResults
    };

    if (claim) {
        requestBody.claim = claim;
    } else if (url) {
        requestBody.url = url;
    }

    try {
        const response = await fetch('/api/check', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        if (data.status === 'error') {
            showError(data.error || 'An error occurred while processing your request');
        } else {
            displayResults(data);
        }
    } catch (error) {
        console.error('Error:', error);
        showError(error.message || 'Failed to connect to the API. Please try again.');
    }
}

// Display results
function displayResults(data) {
    // Verdict badge
    const verdictBadge = document.getElementById('verdict-badge');
    const verdict = data.verdict.toLowerCase().replace(/_/g, '-');
    verdictBadge.className = `verdict-badge ${verdict}`;
    verdictBadge.textContent = data.verdict.replace(/_/g, ' ');

    // Claim text
    document.getElementById('claim-text').textContent = data.claim;

    // Confidence
    const confidence = Math.round(data.confidence * 100);
    const confidenceFill = document.getElementById('confidence-fill');
    const confidenceValue = document.getElementById('confidence-value');

    setTimeout(() => {
        confidenceFill.style.width = `${confidence}%`;
    }, 100);
    confidenceValue.textContent = `${confidence}%`;

    // Net score
    document.getElementById('net-score').textContent = data.net_score.toFixed(2);

    // Processing time
    document.getElementById('processing-time').textContent = `${data.processing_time}s`;

    // Evidences
    const evidencesList = document.getElementById('evidences-list');
    evidencesList.innerHTML = '';

    if (data.evidences && data.evidences.length > 0) {
        data.evidences.forEach((evidence, index) => {
            const card = createEvidenceCard(evidence, index + 1);
            evidencesList.appendChild(card);
        });
    } else {
        evidencesList.innerHTML = '<p style="text-align: center; color: #6b7280;">No evidence sources found.</p>';
    }

    showSection(resultsSection);
}

// Create evidence card
function createEvidenceCard(evidence, index) {
    const card = document.createElement('div');
    card.className = 'evidence-card glass-card';

    const stance = evidence.stance.toLowerCase();
    const stanceClass = stance === 'supports' ? 'supports' : stance === 'refutes' ? 'refutes' : 'neutral';

    const similarity = Math.round(evidence.similarity * 100);
    const stanceScore = Math.round(evidence.stance_score * 100);

    card.innerHTML = `
        <div class="evidence-header">
            <a href="${evidence.url}" target="_blank" rel="noopener noreferrer" class="evidence-url">
                Source ${index}: ${new URL(evidence.url).hostname}
            </a>
            <span class="stance-badge ${stanceClass}">${evidence.stance}</span>
        </div>
        <p class="evidence-text">${evidence.best_sentence}</p>
        <div class="evidence-scores">
            <div>Similarity: <span>${similarity}%</span></div>
            <div>Stance Confidence: <span>${stanceScore}%</span></div>
        </div>
    `;

    return card;
}

// Show error
function showError(message) {
    document.getElementById('error-message').textContent = message;
    showSection(errorSection);
}

// Retry button
document.getElementById('retry-btn').addEventListener('click', () => {
    showSection(inputSection);
});

// Check another button
document.getElementById('check-another-btn').addEventListener('click', () => {
    claimInput.value = '';
    urlInput.value = '';
    showSection(inputSection);
});
