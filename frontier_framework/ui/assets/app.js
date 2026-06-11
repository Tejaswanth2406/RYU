// app.js
document.addEventListener('DOMContentLoaded', () => {
    // Mock Data update loop
    setInterval(updateStats, 2000);
    setInterval(addEvent, 3500);

    // Initial load
    renderGenome();
});

function updateStats() {
    document.getElementById('active-agents').innerText = Math.floor(Math.random() * 20 + 80);
    document.getElementById('fitness-score').innerText = (Math.random() * 0.2 + 0.7).toFixed(3);
    document.getElementById('memory-usage').innerText = Math.floor(Math.random() * 100 + 400) + ' MB';
    document.getElementById('mutation-count').innerText = Math.floor(Math.random() * 500 + 1000);
}

function renderGenome() {
    const genome = {
        'Specialization': 'Synthesizer',
        'Mutation Rate': '0.05',
        'Curiosity': '0.82',
        'Risk Tolerance': '0.64',
        'Cooperation Bias': '0.91'
    };
    
    const container = document.getElementById('genome-details');
    container.innerHTML = '';
    
    for (const [key, value] of Object.entries(genome)) {
        const row = document.createElement('div');
        row.className = 'genome-row';
        row.innerHTML = `<span class="gene-name">${key}</span><span class="gene-value">${value}</span>`;
        container.appendChild(row);
    }
}

function addEvent() {
    const events = [
        "Mutation Applied: Risk Tolerance +0.02",
        "Agent Spawned (ID: a7b9c2)",
        "Memory Added: New Swarm Strategy Learned",
        "Repair Triggered: Invalid Syntax in Output",
        "Mitosis Completed: 3 Survivors Recombined"
    ];
    
    const list = document.getElementById('event-list');
    const li = document.createElement('li');
    
    const now = new Date();
    const timeStr = now.toISOString().split('T')[1].slice(0, 8);
    
    li.innerText = `[${timeStr}] ${events[Math.floor(Math.random() * events.length)]}`;
    list.prepend(li);
    
    if (list.children.length > 50) {
        list.removeChild(list.lastChild);
    }
}
