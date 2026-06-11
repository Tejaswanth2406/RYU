// charts.js
// Mock canvas drawing for Swarm Visualization
document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('swarm-canvas');
    if (!canvas) return;
    
    // Set actual size in memory (scaled to account for extra pixel density)
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = 300; // Fixed height
    
    const ctx = canvas.getContext('2d');
    
    // Draw mock swarm
    function drawSwarm() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        const numNodes = 20;
        const nodes = [];
        
        // Generate nodes
        for(let i=0; i<numNodes; i++) {
            nodes.push({
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                radius: Math.random() * 5 + 3,
                color: Math.random() > 0.5 ? '#3b82f6' : '#10b981'
            });
        }
        
        // Draw edges
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
        ctx.lineWidth = 1;
        for(let i=0; i<numNodes; i++) {
            for(let j=i+1; j<numNodes; j++) {
                if (Math.random() > 0.85) {
                    ctx.beginPath();
                    ctx.moveTo(nodes[i].x, nodes[i].y);
                    ctx.lineTo(nodes[j].x, nodes[j].y);
                    ctx.stroke();
                }
            }
        }
        
        // Draw nodes
        for(let i=0; i<numNodes; i++) {
            ctx.beginPath();
            ctx.arc(nodes[i].x, nodes[i].y, nodes[i].radius, 0, Math.PI * 2);
            ctx.fillStyle = nodes[i].color;
            ctx.fill();
        }
    }
    
    drawSwarm();
    setInterval(drawSwarm, 2000);
});
