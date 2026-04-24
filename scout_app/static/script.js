const chatWindow = document.getElementById('chat-window');
const chatForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');

// Suggestion click handler
document.querySelectorAll('.tip-box li').forEach(li => {
    li.addEventListener('click', () => {
        userInput.value = li.innerText.replace(/"/g, '');
        chatForm.dispatchEvent(new Event('submit'));
    });
});

async function sendMessage(e) {
    e.preventDefault();
    const text = userInput.value.trim();
    if (!text) return;

    // Append User Message
    appendMessage(text, 'user');
    userInput.value = '';

    // Show Typing Indicator
    const typingId = showTypingIndicator();

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: text })
        });

        const data = await response.json();
        
        // Remove typing indicator
        document.getElementById(typingId).remove();

        // Append Bot Reply
        appendBotReply(data.reply, data.players);

    } catch (error) {
        document.getElementById(typingId).remove();
        appendMessage("Lỗi kết nối đến máy chủ AI. Vui lòng kiểm tra lại server Flask.", 'bot');
    }
}

function appendMessage(text, sender) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `chat-message ${sender}`;
    
    // Parse bold text like **word** to <strong>word</strong> for simple Markdown
    const formattedText = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

    let icon = sender === 'user' ? '<i class="ph-fill ph-user"></i>' : '<i class="ph-fill ph-robot"></i>';
    
    msgDiv.innerHTML = `
        <div class="avatar">${icon}</div>
        <div class="message-content glass-chip">
            <p>${formattedText}</p>
        </div>
    `;
    
    chatWindow.appendChild(msgDiv);
    scrollBottom();
}

function showTypingIndicator() {
    const id = 'typing-' + Date.now();
    const msgDiv = document.createElement('div');
    msgDiv.className = `chat-message bot`;
    msgDiv.id = id;
    
    msgDiv.innerHTML = `
        <div class="avatar"><i class="ph-fill ph-robot"></i></div>
        <div class="message-content glass-chip">
            <div class="typing-indicator">
                <span></span><span></span><span></span>
            </div>
        </div>
    `;
    
    chatWindow.appendChild(msgDiv);
    scrollBottom();
    return id;
}

function appendBotReply(replyText, players) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `chat-message bot`;
    
    const formattedText = replyText.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    let htmlContent = `
        <div class="avatar"><i class="ph-fill ph-robot"></i></div>
        <div class="message-content glass-chip" style="width: 100%;">
            <p>${formattedText}</p>
    `;

    // Render Player Cards if any
    if (players && players.length > 0) {
        htmlContent += `<div class="player-results">`;
        
        players.forEach(p => {
            htmlContent += `
                <div class="player-card">
                    <div class="pc-header">
                        <div class="pc-name-group">
                            <h4>${p.name}</h4>
                            <div class="pc-team">
                                <i class="ph-fill ph-shield-chevron"></i> ${p.team} | Vị trí: ${p.position}
                            </div>
                        </div>
                        <div class="pc-match">${p.match}</div>
                    </div>
                    <div class="pc-stats">
                        ${p.dynamic_stats.map(s => `
                        <div class="stat-item">
                            <span class="stat-label">${s.label}</span>
                            <span class="stat-value">${s.value}</span>
                        </div>
                        `).join('')}
                    </div>
                </div>
            `;
        });
        
        htmlContent += `</div>`;
    }

    htmlContent += `</div>`;
    msgDiv.innerHTML = htmlContent;
    
    chatWindow.appendChild(msgDiv);
    scrollBottom();
}

function scrollBottom() {
    chatWindow.scrollTo({
        top: chatWindow.scrollHeight,
        behavior: 'smooth'
    });
}
