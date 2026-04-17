import os
import re

replacements = {
    'âœ¦': '✨',
    'ðŸ"': '📱',
    'ðŸ"…': '📅', 
    'ðŸ¤–': '🤖',
    'ðŸ“Š': '📊',
    'ðŸŽ¯': '🎯',
    'ðŸ‘¤': '👤',
    'ðŸ’¥': '💥',
    'ðŸ“–': '📖',
    'ðŸ“': '📚',
    'ðŸ”Ž': '🔍',
    'ðŸ“¡': '📡',
    'ðŸ“\x83': '📜',
    'ðŸ“\x84': '📄',
    'ðŸ“\x85': '📅',
    'ðŸ“\x86': '📆',
    'ðŸ“\x8c': '📌',
    'ðŸ“\x8d': '📍',
    'ðŸ“\x9a': '📚',
    'ðŸ“\x9b': '📓',
    'ðŸ“\x9c': '📜',
    'ðŸ“\x9d': '📝',
    'ðŸ“\xa1': '📡',
    'ðŸ“\xa2': '📢',
    'ðŸ“\xa3': '📣',
    'ðŸ’': '💡',
    'â€¢': '•',
    'â†’': '→',
    'âœ”': '✔',
    'âœ•': '✕',
    'âš™': '⚙',
    'â–¸': '▸',
    'âš¡': '⚡',
    'âœ✨': '✨',
    'âš\xa0': '⚠',
    'âœ“': '✓'
}

for f in os.listdir(r'c:\Users\Amar\Desktop\SOP\templates'):
    if f.endswith('.html'):
        filepath = os.path.join(r'c:\Users\Amar\Desktop\SOP\templates', f)
        with open(filepath, 'r', encoding='utf-8') as file:
            original_text = file.read()
            
        text = original_text
        for bad, good in replacements.items():
            text = text.replace(bad, good)
        
        # generic catch-all for any remaining ðŸ sequences (except the standard emojis we just placed)
        # we'll map them to a small blue diamond '🔹'
        text = re.sub(r'ðŸ.{1,2}', '🔹', text)
        
        # handle remaining unexpected 'â' characters not covered
        text = re.sub(r'â[\x80-\xff]{1,2}', '•', text)
        text = text.replace('â', '•')
        text = text.replace('ð', '🔹')
        
        if text != original_text:
            with open(filepath, 'w', encoding='utf-8') as file:
                file.write(text)
            print(f'Fixed {f}')
