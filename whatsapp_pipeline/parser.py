import re
from datetime import datetime
from pathlib import Path

# A highly robust combined regex pattern to match WhatsApp message headers:
# Matches bracketed formats like "[12/10/24, 18:24:20] Sender: Msg"
# and dashed formats like "12/10/24, 18:24 - Sender: Msg"
# Handles optional bracket wrapping and optional dash.
LINE_REGEX = re.compile(
    r'^(?:\[?([\d\-\./]+),?\s+([\d:]+(?:\s*[aApP][mM])?)\]?\s*(?:-)?\s*([^:]+):\s*(.*))$'
)

# Robust attachment regex to find any image filename within the message
ATTACH_REGEX = re.compile(r'\b([\w\-.\s]+\.(?:jpg|jpeg|png|webp))\b', re.IGNORECASE)

def clean_control_chars(text):
    """Remove hidden Unicode LTR, RTL, non-breaking spaces and formatting marks."""
    if not text:
        return ""
    text = text.replace('\u200e', '').replace('\u200f', '')
    text = text.replace('\u202f', ' ').replace('\xa0', ' ')
    return text.strip()

def parse_datetime(date_str, time_str):
    """Try parsing various datetime formats."""
    date_str = clean_control_chars(date_str).replace('/', '-').replace('.', '-')
    time_str = clean_control_chars(time_str)
    
    # Try different formats
    for fmt_date in ['%d-%m-%Y', '%d-%m-%y', '%m-%d-%Y', '%m-%d-%y', '%Y-%m-%d']:
        for fmt_time in ['%H:%M:%S', '%H:%M', '%I:%M:%S %p', '%I:%M %p', '%I:%M:%S%p', '%I:%M%p']:
            try:
                dt_str = f"{date_str} {time_str}"
                return datetime.strptime(dt_str, f"{fmt_date} {fmt_time}")
            except ValueError:
                pass
    # Fallback to current time if parsing fails completely
    return datetime.now()

def parse_chat_log(chat_file_path):
    """
    Parses WhatsApp chat file chronologically.
    Returns a list of dicts: [{'timestamp': datetime, 'sender': str, 'text': str, 'attachment': str or None}]
    """
    messages = []
    current_msg = None
    
    with open(chat_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # Clean control characters
            line_str = clean_control_chars(line.strip('\n'))
            if not line_str:
                continue
                
            m = LINE_REGEX.match(line_str)
            if m:
                date_part, time_part, sender, body = m.groups()
                dt = parse_datetime(date_part, time_part)
                
                # Check if body is an attachment
                attachment = None
                attach_match = ATTACH_REGEX.search(body)
                if attach_match:
                    attachment = attach_match.group(1)
                
                current_msg = {
                    "timestamp": dt,
                    "sender": sender.strip(),
                    "text": body.strip() if not attachment else "",
                    "attachment": attachment.strip() if attachment else None
                }
                messages.append(current_msg)
            else:
                # Continuation of the previous message (multiline text)
                if current_msg and not current_msg["attachment"]:
                    current_msg["text"] += "\n" + line_str
                    
    return messages

def group_by_subject(messages, time_threshold_sec=300):
    """
    Identifies groups of 4 images belonging to each person and associates nearby text.
    Handles sequential batches, preceding text, succeeding text, and timestamp collisions.
    """
    subjects = []
    pending_subjects = []
    
    current_images = []
    start_time = None
    sender = None
    
    last_text = ""
    last_text_sender = None
    
    for msg in messages:
        if msg["attachment"]:
            if not current_images:
                start_time = msg["timestamp"]
                sender = msg["sender"]
            
            # If the sender changes or the time gap is too large, flush any partial batch
            if msg["sender"] != sender or (start_time and (msg["timestamp"] - start_time).total_seconds() > time_threshold_sec):
                if current_images:
                    subj = {
                        "sender": sender,
                        "images": current_images,
                        "associated_text": "",
                        "timestamp": start_time
                    }
                    if last_text and last_text_sender == sender:
                        subj["associated_text"] = last_text
                        last_text = ""
                        last_text_sender = None
                    else:
                        pending_subjects.append(subj)
                    subjects.append(subj)
                    
                current_images = []
                start_time = msg["timestamp"]
                sender = msg["sender"]
                last_text = ""
                last_text_sender = None
                
            current_images.append(msg["attachment"])
            
            # If we reach exactly 4 images, flush them immediately
            if len(current_images) == 4:
                subj = {
                    "sender": sender,
                    "images": current_images,
                    "associated_text": "",
                    "timestamp": start_time
                }
                if last_text and last_text_sender == sender:
                    subj["associated_text"] = last_text
                    last_text = ""
                    last_text_sender = None
                else:
                    pending_subjects.append(subj)
                subjects.append(subj)
                
                # Clear accumulator for next batch
                current_images = []
                start_time = None
        else:
            # It's a text message
            text = msg["text"]
            text_sender = msg["sender"]
            
            # If we have subjects waiting for a text message from the same sender
            matched_pending = False
            for subj in pending_subjects:
                if subj["sender"] == text_sender and not subj["associated_text"]:
                    subj["associated_text"] = text
                    pending_subjects.remove(subj)
                    matched_pending = True
                    break
                    
            if not matched_pending:
                # Save as preceding text
                last_text = text
                last_text_sender = text_sender
                
    # Flush any remaining partial batch
    if current_images:
        subj = {
            "sender": sender,
            "images": current_images,
            "associated_text": "",
            "timestamp": start_time
        }
        if last_text and last_text_sender == sender:
            subj["associated_text"] = last_text
        subjects.append(subj)
        
    return subjects
