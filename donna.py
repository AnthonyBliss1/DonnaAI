import os
import pyaudio
import wave
import openai
from pocketsphinx import LiveSpeech, get_model_path
from playsound import playsound
from openai import OpenAI
import json
from datetime import datetime, timedelta
from calendar_utils import create_and_send_invite, get_events, query_calendar, get_upcoming_events
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QSystemTrayIcon, QMenu, QHBoxLayout
from PySide6.QtGui import QPainter, QColor, QPen, QIcon, QPainterPath
from PySide6.QtCore import QTimer, Qt, QPoint, Signal, QObject, QRect
import numpy as np
import threading
import time
import requests
from collections import deque
from dateutil import parser
import pytz
import urllib.request

# Set API keys
openai.api_key = os.getenv('OPENAI_API_KEY')
BLAND_AI_API_KEY = os.getenv('BLAND_AI_API_KEY')

client = OpenAI()

user_name = "Anthony Bliss"

# Wake word configuration
WAKE_WORD = 'hey donna'

MAX_HISTORY = 10  # Maximum number of messages to keep in history
conversation_history = deque(maxlen=MAX_HISTORY)

class VoiceVisualizerWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(300, 80)  # Adjusted height to fit in the smaller window
        self.state = "idle"
        self.amplitude = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(50)  # Update every 50ms

    def set_state(self, state):
        self.state = state
        self.update()

    def set_amplitude(self, amplitude):
        self.amplitude = min(amplitude, 1.0)  # Ensure amplitude is between 0 and 1

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        if self.state == "wake_word":
            color = QColor(255, 165, 0)  # Orange
        elif self.state == "recording":
            color = QColor(255, 0, 0)  # Red
        elif self.state == "responding":
            color = QColor(0, 255, 0)  # Green
        elif self.state == "thinking":
            color = QColor(0, 0, 255)  # Blue
        else:
            color = QColor(128, 128, 128)  # Gray

        painter.setPen(QPen(color, 2))

        rect = self.rect()
        center_y = rect.height() // 2

        # Adjust sensitivity for recording state
        amplitude_multiplier = 220 if self.state == "recording" else 60

        # Increase frequency for more dynamic visualization
        frequency = 0.2 if self.state == "recording" else 0.1

        # Draw a more dynamic waveform
        for x in range(0, rect.width(), 3):  # Decreased step for smoother curve
            y = center_y + int(np.sin(x * frequency) * self.amplitude * amplitude_multiplier)
            y += int(np.cos(x * frequency * 0.5) * self.amplitude * amplitude_multiplier * 0.5)
            painter.drawLine(x, center_y, x, y)

class AssistantSignals(QObject):
    show_window = Signal()
    hide_window = Signal()
    set_visualizer_state = Signal(str)
    set_visualizer_amplitude = Signal(float)

class RoundedCornerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TranslucentBackground)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Create rounded rect path
        path = QPainterPath()
        path.addRoundedRect(QRect(0, 0, self.width(), self.height()), 15, 15)

        # Fill the path with a dark color
        painter.fillPath(path, QColor(40, 40, 40))  # Dark gray background

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Donna Voice Assistant")
        self.visualizer = VoiceVisualizerWidget()
        
        # Create a main layout
        main_layout = QVBoxLayout()
        
        # Create a horizontal layout for centering
        h_layout = QHBoxLayout()
        h_layout.addStretch()
        h_layout.addWidget(self.visualizer)
        h_layout.addStretch()
        
        # Add the horizontal layout to the main layout
        main_layout.addStretch()
        main_layout.addLayout(h_layout)
        main_layout.addStretch()
        
        # Set margins to create some padding
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Create the rounded corner widget and set it as the central widget
        rounded_widget = RoundedCornerWidget(self)
        rounded_widget.setLayout(main_layout)
        self.setCentralWidget(rounded_widget)

        # Create system tray icon
        self.tray_icon = QSystemTrayIcon(self)
        icon = QIcon("brain.png")  # Replace with your icon path
        self.tray_icon.setIcon(icon)
        self.tray_icon.setVisible(True)

        # Create tray menu
        tray_menu = QMenu()
        show_action = tray_menu.addAction("Show")
        show_action.triggered.connect(self.show)
        quit_action = tray_menu.addAction("Quit")
        quit_action.triggered.connect(QApplication.quit)
        self.tray_icon.setContextMenu(tray_menu)

        # Set window flags to remove title bar and make it stay on top
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        # Set initial size (adjust as needed)
        self.setFixedSize(320, 100)  # Decreased height

        self.assistant_signals = AssistantSignals()
        self.assistant_signals.show_window.connect(self.show_window)
        self.assistant_signals.hide_window.connect(self.hide_window)
        self.assistant_signals.set_visualizer_state.connect(self.visualizer.set_state)
        self.assistant_signals.set_visualizer_amplitude.connect(self.visualizer.set_amplitude)

    def closeEvent(self, event):
        event.ignore()
        self.hide()
        self.tray_icon.showMessage(
            "Donna Voice Assistant",
            "Application was minimized to Tray",
            QSystemTrayIcon.Information,
            2000
        )

    def show_window(self):
        # Get the primary screen geometry
        screen_geometry = QApplication.primaryScreen().availableGeometry()
        
        # Calculate position
        window_width = self.width()
        window_height = self.height()
        x = screen_geometry.width() - window_width
        y = 40  # 40 pixels from top edge (to account for menu bar)

        # Set window position
        self.setGeometry(x, y, window_width, window_height)

        # Show and raise window
        self.show()
        self.raise_()
        self.activateWindow()

    def hide_window(self):
        self.hide()

def detect_wake_word(assistant_signals):
    assistant_signals.hide_window.emit()
    assistant_signals.set_visualizer_state.emit("wake_word")
    model_path = get_model_path()
    speech = LiveSpeech(
        verbose=False,
        lm=False,
        keyphrase=WAKE_WORD,
        kws_threshold=1e-20
    )
    print("Listening for wake word...")
    for phrase in speech:
        print("Wake word detected!")
        return

def record_audio(filename, max_duration=30, silence_threshold=500, silence_duration=2.0, visualizer=None):
    chunk = 1024
    audio_format = pyaudio.paInt16
    channels = 1
    rate = 16000
    pa = pyaudio.PyAudio()

    stream = pa.open(format=audio_format,
                     channels=channels,
                     rate=rate,
                     input=True,
                     frames_per_buffer=chunk)

    print("Recording...")
    frames = []
    silent_chunks = 0
    audio_started = False

    if visualizer:
        visualizer.set_visualizer_state.emit("recording")

    for _ in range(0, int(rate / chunk * max_duration)):
        data = stream.read(chunk)
        frames.append(data)

        # Convert audio chunks to numpy array
        audio_data = np.frombuffer(data, dtype=np.int16)
        
        # Calculate volume
        volume = np.abs(audio_data).mean()

        if visualizer:
            visualizer.set_visualizer_amplitude.emit(volume / 32768.0)

        # Check if audio has started
        if not audio_started and volume > silence_threshold:
            audio_started = True

        # If audio has started, check for silence
        if audio_started:
            if volume < silence_threshold:
                silent_chunks += 1
                if silent_chunks > silence_duration * (rate / chunk):
                    break
            else:
                silent_chunks = 0

    print("Recording complete.")

    stream.stop_stream()
    stream.close()
    pa.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(pa.get_sample_size(audio_format))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()

    if visualizer:
        visualizer.set_visualizer_state.emit("idle")
        visualizer.set_visualizer_amplitude.emit(0)

def transcribe_audio(audio_filename):
    with open(audio_filename, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file
        )
    return transcript.text

def get_assistant_functions():
    return [
        {
            "name": "create_event",
            "description": "Create a new event in the user's calendar",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "start_time": {"type": "string", "format": "date-time"},
                    "end_time": {"type": "string", "format": "date-time"},
                    "description": {"type": "string"},
                },
                "required": ["title", "start_time", "end_time"],
            },
        },
        {
            "name": "get_events",
            "description": "Retrieve events from the user's calendar",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {"type": "string", "format": "date"},
                    "end_date": {"type": "string", "format": "date"},
                },
                "required": ["start_date", "end_date"],
            },
        },
        {
            "name": "query_calendar",
            "description": "Query the user's calendar for specific information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        },
        {
            "name": "get_upcoming_events",
            "description": "Get events for the next 7 days",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "make_restaurant_reservation",
            "description": "Make a restaurant reservation using bland.ai. Requires a valid phone number for the restaurant.",
            "parameters": {
                "type": "object",
                "properties": {
                    "phone_number": {"type": "string", "description": "The phone number of the restaurant. This is required."},
                    "party_size": {"type": "integer"},
                    "preferred_time": {"type": "string", "description": "The preferred time in 12-hour format (e.g., '7:30 PM')"},
                    "preferred_date": {"type": "string", "format": "date"},
                    "restaurant_name": {"type": "string"},
                },
                "required": ["phone_number", "party_size", "preferred_time", "preferred_date", "restaurant_name"],
            },
        },
        {
            "name": "play_latest_call_recording",
            "description": "Play the recording of the most recent restaurant reservation call",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    ]

def make_restaurant_reservation(phone_number, party_size, preferred_time, preferred_date, restaurant_name):
    if not phone_number:
        return {"status": "error", "message": "A valid phone number for the restaurant is required to make a reservation."}
    
    # Convert time to 12-hour format if it's in 24-hour format
    try:
        time_obj = datetime.strptime(preferred_time, "%I:%M %p")
    except ValueError:
        try:
            time_obj = datetime.strptime(preferred_time, "%H:%M")
            preferred_time = time_obj.strftime("%I:%M %p").lstrip("0").lower()
        except ValueError:
            return {"status": "error", "message": "Invalid time format. Please use '7:30 PM' or '19:30'."}
    
    url = "https://us.api.bland.ai/v1/calls"
    
    headers = {
        'Authorization': os.getenv('BLAND_AI_API_KEY')
    }
    
    data = {
        "phone_number": phone_number,
        "task": f"Make a reservation under the name {user_name} for {party_size} people on {preferred_date} at {preferred_time}. If that time is not available, ask for the closest available time.",
        "model": "enhanced",
        "language": "en",
        "voice": "june",
        "max_duration": 300,  # 5 minutes
        "wait_for_greeting": True,
        "record": True
    }

    try:
        print("Sending request to bland.ai API...")
        print(f"Payload: {json.dumps(data, indent=2)}")
        response = requests.post(url, json=data, headers=headers)
        print(f"Response status code: {response.status_code}")
        print(f"Response content: {response.text}")
        
        if response.status_code == 200:
            response_data = response.json()
            call_id = response_data.get('call_id')
            
            # Save the call_id to a file with the current date
            current_date = datetime.now().strftime("%Y-%m-%d")
            filename = f"latest_call_id.txt"
            with open(filename, "w") as f:
                f.write(call_id)
            print(f"Saved call_id {call_id} to {filename}")
            
            time.sleep(60)
            
            # Analyze the call
            analyze_url = f"https://api.bland.ai/v1/calls/{call_id}/analyze"
            analyze_payload = {
                "goal": "Make a restaurant reservation",
                "questions": [
                    ["Was the reservation confirmed?", "boolean"],
                    ["What date and time was the reservation confirmed for?", "string"],
                    ["How many people is the reservation for?", "integer"]
                ]
            }
            analyze_headers = {
                "authorization": BLAND_AI_API_KEY,
                "Content-Type": "application/json"
            }
            
            print(f"Sending analyze request for call_id: {call_id}")
            print(f"Analyze payload: {json.dumps(analyze_payload, indent=2)}")
            analyze_response = requests.post(analyze_url, json=analyze_payload, headers=analyze_headers)
            
            print(f"Analyze response status code: {analyze_response.status_code}")
            print(f"Analyze response content: {analyze_response.text}")
            
            if analyze_response.status_code == 200:
                analyze_data = analyze_response.json()
                print(f"Analyze data: {json.dumps(analyze_data, indent=2)}")
                
                if 'answers' in analyze_data and len(analyze_data['answers']) == 3:
                    confirmed = analyze_data['answers'][0]
                    if confirmed:
                        confirmed_datetime = analyze_data['answers'][1]
                        confirmed_party_size = analyze_data['answers'][2]
                        
                        return {
                            "status": "success",
                            "message": "Reservation confirmed",
                            "call_id": call_id,
                            "reservation_details": {
                                "party_size": confirmed_party_size,
                                "confirmed_datetime": confirmed_datetime,
                                "restaurant_name": restaurant_name
                            }
                        }
                    else:
                        return {
                            "status": "failure",
                            "message": "Reservation could not be confirmed",
                            "call_id": call_id
                        }
                else:
                    return {"status": "error", "message": f"Unexpected analyze response structure: {analyze_data}"}
            else:
                return {"status": "error", "message": f"Failed to analyze call: {analyze_response.text}"}
        else:
            return {"status": "error", "message": f"Failed to make call: {response.text}"}
    
    except Exception as e:
        print(f"Error in make_restaurant_reservation: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": f"Failed to process reservation: {str(e)}"}

def create_event(title, start_time, end_time, description=""):
    # Implement this function to add an event to the user's calendar
    # For now, we'll just print the event details
    print(f"Adding event to calendar: {title} on {start_time}")
    return {"status": "success", "message": "Event added to calendar"}

def generate_response(user_text, visualizer=None):
    if visualizer:
        visualizer.set_visualizer_state.emit("thinking")

    # Add the user's message to the conversation history
    conversation_history.append({"role": "user", "content": user_text})

    # Prepare the messages for the API call
    messages = [
        {"role": "system", "content": "You are Donna, a smart, witty, and helpful secretary with access to the user's calendar and the ability to make phone calls on behalf of the user. Use the provided functions when necessary to interact with the calendar or make phone calls. When making restaurant reservations, always ask for and confirm all necessary details before proceeding, especially the restaurant's phone number which is required for the reservation. Ensure that times are provided in 12-hour format (e.g., '7:30 PM'). DO NOT CREATE NUMBERED LISTS IN YOUR RESPONSES."},
    ]
    messages.extend(list(conversation_history))

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        functions=get_assistant_functions(),
        function_call="auto"
    )

    message = response.choices[0].message

    if message.function_call:
        function_name = message.function_call.name
        function_args = json.loads(message.function_call.arguments)
        
        try:
            if function_name == "create_event":
                result = create_event(**function_args)
            elif function_name == "get_events":
                result = get_events(**function_args)
            elif function_name == "query_calendar":
                result = query_calendar(**function_args)
            elif function_name == "get_upcoming_events":
                result = get_upcoming_events()
            elif function_name == "make_restaurant_reservation":
                if 'phone_number' not in function_args or not function_args['phone_number']:
                    return "I'm sorry, but I need the restaurant's phone number to make a reservation. Can you please provide the phone number?"
                result = make_restaurant_reservation(**function_args)
                print(f"Reservation result: {result}")
                
                if result["status"] == "success":
                    # Add the confirmed reservation to the calendar
                    reservation_details = result["reservation_details"]
                    restaurant_name = reservation_details['restaurant_name']
                    
                    # Parse the datetime using dateutil
                    try:
                        event_datetime = parser.parse(reservation_details['confirmed_datetime'])
                        # If no year is provided, assume it's for the next occurrence of that date
                        if event_datetime.year == datetime.now().year and event_datetime < datetime.now():
                            event_datetime = event_datetime.replace(year=event_datetime.year + 1)
                        # Ensure the datetime is timezone-aware
                        if event_datetime.tzinfo is None:
                            eastern = pytz.timezone('US/Eastern')
                            event_datetime = eastern.localize(event_datetime)
                        else:
                            event_datetime = event_datetime.astimezone(pytz.timezone('US/Eastern'))
                    except ValueError as e:
                        print(f"Error parsing date: {e}")
                        return f"I'm sorry, but there was an error processing the reservation date. Can you please try again?"

                    end_datetime = event_datetime + timedelta(hours=2)  # Assume 2-hour duration
                    
                    event_title = f"Reservation at {restaurant_name}"
                    event_description = f"Reservation for {reservation_details['party_size']} people at {restaurant_name}."
                    
                    calendar_result = create_and_send_invite(
                        title=event_title,
                        start_time=event_datetime.isoformat(),
                        end_time=end_datetime.isoformat(),
                        description=event_description,
                    )
                    
                    print(f"Calendar invite creation result: {calendar_result}")
                    
                    if calendar_result["status"] == "success":
                        response_text = (f"Great news! I've successfully made a reservation for {user_name} "
                                         f"at {restaurant_name} for {reservation_details['party_size']} people on {event_datetime.strftime('%A, %B %d, %Y')} "
                                         f"at {event_datetime.strftime('%I:%M %p')}. I've sent a calendar invite to your email. "
                                         f"Is there anything else you need?")
                    else:
                        response_text = (f"I've successfully made a reservation for {user_name} at {restaurant_name}, but there was an issue sending the calendar invite: "
                                         f"{calendar_result['message']}. The reservation is for {reservation_details['party_size']} people on "
                                         f"{event_datetime.strftime('%A, %B %d, %Y')} at {event_datetime.strftime('%I:%M %p')}. "
                                         f"You might want to add this to your calendar manually. Is there anything else you need?")
                elif result["status"] == "failure":
                    response_text = f"I'm sorry, but I couldn't confirm the reservation at {function_args['restaurant_name']}. The restaurant might be fully booked or there might have been an issue with the call. Would you like me to try again or assist you with something else?"
                else:
                    response_text = f"I'm sorry, but there was an error while trying to make the reservation at {function_args['restaurant_name']}: {result['message']}. Would you like me to try again or assist you with something else?"
                
                return response_text
            elif function_name == "play_latest_call_recording":
                result = play_latest_call_recording()
            else:
                result = {"status": "error", "message": f"Unknown function: {function_name}"}
        except Exception as e:
            print(f"Error executing {function_name}: {str(e)}")  # Debug print
            return f"I'm sorry, but there was an error while processing your request: {str(e)}. Can I assist you with something else?"

        follow_up_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are Donna, a smart, witty, and helpful secretary. Respond to the function result."},
                {"role": "function", "name": function_name, "content": json.dumps(result)},
            ]
        )
        if visualizer:
            visualizer.set_visualizer_state.emit("responding")
        return follow_up_response.choices[0].message.content
    else:
        if visualizer:
            visualizer.set_visualizer_state.emit("responding")
        # Add the assistant's response to the conversation history
        conversation_history.append({"role": "assistant", "content": message.content})
        return message.content

def text_to_speech(text, visualizer=None):
    if visualizer:
        visualizer.set_visualizer_state.emit("responding")

    response = client.audio.speech.create(
        model='tts-1',
        voice='nova',
        input=text
    )
    
    audio_filename = "response.mp3"
    with open(audio_filename, 'wb') as f:
        for chunk in response.iter_bytes():
            f.write(chunk)
    
    # Convert mp3 to wav for easier processing
    from pydub import AudioSegment
    sound = AudioSegment.from_mp3(audio_filename)
    wav_filename = "response.wav"
    sound.export(wav_filename, format="wav")

    # Read the wav file
    with wave.open(wav_filename, 'rb') as wf:
        # Get the parameters of the wav file
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()

        # Read all frames
        frames = wf.readframes(n_frames)

    # Convert frames to numpy array
    audio_data = np.frombuffer(frames, dtype=np.int16)

    # Calculate chunk size for approximately 50ms of audio
    chunk_size = int(framerate * 0.05)

    def update_visualizer():
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i+chunk_size]
            if len(chunk) > 0:
                amplitude = np.abs(chunk).max() / 32768.0  # Normalize to 0-1
                visualizer.set_visualizer_amplitude.emit(amplitude)
            time.sleep(0.05)  # Sleep for 50ms

    # Start the visualizer update in a separate thread
    visualizer_thread = threading.Thread(target=update_visualizer)
    visualizer_thread.start()

    # Play the audio
    playsound(wav_filename)

    # Wait for the visualizer thread to finish
    visualizer_thread.join()

    # Clean up
    os.remove(audio_filename)
    os.remove(wav_filename)

    if visualizer:
        visualizer.set_visualizer_state.emit("idle")
        visualizer.set_visualizer_amplitude.emit(0)

def check_event_created(event_title):
    events = get_upcoming_events()
    return any(event['title'] == event_title for event in events)

def play_latest_call_recording():
    try:
        # Get the most recent call_id
        current_date = datetime.now().strftime("%Y-%m-%d")
        filename = f"latest_call_id.txt"
        
        if not os.path.exists(filename):
            return {"status": "error", "message": "No recent call found for today"}
        
        with open(filename, "r") as f:
            call_id = f.read().strip()
        
        # Request the call recording
        url = f"https://api.bland.ai/v1/calls/{call_id}/recording"
        headers = {
            "Authorization": os.getenv('BLAND_AI_API_KEY')
        }
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            if data["status"] == "success":
                mp3_url = data["url"]
                
                # Download the MP3 file
                mp3_filename = f"call_recording_{call_id}.mp3"
                urllib.request.urlretrieve(mp3_url, mp3_filename)
                
                # Play the MP3 file
                playsound(mp3_filename)
                
                # Clean up the downloaded file
                os.remove(mp3_filename)
                
                return {"status": "success", "message": "Call recording played successfully"}
            else:
                return {"status": "error", "message": "Failed to get recording URL"}
        else:
            return {"status": "error", "message": f"Failed to retrieve recording: {response.text}"}
    
    except Exception as e:
        print(f"Error in play_latest_call_recording: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": f"Failed to play recording: {str(e)}"}

def run_assistant(assistant_signals):
    global conversation_history
    while True:
        detect_wake_word(assistant_signals)
        
        assistant_signals.show_window.emit()
        conversation_active = True
        conversation_history.clear()  # Reset conversation history
        while conversation_active:
            audio_filename = 'user_input.wav'
            record_audio(audio_filename, max_duration=30, silence_threshold=500, silence_duration=2.0, visualizer=assistant_signals)
            user_text = transcribe_audio(audio_filename)
            print(f'User: {user_text}')
            
            if "thank you" in user_text.lower():
                print('Donna: You\'re welcome!')
                conversation_active = False
                break
            
            print('Donna: ', end='', flush=True)
            assistant_response = generate_response(user_text, visualizer=assistant_signals)
            print(assistant_response)
            text_to_speech(assistant_response, visualizer=assistant_signals)
            
            os.remove(audio_filename)
        
        if os.path.exists(audio_filename):
            os.remove(audio_filename)
        
        assistant_signals.hide_window.emit()

def main():
    app = QApplication([])
    app.setQuitOnLastWindowClosed(False)  # Prevent app from quitting when window is closed
    window = MainWindow()
    window.hide()  # Start with the window hidden

    assistant_thread = threading.Thread(target=run_assistant, args=(window.assistant_signals,))
    assistant_thread.start()

    app.exec()

if __name__ == '__main__':
    main()
