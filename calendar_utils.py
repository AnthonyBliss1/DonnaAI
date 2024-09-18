import datetime
import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pytz

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/calendar', 'https://www.googleapis.com/auth/calendar.events']

class CalendarManager:
    def __init__(self):
        self.service = self.authenticate()

    def authenticate(self):
        creds = None
        if os.path.exists('token.json'):
            try:
                creds = Credentials.from_authorized_user_file('token.json', SCOPES)
            except Exception:
                # If there's any error loading the credentials, we'll ignore it and create new ones
                os.remove('token.json')
                creds = None

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception:
                    # If refresh fails, we'll create new credentials
                    flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
                    creds = flow.run_local_server(port=0)
            else:
                flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)
        
            # Save the credentials for the next run
            with open('token.json', 'w') as token:
                token.write(creds.to_json())

        return build('calendar', 'v3', credentials=creds)

    def create_event(self, title, start_time, end_time, description="", calendar_id='primary'):
        event = {
            'summary': title,
            'description': description,
            'start': {'dateTime': start_time, 'timeZone': 'UTC'},
            'end': {'dateTime': end_time, 'timeZone': 'UTC'},
        }

        try:
            event = self.service.events().insert(calendarId=calendar_id, body=event).execute()
            return {"status": "success", "message": f"Event created: {event.get('htmlLink')}"}
        except HttpError as error:
            return {"status": "error", "message": f"An error occurred: {error}"}

    def create_and_send_invite(self, title, start_time, end_time, description="", attendees=[], calendar_id='primary'):
        event = {
            'summary': title,
            'description': description,
            'start': {'dateTime': start_time, 'timeZone': 'UTC'},
            'end': {'dateTime': end_time, 'timeZone': 'UTC'},
            'attendees': [{'email': attendee} for attendee in attendees],
            'reminders': {
                'useDefault': False,
                'overrides': [
                    {'method': 'email', 'minutes': 24 * 60},
                    {'method': 'popup', 'minutes': 30},
                ],
            },
        }

        try:
            event = self.service.events().insert(calendarId=calendar_id, body=event, sendUpdates='all').execute()
            return {"status": "success", "message": f"Event created and invite sent: {event.get('htmlLink')}"}
        except HttpError as error:
            return {"status": "error", "message": f"An error occurred: {error}"}

    def get_events(self, start_date, end_date, calendar_id='primary'):
        try:
            # Convert string dates to datetime objects if they're not already
            if isinstance(start_date, str):
                start_date = datetime.datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            if isinstance(end_date, str):
                end_date = datetime.datetime.fromisoformat(end_date.replace('Z', '+00:00'))

            # Ensure the dates are in UTC
            start_date = start_date.astimezone(pytz.UTC)
            end_date = end_date.astimezone(pytz.UTC)

            events_result = self.service.events().list(
                calendarId=calendar_id,
                timeMin=start_date.isoformat(),
                timeMax=end_date.isoformat(),
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            events = events_result.get('items', [])

            return [
                {
                    "title": event['summary'],
                    "start": event['start'].get('dateTime', event['start'].get('date')),
                    "end": event['end'].get('dateTime', event['end'].get('date')),
                    "description": event.get('description', '')
                }
                for event in events
            ]
        except HttpError as error:
            print(f"An error occurred: {error}")
            return []

    def query_calendar(self, query, calendar_id='primary'):
        try:
            events_result = self.service.events().list(
                calendarId=calendar_id,
                q=query,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            events = events_result.get('items', [])

            if not events:
                return {"status": "no results", "message": f"No events found matching '{query}'."}
            else:
                return {
                    "status": "success",
                    "events": [
                        {
                            "title": event['summary'],
                            "start": event['start'].get('dateTime', event['start'].get('date')),
                            "end": event['end'].get('dateTime', event['end'].get('date')),
                            "description": event.get('description', '')
                        }
                        for event in events
                    ]
                }
        except HttpError as error:
            return {"status": "error", "message": f"An error occurred: {error}"}

    def get_upcoming_events(self):
        now = datetime.datetime.now(pytz.UTC)
        week_from_now = now + datetime.timedelta(days=7)
        return self.get_events(now, week_from_now)

    def check_conflicts(self, start_time, end_time):
        events = self.get_events(start_time, end_time)
        return len(events) > 0

# Initialize the CalendarManager
calendar_manager = CalendarManager()

# Wrapper functions to maintain the same interface as before
def create_event(title, start_time, end_time, description=""):
    return calendar_manager.create_event(title, start_time, end_time, description)

def get_events(start_date, end_date):
    return calendar_manager.get_events(start_date, end_date)

def query_calendar(query):
    return calendar_manager.query_calendar(query)

def get_upcoming_events():
    return calendar_manager.get_upcoming_events()

def check_conflicts(start_time, end_time):
    return calendar_manager.check_conflicts(start_time, end_time)

def create_and_send_invite(title, start_time, end_time, description="", attendees=[]):
    return calendar_manager.create_and_send_invite(title, start_time, end_time, description, attendees)
