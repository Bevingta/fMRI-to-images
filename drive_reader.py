import os.path

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Define the scopes for access
SCOPES = ['https://www.googleapis.com/auth/drive']

def authenticate():
    creds = None

    # Check if token file exists
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json')
    
    # If no valid credentials available, prompt user to login
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    
    return creds

def get_file_metadata(file_id, creds):
    service = build('drive', 'v3', credentials=creds)
    file = service.files().get(fileId=file_id).execute()
    return file

def get_folder_contents(folder_id, creds):
    service = build('drive', 'v3', credentials=creds)
    results = service.files().list(
        q=f"'{folder_id}' in parents",
        fields="nextPageToken, files(id, name)"
    ).execute()
    items = results.get('files', [])

    if len(items) == 0:
        print("No files found.")

    return items

def main():
    # Authenticate user
    creds = authenticate()

    # File ID of the shared file
    #file_id = input("Enter the file ID: ")
    folder_id = "1iTp-qLLoKSyUCYyvCAQwNwSKzpvN1NiF"

    commanding = True

    file_metadata = get_file_metadata(folder_id, creds)
    
    while commanding:
        print()
        #gets the terminal command
        command = input(f"{file_metadata['name']} > ")

        #cd command using <id> as navigation
        if command[:2] == 'cd':
            file_metadata = get_file_metadata(command[3:], creds)

        #ls command to list files in the folder
        elif command == 'ls':
            file_info = get_folder_contents(file_metadata['id'], creds)
            # Print names and IDs of files in the folder
            for item in file_info:
                print(f"Name: {item['name']}, ID: {item['id']}")

        else:
            print("Invalid command. Type 'ls' to list files or 'cd <folder_id>' to navigate.")

if __name__ == '__main__':
    main()
