"""
Ultamation 2026 Programming Challenge
Example utility functions to fetch data and send solutions to the server
"""

import json
import requests # Must install via 'pip install requests' or use the included run.bat file

BASE_URL = "http://172.16.30.223:8081"
PLAYER_ID = "5d64dc5f-901e-42b6-a65b-6c18c95a90c9" # Replace with your actual player ID from the competition dashboard

def get_data(stage=1):
    headers = {
        'stage': str(stage),
        "playerid": PLAYER_ID,
        'player-id': str(PLAYER_ID)
    }
    response = requests.get(f"{BASE_URL}/data", headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        response.raise_for_status()

def send_solution(stage, solution):
    headers = {
        'stage': str(stage),
        'playerid': str(PLAYER_ID),
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    data = {
        'answer': str(solution_to_string(solution))
    }
    response = requests.post(f"{BASE_URL}/answer", headers=headers, data=data)
    return response.text

def solution_to_string(solution):
    result = ",".join(f"{task_id}-{server_id}" for task_id, server_id in solution)
    return result
    