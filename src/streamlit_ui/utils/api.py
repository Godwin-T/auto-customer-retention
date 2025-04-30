import requests


def make_api_call(endpoint, method="GET", data=None, files=None):
    """
    Make API calls to the backend services

    Args:
        endpoint (str): Full URL of the endpoint
        method (str): HTTP method (GET or POST)
        data (dict): JSON data to send (for POST)
        files (dict): Files to upload (for POST)

    Returns:
        tuple: (response_data, error_message)
    """
    try:
        if method == "GET":
            response = requests.get(endpoint)
        elif method == "POST":
            if files:
                response = requests.post(endpoint, files=files)
            else:
                response = requests.post(endpoint, json=data)

        if response.status_code in [200, 201]:
            return response.json(), None
        else:
            return None, f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return None, f"Exception: {str(e)}"
