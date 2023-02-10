import os


class Endpoint:
    def __init__(self):
        self._url = None
    
    @property
    def url(self):
        if self._url is None:
            self._url = self.get_url()
        
        return self._url
    
    def get_url(self):
        endpoint = os.environ.get("endpoint")

        return endpoint 


def get_token():
    token = os.environ.get("auth_token")

    if token is None:
        raise ValueError("auth-token not found in environment variables")
    
    return token
