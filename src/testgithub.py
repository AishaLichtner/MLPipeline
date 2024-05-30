import requests
import tempfile
import pickle
import os
def download_and_load_model(github_url: str, local_path: str = None):
    """
        Download a model from GitHub and load it from a pickle file.
        
        Args:
            github_url (str): The GitHub URL to the pickle file containing the model.
            local_path (str): The local path to save the downloaded pickle file.
        
        Returns:
            model: The loaded model.
     """

    # Download the model file
    response = requests.get(github_url)
    response.raise_for_status()

    if local_path is None:
        local_path = os.path.join(tempfile.gettempdir(), 'model.pkl')
    
    # Save the model to a local file in write-binary mode
    with open(local_path, 'wb') as file:
        file.write(response.content)
    
    # Check if the file was downloaded correctly
    file_size = os.path.getsize(local_path)
    if file_size == 0:
        raise ValueError("Downloaded file is empty")

    # Load the model from the local file in read-binary mode
    with open(local_path, 'rb') as file:
        model = pickle.load(file)
    
    return model

if __name__ == "__main__":
    download_and_load_model("https://github.com/AishaLichtner/MLPipeline/blob/main/outputs/models/1036708_model.pkl")
  