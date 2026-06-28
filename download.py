from dotenv import load_dotenv
import os
from roboflow import Roboflow

load_dotenv()
api_key = os.getenv("ROBOFLOW_API_KEY")

rf = Roboflow(api_key=api_key)
project = rf.workspace("iot-project").project("damaged-package-detection")
dataset = project.version(2).download("folder")


