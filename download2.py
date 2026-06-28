from dotenv import load_dotenv
import os
from roboflow import Roboflow

load_dotenv()
rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
project = rf.workspace("damages-intact-box-dataset").project("damaged-intact-box-dataset")
dataset = project.version(1).download("folder")
