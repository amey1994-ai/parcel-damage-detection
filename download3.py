# from dotenv import load_dotenv
# import os
# from roboflow import Roboflow

# load_dotenv()
# rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))

# project = rf.workspace("ameesh").project("package_damage_classification-1uqzq")
# dataset = project.version(1).download("folder")










from dotenv import load_dotenv
import os
from roboflow import Roboflow

load_dotenv()
rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))

project = rf.workspace("ameesh").project("package_damage_classification-1uqzq")

# Try versions
for v in [1, 2, 3, 4]:
    try:
        dataset = project.version(v).download("folder")
        print(f"Done! Version {v} worked.")
        break
    except Exception as e:
        print(f"Version {v} failed: {e}")