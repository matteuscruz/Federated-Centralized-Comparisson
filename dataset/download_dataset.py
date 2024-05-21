from roboflow import Roboflow
rf = Roboflow(api_key="TycSacoLGqa1qKaf5k0X")
project = rf.workspace("inatel").project("pneumonia-classification-imrcv")
dataset = project.version(1).download("folder")
