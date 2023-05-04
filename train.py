import os

if __name__ == "__main__":
    
    HOME = os.getcwd()
    os.environ["HOME"] = HOME


    if not os.path.exists("WAND-4"):
        from roboflow import Roboflow
        rf = Roboflow(api_key="yOphezEoorbTPlrhX0UT")
        project = rf.workspace("pomvom").project("wand")
        dataset = project.version(4).download("yolov8")

    from ultralytics import YOLO

    # Load a model
    model = YOLO('yolov8n-seg.yaml')  # build a new model from YAML
    model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)
    model = YOLO('yolov8n-seg.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

    model.train(data=f"{HOME}/WAND-4/data.yaml", epochs=300, batch=32, device=0)  # train the mode