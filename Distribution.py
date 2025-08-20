import os

def listFolder(path):
    dirs = []
    for root, subdirs, files in os.walk(path):
        if not subdirs:
            filenames = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if filenames:
                dirs.append({"path": root, "filenames": filenames})
    return dirs

