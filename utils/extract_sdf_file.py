def extract_sdf_file(gninatypes_file, datadir):
    path = gninatypes_file.split("/")
    base_name = path[1].split(".")[0]
    base_name = base_name.rsplit("_", 1)[0]
    base_name += ".sdf"
    return datadir + "/" + path[0] + "/" + base_name
