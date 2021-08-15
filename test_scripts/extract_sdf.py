def print_inventory(dct):
    for item, amount in dct.items():
        print("{} ({})".format(item, amount))


directory = set()
files={}
def extract_sdf_file(gninatypes_file):
    datadir = "/scratch/shubham/crossdock_data"
    path = gninatypes_file.split("/")
    base_name = path[1].split(".")[0]
    base_name = base_name.rsplit("_", 1)[0]
    base_name += ".sdf"
    return datadir + "/structs/" + path[0] + "/" + base_name

with open("../types/training_example.types") as logfile:
    for line in logfile:
        path = line.split(" ")[1].split("/")
        directory.add(path[0])
        total_docked = path[1].split(".")[0]
        result = total_docked.rsplit('_', 1)[0]
#        print(extract_sdf_file(line.split(" ")[1]))
        if result not in files.keys():
            files[result] = 1
        else:
            files[result] += 1
#print(files)
#print_inventory(files)
for dir in directory:
    print(dir)
