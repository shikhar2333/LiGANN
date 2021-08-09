def print_inventory(dct):
    for item, amount in dct.items():
        print("{} ({})".format(item, amount))


directory = set()
files={}
with open("custom_cd.types") as logfile:
    for line in logfile:
        path = line.split(" ")[3].split("/")
        directory.add(path[0])
        total_docked = path[1].split(".")[0]
        result = total_docked.rsplit('_', 1)[0]       
        if result not in files.keys():
            files[result] = 1
        else:
            files[result] += 1
# print(files)
print_inventory(files)
# print(directory)