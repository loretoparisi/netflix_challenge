data = open("../../data/um/all.dta", "r")
idx = open("../../data/um/all.idx", "r")
out = open("../../data/probe_um.dta", "w")

for line in data:
    index = int(idx.readline())
    if index == 4:
        out.write(line)

data.close()
idx.close()
out.close()
