
def createsockets(pos, ips, outputs):
    outputs.append([])
    for idx in range(0,len(ips)):
        if idx == pos:
            outputs[-1].append("N")
            continue
        
        if pos % 2 == 0:
            if idx % 2 != 0:
                outputs[-1].append("C")
            else:
                if idx < pos:
                    outputs[-1].append("C")
                else:
                    outputs[-1].append("S")
        else:
            if idx % 2 == 0:
                outputs[-1].append("S")
            else:
                if idx < pos:
                    outputs[-1].append("S")
                else:
                    outputs[-1].append("C")


ips = [1,2,3.4,5,6,7,8,9]
outputs = []

for i in range(0,len(ips)):
    createsockets(i,ips,outputs)

# check output
for idx in range(0, len(outputs)):
    for i in range(0,len(outputs[idx])):
        if i == idx:
            continue
        print(str(i) + ": " + outputs[idx][i] + " : " + outputs[i][idx])
    print("")

pass
