
for i in range(19):
    fp=open("job_tmpl.sh")
    with open("job{:02d}.sh".format(i),"w") as ofp:
        for line in fp:
            ofp.write(line.format(i))

