import os
import glob

import sys
#name=sys.argv[1]
filelist=[]
for check_filename in glob.glob("./*.total.png"):
    name_,_=os.path.splitext(os.path.basename(check_filename))
    name,_=os.path.splitext(name_)
    if os.path.exists(name+".total.png"):
        print(name)
        for filename in glob.glob(name+"/trial*/model/model.*.checkpoint"):
            filelist.append(filename)
print("#file:",len(filelist))
for filepath in filelist:
    os.remove(filepath)
    pass

