import os
wdir='/storage/zhang/mfMap.py'
os.chdir(wdir)
oo=["logs", "results", "ssd", "table"]
organ="COADREAD"
for x in oo:
    if os.path.exists(x):
        os.system("rm -rf {}".format(x))
    os.makedirs(os.path.join(x,organ))
if os.path.exists("data"):
    os.system("rm -rf {}".format('data'))
os.makedirs('data')
os.system("cp -r ./data_bak/* ./data/")
