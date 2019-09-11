import os
import glob


for filename in glob.glob("/home/nguyen.viet.anhd/Downloads/data_bachuc/*/*.m4a"):
    new_name = filename.replace(" ", "_")
    os.rename(filename, new_name)
    print(new_name)
    os.system("autosub  {} -S vi -D vi".format(new_name))