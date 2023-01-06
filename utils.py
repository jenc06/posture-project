import os
import glob

if __name__ == "__main__":
    # remove timestamps from all the existing files
    filenames = glob.glob("./data/good/*.csv") + glob.glob("./data/mild/*.csv") + glob.glob("./data/bad/*.csv")
    print(filenames[0])

    for file in filenames:
        folder = os.path.split(file)[0]
        filename = os.path.split(file)[-1]
        start_ndx = filename.find('_ts_')

        if start_ndx >= 0:
            new_filename = filename[:start_ndx] + filename[start_ndx+22:]
            new_filename = os.path.join(folder, new_filename.replace(":", "_"))
            print(f"renaming {file} to {new_filename}")

            os.rename(file, new_filename)
