import argparse
import os
# take path from arguement
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str,
                    help='path to input data')
args = parser.parse_args()
sets = ["train", "test", "valid"]
for set in sets:
    fileNames = [*os.listdir(args.input+set+"/labels")]
    print('There are {} labels in the dataset'.format(len(fileNames)))
    for fileName in fileNames:
        fileIn = open(args.input+set+"/labels/"+fileName, "r")
        if not os.path.exists(args.input+set+"/labelsNew/"):
            os.mkdir(args.input+set+"/labelsNew/")
        fileOut = open(args.input+set+"/labelsNew/"+fileName, "w")
        lines = fileIn.readlines()
        for line in lines:
            newline = line
            if newline[0] == '2':
                newline = '0'+newline[1:]
            elif newline[0] == '0':
                newline = '1'+newline[1:]
            fileOut.write(newline)
        fileIn.close()