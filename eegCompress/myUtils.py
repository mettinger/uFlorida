# basic convenience utilities

import datetime
import pytz
timeZone = pytz.timezone('America/Los_Angeles')

# print to console and also log file to deal with notebook disconnects
def myPrint(stringToPrint):
    print(stringToPrint)
    with open('/home/jmark.ettinger/github/uFlorida/eegCompress/notebookLog.txt', 'a') as file:
        timestamp = str(datetime.datetime.now().astimezone(timeZone).strftime('%m-%d %H:%M'))
        file.write(timestamp + ": " + stringToPrint + "\n")