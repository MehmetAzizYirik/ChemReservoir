from scripts.logFileSetting import setup_logger
from scripts.chemReservoir import chemReservoir


loggerShortMemory = setup_logger("shortMemoryTaskLoggerFile")
shortMemoryTask = chemReservoir(1, loggerShortMemory, memoryTask="short", runTime=10, repetition=2, networkOptMaxTime=200) #run time 50, max time 500
shortMemoryTask.INT_BOUNDS = {
            'nodeValues': (50, 300),
            'moleculeInflow': (50, 200),
            'chordDistance': (5, 25),
            'chordStep': (5, 25)
        }
shortMemoryResult=shortMemoryTask.run(max_time=1000) #10000
print("Short memory results", shortMemoryResult)

loggerLongMemory = setup_logger("longMemoryTaskLoggerFile")
longMemoryTask = chemReservoir(1, loggerLongMemory, memoryTask="long", tau=6, runTime=50, repetition=2, networkOptMaxTime=200) #run time 50, max time 500
longMemoryTask.INT_BOUNDS = {
            'nodeValues': (50, 300),
            'moleculeInflow': (50, 200),
            'chordDistance': (5, 25),
            'chordStep': (5, 25)
        }
longMemoryResult=longMemoryTask.run(max_time=500) #10000
print("Long memory results", longMemoryResult)

