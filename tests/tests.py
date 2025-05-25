from scripts.logFileSetting import setup_logger
from scripts.chemReservoir import chemReservoir

loggerShortMemory = setup_logger("shortMemoryTaskLoggerFile")
shortMemoryTask = chemReservoir(1, loggerShortMemory, memoryTask="short", runTime=50, repetition=2)
shortMemoryTask.INT_BOUNDS = {
            'nodeValues': (50, 300),
            'moleculeInflow': (50, 200),
            'chordDistance': (5, 25),
            'chordStep': (5, 25)
        }
shortMemoryResult=shortMemoryTask.main()

loggerLongMemory = setup_logger("longMemoryTaskLoggerFile")
longMemoryTask = chemReservoir(1, loggerLongMemory, memoryTask="long", tau=6, runTime=50, repetition=2)
longMemoryTask.INT_BOUNDS = {
            'nodeValues': (50, 300),
            'moleculeInflow': (50, 200),
            'chordDistance': (5, 25),
            'chordStep': (5, 25)
        }
longMemoryResult=longMemoryTask.main()

