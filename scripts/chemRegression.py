import math
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error



def storeStochSimData(stochDataLength, numberOfMolecules, stochData):
    """
    Storing stochastic simulation data in proper array form separate for the molecule count and time arrays.
    Also not excluding the food molecule data.

    Args:
        stochDataLength (int): length of the array
        numberOfMolecules (int): number of molecules for the array size.
        stochData (np.array): the molecule count from the stochastic simulation.
    Returns:
        tuple: (np.array, np.array) the updated stochastic simulation data.
    """

    concentrations=np.empty((stochDataLength, numberOfMolecules), float)
    stochasticTimeArray= stochData[:,0].astype(float)
    for i in range(0, stochDataLength):
        concentrations[i] = stochData[i][2:].astype(float)
    return stochasticTimeArray, concentrations


def filtering(interpConcentrations):
    """
    Filtering the columns half of which filled by zeros.

    Args:
        interpConcentrations (np.array): interpolated molecule count.

    Returns:
        np.array: the filtering interpolated stochastic simulation data.
    """
    filteringThreshold = int(interpConcentrations.shape[0] * 0.5)
    less_than_one = interpConcentrations < 1
    count_less_than_one = np.sum(less_than_one, axis=0)
    valid_columns = count_less_than_one < filteringThreshold
    newArray = interpConcentrations[:, valid_columns]
    return newArray


def getShortMemoryTargetData(interpolatedFlow):
    """
    Generating target data for short term memory task.

    Args:
        interpolatedFlow (np.array): interpolated input signal.

    Returns:
        np.array: target data for short term memory task.
    """
    size=len(interpolatedFlow)
    target = np.zeros(size-2)
    for i in range(2, size):
        target[i-2]=interpolatedFlow[i-1]+2*interpolatedFlow[i-2]
    return target


def getLongMemoryTargetData(interpolatedFlow, tau=10):
    """
    Generating target data for long term memory task.

    Args:
        interpolatedFlow (np.array): interpolated input signal.

    Returns:
        np.array: target data for long term memory task.
    """
    size=len(interpolatedFlow)
    target = np.zeros(int(size-int((3/2)*tau)))
    for i in range(int((3/2)*tau), size):
        target[i-int((3/2)*tau)]=interpolatedFlow[i-tau]+(1/2)*interpolatedFlow[i-int((3/2)*tau)]
    return target


def getErrorValues(y_train, y_train_pred, y_test, y_test_pred):
    """
    Calculating normalized root mean square error (NRMSE) value for training and test data.

    Args:
        y_train (np.array): original training target
        y_train_pred (np.array): predicted training target
        y_test (np.array): original test target
        y_test_pred (np.array): predicted test target
    Returns:
        float: NRMSE
    """

    rmseTrain = np.sqrt(mean_squared_error(y_train, y_train_pred))
    diffTrain=(y_train.max() - y_train.min())
    nrmseTrain = rmseTrain / diffTrain if diffTrain!=0 else rmseTrain
    rmseTest = np.sqrt(mean_squared_error(y_test, y_test_pred))
    diffTest=(y_test.max() - y_test.min())
    nrmseTest = rmseTest / diffTest if diffTest!=0 else rmseTest
    return nrmseTest


def regressionMemoryTask(train_split, sample_values, interpConcentrations):
    """
    Regression function for memory tasks, short or long term memory.

    Args:
        train_split (float): splitting the target data into train and test, default 0.70.
        sample_values (np.array): target data.
        interpConcentrations (np.array): the interpolated molecule count from the stochastic simulation.

    Returns:
        float: NRMSE
    """

    regression = RidgeCV()
    train_size = int(len(interpConcentrations) * train_split)
    regression.fit(interpConcentrations[0:train_size], sample_values[0:train_size])
    y_train_pred=regression.predict(interpConcentrations[0:train_size])
    y_test_pred=regression.predict(interpConcentrations[train_size:])
    error= getErrorValues(sample_values[0:train_size], y_train_pred, sample_values[train_size:], y_test_pred[:len(sample_values[train_size:])])
    return error


class chemReg:
        
        def __init__(self):
            """
            Initializing the instances for chemRegression class.
            """
            self.experimentalTimes=None
            self.numberOfMolecules=None
            self.modStochSim = None
            self.interpConcentrations=None
            self.train_split = None
            self.split_idx = None
            self.train_idx = None
            self.test_idx  = None
            self.interpConcentrationsLength=None
            self.concentrationsLength = None
            self.scaled_data=None
            self.input=None
            self.sineInput=None
            self.sineInputTime=None
            self.interpolatedFlow=None
            self.firstMoleculeFlow=None
            self.inputFlowTimes=None
            self.U_lorenz=None
            self.X_lorenz=None
        
        def initializeArrays(self, stochData):
            """
            Initializing stochastic simulated related instances variables.
            """

            self.stochDataLength= len(stochData)
            self.numberOfMolecules-=1

        def interpolationWithExperimentalTimeSeries(self, stochasticTimeArray, concentrations):
            """
            Interpolation with experimental data time series to be able to get the interpolated concentrations
            from the stochsim data.

            Args:
                stochasticTimeArray (np.array): stochastic simulation run times
                concentrations (np.array): molecule counts over time.

            Returns:
                np.array: interpolated molecule counts from stochastic simulation.
            """
            maxTime=int(max(self.inputFlowTimes))
            maxTime+=1
            experimentalTimes = np.arange(0,  maxTime)
            self.getInterpolatedFlow(experimentalTimes)
            self.concentrationsLength=len(experimentalTimes)
            interpConcentrations = np.empty((self.concentrationsLength, self.numberOfMolecules), float)
            count=0
            for i in range(self.numberOfMolecules):
                temp=self.interpolation(
                        experimentalTimes, stochasticTimeArray, concentrations[:,i]
                )
                interpConcentrations[:, count] = temp
                count+=1

            interpConcentrations = filtering(interpConcentrations)
            self.setTrainTestIDX()
            return interpConcentrations

        def preprocessingStochData(self, stochasticTimeArray, concentrations):
            """
            Preprocessing the stochastic simulation data.

            Args:
                stochasticTimeArray (np.array): stochastic simulation run times
                concentrations (np.array): molecule counts over time.

            Returns:
                np.array: preprocessed molecule count.
            """

            self.getInterpolatedFlow(stochasticTimeArray)
            self.concentrationsLength=len(stochasticTimeArray)
            concentrations = filtering(concentrations)
            self.setTrainTestIDX()
            return concentrations

        def setTrainTestIDX(self):
            """
            Setting the test and train index values for the input array before
            the regression step.
            """
            self.split_idx = math.floor(self.concentrationsLength* self.train_split)
            self.train_idx = [*range(0, self.split_idx)]
            self.test_idx  = [*range(self.split_idx, self.concentrationsLength)]
            
        def getInterpolatedFlow(self, stochasticTimeArray):
            """
            The interpolated inflow as the input for basic functions.
            The original target data is already fed with the same input flow as Huck experiments.
            Here, the interpolated flow is used as input
            """
            self.interpolatedFlow=np.interp(stochasticTimeArray, self.inputFlowTimes, self.moleculeFlow)
                
        def interpolation(self, inputXAxis, xAxis, yAxis):
            """
            Interpolation function, used in input signal and stochastic simulation
            preprocessing steps.

            Args:
                inputXAxis (np.array): target input time series
                xAxis (np.array): original time series
                yAxis (np.array): original molecule counts over time.

            Returns:
                np.array: interpolated molecule count.
            """
            return np.interp(inputXAxis, xAxis, yAxis)

        def regressionShortMemoryTargetData(self, interpConcentrations):
            """
            Regression function for short memory task.

            Args:
                interpConcentrations (np.array): interpolated molecule count from stochastic simulation.

            Returns:
                float: NRMSE.
            """
            targetData= getShortMemoryTargetData(self.interpolatedFlow)
            error= regressionMemoryTask(self.train_split, targetData, interpConcentrations)
            return error

        def regressionLongMemoryTargetData(self, interpConcentrations, tauValue=10):
            """
            Regression function for long memory task.

            Args:
                interpConcentrations (np.array): interpolated molecule count from stochastic simulation.

            Returns:
                float: NRMSE.
            """
            delay = int((3 / 2) * tauValue)
            targetData= getLongMemoryTargetData(self.interpolatedFlow, tau=tauValue)
            error= regressionMemoryTask(self.train_split, targetData, interpConcentrations[delay:])
            return error

        def runShortMemoryTask(self, stochData, trainSplit=0.7):
            """
            Regression main function for short memory task.

            Args:
                stochData (np.array): molecule count from stochastic simulation.

            Returns:
                float: NRMSE.
            """
            self.train_split = trainSplit
            self.initializeArrays(stochData)
            stochasticTimeArray, concentrations = storeStochSimData(self.stochDataLength, self.numberOfMolecules, stochData)
            interpConcentrations = self.interpolationWithExperimentalTimeSeries(stochasticTimeArray, concentrations)
            zero_columns = np.all(interpConcentrations == 0, axis=0)
            num_zero_columns = np.sum(zero_columns)
            if num_zero_columns != 0: print(f"Number of all-zero columns: {interpConcentrations.shape}, {num_zero_columns}")
            floatError = self.regressionShortMemoryTargetData(interpConcentrations)
            return floatError
            
        def runLongMemoryTask(self, stochData, trainSplit=0.7, tau=10):
            """
            Regression main function for long memory task.

            Args:
                stochData (np.array): molecule count from stochastic simulation.

            Returns:
                float: NRMSE.
            """
            self.train_split=trainSplit
            self.initializeArrays(stochData)
            stochasticTimeArray, concentrations= storeStochSimData(self.stochDataLength, self.numberOfMolecules, stochData)
            interpConcentrations=self.interpolationWithExperimentalTimeSeries(stochasticTimeArray, concentrations)
            zero_columns = np.all(interpConcentrations == 0, axis=0)
            num_zero_columns = np.sum(zero_columns)
            if num_zero_columns !=0 : print(f"Number of all-zero columns: {interpConcentrations.shape}, {num_zero_columns}")
            floatError=self.regressionLongMemoryTargetData(interpConcentrations, tauValue=tau)
            return floatError
