"""
Functions for the regression of the MOD Stochsim.

@author Mehmet Aziz Yirik
"""

import sys, os
import csv, math
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
import pandas as pd
import seaborn as sns
import random
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import linear_model
import os
import pywt
from sklearn.metrics import r2_score


def storeStochSimData(stochDataLength, numberOfMolecules, stochData):
    '''
    Storing time and concentration data into two separate
    arrays for the regression.
    '''

    concentrations=np.empty((stochDataLength, numberOfMolecules), float)
    stochasticTimeArray= stochData[:,0].astype(float) #self.stochData[:,0].astype(float)
    for i in range(0, stochDataLength):
        #self.concentrations[i]=self.stochData[i][2:].astype(float)
        concentrations[i] = stochData[i][2:].astype(float) #self.stochData[i][2:].astype(float)
    return stochasticTimeArray, concentrations


def filtering(interpConcentrations):
    filteringThreshold = int(interpConcentrations.shape[0] * 0.5)
    less_than_one = interpConcentrations < 1
    count_less_than_one = np.sum(less_than_one, axis=0)
    valid_columns = count_less_than_one < filteringThreshold
    newArray = interpConcentrations[:, valid_columns]
    return newArray


def getShortMemoryTargetData(interpolatedFlow):
    size=len(interpolatedFlow)
    target = np.zeros(size-2)
    for i in range(2, size):
        target[i-2]=interpolatedFlow[i-1]+2*interpolatedFlow[i-2]
    return target


def getLongMemoryTargetData(interpolatedFlow, tau=10):
    size=len(interpolatedFlow)
    target = np.zeros(int(size-int((3/2)*tau)))
    for i in range(int((3/2)*tau), size):
        target[i-int((3/2)*tau)]=interpolatedFlow[i-tau]+(1/2)*interpolatedFlow[i-int((3/2)*tau)]
    #print("tau data", tau, len(target), size)
    return target


def getErrorValues(y_train, y_train_pred, y_test, y_test_pred):
    '''
    Calculating the MSE value for training and test data.
    '''

    rmseTrain = np.sqrt(mean_squared_error(y_train, y_train_pred))
    diffTrain=(y_train.max() - y_train.min())
    nrmseTrain = rmseTrain / diffTrain if diffTrain!=0 else rmseTrain
    rmseTest = np.sqrt(mean_squared_error(y_test, y_test_pred))
    diffTest=(y_test.max() - y_test.min())
    nrmseTest = rmseTest / diffTest if diffTest!=0 else rmseTest
    return nrmseTest


def functionRegressionShortMemory(train_split, sample_values, interpConcentrations):
    regression = RidgeCV() #instead of 0 all started from 2 due to the shorttermmemory error.
    train_size = int(len(interpConcentrations) * train_split)
    regression.fit(interpConcentrations[0:train_size], sample_values[0:train_size])
    y_train_pred=regression.predict(interpConcentrations[0:train_size])
    y_test_pred=regression.predict(interpConcentrations[train_size:])
    error= getErrorValues(sample_values[0:train_size], y_train_pred, sample_values[train_size:], y_test_pred[:len(sample_values[train_size:])])
    #print("weights: ", regression.coef_)
    return error


class modRegression:
        
        def __init__(self):
            #self.scalingRate=1
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
            self.stochDataLength= len(stochData) #len(self.stochData)
            #self.stochasticTimeArray=np.zeros((self.stochDataLength))
            self.numberOfMolecules-=1
            #self.concentrations=np.empty(((self.stochDataLength),self.numberOfMolecules), float)
            self.dfconcentrations=pd.DataFrame()

        def interpolationWithExperimentalTimeSeries(self, stochasticTimeArray, concentrations):
            '''
            Interpolation with experimental data time series
            to be able to get the interpolated concentrations
            from the stochsim.
            '''
            maxTime=int(max(self.inputFlowTimes))
            maxTime+=1
            #states=1
            #self.experimentalTimes now static
            experimentalTimes = np.arange(0,  maxTime) #stochasticTimeArray when there is stepSize then it would be different but now nothing.
            self.getInterpolatedFlow(experimentalTimes)
            self.concentrationsLength=len(experimentalTimes) #len(stochasticTimeArray) #self.interpConcentrationsLength=len(self.experimentalTimes)
            interpConcentrations = np.empty((self.concentrationsLength, self.numberOfMolecules), float) #self.interpConcentrationsLength
            count=0
            for i in range(self.numberOfMolecules):
                temp=self.interpolation(
                        experimentalTimes, stochasticTimeArray, concentrations[:,i] #self.concentrations[:,i]
                )
                interpConcentrations[:, count] = temp
                #self.interpConcentrations[:,count]=temp
                count+=1

            interpConcentrations = filtering(interpConcentrations)
            self.setTrainTestIDX()
            return interpConcentrations

        def preprocessingStochData(self, stochasticTimeArray, concentrations):
            '''
            Interpolation with experimental data time series
            to be able to get the interpolated concentrations
            from the stochsim.
            '''
            self.getInterpolatedFlow(stochasticTimeArray)
            self.concentrationsLength=len(stochasticTimeArray)
            concentrations = filtering(concentrations)
            self.setTrainTestIDX()
            return concentrations

        def exponential_moving_average_2d(self, data, alpha):
            num_rows, num_cols = data.shape
            smoothed_data = np.zeros_like(data)

            for i in range(num_cols):
                smoothed_data[:, i] = self.exponential_moving_average(data[:, i], alpha)

            return smoothed_data
        
        def moving_average_1d(self, data, window_size):
            cumsum = np.cumsum(data)
            moving_avg = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
            return np.concatenate([cumsum[:window_size-1] / np.arange(1, window_size),
                                    moving_avg])
        
        def movingAverage(self, matrix):
            window_size = 20
            smoothed_matrix = np.apply_along_axis(self.moving_average_1d, axis=0, arr=matrix, window_size=window_size)
            return smoothed_matrix
        
        def wavelet_denoise2D(self, data, wavelet='db4', threshold_type='soft', threshold_scale=1.0):
            level=1000
            count=0
            numberOfColumns=data.shape[1]
            denoisedMatrix=np.empty((self.interpConcentrationsLength+1, numberOfColumns), float)
            for i in range(numberOfColumns):
                temp=self.wavelet_denoise1D(self.interpConcentrations[:,i], wavelet=wavelet, level=level, threshold_type=threshold_type, threshold_scale=threshold_scale)
                denoisedMatrix[:,count]=temp
                count+=1
            return denoisedMatrix
            
        def wavelet_denoise1D(self, data, wavelet='db4', level=100, threshold_type='soft', threshold_scale=1.0):
            coeffs = pywt.wavedec(data, wavelet, level=level)
            threshold = threshold_scale * np.sqrt(2 * np.log(len(data)))
            thresholded_coeffs = [pywt.threshold(coeff, threshold, mode=threshold_type) for coeff in coeffs]
            denoised_data = pywt.waverec(thresholded_coeffs, wavelet)
            return denoised_data
            
        def exponential_moving_average(self, data, alpha):
            smoothed_data = np.zeros_like(data)
            smoothed_data[0] = data[0]
            for t in range(1, len(data)):
                smoothed_data[t] = alpha * data[t] + (1 - alpha) * smoothed_data[t-1]

            return smoothed_data

        """
        Basic functions to deal with arrays
        """
        
        def setTrainTestIDX(self):
            '''
            Setting the test and train index values for the input array before
            the regression step.
            '''
            self.split_idx = math.floor(self.concentrationsLength* self.train_split)
            self.train_idx = [*range(0, self.split_idx)]
            self.test_idx  = [*range(self.split_idx, self.concentrationsLength)]
            
        def getInterpolatedFlow(self, stochasticTimeArray):
            '''
            The interpolated inflow as the input for basic functions.
            The original target data is already fed with the same input flow as Huck experiments.
            Here, the interpolated flow is used as input
            '''
            '''
            plt.plot(self.moleculeFlow, marker='o')
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.title('molecule inflow')
            plt.grid(True)
            plt.show()
            '''
            self.interpolatedFlow=np.interp(stochasticTimeArray, self.inputFlowTimes, self.moleculeFlow)
            
        def generateSineData(self, amplitude=0.1, frequency=1):
            '''
            Generate sine data as a target data for interpolatedFlow input.
            '''
            targetSineData = amplitude * np.sin(frequency * self.interpolatedFlow)
            return targetSineData
                
        def generateLinearData(self, slope=2, intercept=3):
            '''
            Generate linear data as a target data for interpolatedFlow input.
            '''
            targetLinearData = slope * self.interpolatedFlow + intercept
            return targetLinearData
        
        def plotStepFunction(self, timeSeconds, stepFunctionValues):
            '''
            Plot the step function without connecting lines
            '''
            plt.step(timeSeconds, stepFunctionValues, where='post', color='blue', label='Step Function')

            for i in range(1, len(stepFunctionValues)):
                plt.vlines(timeSeconds[i], stepFunctionValues[i - 1], stepFunctionValues[i], linestyle='dashed', color='red')

            plt.xlabel('Time (seconds)')
            plt.ylabel('Y-axis')
            plt.legend()
            plt.show()

        def generateStepFunction(self):
            stepThreshold = np.mean(self.interpolatedFlow)
            maxLen=int(len(self.interpolatedFlow))
            target_step = np.where(self.interpolatedFlow >= stepThreshold, 1, 0)
            return target_step

        def standardScaler(self):
            '''
            Performing standard scaling on the interpolated concentrations array
            before the regression step.
            '''
            scaler=StandardScaler()
            data=self.interpConcentrations.reshape(-1, self.interpConcentrations.shape[1])
            scaler.fit(data[self.train_idx])
            self.scaled_data = scaler.transform(data)
                
        def interpolation(self, inputXAxis, xAxis, yAxis):
            return np.interp(inputXAxis, xAxis, yAxis)
           
        def functionRegression(self, sample_values):
            regression = RidgeCV()
            N_future = 30
            N_skip = 100
            N_train = 500+N_skip
            N_test = 1000+N_train
            regression.fit(self.scaled_data[N_skip:N_train], sample_values[N_skip+N_future:N_train+N_future])
            weights = regression.coef_
            y_train_pred=regression.predict(self.scaled_data[N_skip:N_train])
            y_test_pred=regression.predict(self.scaled_data[N_train:N_test])
            maxLen=int(len(self.interpolatedFlow))
            trainIdx=int(len(y_train_pred))
            predictLen=int(len(y_test_pred))
            getErrorValues(sample_values[N_skip+N_future:N_train+N_future], y_train_pred, sample_values[N_train+N_future:N_test+N_future], y_test_pred)
            return y_test_pred
        
        def functionRegressionOneByOne(self, sample_values):
            regression = RidgeCV()
            N_future = 2
            N_skip = 5
            N_train = 40+N_skip
            N_test = 45+N_train
            columns=sample_values.shape[1]
            errors=0.0
            for i in range(columns):
                regression.fit(self.scaled_data[N_skip:N_train], sample_values[N_skip+N_future:N_train+N_future, i])
                weights = regression.coef_
                y_train_pred=regression.predict(self.scaled_data[N_skip:N_train])
                y_test_pred=regression.predict(self.scaled_data[N_train:N_test])
                maxLen=int(len(self.interpolatedFlow))
                trainIdx=int(len(y_train_pred))
                predictLen=int(len(y_test_pred))
                error= getErrorValues(sample_values[N_skip + N_future:N_train + N_future, i], y_train_pred, sample_values[N_train + N_future:N_test + N_future, i], y_test_pred)
                errors+=error
            return y_test_pred

        def plotTestPrediction(self, sample_values, regression_data):
            N_future = 30
            N_skip = 100
            N_train = 500+N_skip
            N_test = 1000+N_train
            length=len(regression_data)
            columns=regression_data.shape[1]

            for i in range(columns):
                plt.plot(self.experimentalTimes[:N_train], sample_values[:N_train, i], label="no skip", color="red")
                plt.axvline(x=N_skip, color='r', linestyle='--', linewidth=1)
                plt.plot(self.experimentalTimes[N_skip:N_train], sample_values[N_skip:N_train, i], label="no delay", color="blue")
                plt.plot(self.experimentalTimes[N_skip+N_future:N_train+N_future], sample_values[N_skip+N_future:N_train+N_future, i], label="with delay", color="green")
                plt.axvline(x=N_skip+N_future, color='r', linestyle='--', linewidth=1)
                plt.axvline(x=N_train, color='r', linestyle='--', linewidth=1)
                plt.xlabel('Time')
                plt.ylabel('Amplitude')
                plt.legend()
                plt.show()
                plt.plot(self.experimentalTimes[N_train+N_future:N_test+N_future], regression_data[:, i], label='Predicted Sine Curve', color='green')
                plt.plot(self.experimentalTimes[N_train+N_future:N_test+N_future], sample_values[N_train+N_future:N_test+N_future, i], label='Actual Sine Curve', color='blue')
                plt.xlabel('Time')
                plt.ylabel('Amplitude')
                plt.legend()
                plt.title('Actual vs Predicted Sine Curve')
                plt.show()
        
        def plotTestPredictionMemory(self, sample_values, regression_data):
            train_size = int(len(self.scaled_data) * self.train_split)
            plt.plot(self.experimentalTimes[train_size:-3], regression_data[:-3], label='Predicted Sine Curve', color='green')
            plt.plot(self.experimentalTimes[train_size:-3], sample_values[train_size:], label='Actual Sine Curve', color='blue')
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.title('Actual vs Predicted Sine Curve')
            plt.show()
            
        def plotTestSinglePrediction(self, sample_values, regression_data):
            N_future = 2
            N_skip = 5
            N_train = 40
            N_test = 45
            length=len(regression_data)
            plt.plot(self.experimentalTimes[:N_train], sample_values[:N_train], label="no skip", color="red")
            plt.axvline(x=N_skip, color='r', linestyle='--', linewidth=1)
            plt.plot(self.experimentalTimes[N_skip:N_train], sample_values[N_skip:N_train], label="no delay", color="blue")
            plt.plot(self.experimentalTimes[N_skip+N_future:N_train+N_future], sample_values[N_skip+N_future:N_train+N_future], label="with delay", color="green")
            plt.axvline(x=N_skip+N_future, color='r', linestyle='--', linewidth=1)
            plt.axvline(x=N_train, color='r', linestyle='--', linewidth=1)
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.show()
            plt.plot(self.experimentalTimes[N_train+N_future:N_test+N_future], regression_data[:], label='Predicted Sine Curve', color='green')
            plt.plot(self.experimentalTimes[N_train+N_future:N_test+N_future], sample_values[N_train+N_future:N_test+N_future], label='Actual Sine Curve', color='blue')
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.title('Actual vs Predicted Sine Curve')
            plt.show()
            
        def create_lagged_features(self, df, col, delay_time=5):
            '''
            Function to create lagged features for a variable
            '''
            for i in range(1, delay_time + 1):
                lagged_col = df[col].shift(i).to_frame(f'{col}_lag_{i}')
                df = pd.concat([df, lagged_col], axis=1)
            return df

        def getLaggedFeaturesRegressionMatrix(self, delay_time=5):
            df_simulation_output_lagged = pd.DataFrame()
            for col in self.dfconcentrations.columns:
                lagged_col = self.create_lagged_features(self.dfconcentrations[[col]], col, delay_time)
                if df_simulation_output_lagged.empty:
                    df_simulation_output_lagged = lagged_col
                else:
                    df_simulation_output_lagged = pd.concat([df_simulation_output_lagged, lagged_col], axis=1)
            return df_simulation_output_lagged
        
        def regressionForecasting(self):
            N_future = 12
            N_skip = 60
            N_train = 180 + N_skip
            N_test = 1200 + N_train
            index = U_lorenz.index
            train_data = X_lorenz.loc[index].values

            reg = linear_model.Ridge(alpha=0.5*1e-4)
            reg.fit(train_data[N_skip:N_train], U_lorenz[N_skip+N_future:N_train+N_future])

            U_train = reg.predict(train_data[N_skip:N_train])
            U_pred = reg.predict(train_data[N_train:N_test])
            train_mse = nmse(U_train, U_lorenz[N_skip+N_future:N_train+N_future])
            test_mse = nmse(U_pred, U_lorenz[N_train+N_future:N_test+N_future])
            
        def regressionForIdenticalTargetData(self):
            '''
            When input data and target data are identical.
            '''
            targetData= np.array(self.interpolatedFlow)
            regression_data=self.functionRegressionOneByOne(targetData)
        
        def regressionShortMemoryTargetData(self, interpConcentrations):
            targetData= getShortMemoryTargetData(self.interpolatedFlow)
            regression_data= functionRegressionShortMemory(self.train_split, targetData, interpConcentrations)
            return regression_data

        def regressionLongMemoryTargetData(self, interpConcentrations, tauValue=10):
            '''
            plt.plot(self.interpolatedFlow, marker='o')
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.title('interpolated inflow')
            plt.grid(True)
            plt.show()
            '''
            delay = int((3 / 2) * tauValue)
            targetData= getLongMemoryTargetData(self.interpolatedFlow, tau=tauValue)
            '''
            print("tau: ", tauValue)
            print(targetData)
            plt.plot(targetData, marker='o')
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.title('target data')
            plt.grid(True)
            plt.show()
            print("len of target and input: ", len(targetData), len(interpConcentrations))
            '''
            regression_data= functionRegressionShortMemory(self.train_split, targetData, interpConcentrations[delay:])
            return regression_data
            
        def regressionForecastingDelayTime(self, delay_time=0, dataName='mackey_glass'):
            df_simulation_output_lagged = self.getLaggedFeaturesRegressionMatrix(delay_time=delay_time)
            df_combined = pd.concat([df_simulation_output_lagged.iloc[delay_time:], pd.Series(self.interpolatedFlow[delay_time:], name=dataName)], axis=1)
            df_combined = df_combined.dropna()

            train_size = int(len(df_combined) * self.train_split)
            train, test = df_combined[:train_size], df_combined[train_size:]

            X_train = train.drop(dataName, axis=1)
            y_train = train[dataName]
            X_test = test.drop(dataName, axis=1)
            y_test = test[dataName]

            ridge_model = RidgeCV()
            ridge_model.fit(X_train, y_train)

            predictions = ridge_model.predict(X_test)

            test_error = mean_squared_error(y_test, predictions)
                        
            plt.plot(self.experimentalTimes[-len(predictions):], y_test, label='Original %s Data' %dataName)
            plt.plot(self.experimentalTimes[-len(predictions):], predictions, label='Predicted %s Data' %dataName, linestyle='--')
            plt.xlabel('Time')
            plt.ylabel('%s Data' %dataName)
            plt.legend()
            plt.show()
        
        def retrieveDataWithoutMilliseconds(self, maximumTime):
            second=0
            count=0
            self.cleanedConcentrations=np.empty((maximumTime, self.numberOfMolecules), float)
            while second < maximumTime:
                for i in range(0, self.stochDataLength):
                    if second==int(self.stochasticTimeArray[i]):
                        self.cleanedConcentrations[second,:]=self.concentrations[i, :]
                        second+=1
            self.concentrations=self.cleanedConcentrations
            
        def runTimeSeriesPredictionTask(self, trainSplit=0.7, stepSize=3):
            self.train_split=trainSplit
            self.initializeArrays()
            storeStochSimData()
            self.interpolationWithExperimentalTimeSeries(stepSize=stepSize)
            self.regressionForecastingDelayTime()
        
        def runBasicTask(self, trainSplit=0.7, stepSize=3):
            self.train_split=trainSplit
            self.initializeArrays()
            timeArray= storeStochSimData()
            self.interpolationWithExperimentalTimeSeries(stepSize=stepSize)
            self.regressionForIdenticalTargetData()
        
        def runShortMemoryTask(self, stochData, trainSplit=0.7):
            self.train_split=trainSplit
            self.initializeArrays(stochData)
            stochasticTimeArray, concentrations = storeStochSimData(self.stochDataLength, self.numberOfMolecules,
                                                                    stochData)
            interpConcentrations = self.preprocessingStochData(stochasticTimeArray, concentrations)
            thresholdSize = int(self.numberOfMolecules / 2)
            floatError = float(-10.0)
            if interpConcentrations.shape[1] < thresholdSize:
                # if self.interpConcentrations.shape[1]<thresholdSize:
                floatError = float(20)  # float(1e6)#(-10.0)
            else:
                floatError =  self.regressionShortMemoryTargetData(interpConcentrations)
            return floatError
            
        def runLongMemoryTask(self, stochData, trainSplit=0.7, tau=10):
            self.train_split=trainSplit
            self.initializeArrays(stochData)
            stochasticTimeArray, concentrations= storeStochSimData(self.stochDataLength, self.numberOfMolecules, stochData)
            thresholdSize = int(self.numberOfMolecules / 5)
            interpConcentrations=self.interpolationWithExperimentalTimeSeries(stochasticTimeArray, concentrations) #self.preprocessingStochData(stochasticTimeArray, concentrations)
            zero_columns = np.all(interpConcentrations == 0, axis=0)
            num_zero_columns = np.sum(zero_columns)
            if num_zero_columns !=0 : print(f"Number of all-zero columns: {interpConcentrations.shape}, {num_zero_columns}")
            floatError=float(-10.0)
            '''
            if interpConcentrations.shape[1]<thresholdSize:
            #if self.interpConcentrations.shape[1]<thresholdSize:
                floatError=float(20) #float(1e6)#(-10.0)
            else:
            '''
            floatError=self.regressionLongMemoryTargetData(interpConcentrations, tauValue=tau)
            return floatError
