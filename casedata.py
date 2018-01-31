#!/usr/bin/python3
# -*- coding: utf-8 -*-
import re
import sys
import os
import struct
import math
import warnings
import numpy as np
import pandas as pd
import scipy.io as sio


class CaseData:
    """
    CaseData class for SKLOE
    Please read the readme file.
    """

    def __init__(self, filename, lam=1, sseg='all'):
        """
        @param: filename - input file name
        @param: lam      - scale ratio
        @param: sSeg    - selected segment
        @param: debug    - if debug
        """

        if os.path.exists(filename):
            self.filename = filename
        else:
            warnings.warn(
                "File {:s} does not exist. Breaking".format(filename))
            sys.exit()

        self.lam = lam
        self.fs = 0
        self.chN = 0
        self.segN = 0
        self.scale = 'model'
        if not (isinstance(sseg, int) or sseg == 'all'):
            print("Error: Input 'sseg' is illegal (should be int or 'all').")
            raise

        self._read(sseg)

    def _read(self, sseg):
        """read the *.out file"""

        print('Reading file {}...'.format(self.filename), end='')

        with open(self.filename, 'rb') as fIn:
            # file head
            fmtstr = '=hhlhh2s2s240s'
            buf = fIn.read(256)
            if not buf:
                warnings.warn("Reading data file {0} failed, exiting...".format(
                    self.filename))
            tmp = struct.unpack(fmtstr, buf)
            index, self.chN, self.fs, self.segN =\
                tmp[0], tmp[1], tmp[3], tmp[4]
            dateMonth, dateDay = tmp[5].decode('utf-8'), tmp[6].decode('utf-8')
            # date mm/dd
            self.date = '{0:2s}/{1:2s}'.format(dateMonth, dateDay)
            # global description
            self.desc = tmp[7].decode('utf-8').rstrip()

            # read the name of each channel
            fmtstr = self.chN * '16s'
            buf = struct.unpack(fmtstr, fIn.read(self.chN * 16))
            chName = []
            chIdx = []
            for idx, item in enumerate(buf):
                chName.append(buf[idx].decode('utf-8').rstrip())
                chIdx.append('Ch{0:02d}'.format(idx + 1))
            # read the unit of each channel
            fmtstr = self.chN * '4s'
            buf = struct.unpack(fmtstr, fIn.read(self.chN * 4))
            chUnit = []
            for idx, item in enumerate(buf):
                chUnit.append(buf[idx].decode('utf-8').rstrip())

            # read the coefficient of each channel
            fmtstr = '=' + self.chN * 'f'
            buf = fIn.read(self.chN * 4)
            chCoef = struct.unpack(fmtstr, buf)

            # read the id of each channel, if there are
            if (index < -1):
                fmtstr = '=' + self.chN * 'h'
                buf = fIn.read(self.chN * 2)
                chIdx = struct.unpack(fmtstr, buf)
            else:
                chIdx = list(range(1, self.chN + 1))

            chInfoDict = {'Index': chIdx,
                          'Name': chName,
                          'Unit': chUnit,
                          'Coef': chCoef}

            column = ['Name', 'Unit', 'Coef']
            self.chInfo = pd.DataFrame(chInfoDict, index=chIdx, columns=column)

            # sampNum[i] is the number of samples in segment i
            sampNum = [0 for i in range(self.segN)]
            # segInfo[i] is the information of the segment i
            # [seg_index, segChN, sampNum, ds, s, min, h, desc]
            segInfo = [[] for i in range(self.segN)]
            # seg_satistic[i] is the satistical values of the segment i
            # [mean[segChN], std[segChN], max[segChN], min[segChN]]
            segStatis = [[] for i in range(self.segN)]
            # dataRaw[i] are the data of the segment i
            dataRaw = [[] for i in range(self.segN)]
            # note for each segment
            note = [[] for i in range(self.segN)]

            # read data of each segment
            for iseg in range(self.segN):
                # jump over the blank section
                p_cur = fIn.tell()
                fIn.seek(128 * math.ceil(p_cur / 128))

                # read segment informantion
                fmtstr = '=hhlBBBBBBBB240s'
                buf = fIn.read(256)
                segInfo[iseg] = struct.unpack(fmtstr, buf)

                segChN = segInfo[iseg][1]
                sampNum[iseg] = segInfo[iseg][2] - 5
                note[iseg] = segInfo[iseg][11].decode('utf-8').rstrip()

                # read the statiscal values of each channel
                fmtstr = '=' + segChN * 'h' + segChN * 'f' + segChN * 2 * 'h'
                buf = fIn.read(segChN * (2 * 3 + 4))
                segStatis[iseg] = struct.unpack(fmtstr, buf)

                # read the data in each channel
                dataRaw[iseg] = np.frombuffer(fIn.read(sampNum[iseg] *
                                                       segChN * 2),
                                              dtype=np.int16).reshape(
                    (sampNum[iseg], segChN))

        # segment information
        segType = []
        startTime = []
        stopTime = []
        index = []
        duration = []
        for n in range(self.segN):
            # type: 0 - 采样段，1 - 前标定段，2 - 后标定段
            segType.append(segInfo[n][0])
            startTime.append('{0:02d}:{1:02d}:{2:02d}.{3:1d}'.format(
                segInfo[n][6], segInfo[n][5], segInfo[n][4], segInfo[n][3]))
            stopTime.append('{0:02d}:{1:02d}:{2:02d}.{3:1d}'.format(
                segInfo[n][10], segInfo[n][9], segInfo[n][8], segInfo[n][7]))
            index.append('Seg{0:2d}'.format(n))
            duration.append('{0:8.1f}s'.format((sampNum[n] - 1) / self.fs))
        segInfoDict = {'Type': segType,
                       'Start': startTime,
                       'Stop': stopTime,
                       'Duration': duration,
                       'N sample': sampNum,
                       'Note': note}
        column = ['Type', 'Start', 'Stop', 'Duration', 'N sample', 'Note']
        segInfo = pd.DataFrame(segInfoDict, index=index, columns=column)

        # convert the statistics into data matrix
        self.segStatis = []
        for n in range(self.segN):
            segStatis_temp = np.reshape(
                np.array(segStatis[n], dtype='float64'), (4, self.chN)).transpose()
            for m in range(self.chN):
                segStatis_temp[m] *= chCoef[m]
            column = ['Mean', 'STD', 'Max', 'Min']
            self.segStatis.append(pd.DataFrame(
                segStatis_temp, index=chName, columns=column))
            self.segStatis[n]['Unit'] = chUnit

        # convert the dataRaw into data matrix
        self.data = [[] for i in range(self.segN)]
        for n in range(self.segN):
            data_temp = dataRaw[n].astype('float64')
            for m in range(self.chN):
                data_temp[:, m] *= chCoef[m]
            index = np.arange(segInfo['N sample'].iloc[n]) / self.fs
            self.data[n] = pd.DataFrame(
                data_temp, index=index, columns=chName, dtype='float64')

        if sseg == 'all':
            self.segInfo = segInfo
        else:
            self.data = [self.data[sseg]]
            self.segInfo = segInfo[sseg:sseg + 1]
            self.segN = 1
            self.segStatis = [self.segStatis[sseg]]

        print(" Done!")

    def _write(self, filename, sseg=0):
        """write the *.out file"""

        if not filename.endswith('.out'):
            filename += '.out'

        # write data of selected segment(s)
        if sseg == 'all':
            sseg = list(range(self.segN))
        elif type(sseg) is int:
            sseg = [sseg]

        print('Saving segment(s) No. {} in file {}'.format(sseg, filename))

        with open(filename, 'wb') as fOut:
            # file head
            datemmdd = self.date.split('/')
            buf = struct.pack('=hhlhh', -2, self.chN, 0x0d, self.fs, len(sseg)) \
                + struct.pack('2s2s240s', datemmdd[0].encode('utf-8'),
                              datemmdd[1].encode('utf-8'),
                              self.desc.encode('utf-8')).replace(b'\x00', b' ')
            if fOut.write(buf) != 256:
                print("Error when saving out file!")
                raise

            # write the name of each channel
            fOut.write(struct.pack(self.chN * '16s',
                                   *[self.chInfo['Name'].iloc[i].encode('utf-8')
                                     for i in range(self.chN)]).replace(b'\x00', b' '))

            # write the unit of each channel
            fOut.write(struct.pack(self.chN * '4s',
                                   *[self.chInfo['Unit'].iloc[i].encode('utf-8')
                                     for i in range(self.chN)]).replace(b'\x00', b' '))

            # write the coefficient of each channel
            # calculate new coefficient for each channel
            # max short: 32767
            chMagMax = np.amax(np.array(
                [np.amax(abs(self.data[i].values), axis=0) for i in sseg]),
                axis=0)
            chCoef_ = (chMagMax / 32767).astype(np.float32)
            fOut.write(struct.pack('=' + self.chN * 'f', *chCoef_))

            # write the index of each channel (write case -2)
            fOut.write(struct.pack('=' + self.chN * 'h', *self.chInfo.index))

            for iseg in sseg:
                # jump over the blank section
                p_cur = fOut.tell()
                fOut.seek(128 * math.ceil(p_cur / 128))

                # write segment informantion
                fOut.write(struct.pack('=h', self.segInfo['Type'][iseg]))
                fOut.write(struct.pack('=h', self.chN))
                fOut.write(struct.pack(
                    '=l', self.segInfo['N sample'][iseg] + 5))
                fOut.write(struct.pack(8 * 'B', *list(
                    map(int, re.split(':|\.', self.segInfo.Start[iseg])[::-1] +
                        re.split(':|\.', self.segInfo.Stop[iseg])[::-1]))))
                fOut.write(struct.pack(
                    '240s', self.segInfo.Note[iseg].encode('utf-8')).replace(b'\x00', b' '))

                # calculate the statistical information of each channel
                # as short
                mean_ = np.mean(self.data[iseg].values, axis=0) / chCoef_
                # as float
                std_ = np.std(self.data[iseg].values, axis=0) / chCoef_
                # as short
                max_ = np.amax(self.data[iseg].values, axis=0) / chCoef_
                min_ = np.amin(self.data[iseg].values, axis=0) / chCoef_
                # write the statistical information of each channel
                fOut.write(struct.pack('=' + self.chN * 'h',
                                       *np.round(mean_).astype(np.int16)))
                fOut.write(struct.pack('=' + self.chN * 'f', *std_))
                fOut.write(struct.pack('=' + self.chN * 'h',
                                       *np.round(max_).astype(np.int16)))
                fOut.write(struct.pack('=' + self.chN * 'h',
                                       *np.round(min_).astype(np.int16)))

                # write the data in each channel
                raw_ = np.round(self.data[iseg].values / np.repeat(chCoef_.reshape(
                    1, -1), self.segInfo['N sample'][iseg], axis=0)).astype(np.int16)
                fOut.write(raw_.tobytes())

    def pInfo(self, printTxt=False, printExcel=False):
        print('-' * 50)
        print('Segment: {0:2d}; Channel: {1:3d}; Sampling frequency: {2:4d}Hz.'.format(
            self.segN, self.chN, self.fs))
        print(self.segInfo.to_string(justify='center'))
        print('-' * 50)
        path = os.getcwd()
        path += '/' + os.path.splitext(self.filename)[0]
        if printTxt:
            fname = path + '_Info.txt'
            self.segInfo.to_csv(path_or_buf=fname, sep='\t')
        if printExcel:
            fname = path + '_Info.xlsx'
            self.segInfo.to_excel(fname, sheet_name='Sheet01')

    def pChInfo(self,
                printTxt=False,
                printExcel=False):
        print('-' * 50)
        print(self.chInfo.to_string(justify='center'))
        print('-' * 50)
        if printTxt:
            path = os.getcwd()
            fname = path + '/' + \
                os.path.splitext(self.filename)[0] + 'ChInfo.txt'
            infoFile = open(fname, 'w')
            infoFile.write('Channel total: {0:3d} \n'.format(self.chN))
            formatters = {'Name': "{:16s}".format,
                          "Unit": "{:4s}".format,
                          "Coef": "{: .7f}".format}
            infoFile.write(self.chInfo.to_string(
                formatters=formatters, justify='center'))
            infoFile.close()
        if printExcel:
            file_name = path + '/' + \
                os.path.splitext(self.filename)[0] + '_ChInfo.xlsx'
            self.chInfo.to_excel(file_name, sheet_name='Sheet01')

    def to_dat(self,
               sSeg='all'):
        def writefile(self, idx):
            path = os.getcwd()
            file_name = path + '/' + \
                os.path.splitext(self.filename)[
                    0] + '_seg{0:02d}.txt'.format(idx)
            comments = '\t'.join(
                self.chInfo['Name']) + '\n' + '\t'.join(self.chInfo['Unit']) + '\n'
            hearder_fmt_str = 'File: {0:s}, Seg{1:02d}, fs:{2:4d}Hz\nDate: {3:5s} from: {4:8s} to {5:8s}\nNote:{6:s}\n'
            header2write = hearder_fmt_str.format(
                self.filename, idx, self.fs, self.date, self.segInfo['Start'].iloc[idx], self.segInfo['Stop'].iloc[idx], self.segInfo['Note'].iloc[idx])
            header2write += comments
            infoFile = open(file_name, 'w')
            infoFile.write(header2write)
            data_2write = self.data[idx].to_string(header=False,
                                                   index=False, justify='left', float_format='% .5E')
            infoFile.write(data_2write)
            infoFile.close()
            print('Export: {0:s}'.format(file_name))

        if sSeg == 'all':
            for idx in range(self.segN):
                writefile(self, idx)
        elif isinstance(sSeg, int):
            if sSeg <= self.segN:
                writefile(self, sSeg)
            else:
                warnings.warn('seg exceeds the max.')
        else:
            warnings.warn('Input sSeg is illegal. (int or defalt)')

    def pst(self,
            printTxt=False,
            printExcel=False):
        print('-' * 50)
        print('Segment total: {0:02d}'.format(self.segN))
        for idx, istatictis in enumerate(self.segStatis):
            print('')
            print('Seg{0:02d}'.format(idx))
            print(istatictis.to_string(float_format='% .3E', justify='center'))
            print('')
        print('-' * 50)
        path = os.getcwd()
        if printTxt:
            file_name = path + '/' + \
                os.path.splitext(self.filename)[0] + '_statistic.txt'
            infoFile = open(file_name, 'w')
            infoFile.write('Segment total: {0:02d}\n'.format(self.segN))
            for idx, istatictis in enumerate(self.segStatis):
                infoFile.write('\n')
                infoFile.write('Seg{0:02d}\n'.format(idx))
                infoFile.write(istatictis.to_string(
                    float_format='% .3E', justify='center'))
            infoFile.close()
            print('Export: {0:s}'.format(file_name))
        if printExcel:
            file_name = path + '/' + \
                os.path.splitext(self.filename)[0] + '_statistic.xlsx'
            for idx, istatictis in enumerate(self.segStatis):
                istatictis.to_excel(
                    file_name, sheet_name='SEG{:02d}'.format(idx))
            print('Export: {0:s}'.format(file_name))

    def to_mat(self, sSeg=0):
        if isinstance(sSeg, int):
            if sSeg <= self.segN:
                data_dic = {'Data': self.data[sSeg].values,
                            'chInfo': self.chInfo,
                            'Date': self.date,
                            'Nseg': 1,
                            'fs': self.fs,
                            'chN': self.chN,
                            'Seg_sta': self.segStatis[sSeg],
                            'SegInfo': self.segInfo[sSeg:sSeg + 1],
                            'Readme': 'Generated by CaseData from python'
                            }
                path = os.getcwd()
                fname = path + '/' + \
                    os.path.splitext(self.filename)[
                        0] + 'seg{:02d}.mat'.format(sSeg)
                sio.savemat(fname, data_dic)
                print('Export: {0:s}'.format(fname))
            else:
                warnings.warn('seg exceeds the max.')
        else:
            warnings.warn('Selected segment id is illegal (should be int).')

    def fix_unit(self, c_chN, unit, pInfo=False):
        self.chInfo['Unit'].loc[c_chN] = unit
        if pInfo:
            print('-' * 50)
            print(self.chInfo.to_string(justify='center'))
            print('-' * 50)

    def to_fullscale(self, rho=1.025, lam=60, g=9.807, pInfo=False):
        if self.scale == 'prototype':
            print('The data is already upscaled.')
            return
        else:
            print('Please make sure the channel units are all checked!')
            if pInfo:
                print(self.chInfo.to_string(
                    justify='center', columns=['Name', 'Unit']))
            self.rho = rho
            self.lam = lam
            self.scale = 'prototype'
            trans_dic = {'kg': ['kN', np.array([g * 0.001, 1.0, 3.0])],
                         'cm': ['m', np.array([0.01, 0.0, 1.0])],
                         'mm': ['m', np.array([0.001, 0.0, 1.0])],
                         'm': ['m', np.array([1, 0.0, 1.0])],
                         's': ['s', np.array([1, 0.0, 0.5])],
                         'deg': ['deg', np.array([1, 0.0, 0.0])],
                         'rad': ['rad', np.array([1, 0.0, 0.0])],
                         'N': ['kN', np.array([0.001, 1.0, 3.0])]
                         }

            def findtrans(trans_dic, unit):
                unit = unit.lower()
                if unit in trans_dic:
                    trans = trans_dic[unit]
                    return trans
                elif '/' in unit:
                    unitUpper, unitLower = unit.split('/')
                    transUpper = findtrans(trans_dic, unitUpper)
                    transLower = findtrans(trans_dic, unitLower)
                    trans = [transUpper[0] + '/' +
                             transLower[0], np.array([0.0, 0.0, 0.0])]
                    trans[1][0] = transUpper[1][0] / transLower[1][0]
                    trans[1][1] = transUpper[1][1] - transLower[1][1]
                    trans[1][2] = transUpper[1][2] - transLower[1][2]
                    return trans
                elif '.' in unit:
                    unitWithDot = unit.split('.')
                    transU = []
                    transN1 = np.array([])
                    transN2 = np.array([])
                    transN3 = np.array([])
                    for idx, uWithDot in enumerate(unitWithDot):
                        transWithDot = findtrans(trans_dic, uWithDot)
                        transU.append(transWithDot[0])
                        transN1 = np.append(transN1, transWithDot[1][0])
                        transN2 = np.append(transN2, transWithDot[1][1])
                        transN3 = np.append(transN3, transWithDot[1][2])
                    trans = ['.'.join(transU), np.array([1.0, 0.0, 0.0])]
                    for x in np.nditer(transN1):
                        trans[1][0] *= x
                    trans[1][1] = transN2.sum()
                    trans[1][2] = transN3.sum()
                    return trans
                elif unit[-1].isdigit():
                    n = int(unit[-1])
                    unit = unit[0:-1]
                    if unit in trans_dic:
                        trans_temp = trans_dic[unit]
                        trans = [trans_temp[0] +
                                 str(n), np.array([1.0, 0.0, 0.0])]
                        trans[1][0] = trans_temp[1][0]**n
                        trans[1][1] = trans_temp[1][1] * n
                        trans[1][2] = trans_temp[1][2] * n
                        return trans
                    else:
                        warnings.warn(
                            "input unit cannot identified, please check the unit.")
                else:
                    warnings.warn(
                        "input unit cannot identified, please check the unit.")

            transUnit = []
            transCoeffunit = np.zeros(self.chN)
            transCoeffrho = np.zeros(self.chN)
            transCoefflam = np.zeros(self.chN)
            for idx, unit in enumerate(self.chInfo['Unit']):
                trans_temp = findtrans(trans_dic, unit)
                transUnit.append(trans_temp[0])
                transCoeffunit[idx] = trans_temp[1][0]
                transCoeffrho[idx] = trans_temp[1][1]
                transCoefflam[idx] = trans_temp[1][2]
            self.chInfo['Unit'] = transUnit
            self.chInfo['Coeffunit'] = transCoeffunit
            self.chInfo['Coeffrho'] = transCoeffrho
            self.chInfo['Coefflam'] = transCoefflam

            del self.chInfo['Coef']

            if pInfo:
                print(self.chInfo.to_string(justify='center'))

            for idx1 in range(self.segN):
                for idx2, name in enumerate(self.chInfo['Name']):
                    C1 = self.chInfo['Coeffunit'].iloc[idx2]
                    C2 = rho ** self.chInfo['Coeffrho'].iloc[idx2]
                    C3 = lam ** self.chInfo['Coefflam'].iloc[idx2]
                    C = C1 * C2 * C3
                    self.data[idx1][name] *= C
            print('The data is upscaled.')
