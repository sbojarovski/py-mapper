#!/usr/bin/env python2
# -*- coding: utf-8 -*-
'''
This file is part of the Python Mapper package, an open source tool
for exploration, analysis and visualization of data.

Copyright 2011–2015 by the authors:
    Daniel Müllner, http://danifold.net
    Aravindakshan Babu, anounceofpractice@hotmail.com

Python Mapper is distributed under the GPLv3 license. See the project home page

    http://danifold.net/mapper

for more information.
'''

'''GUI for the Mapper algorithm'''

GUIversion = '0.1.2'
GUIdate = 'April 28, 2015'

#import wxversion
#wxversion.select('3.0', '2.8')
import wx
import os
import sys
import traceback
import numpy as np

# TBD: deferred import
import collections
import pickle

# More imports below!

BORDER = 4

wxversion = [int(x) for x in wx.version().split(' ')[0].split('.')]
if wxversion[:2] < [2, 8]:
    print 'Warning: The Mapper GUI was coded for wxPython version 2.8 or '
    'newer.\nIf you experience glitches or errors in the interface, please '
    'update wxPython.'
oldwx = (wxversion[:2] < [2, 9])

'''##############################################################
   ###  Mapper worker class
   ##############################################################'''

from threading import Thread, Lock, Condition
from errno import EINTR

class JobInterrupt:
    pass

class ParameterError:
    ShowErrorDialog = True

    def __init__(self, msg, parent):
        self.msg = msg
        if ParameterError.ShowErrorDialog:
            while not isinstance(parent, CollapsiblePane):
                parent = parent.Parent
            parent.Collapse(False)
            ErrorDialog(parent, msg)

    def __str__(self):
        return self.msg

WorkerProcess = None  # Global variable for the worker process. We need this
# to stop the process in case an exception is raised before the main
# window is established.

class MapperInterfaceThread(Thread):
    '''Mapper interface thread.

    This thread starts a subprocess and does nothing else than passing commands
    through to the subprocess. We have the combination of a thread plus a
    process for the following reason:

      * The thread can generate events in the main application thread. This
        way, the application can be notified about finished computations
        without waiting for the result and blocking the user interface.

      * The subprocess can be safely terminated on user request and started
        again. This way, the user can interrupt and restart the computation
        without special breakpoints and without damage to the user interface
        process.
    '''
    def __init__(self, notify_dest, event):
        Thread.__init__(self, name='Mapper_answer_thread')
        self._notify_dest = notify_dest
        self._event = event
        self._want_terminate = False
        self._lock = Lock()
        self._want_acquire = False
        self._condition = Condition(Lock())
        # self.daemon = False
        # Do not automatically terminate when the main terminates: we need this
        # thread to terminate the subprocess cleanly.
        self.jobnr = 0
        self.StartWorker()
        self.start()

    def run(self):
        # Wait for the answer (while not blocking the other thread(s))
        while True:
            self._condition.acquire()
            while self._want_acquire:
                self._condition.wait()
            self._condition.release()
            answer = None
            with self._lock:
                if self._want_terminate:
                    break
                try:
                    if self.Conn.poll(.5):
                        answer = self.Conn.recv()
                except IOError as e:
                    # See
                    # http://stackoverflow.com/questions/4952247/
                    # interrupted-system-call-with-processing-queue
                    if e.errno != EINTR:
                        print 'IO error detected.'
                        raise

            if answer == None:
                continue
            elif answer[0] == 'JobFinished':
                jobnr = answer[1]
                sys.stderr.write('Job {0} is finished.\n'.format(jobnr))
                if jobnr not in self.jobs:
                    raise ValueError('Job no. {0} is not in the ' \
                                         'list of current jobs.\n'\
                                         .format(jobnr))
                else:
                    self.jobs.remove(jobnr)
                if self.jobs:
                    continue
                else:
                    answer = ('NoPending', {})
            elif answer[0] != 'Progress':
                sys.stderr.write('Answer received by thread: {0}.\n'\
                                     .format(answer[0]))
            event = self._event(answer=answer)
            # Notify the main thread about the answer
            wx.PostEvent(self._notify_dest, event)

        sys.stderr.write('The Mapper interface thread has stopped.\n')

    def StartWorker(self):
        global WorkerProcess

        from multiprocessing import Process, Pipe
        child_conn, self.Conn = Pipe(duplex=True)
        self.Worker = Process(target=MapperWorkerProcess,
                              args=(child_conn,))
        WorkerProcess = self.Worker
        self.jobs = set()
        self.Worker.start()
        sys.stderr.write('A new Mapper worker process was started.\n')

    def RestartWorker(self):
        self._condition.acquire()
        self._want_acquire = True
        self._lock.acquire()
        self._want_acquire = False
        self._condition.notify()
        self._condition.release()
        self.StopWorker()
        self.StartWorker()
        self._lock.release()

    def StopInterface(self):
        self._condition.acquire()
        self._want_acquire = True
        self._lock.acquire()
        self._want_acquire = False
        self._condition.notify()
        self._condition.release()
        self.StopWorker()
        self._want_terminate = True
        self._lock.release()

    def StopWorker(self):
        if self.Worker.is_alive():
            self.SendJob(('Stop', {}))
            self.Worker.join(timeout=.5)
            if self.Worker.is_alive():
                sys.stderr.write('Kill the Mapper worker process... ')
                self.Worker.terminate()
                self.Worker.join()
                sys.stderr.write('Done.\n')
        else:
            sys.stderr.write('The Mapper worker thread has stopped '
                             'prematurely.\n')
        #self.Conn.close() # Can raise a "buffer overflow"!!!
        sys.stderr.write('The Mapper worker process has stopped.\n')

    def SendJob(self, job):
        self.jobnr += 1
        self.jobs.add(self.jobnr)
        self.Conn.send((job, self.jobnr))


class MapperWorkerProcess:
    '''Mapper worker process

    Import the mapper module here, since it takes a long time to load.

    The Mapper input data may undergo several transformations. We label the
    different stages as follows. All steps are optional, so each of the
    variables might point to the same array as the variable before in the list.

      * self.InputData     : raw data as loaded
      * self.InputDataP    : after preprocessing
      * self.InputDataPS   : after taking a subset according to the mask
      * self.InputDataPSI  : after computing the intrinsic metric
      * self.InputDataPSIT : after the filter transformation

    Currently, the PSI and PSIT variants are always the same; the distinction
    is only for bookkeeping.
    '''
    def __init__(self, conn):
        self.conn = conn

        try:
            import mapper
        except ImportError as e:
            self.conn.send(('Error', (u'The ‘mapper’ module could not be '
                                      'imported:\n\n'
                                      'ImportError: ' + str(e))))
            raise
        self.mapper = mapper

        self.Commands = {
            'Start': self.Startup,
            'LoadData' : self.LoadDataJob,
            'RunMapper' : self.RunMapper,
            'GenerateScaleGraph' : self.GenerateScaleGraph,
            'FilterHistogram' : self.FilterHistogram,
            'MinNnghbrs' : self.MinNnghbrs,
            'ViewData' : self.ViewData,
            'GenerateScript' : self.GenerateScript,
            'Stop' : self.Stop,
            }

        self.InputPar = None
        self.PreprocessPar = None
        self.MetricPar = None
        self.FilterPar = None
        self.FilterTrafoPar = None
        self.MapperPar = None
        self.CutoffPar = None
        self.ToComplexPar = None
        self.NodeColorCode = None
        self.ComplexGenerated = False

        self.PointLabels = None
        self.Filter = None

        self.SyntheticShapes = { 'Circle' : self.mapper.shapes.circle,
                                 '2-Torus' : self.mapper.shapes.torus,
                                 }
        self.SyntheticShapeNames = { 'Circle' : 'mapper.shapes.circle',
                                     '2-Torus' : 'mapper.shapes.torus',
                                     }
        self.jobnr = False
        self.run()

    def run(self):
        self.keepalive = True
        while self.keepalive:
            assert not self.jobnr
            try:
                msg, self.jobnr = self.conn.recv()
            except EOFError:
                raise RuntimeError('Connection to the main process was lost.')
            print 'Job {0} received by subprocess: {1}.'.format(self.jobnr,
                                                                msg[0])
            assert isinstance(msg, tuple)
            assert len(msg) == 2
            cmd, data = msg
            assert isinstance(data, dict)

            assert cmd in self.Commands
            self.Commands[cmd](**data)

    def JobFinished(self):
        if not self.jobnr:
            raise AssertionError('Job number expected. Something has gone '
                                 'wrong with the inter-process communication.')
        self.conn.send(('JobFinished', self.jobnr))
        self.jobnr = False

    def Startup(self):
        self.conn.send(('StartupSuccess', {'MapperPath' : self.mapper.__path__}))
        self.JobFinished()

    def Stop(self):
        self.conn.close()
        self.keepalive = False

    def Progress(self, value):
        self.conn.send(('Progress', value))

    def FilterHistogram(self, **kwargs):
        try:
            self.GetInput(**kwargs)
            self.GetPreprocess(**kwargs)
            self.GetMetric(**kwargs)
            self.GetFilter(**kwargs)
            self.GetFilterTrafo(**kwargs)

            filt = self.FilterTransformed if self.Mask is None \
                else self.FilterTransformed[self.Mask]
            hist = np.histogram(filt, bins=kwargs['Bins'])[0]
            self.conn.send(('FilterHistogram',
                            (hist, filt.min(), filt.max())))
        except JobInterrupt:
            self.conn.send(('FilterHistogram', None))
        self.JobFinished()

    def MinNnghbrs(self, **kwargs):
        try:
            self.GetInput(**kwargs)
            self.GetPreprocess(**kwargs)

            MetricPar = kwargs['Metric']
            if self.is_vector_data:
                metricpar = self.pdist_parameters(*MetricPar['Metric'])
            else:
                metricpar = {}

            Min = self.mapper.metric.minimal_k_to_make_dataset_connected(
                self.InputDataPS,
                metricpar=metricpar, callback=self.Progress)
            self.conn.send(('MinNnghbrs', Min))
        except JobInterrupt:
            pass
        self.JobFinished()

    def ViewData(self, **kwargs):
        try:
            self.GetInput(**kwargs)
            self.GetPreprocess(**kwargs)
            if not self.is_vector_data:
                self.conn.send(('Error', ('Distance matrix data cannot be '
                                          'displayed, only vector data.')))
                raise JobInterrupt
            if self.InputDataPS.shape[1] not in (2, 3):
                self.conn.send(('Error',
                                ('Cannot display {0}-dimensional data.'.format(
                                self.InputDataPS.shape[1]))))
                raise JobInterrupt

            self.GetMetric(**kwargs)
            self.GetFilter(**kwargs)
            self.GetFilterTrafo(**kwargs)

            self.conn.send(('InputData', (self.InputDataPS,
                                          self.FilterTransformed,
                                          self.Mask)))
        except JobInterrupt:
            pass
        self.JobFinished()

    def GenerateScaleGraph(self, **kwargs):
        try:
            self.GetInput(**kwargs)
            self.GetPreprocess(**kwargs)
            self.GetMetric(**kwargs)
            self.GetFilter(**kwargs)
            self.GetFilterTrafo(**kwargs)
            self.Mapper(**kwargs)
            self.Cutoff(**kwargs)
            self.conn.send(('ScaleGraph',
                            { 'MapperOutput' : self.MapperOutput }))
        except JobInterrupt:
            pass
        self.JobFinished()

    def RunMapper(self, **kwargs):
        try:
            self.GetInput(**kwargs)
            self.GetPreprocess(**kwargs)
            self.GetMetric(**kwargs)
            self.GetFilter(**kwargs)
            self.GetFilterTrafo(**kwargs)
            self.Mapper(**kwargs)
            self.Cutoff(**kwargs)
            self.ToComplex(**kwargs)
            self.GetNodeColor(**kwargs)
            MinSizes = kwargs['MinSizes']
            self.conn.send(('MapperOutput', \
                            { 'MapperOutput' : self.MapperOutput,
                              'MinSizes' : MinSizes }))
        except JobInterrupt:
            pass
        self.JobFinished()

    def GetInput(self, **kwargs):
        if kwargs['Input'] == self.InputPar:
            print 'Reuse data.'
        else:
            self.LoadData(**kwargs)
            if not self.InputPar: raise JobInterrupt

    def LoadDataJob(self, **kwargs):
        self.LoadData(**kwargs)
        self.JobFinished()

    def LoadData(self, **kwargs):
        print 'Load data.'
        self.InputPar = kwargs['Input']
        self.PreprocessPar = None
        self.MetricPar = None
        self.FilterPar = None
        self.FilterTrafoPar = None
        self.MapperPar = None
        self.CutoffPar = None
        self.ToComplexPar = None
        self.NodeColorCode = None

        if self.InputPar[0] == 'File':
          File = self.InputPar[1]
          if File == '':
              self.InputPar = None
          else:
              try:
                  se = os.path.splitext(File)
                  ext = se[1]
                  if ext == '.gz':
                      ext = os.path.splitext(se[0])[1] + ext
                  if ext == '.npy':
                      self.InputData = np.load(File).astype(np.float)
                  elif ext == '.csv':
                      self.InputData = np.loadtxt(File, delimiter=',',
                                                  dtype=np.float)
                  elif ext == '.csv.gz':
                      import gzip
                      with gzip.open(File, 'r') as f:
                          self.InputData = np.loadtxt(f, delimiter=',',
                                                      dtype=np.float)
                  elif ext == '.pkl' or ext == '.pickle':
                      with open(File) as f:
                          self.InputData = np.array(pickle.load(f),
                                                    dtype=np.float)
                          # TBD: Move this logic into the DistanceMatrix or
                          # PointCloud classes.
                          #
                          # TBD: Update the Script_Input function to conform
                          # with the below logic
                          try:
                              # We check to see if the .pkl file has the
                              # optional labels
                              # and dists_info objects. These are seq and dict
                              # objects resp.
                              next_data = pickle.load(f)
                              if isinstance(next_data,
                                            collections.MutableMapping):
                                  print "Found data information dictionary."
                                  self.Data_info = next_data
                              else:
                                  print "Found data labels."
                                  self.PointLabels = np.array(next_data)
                                  print "Found data information dictionary."
                                  self.Data_info = pickle.load(f)
                          except EOFError:
                              pass
                  elif ext == '.mat':
                      from scipy.io import loadmat
                      X = loadmat(File, squeeze_me=True)
                      del X['__globals__']
                      del X['__header__']
                      del X['__version__']
                      assert len(X) == 1
                      self.InputData = X.values()[0]
                  else:
                      self.InputData = np.loadtxt(File, dtype=np.float)
              except Exception as e:
                  traceback.print_exc(None, sys.stderr)
                  self.conn.send(('Error', 'Read error: ' + str(e)))
                  self.conn.send(('Data info', None))
                  self.InputPar = None

        elif self.InputPar[0] == 'Shape':
            Shape, Parameters = self.InputPar[1]
            assert Shape in self.SyntheticShapes
            self.InputData = self.SyntheticShapes[Shape](**Parameters)
        else:
            self.conn.send(('Error', 'Not yet implemented.'))
            self.InputPar = None

        if self.InputPar is None:
            self.InputData = None
            self.is_vector_data = None
            self.conn.send(('Data info', None))
            return
        self.is_vector_data = self.InputData.ndim != 1
        NumObs = self.InputData.shape[0] if self.is_vector_data \
            else self.mapper.n_obs(self.InputData)

        self.conn.send(('Data info',
                        { 'IsVector' : self.is_vector_data,
                          'Shape' : self.InputData.shape,
                          'NumObs' : NumObs,
                          }))

    def GetPreprocess(self, **kwargs):
        PreprocessEnabled = kwargs['PreprocessEnabled']
        Preprocess = kwargs['Preprocess']
        PreprocessPar = Preprocess if PreprocessEnabled else ''
        if PreprocessPar == self.PreprocessPar:
            print 'Reuse preprocessing.'
            return
        self.PreprocessPar = PreprocessPar
        self.MetricPar = None
        self.FilterPar = None
        self.FilterTrafoPar = None
        self.MapperPar = None
        self.CutoffPar = None
        self.ToComplexPar = None
        self.NodeColorCode = None

        if len(PreprocessPar):
            print 'Do preprocessing.'
            try:
                ldict = {'data' : self.InputData,
                         'mask' : None,
                         'Gauss_density' : self.mapper.filters.Gauss_density,
                         'kNN_distance' : self.mapper.filters.kNN_distance,
                         'crop' : self.mapper.crop,
                         'np' : np,
                         }
                exec PreprocessPar in {}, ldict
                self.InputDataP = ldict['data']
                PreprocessingMask = ldict['mask']
                self.InputDataPS, self.PointLabelsSubset = \
                    self.mapper.mask_data(self.InputDataP,
                                          PreprocessingMask,
                                          self.PointLabels)
            except Exception as e:
                traceback.print_exc(None, sys.stderr)
                self.conn.send(('Error', 'Preprocessing error: {0}'.\
                                format(repr(e))))
                self.PreprocessPar = None
                raise JobInterrupt
        else:
            print 'No preprocessing.'
            self.InputDataP = self.InputDataPS = self.InputData
            self.PointLabelsSubset = self.PointLabels

        self.is_vector_data = self.InputDataPS.ndim != 1

        NumObs = self.InputDataPS.shape[0] \
            if self.is_vector_data \
            else self.mapper.n_obs(self.InputDataPS)

        self.conn.send(('Data info',
                        { 'IsVector' : self.is_vector_data,
                          'Shape' : self.InputDataPS.shape,
                          'NumObs' : NumObs,
                          }))

    @staticmethod
    def my_str(x):
        if isinstance(x, (float, np.floating)) and np.isposinf(x):
            return 'np.inf'
        elif isinstance(x, (str, unicode)):
            return "'" + x.replace("'", "\\'").replace("\n", "\\n") + "'"
        else:
            return str(x)

    @staticmethod
    def dict_to_str(x):
        return ', '.join([k + '=' + MapperWorkerProcess.my_str(v)
                          for k, v in x.items()])

    def Script_Input(self, **kwargs):
        ret = (
            "'''\n"
            '    Step 1: Input\n'
            "'''\n")
        InputPar = kwargs['Input']
        if InputPar[0] == 'File':
            File = InputPar[1]
            FileStr = self.my_str(File)
            if File == '':
                ret += 'data = None # Empty data - this is invalid!\n'
            else:
                se = os.path.splitext(File)
                ext = se[1]
                if ext == '.gz':
                    ext = os.path.splitext(se[0])[1] + ext
                if ext == '.npy':
                    ret += (
                        "filename = {0}\n"
                        "data = np.load(filename).astype(np.float)\n").\
                        format(FileStr)
                elif ext == '.csv':
                    ret += (
                        "filename = {0}\n"
                        "data = np.loadtxt(str(filename), delimiter=','"
                        ", dtype=np.float)\n").\
                        format(FileStr)
                elif ext == '.csv.gz':
                    ret += (
                        "import gzip\n"
                        "filename = {0}\n"
                        "with gzip.open(filename, 'r') as inputfile:\n"
                        "    data = np.loadtxt(inputfile, delimiter=','"
                        ", dtype=np.float)\n").\
                        format(FileStr)
                elif ext == '.pkl' or ext == '.pickle':
                    ret += (
                        'import Pickle\n'
                        "with open({0}) as f:\n"
                        '    data = np.array(Pickle.load(f), '
                        'dtype=np.float)\n').\
                        format(FileStr)
                else:
                    ret += (
                        'filename = {0}\n'
                        'data = np.loadtxt(filename, '
                        'dtype=np.float)\n').\
                        format(FileStr)
        elif InputPar[0] == 'Shape':
            Shape, Parameters = InputPar[1]
            assert Shape in self.SyntheticShapes
            ret += 'data = {0}({1})\n'.\
                format(self.SyntheticShapeNames[Shape],
                       self.dict_to_str(Parameters))
        else:
            self.conn.send(('Error', 'Not yet implemented.'))
            self.conn.send(('Data info', None))
            ret += "raise ValueError('Not yet implemented: {}')\n".\
                   format(InputPar[0])
        return ret


    def Script_Preprocessing(self, **kwargs):
        ret = '# Preprocessing\n'
        if not kwargs['PreprocessEnabled']:
            ret += "'''\n"
        ret += (
            'point_labels = None\n'
            'mask = None\n'
            'Gauss_density = mapper.filters.Gauss_density\n'
            'kNN_distance  = mapper.filters.kNN_distance\n'
            'crop = mapper.crop\n'
            '# Custom preprocessing code\n'
            '{}\n'
            '# End custom preprocessing code\n'
            ).format(kwargs['Preprocess'])
        ret += (
            'data, point_labels = mapper.mask_data(data, mask, point_labels)\n'
            )
        if not kwargs['PreprocessEnabled']:
            ret += "'''\n"
        return ret

    def GetMetric(self, **kwargs):
        MetricPar = kwargs['Metric']
        if self.MetricPar == MetricPar:
            print 'Reuse metric.'
            return
        self.MetricPar = MetricPar
        self.FilterPar = None
        self.FilterTrafoPar = None
        self.MapperPar = None
        self.CutoffPar = None
        self.ToComplexPar = None
        self.NodeColorCode = None

        if MetricPar['Intrinsic']:
            Metric = MetricPar['Metric']
            assert isinstance(Metric, tuple)
            assert len(Metric) == 2
            if self.is_vector_data:
                if Metric[0] != 'Euclidean':
                    self.conn.send(('Error', 'Only the Euclidean metric has '
                                    'been implemented so far.'))
                    self.MetricPar = None
                    raise JobInterrupt
            try:
                self.InputDataPSI = self.mapper.metric.intrinsic_metric(\
                    self.InputDataPS,
                    MetricPar['Nnghbrs'],
                    MetricPar['eps'],
                    callback=self.Progress,
                    )
            except AssertionError as e:
                traceback.print_exc(None, sys.stderr)
                self.conn.send(('Error', e.message))
                self.MetricPar = None
                raise JobInterrupt

            self.IsPSIVectorData = False
        else:
            self.InputDataPSI = self.InputDataPS
            self.IsPSIVectorData = self.is_vector_data

    def Script_Metric(self, **kwargs):
        ret = (
            "'''\n"
            '    Step 2: Metric\n'
            "'''\n")
        MetricPar = kwargs['Metric']
        Metric = MetricPar['Metric']
        assert isinstance(Metric, tuple)
        assert len(Metric) == 2
        ret += 'intrinsic_metric = {}\n'.format(MetricPar['Intrinsic'])
        ret += ('if intrinsic_metric:\n'
                '    is_vector_data = data.ndim != 1\n'
                '    if is_vector_data:\n'
                '        metric = {}\n').format(Metric[0])
        ret += ("        if metric != 'Euclidean':\n"
                "            raise ValueError('Not implemented')\n"
                '    data = mapper.metric.intrinsic_metric(data, '
                'k={0}, eps={1})\n').\
                format(MetricPar['Nnghbrs'], MetricPar['eps'])
        ret += 'is_vector_data = data.ndim != 1\n'
        return ret

    def GetFilter(self, **kwargs):
        FilterPar = kwargs['FilterFn']
        if self.FilterPar == FilterPar:
            print 'Reuse filter.'
            return
        self.FilterPar = FilterPar
        self.FilterTrafoPar = None
        self.MapperPar = None
        self.CutoffPar = None
        self.ToComplexPar = None
        self.NodeColorCode = None

        vector_filters = { }
        universal_filters = { 'kNN distance' : \
                                  self.mapper.filters.kNN_distance,
                              'Distance to a measure' : \
                                  self.mapper.filters.distance_to_measure,
                              'Eccentricity' : \
                                  self.mapper.filters.eccentricity,
                              'Density, Gaussian kernel' : \
                                  self.mapper.filters.Gauss_density,
                              'Graph Laplacian' : \
                                  self.mapper.filters.graph_Laplacian,
                              'Distance matrix eigenvector' : \
                                  self.mapper.filters.dm_eigenvector,
                              'No filter' :
                                  self.mapper.filters.zero_filter,
                              }
        dm_filters = { }
        Filterfn, Parameters = FilterPar

        if self.IsPSIVectorData:
            metricpar = self.pdist_parameters(*self.MetricPar['Metric'])
            if Filterfn in vector_filters:
                self.Filter = \
                    vector_filters[Filterfn](self.InputDataPSI,
                                             metricpar=metricpar,
                                             callback=self.Progress,
                                             **Parameters)
            elif Filterfn in universal_filters:
                self.Filter = \
                    universal_filters[Filterfn](self.InputDataPSI,
                                                metricpar=metricpar,
                                                callback=self.Progress,
                                                **Parameters)
            elif Filterfn in dm_filters:
                print 'Warning: Inefficient filter function!'
                self.InputDataPSI = \
                    self.mapper.metric.dm_from_data(self.InputDataPSI,
                                                    **metricpar)
                self.Filter = \
                    dm_filters[Filterfn](self.InputDataPSI,
                                         callback=self.Progress,
                                         **Parameters)
            else:
                self.conn.send(('Error', 'Invalid filter choice.'))
                self.FilterPar = None
                raise JobInterrupt
        else:
            if Filterfn in universal_filters:
                self.Filter = \
                    universal_filters[Filterfn](self.InputDataPSI,
                                                callback=self.Progress,
                                                **Parameters)
            elif Filterfn in dm_filters:
                self.Filter = \
                    dm_filters[Filterfn](self.InputDataPSI,
                                         callback=self.Progress,
                                         **Parameters)
            else:
                self.conn.send(('Error', 'Invalid filter choice.'))
                self.FilterPar = None
                raise JobInterrupt

    def Script_Filter(self, **kwargs):
        ret = (
            "'''\n"
            '    Step 3: Filter function\n'
            "'''\n")
        MetricPar = kwargs['Metric']
        FilterPar = kwargs['FilterFn']

        vector_filters = { }
        universal_filters = { 'kNN distance' : 'mapper.filters.kNN_distance',
                              'Distance to a measure' : \
                                  'mapper.filters.distance_to_measure',
                              'Eccentricity' : 'mapper.filters.eccentricity',
                              'Density, Gaussian kernel' : \
                                  'mapper.filters.Gauss_density',
                              'Distance matrix eigenvector' : \
                                  'mapper.filters.dm_eigenvector',
                              'Graph Laplacian' : \
                                  'mapper.filters.graph_Laplacian',
                              'No filter' : \
                                  'mapper.filters.zero_filter',
                              }
        dm_filters = { }
        Filterfn, Parameters = FilterPar

        ret += ('if is_vector_data:\n'
                '    metricpar = {}\n').\
                format(self.pdist_parameters(*MetricPar['Metric']))

        if Filterfn in vector_filters:
            ret += ('    f = {0}(data,\n'
                    '        metricpar=metricpar,\n'
                    '        {1})\n').\
                    format(vector_filters[Filterfn],
                           self.dict_to_str(Parameters))
        elif Filterfn in universal_filters:
            ret += ('    f = {0}(data,\n'
                    '        metricpar=metricpar,\n'
                    '        {1})\n').\
                    format(universal_filters[Filterfn],
                           self.dict_to_str(Parameters))
        elif Filterfn in dm_filters:
            ret += ('    # Warning! Inefficient filter function\n'
                    '    data = mapper.metric.dm_from_data(data, {0})\n'
                    '    f = {1}(data,\n'
                    '        {2})\n').\
                    format(self.dict_to_str(metricpar),
                           dm_filters[Filterfn],
                           self.dict_to_str(Parameters))
        else:
            ret += "    raise ValueError('Invalid filter choice: {}.')\n".\
                   format(Filterfn)
        ret += 'else:\n'
        if Filterfn in universal_filters:
            ret += ('    f = {0}(data,\n'
                    '        {1})\n').\
                    format(universal_filters[Filterfn],
                           self.dict_to_str(Parameters))
        elif Filterfn in dm_filters:
            ret += ('    f = {0}(data,\n'
                    '        {1})\n').\
                    format(dm_filters[Filterfn],
                           self.dict_to_str(Parameters))
        else:
            ret += "    raise ValueError('Invalid filter choice: {}.')\n".\
                   format(Filterfn)
        return ret

    @staticmethod
    def pdist_parameters(metric, parameter=None):
        if metric == 'Euclidean':
            return {'metric' : 'euclidean'}
        elif metric == 'Minkowski':
            p = parameter['exponent']
            if p == np.inf:
                return {'metric' : 'chebyshev'}
            else:
                return {'metric' : 'minkowski', 'p' : p}
        elif metric == 'Chebychev':
            return {'metric' : 'chebychev'}
        else:
            raise ValueError('Metric is not implemented')

    def GetFilterTrafo(self, **kwargs):
        FilterTrafoPar = kwargs['FilterTrafo']
        if self.FilterTrafoPar == FilterTrafoPar:
            print 'Reuse filter transformation.'
            return
        self.FilterTrafoPar = FilterTrafoPar
        self.MapperPar = None
        self.CutoffPar = None
        self.ToComplexPar = None
        self.NodeColorCode = None

        if FilterTrafoPar[0]:
            print 'Transform filter.'
            Mask = None
            try:
                ldict = {'f' : self.Filter,
                         'data' : self.InputDataPSI,
                         'mask' : Mask,
                         'np' : np,
                         'crop' : self.mapper.crop}
                exec FilterTrafoPar[1] in {}, ldict
                self.FilterTransformed = ldict['f']
                # self.InputDataPSIT = ldict['data']
                self.InputDataPSIT = self.InputDataPSI
                self.Mask = ldict['mask']
            except Exception as e:
                traceback.print_exc(None, sys.stderr)
                self.conn.send(('Error', 'Filter transformation error: {0}'.\
                                format(repr(e))))
                self.FilterTrafoPar = None
                raise JobInterrupt
        else:
            self.FilterTransformed = self.Filter
            self.InputDataPSIT = self.InputDataPSI
            self.Mask = None

    def Script_FilterTrafo(self, **kwargs):
        FilterTrafoPar = kwargs['FilterTrafo']
        ret = '# Filter transformation\n'
        if not FilterTrafoPar[0]:
            ret += "'''\n"
        ret += ('mask = None\n'
                'crop = mapper.crop\n'
                '# Custom filter transformation\n')
        ret += FilterTrafoPar[1] + '\n'
        ret += '# End custom filter transformation\n'
        if not FilterTrafoPar[0]:
            ret += "'''\n"
        return ret

    def Mapper(self, **kwargs):
        MapperPar = kwargs['MapperParameters']
        if self.MapperPar == MapperPar:
            print 'Reuse Mapper output.'
            return
        self.MapperPar = MapperPar
        self.CutoffPar = None
        self.ToComplexPar = None
        self.NodeColorCode = None

        self.MapperOutput = None

        cluster_choices = { \
            'Single'   : self.mapper.single_linkage(),
            'Complete' : self.mapper.complete_linkage(),
            'Average'  : self.mapper.average_linkage(),
            'Weighted' : self.mapper.weighted_linkage(),
            'Centroid' : self.mapper.centroid_linkage(),
            'Median'   : self.mapper.median_linkage(),
            'Ward'     : self.mapper.ward_linkage() }

        cover_choices = { \
            'Uniform 1-d cover' : \
                self.mapper.cover.cube_cover_primitive,
            'Balanced 1-d cover' : \
                self.mapper.cover.balanced_cover_1d,
            'Subrange decomposition' : \
                self.mapper.cover.subrange_decomposition_cover_1d }

        try:
            Cover, Parameters = MapperPar['Cover']
            self.Cover = cover_choices[Cover](**Parameters)
            Clustering, Parameters = MapperPar['Clustering']
            cluster = cluster_choices[Clustering]
            assert Parameters == {}
            if self.IsPSIVectorData:
                metricpar = self.pdist_parameters(*kwargs['Metric']['Metric'])
            else:
                metricpar = {}

            # TBD: Update Script_Mapper to contain the point_labels argument

            self.MapperOutput = self.mapper.mapper(\
                self.InputDataPSIT, self.FilterTransformed,
                cover=self.Cover,
                cutoff=None,
                mask=self.Mask,
                cluster=cluster,
                point_labels=self.PointLabelsSubset,
                metricpar=metricpar)
        except Exception as e:
            if isinstance(e, AssertionError):
                if len(str(e)):
                    msg = 'Assertion error: ' + str(e)
                else:
                    msg = 'Assertion error.'
            else:
                msg = 'Mapper error: ' + str(e)
            self.conn.send(('Error', msg + '\n\n'
                            'Restart the Mapper process with the '
                            u"‘Interrupt’ button."))
            raise
            # self.MapperPar = None
            # raise JobInterrupt

    def Script_Mapper(self, **kwargs):
        ret = (
            "'''\n"
            '    Step 4: Mapper parameters\n'
            "'''\n")
        MapperPar = kwargs['MapperParameters']

        cluster_choices = { 'Single'   : 'mapper.single_linkage()',
                            'Complete' : 'mapper.complete_linkage()',
                            'Average'  : 'mapper.average_linkage()',
                            'Weighted' : 'mapper.weighted_linkage()',
                            'Centroid' : 'mapper.centroid_linkage()',
                            'Median'   : 'mapper.median_linkage()',
                            'Ward'     : 'mapper.ward_linkage()' }

        cover_choices = { 'Uniform 1-d cover' : \
                              'mapper.cover.cube_cover_primitive',
                          'Balanced 1-d cover' : \
                              'mapper.cover.balanced_cover_1d',
                          'Subrange decomposition' : \
                              'mapper.cover.subrange_decomposition_cover_1d' }

        Cover, Parameters = MapperPar['Cover']
        ret += 'cover = {0}({1})\n'.\
            format(cover_choices[Cover],
                   self.dict_to_str(Parameters))

        Clustering, Parameters = MapperPar['Clustering']
        ret += 'cluster = ' + cluster_choices[Clustering] + '\n'
        assert Parameters == {}

        ret += ('if not is_vector_data:\n'
                '    metricpar = {}\n'
                'mapper_output = mapper.mapper(data, f,\n'
                '    cover=cover,\n'
                '    cluster=cluster,\n'
                '    point_labels=point_labels,\n'
                '    cutoff=None,\n'
                '    metricpar=metricpar)\n')
        return ret

    def Cutoff(self, **kwargs):
        # Require Mapper to have being run first, so that the self.Cover
        # attribute is correct.
        CutoffPar = kwargs['Cutoff']
        if self.CutoffPar == CutoffPar:
            print 'Reuse cutoff criterion.'
            return
        self.CutoffPar = CutoffPar
        self.ToComplexPar = None
        self.NodeColorCode = None
        Cutoff, Parameters = CutoffPar

        cutoff_choices = { 'First gap' : self.mapper.cutoff.first_gap,
                           'Histogram method' : \
                               self.mapper.cutoff.histogram,
                           'Biggest gap' : \
                               self.mapper.cutoff.variable_exp_gap,
                           'Biggest gap 2' : \
                               self.mapper.cutoff.variable_exp_gap2,
                           }
        sg_choices = ('Scale graph algorithm',)

        if Cutoff in cutoff_choices:
            cutoff = cutoff_choices[Cutoff](**Parameters)
            self.MapperOutput.path_from_cutoff(cutoff)
        elif Cutoff in sg_choices:
            try:
                self.mapper.do_scale_graph(\
                    self.MapperOutput,
                    callback=self.Progress,
                    **Parameters)
            except Exception as e:
                self.conn.send(('Error',
                                (u'Scale graph error: No path can '
                                 u'be found.\n\n'
                                 u'The original Python error message '
                                 u'is: “' + str(e) + u'”.\n\n'
                                 u'(' + str(type(e)) + ')')))
                raise
        else:
            raise ValueError('Unknown cutoff strategy.')
        self.MapperOutput.nodes_from_path(self.FilterTransformed)

    def ToComplex(self, **kwargs):
        ToComplexPar = kwargs['SimpleComplex']
        if ToComplexPar == self.ToComplexPar:
            return
        self.MapperOutput.complex_from_nodes(cover=self.Cover,
                                             simple=ToComplexPar)
        self.ToComplexPar = ToComplexPar

    def Script_Cutoff(self, **kwargs):
        Cutoff, Parameters = kwargs['Cutoff']
        SimpleComplex = kwargs['SimpleComplex']

        cutoff_choices = { 'First gap' : 'mapper.cutoff.first_gap',
                           'Histogram method' : 'mapper.cutoff.histogram',
                           'Biggest gap' : 'mapper.cutoff.variable_exp_gap',
                           'Biggest gap 2' : 'mapper.cutoff.variable_exp_gap2',
                           }
        sg_choices = ('Scale graph algorithm',)

        if Cutoff in cutoff_choices:
            ret = ('cutoff = {0}({1})\n'
                   'mapper_output.cutoff(cutoff, f, cover=cover, '
                   'simple={2})\n').\
                   format(cutoff_choices[Cutoff],
                          self.dict_to_str(Parameters),
                          SimpleComplex)
        elif Cutoff in sg_choices:
            ret = ('mapper.scale_graph(mapper_output, f, cover=cover,\n'
                   '    {0},\n'
                   '    simple={1})\n').\
                   format(self.dict_to_str(Parameters), SimpleComplex)
        else:
            raise ValueError('Unknown cutoff strategy.')
        ret += ('mapper_output.draw_scale_graph()\n'
                "plt.savefig('scale_graph.pdf')\n")
        return ret

    def GetNodeColor(self, **kwargs):
        NodeColorEnabled = kwargs['NodeColorEnabled']
        NodeColor = kwargs['NodeColor']
        NodeColorCode = NodeColor if NodeColorEnabled else ''
        if NodeColorCode == self.NodeColorCode:
            print 'Reuse display parameters.'
            return

        if len(NodeColorCode):
            print 'Compute node colors.'
            try:
                ldict = {'data' : self.InputDataPSIT,
                         'f' : self.FilterTransformed,
                         'nodes' : self.MapperOutput.nodes,
                         'node_color' : None,
                         'point_color' : None,
                         'name' : 'custom scheme',
                         'np' : np,
                         }
                exec NodeColorCode in {}, ldict
                NodeColors = ldict['node_color']
                PointColors = ldict['point_color']
                NodeColorsScheme = ldict['name']
                if type(NodeColorsScheme) != str:
                    raise AssertionError('Node color scheme name must be a string')
                NodeColors = self.MapperOutput.postprocess_node_color(
                    NodeColors, PointColors, self.PointLabelsSubset)
            except Exception as e:
                traceback.print_exc(None, sys.stderr)
                self.conn.send(('Error', 'Node color error: {0}'.\
                                format(repr(e))))
                self.NodeColorCode = None
                raise JobInterrupt
        else:
            print 'No display parameters.'
            NodeColors = None
            NodeColorsScheme = None

        self.NodeColorCode = NodeColorCode
        self.conn.send(('Node colors', (NodeColors, NodeColorsScheme)))

    def Script_Display(self, **kwargs):
        ret = (
            "'''\n"
            '    Step 5: Display parameters\n'
            "'''\n")
        NodeColorEnabled = kwargs['NodeColorEnabled']
        NodeColor = kwargs['NodeColor']
        ret += '# Node coloring\n'
        if not kwargs['NodeColorEnabled']:
            ret += "'''\n"
        ret += ('nodes = mapper_output.nodes\n'
                'node_color = None\n'
                'point_color = None\n'
                "name = 'custom scheme'\n"
                '# Custom node coloring\n')
        ret += NodeColor + '\n'
        ret += ('# End custom node coloring\n'
                'node_color = mapper_output.postprocess_node_color(node_color, '
                'point_color, point_labels)\n')
        ret += 'minsizes = {0}\n'.format(kwargs['MinSizes'])
        ret += (
            #"plt.gca().cla()\n"
            'mapper_output.draw_2D(minsizes=minsizes,\n'
            '    node_color=node_color,\n'
            '    node_color_scheme=name)\n'
            "plt.savefig('mapper_output.pdf')\n"
            'plt.show()\n')
        return ret


    def GenerateScript(self, **kwargs):
        script = ("'''\n"
                  "    Python Mapper script\n"
                  "    Generated by the Python Mapper GUI\n"
                  "'''\n"
                  "\n"
                  "import mapper\n"
                  "import numpy as np\n"
                  "import matplotlib.pyplot as plt\n"
                  "\n")
        script += self.Script_Input(**kwargs)
        script += self.Script_Preprocessing(**kwargs)
        script += self.Script_Metric(**kwargs)
        script += self.Script_Filter(**kwargs)
        script += self.Script_FilterTrafo(**kwargs)
        script += self.Script_Mapper(**kwargs)
        script += self.Script_Cutoff(**kwargs)
        script += self.Script_Display(**kwargs)
        self.conn.send(('Script', script))
        self.JobFinished()


'''##############################################################
   ###  Figure Frame
   ##############################################################'''

class StatusUpdate:
    def PostStatusUpdate(self, message):
        event = EvtUpdateStatus(self.GetId())
        event.message = message
        wx.PostEvent(self, event)

    def ProgressUpdate(self, value):
        event = EvtUpdateStatus(self.GetId(), message=None, gauge=value)
        wx.PostEvent(self, event)

class ResizeableFrame():
    def OnManualResize(self, event):
        resx = resy = None
        TextEntryDialog = wx.TextEntryDialog(
            self, 'Set resolution in pixels:',
            caption='Set resolution', defaultValue='1024x768')
        if TextEntryDialog.ShowModal() == wx.ID_OK:
            res = TextEntryDialog.GetValue().split('x')
            try:
                if len(res) == 1:
                    resx = int(res[0])
                    if resx < 1: raise ValueError
                elif len(res) == 2:
                    resx, resy = map(int, res)
                    if resx < 1 or resy < 1: raise ValueError
                else:
                    raise ValueError
            except ValueError:
                self.Parent.PostStatusUpdate(\
                    "Invalid resolution: {0}.".\
                        format(TextEntryDialog.GetValue()))
        TextEntryDialog.Destroy()
        if resx:
            if not resy:
                resy = self.OptimalHeight(resx)
            self.Maximize(False)
            self.SetClientSize((resx, resy))

    def OptimalHeight(self, resx):
        w, h = self.GetClientSize()
        return int(np.ceil(float(h) / w * resx))

class FigureFrame(wx.Frame, ResizeableFrame, StatusUpdate):
    '''A frame to display a Matplotlib figure.

    The figure allows zooming and panning. Also, clicks on the figure area are
    detected and can be processed by redefining the OnClick method.

      - Area selection with the left mouse button pressed: zoom.
      - Mouse wheel, '+' and '-' keys: zoom from the center
      - '1' key: restore original view
      - Drag the frame content with the CTRL key pressed: pan the figure
      - Click somewhere on the figure: ??? (user definable)
    '''
    # Class variables
    LastSaveDir = '.'
    LastSaveFilename = 'MapperFigure.pdf'

    def __init__(self, parent, title='', size=(640, 480),
                 axesargs=((0, 0, 1, 1),),
                 axeskwargs={'aspect' : 'equal',
                             'frameon' : False },
                 ):
        wx.Frame.__init__(self, parent, title=title, size=size)
        self.Overlay = wx.Overlay()

        # 1:1 aspect ratio?
        self.aspect_ratio_1 = \
            'aspect' in axeskwargs and axeskwargs['aspect'] == 'equal'

        from matplotlib.figure import Figure
        from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
        # NavigationToolbar2WxAgg

        self.figure = Figure(facecolor='w')  # dpi=
        self.canvas = FigureCanvasWxAgg(parent=self, id=wx.ID_ANY,
                                        figure=self.figure)
        self.axes = self.figure.add_axes(*axesargs, **axeskwargs)

        self.Operation = 0  # 0: nothing
                           # 1: click
                           # 2: zoom
                           # 3: pan
        self.ImageModified = False

        self.Bbox = None  # Bounding box of the full image
        self.AtMargin = np.ones(4, dtype=np.bool)

        # Matplotlib event handling - not used
        # def on_press(event):
        #    print 'you pressed', event.button, event.xdata, event.ydata
        # cid = self.canvas.mpl_connect('button_press_event', on_press)

        self.canvas.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown)
        self.canvas.Bind(wx.EVT_KEY_UP, self.OnKeyUp)
        self.canvas.Bind(wx.EVT_LEFT_DOWN, self.OnLeftDown)
        self.canvas.Bind(wx.EVT_LEFT_UP, self.OnLeftUp)
        self.canvas.Bind(wx.EVT_MOTION, self.OnMotion)
        self.canvas.Bind(wx.EVT_MOUSEWHEEL, self.OnMousewheel)
        self.canvas.Bind(wx.EVT_CHAR, self.OnChar)
        self.canvas.SetFocus()

    def Clear(self):
        self.axes.clear()

    def ShowFrame(self):
        self.canvas.draw()
        self.Show()
        self.Iconize(False)
        self.Raise()

    def StartZoom(self, pos):
        self.Operation = 2
        self.DrawOverlay(pos)
        wx.SetCursor(wx.StockCursor(wx.CURSOR_SIZING))

    def ResetZoom(self):
        if self.Operation == 2:
            self.ClearOverlay()
            wx.SetCursor(wx.StockCursor(wx.CURSOR_ARROW))

    def StartPan(self):
        self.Operation = 3
        self.LastDelta = (0, 0)
        self.TriggerPannedRedraw(test=True)

    def ResetPan(self, pos=None):
        if self.Operation == 3:
            if self.ImageModified:
                self.PanToPos(pos if pos else self.StartPos)
                self.ImageModified = False

    def OnKeyDown(self, event):
        keycode = event.GetKeyCode()
        if keycode == wx.WXK_ESCAPE:
            # Cancel all operations
            self.ResetZoom()
            self.ResetPan()
            self.Operation = 0
        elif keycode == wx.WXK_LEFT:
            self.PanByFraction(.2, 0)
        elif keycode == wx.WXK_RIGHT:
            self.PanByFraction(-.2, 0)
        elif keycode == wx.WXK_UP:
            self.PanByFraction(0, -.2)
        elif keycode == wx.WXK_DOWN:
            self.PanByFraction(0, .2)
        elif self.Operation:
            if keycode == wx.WXK_CONTROL and self.Operation != 3:
                # switch to panning mode
                self.ResetZoom()
                self.StartPan()
        else:
            event.Skip()

    def OnKeyUp(self, event):
        if self.Operation:
            keycode = event.GetKeyCode()
            if keycode == wx.WXK_CONTROL and self.Operation != 2:
                # switch to zoom mode
                self.ResetPan()
                self.StartZoom(event.GetPosition())
        else:
            event.Skip()

    def OnChar(self, event):
        if event.HasModifiers():
            event.Skip()
            return
        key = event.GetKeyCode()
        if key == ord('1'):
            self.OriginalView()
        elif key == ord('+'):
            self.Zoom(1.2)
        elif key == ord('-'):
            self.Zoom(1 / 1.2)
        else:
            event.Skip()

    def OriginalView(self):
        M = self.MarginCoords()
        if self.aspect_ratio_1:
            self.ExpandToFullSize(M)
        self.axes.set_xlim(M[0], M[2])
        self.axes.set_ylim(M[1], M[3])
        self.AtMargin[:] = True
        self.canvas.draw()

    def OnMousewheel(self, event):
        self.Zoom(1.2 ** (\
                float(event.GetWheelRotation()) / event.GetWheelDelta()))

    def OnLeftDown(self, event):
        self.Operation = 1
        self.StartPos = event.GetPosition()
        self.StartCoor = self.DataCoords(self.StartPos)
        self.CoorTrafo = self.axes.transData.inverted().transform
        self.xlim = self.axes.get_xlim()
        self.ylim = self.axes.get_ylim()

    def ClickOrDrag(self, pos):
        delta = pos - self.StartPos
        return np.abs(delta).sum() > 15

    def OnMotion(self, event):
        if not self.Operation: return
        pos = event.GetPosition()
        if self.Operation == 1:  # click
            if event.CmdDown():
                self.StartPan()
            elif self.ClickOrDrag(pos):
                self.StartZoom(pos)
        elif self.Operation == 2:
            self.DrawOverlay(pos)
        elif self.Operation == 3:
            self.TriggerPannedRedraw()

    def OnLeftUp(self, event):
        if not self.Operation: return
        # self.OnMotion(event) # necessary?
        pos = event.GetPosition()
        if self.Operation == 2:  # Zoom
            self.ResetZoom()
            if self.ClickOrDrag(pos):
                # Zoom
                x1, y1 = self.StartCoor
                x2, y2 = self.DataCoords(pos)
                if self.aspect_ratio_1:
                    center = np.array((x1 + x2, y1 + y2)) * .5
                    CS = np.float_(np.maximum(self.GetClientSize(), 1))
                    CS *= max(abs(x2 - x1) / CS[0], abs(y2 - y1) / CS[1]) * .5
                    x1, y1 = center - CS
                    x2, y2 = center + CS
                else:
                    x1, x2 = sorted((x1, x2))
                    y1, y2 = sorted((y1, y2))
                self.axes.set_xlim(x1, x2)
                self.axes.set_ylim(y1, y2)
                self.canvas.draw()
                self.Overlay.Reset()
                M = self.MarginCoords()
                self.AtMargin[:] = (M[0] >= x1, M[1] >= y1, M[2] <= x2, M[3] <= y2)
            else:
                self.Operation == 1
        if self.Operation == 1:  # click -> decision left to subclass
            self.OnClick(self.StartCoor, event)
        elif self.Operation == 3:  # translate
            self.ResetPan(pos)
        self.Operation = 0

    def OnSize(self, event):
        if self.aspect_ratio_1:
            x1, x2 = self.axes.get_xlim()
            y1, y2 = self.axes.get_ylim()
            x1, y1, x2, y2 = self.FitCoords(x1, y1, x2, y2)
            self.axes.set_xlim(x1, x2)
            self.axes.set_ylim(y1, y2)
        # self.Overlay.Reset()
        event.Skip()

    def FitCoords(self, x1, y1, x2, y2):
        center = np.array((x1 + x2, y1 + y2))
        center *= .5
        CS = self.GetClientSize()
        # Treat size 0 like 1 pixel. If the window is that small, the
        # coordinates don't make sense anyway, so we just take care that
        # we get finite values.
        #
        # Daniel: I have no idea why scaling with np.float_ works but not
        # with float.
        CS *= max(np.float_(abs(x2 - x1)) / max(CS[0], 1),
                  np.float_(abs(y2 - y1)) / max(CS[1], 1))
        M = self.MarginCoords()
        if not self.AtMargin[2]:
            t = M[0] + CS[0] if self.AtMargin[0] else center[0] + .5 * CS[0]
            if t < M[2]:
                M[2] = t
            else:
                self.AtMargin[2] = True
        if not self.AtMargin[0]:
            t = M[2] - CS[0]
            if t > M[0]:
                M[0] = t
            else:
                self.AtMargin[0] = True

        if not self.AtMargin[3]:
            t = M[1] + CS[1] if self.AtMargin[1] else center[1] + .5 * CS[1]
            if t < M[3]:
                M[3] = t
            else:
                self.AtMargin[3] = True
        if not self.AtMargin[1]:
            t = M[3] - CS[1]
            if t > M[1]:
                M[1] = t
            else:
                self.AtMargin[1] = True
        self.ExpandToFullSize(M)
        return M

    def ExpandToFullSize(self, bbox):
        # Expand coordinates if necessary to fill the full client area
        # Changes bbox!
        CS = np.float_(np.maximum(self.GetClientSize(), 1))
        if (bbox[2] - bbox[0]) * CS[1] > (bbox[3] - bbox[1]) * CS[0]:
            d = .5 * ((bbox[2] - bbox[0]) * CS[1] / CS[0] - bbox[3] + bbox[1])
            bbox[1] -= d
            bbox[3] += d
        else:
            d = .5 * ((bbox[3] - bbox[1]) * CS[0] / CS[1] - bbox[2] + bbox[0])
            bbox[0] -= d
            bbox[2] += d

    def ClearOverlay(self):
        dc = wx.ClientDC(self.canvas)
        odc = wx.DCOverlay(self.Overlay, dc)
        odc.Clear()

    def DrawOverlay(self, pos):
        dc = wx.ClientDC(self.canvas)
        odc = wx.DCOverlay(self.Overlay, dc)
        odc.Clear()
        black = wx.Colour(0, 0, 0)
        dc.SetPen(wx.Pen(black, 1, wx.DOT))
        transparent_brush = wx.Brush(black, style=wx.TRANSPARENT)
        dc.SetBrush(transparent_brush)
        rect = wx.RectPP(self.StartPos, pos)
        dc.SetClippingRect(rect)  # necessary?
        dc.DrawRectangleRect(rect)

    def TriggerPannedRedraw(self, test=False):
        pos = self.ScreenToClient(wx.GetMousePosition())
        if test and pos == self.StartPos: return

        '''
        bbox = self.axes.bbox
        x0 = 0 if bbox.xmin<.1 else int(round(bbox.xmin))+1
        x1 = int(round(bbox.xmax))
        s = self.GetClientSize().y
        y0 = 0 if abs(s-bbox.ymax)<.5 else s-int(round(bbox.ymax))+1
        y1 = s-int(round(bbox.ymin))
        w = x1-x0
        h = y1-y0
        dc = wx.ClientDC(self.canvas)
        size = dc.GetSize()
        if delta[0]>0:
            dx0, dx1 = delta[0], 0
        else:
            dx0, dx1 = 0, -delta[0]
        if delta[1]>0:
            dy0, dy1 = delta[1], 0
        else:
            dy0, dy1 = 0, -delta[1]
        dc.Blit(x0+dx0,y0+dy0,w-abs(delta[0]),h-abs(delta[1]),dc,x0+dx1,y0+dy1)
        # clear the border
        dc.SetPen(wx.NullPen)
        dc.SetBrush(wx.Brush((255,255,255),wx.SOLID))
        if delta[0]>max(0,self.LastDelta[0]):
            dc.DrawRectangle(x0,y0,delta[0],h)
        elif delta[0]<min(0,self.LastDelta[0]):
            dc.DrawRectangle(x1+delta[0],y0,-delta[0],h)
        if delta[1]>max(0, self.LastDelta[1]):
            dc.DrawRectangle(x0,y0,w,delta[1])
        elif delta[1]<min(0, self.LastDelta[1]):
            dc.DrawRectangle(x0,y1+delta[1],w,-delta[1])
        self.LastDrawnPos = pos
        self.LastDelta = delta
        '''
        self.ImageModified = True

        # pos = self.ScreenToClient(wx.GetMousePosition())
        self.PanToPos(pos)

    def PanToPos(self, pos):
        pixels = self.axes.transAxes.transform(((0, 0), (1, 1)))
        pixels += (pos - self.StartPos) * np.array((-1, 1))
        limits = self.CoorTrafo(pixels)
        self.DrawPannedImage(*limits.T)

    def PanByFraction(self, dx, dy):
        pixels = self.axes.transAxes.transform(((0, 0), (1, 1)))
        pixels -= self.GetClientSize() * np.array((dx, dy))
        limits = self.axes.transData.inverted().transform(pixels)
        self.DrawPannedImage(*limits.T)

    def DrawPannedImage(self, xlim, ylim):
        self.axes.set_xlim(xlim)
        self.axes.set_ylim(ylim)
        self.canvas.draw()  # draw_idle?
        self.Overlay.Reset()
        M = self.MarginCoords()
        self.AtMargin[:] = (M[0] >= xlim[0], M[1] >= ylim[0],
                            M[2] <= xlim[1], M[3] <= ylim[1])

    def DataCoords(self, pos):
        '''Transformation from client pixels to data coordinates.'''
        return self.axes.transData.inverted().transform(
            [(pos.x, self.GetClientSize().y - pos.y)]).squeeze()

    def MarginCoords(self, size=None):
        minx, miny, maxx, maxy, m = self.Bbox
        if m:
            if not size: size = self.GetClientSize()
            dpi = float(self.axes.get_figure().dpi)
            # The last m can be multiplied by a factor different from 1.
            # It is there to make the coordinates not too big in small
            # windows. If the window size is only slightly smaller than the
            # node diameter, the node will partially be outside the window
            # borders.
            scale1 = max(float(maxx - minx), 0.) / max(size[0] / dpi - 2 * m, m)
            scale2 = max(float(maxy - miny), 0.) / max(size[1] / dpi - 2 * m, m)
            ms = m * max(scale1, scale2)
            minx -= ms
            maxx += ms
            miny -= ms
            maxy += ms
        else:
            ms = 0.
        if minx == maxx:
            minx -= .5
            maxx += .5
        if miny == maxy:
            miny -= .5
            maxy += .5
        return np.array((minx, miny, maxx, maxy))

    def AdjustFrameSize(self):
        '''Adjust the frame size to the aspect ration of the contained
        image.'''
        x1, y1, x2, y2 = self.MarginCoords((self.startsize, self.startsize))
        self.axes.set_xlim(x1, x2)
        self.axes.set_ylim(y1, y2)
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        if dx >= dy:
            size = (int(self.startsize), int(np.ceil(dy / dx * self.startsize)))
        else:
            size = (int(np.ceil(dx / dy * self.startsize)), int(self.startsize))
        self.SetClientSize(size)
        self.Bind(wx.EVT_SIZE, self.OnSize)

    def AdjustFigureSize(self):
        M = self.MarginCoords()
        self.ExpandToFullSize(M)
        self.axes.set_xlim(M[0], M[2])
        self.axes.set_ylim(M[1], M[3])

    def SetBbox(self, Bbox):
        '''Set the bounding box of the original view.'''
        self.Bbox = Bbox

    def OnClick(self, pos, event):
        '''Dummy, no-op, may be overwritten by subclasses.'''
        pass

    def Zoom(self, factor):
        '''Zoom in if factor>1, out if factor<1.'''
        pixels = self.axes.transAxes.transform(((0, 0), (1, 1)))
        c = np.mean(pixels, axis=0)
        pixels = c + (pixels - c) / float(factor)
        limits = self.axes.transData.inverted().transform(pixels)
        x, y = limits.T
        # Do not allow zooming out where this would only create an empty margin
        if self.aspect_ratio_1:
            self.AtMargin[:] = False
            x[0], y[0], x[1], y[1] = self.FitCoords(x[0], y[0], x[1], y[1])
        else:
            M = self.MarginCoords()
            self.AtMargin[:] = (M[0] >= x[0], M[1] >= y[0], M[2] <= x[1], M[3] <= y[1])
            x = np.where(self.AtMargin[[0, 2]], M[[0, 2]], x)
            y = np.where(self.AtMargin[[1, 3]], M[[1, 3]], y)
        x0 = self.axes.get_xlim()
        y0 = self.axes.get_ylim()
        if np.any(x0 != x) or np.any(y0 != y):
            self.axes.set_xlim(x)
            self.axes.set_ylim(y)
            self.canvas.draw()

    def OnToFile(self, event):
        endings = sorted(self.canvas.get_supported_filetypes().keys())
        endingslist = ';'.join(['*.' + e for e in endings])
        wildcard = 'Supported file types ({0})|{0}'.format(endingslist)
        filetypes = self.canvas.get_supported_filetypes_grouped()
        for description in sorted(filetypes):
            endings = filetypes[description]
            endingslist = ';'.join(['*.' + e for e in endings])
            wildcard += '|{0} ({1})|{1}'.format(description, endingslist)

        FileDialog = wx.FileDialog(self, 'Mapper output figure',
                                   defaultDir=FigureFrame.LastSaveDir,
                                   style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
                                   wildcard=wildcard,
                                   defaultFile=FigureFrame.LastSaveFilename)
        if FileDialog.ShowModal() == wx.ID_OK:
            path = FileDialog.GetPath()
            try:
                FigureFrame.LastSaveDir, FigureFrame.LastSaveFilename = \
                    os.path.split(path)
                self.figure.savefig(path,
                                    dpi=self.figure.get_dpi(),
                                    )
                self.Parent.PostStatusUpdate(\
                    "Mapper figure was saved to {0}.".format(path))
            except Exception as e:
                msg = 'The figure could not be saved: {0}'.format(e)
                print msg
                traceback.print_exc(None, sys.stderr)
                ErrorDialog(self, msg)
                self.Parent.PostStatusUpdate(msg.split('\n')[0])
        else:
            self.Parent.PostStatusUpdate("Canceled.")
        FileDialog.Destroy()

'''##############################################################
   ###  WxGL canvas
   ##############################################################'''
'''
All methods that start with "GL_deferred" use OpenGL commands that can only be
called after the GLCanvas has been initialized. Unless a method with prefix
GL_deferred is called from within such a prefixed method, it must always be
called as

    self.CallAfterInit(self.GL_deferred_XXX)

instead of

    self.GL_deferred_XXX()
'''

class DeferredExecution:
    def __init__(self, parent):
        self.toDo = []

    def CallAfterInit(self, func):
        self.toDo.append(func)

    def ProcessDeferredCommands(self):
        for func in self.toDo:
            func()
        del self.toDo
        self.CallAfterInit = lambda x: x()

import OpenGL.GL as GL
import wx.glcanvas as glcanvas

class WxGLCanvas(glcanvas.GLCanvas, DeferredExecution):
    # Class variables
    border2d = 4.

    def __init__(self, parent):
        attribList = (glcanvas.WX_GL_RGBA,  # RGBA
                      # glcanvas.WX_GL_SAMPLE_BUFFERS, GL.GL_TRUE,
                      glcanvas.WX_GL_DOUBLEBUFFER,  # Double Buffered
                      glcanvas.WX_GL_DEPTH_SIZE, 24)  # 24 bit

        glcanvas.GLCanvas.__init__(self, parent, attribList=attribList)
        DeferredExecution.__init__(self, parent)
        self.GLContext = glcanvas.GLContext(self)

        self.Data = None
        self.highlight = False
        self.Max = np.ones(2)

        self.ThreeD = False

        self.View = Quaternion()

        self.PointSize = 5
        self.Dragging = False
        self.MousePos = (0, 0)

        self.Bind(wx.EVT_PAINT, self.OnFirstPaint)
        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_ERASE_BACKGROUND, self.OnEraseBackground)

        self.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown)
        self.Bind(wx.EVT_CHAR, self.OnChar)
        self.Bind(wx.EVT_LEFT_DOWN, self.OnLeftDown)
        self.Bind(wx.EVT_LEFT_UP, self.OnLeftUp)
        self.Bind(wx.EVT_MOTION, self.OnMotion)
        self.Bind(wx.EVT_MOUSEWHEEL, self.OnMousewheel)
        try:
            self.SetFocus()  # Receive keyboard events
        except wx.PyAssertionError:
            ErrorDialog(self, 'The OpenGL window cannot be initialized.')
            raise EnvironmentError()

    def OnEraseBackground(self, event):
        '''Process the erase background event.'''
        pass  # Do nothing, to avoid flashing on MSWin

    def OnSize(self, event):
        self.CallAfterInit(self.GL_deferred_DoSetViewport)

    def GL_deferred_DoSetViewport(self):
        # For OS X
        self.SetCurrent(self.GLContext)

        w, h = self.GetClientSize()
        GL.glViewport(0, 0, w, h)

        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glPopMatrix()
        GL.glLoadIdentity()
        # Some z-coordinates (eg. the grey bounding box) might be outside of
        # [-1,1]. We scale the z-coordinate so that these elements are still
        # displayed. (sqrt(1/3) would be enough, take .57 for simplicity.)
        # w, h = self.GetSize()
        M = float(max(w, h))
        m = float(min(w, h))

        if self.ThreeD:
            GL.glScaled(h / M, w / M, .57)
            # slight perspective
            perspective_factor = .2
            GL.glMultMatrixd(((1., 0., 0., 0.), (0., 1., 0., 0.),
                              (0., 0., 1., perspective_factor),
                              (0., 0., 0., 1.)))
            x = (np.sqrt(8 * perspective_factor * perspective_factor + 1) - 1) / \
                (4 * perspective_factor)
            y = np.sqrt(1 - x * x) * (1 + perspective_factor * x)
            # correction for nonzero point size
            c = (m - self.PointSize) / (m * y)
            GL.glScaled(c, c, c)
            GL.glPushMatrix()
            # Rotation to view angle
            self.GL_deferred_Project()
        else:
            # 2 pixel extra margin
            S = min(np.maximum((w - self.PointSize - self.border2d,
                                h - self.PointSize - self.border2d), 0) / \
                                self.Max) / w / h
            GL.glScaled(S * h, S * w, S * M * .57)
            GL.glPushMatrix()

    def OnFirstPaint(self, event):
        self.InitGL()
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        #self.OnPaint(event) # remove ?

    def OnPaint(self, event):
        dc = wx.PaintDC(self) # do not remove: http://wxpython.org/Phoenix/docs/html/PaintDC.html
        self.SetCurrent(self.GLContext) # remove?

        # Make sure that the last image is displayed - avoids flickering
        GL.glFinish()
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glPointSize(self.PointSize)
        if self.highlight:
            GL.glCallList(self.DisplayListStart)
            GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
        GL.glCallList(self.DisplayListStart + 1)
        GL.glCallList(self.DisplayListStart + 2)
        self.SwapBuffers()

    def InitGL(self):
        dc = wx.PaintDC(self) # do not remove: http://wxpython.org/Phoenix/docs/html/PaintDC.html
        self.SetCurrent(self.GLContext) # do not remove

        GL.glClearColor(1, 1, 1, 1)

        # GL.glEnable(GL.GL_POINT_SMOOTH)

        GL.glEnable(GL.GL_ALPHA_TEST);
        GL.glAlphaFunc(GL.GL_GREATER, 0.1);  # TBD

        # GL.glEnable(GL.GL_BLEND);
        # GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA);

        # GL.glEnable(GL.GL_MULTISAMPLE)
        # print GL.glGetIntegerv(GL.GL_SAMPLE_BUFFERS);
        # print GL.glGetIntegerv(GL.GL_SAMPLES);
        # print GL.glGetString(GL.GL_EXTENSIONS)
        # GL.glHint(GL.GL_POINT_SMOOTH_HINT, GL.GL_NICEST)

        self.DisplayListStart = GL.glGenLists(3)
        assert self.DisplayListStart > 0, \
            'OpenGL error: no display lists available.'

        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GL.glPushMatrix()

        self.PointSizes = GL.glGetFloat(GL.GL_ALIASED_POINT_SIZE_RANGE);

        GL.glEnable(GL.GL_POINT_SPRITE)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)  # Byte alignment
        GL.glTexEnvi(GL.GL_TEXTURE_ENV, GL.GL_TEXTURE_ENV_MODE,
                     GL.GL_MODULATE);
        GL.glTexEnvi(GL.GL_POINT_SPRITE, GL.GL_COORD_REPLACE, GL.GL_TRUE)
        a = GL.glGenTextures(2)
        self.TextureIndices = GL.glGenTextures(2)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.TextureIndices[0])

        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER,
                           GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER,
                           GL.GL_NEAREST)
        # GL.glTexParameteri(GL.GL_TEXTURE_2D,GL.GL_TEXTURE_MAG_FILTER,
        #                   GL.GL_LINEAR)
        # GL.glTexParameteri(GL.GL_TEXTURE_2D,GL.GL_TEXTURE_MIN_FILTER,
        #                   GL.GL_LINEAR)
        # GL.glTexParameteri(GL.GL_TEXTURE_2D,GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP)
        # GL.glTexParameteri(GL.GL_TEXTURE_2D,GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP)
        GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_PRIORITY, 1.)

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.TextureIndices[1])

        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER,
                           GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER,
                           GL.GL_NEAREST)
        # GL.glTexParameteri(GL.GL_TEXTURE_2D,GL.GL_TEXTURE_MAG_FILTER,
        #                   GL.GL_LINEAR)
        # GL.glTexParameteri(GL.GL_TEXTURE_2D,GL.GL_TEXTURE_MIN_FILTER,
        #                   GL.GL_LINEAR)
        # GL.glTexParameteri(GL.GL_TEXTURE_2D,GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP)
        # GL.glTexParameteri(GL.GL_TEXTURE_2D,GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP)
        GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_PRIORITY, 0.9)

        self.GL_deferred_GenerateTexture()

        self.ProcessDeferredCommands()

    def GL_deferred_GenerateTexture(self):
        w = self.PointSize
        w2 = .5 * w - .5
        rs = (w2 + .5) * (w2 + .5)
        rs2 = (w2 - .5) * (w2 - .5) if w > 2 else np.inf
        if w == 3:
            rs2 = (w2 + .2) * (w2 + .2)

        texture1 = np.empty((w, w, 2), dtype=np.uint8)
        texture2 = np.empty((w, w), dtype=np.uint8)
        for i in range(w):
            for j in range(w):
                ds = (i - w2) * (i - w2) + (j - w2) * (j - w2)
                a = (ds < rs) * 255
                c = (ds < rs2) * 255
                texture1[i, j, :] = (c, a)
                texture2[i, j] = a

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.TextureIndices[0])
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0,  # level
                        GL.GL_LUMINANCE_ALPHA,  # internalFormat
                        w, w,  # width, height
                        0,  # border
                        GL.GL_LUMINANCE_ALPHA,  # Format
                        GL.GL_UNSIGNED_BYTE,  # Type
                        texture1)

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.TextureIndices[1])
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0,  # level
                        GL.GL_ALPHA,  # internalFormat
                        w, w,  # width, height
                        0,  # border
                        GL.GL_ALPHA,  # Format
                        GL.GL_UNSIGNED_BYTE,  # Type
                        texture2)

    def OnChar(self, event):
        if event.HasModifiers():
            event.Skip()
            return
        key = event.GetKeyCode()
        if key == ord('x'):
            self.RotatePre(5, (1, 0, 0))
        elif key == ord('X'):
            self.RotatePre(-5, (1, 0, 0))
        elif key == ord('y'):
            self.RotatePre(5, (0, 1, 0))
        elif key == ord('Y'):
            self.RotatePre(-5, (0, 1, 0))
        elif key == ord('z'):
            self.RotatePre(5, (0, 0, 1))
        elif key == ord('Z'):
            self.RotatePre(-5, (0, 0, 1))
        elif key == ord('+'):
            self.ChangePointSize(1)
        elif key == ord('-'):
            self.ChangePointSize(-1)
        else:
            event.Skip()
            return
        self.Refresh()

    def OnKeyDown(self, event):
        key = event.GetKeyCode()
        if key == wx.WXK_LEFT:
            self.RotatePost(5, (0, 1, 0))
        elif key == wx.WXK_RIGHT:
            self.RotatePost(-5, (0, 1, 0))
        elif key == wx.WXK_UP:
            self.RotatePost(5, (1, 0, 0))
        elif key == wx.WXK_DOWN:
            self.RotatePost(-5, (1, 0, 0))
        elif key == wx.WXK_PAGEUP:
            self.RotatePost(5, (0, 0, 1))
        elif key == wx.WXK_PAGEDOWN:
            self.RotatePost(-5, (0, 0, 1))
        else:
            event.Skip()
            return
        self.Refresh()

    '''
    def ScalePointSize(self, factor):
        self.PointSize = np.clip(self.PointSize*factor, *self.PointSizes)
    '''

    def ChangePointSize(self, d):
        self.PointSize = int(np.clip(self.PointSize + d, *self.PointSizes))
        self.CallAfterInit(self.GL_deferred_GenerateTexture)
        self.CallAfterInit(self.GL_deferred_DoSetViewport)

    def RotatePre(self, a, v):
        if self.ThreeD:
            self.View.prerotate(a, v)
            self.CallAfterInit(self.GL_deferred_Project)

    def RotatePost(self, a, v):
        if self.ThreeD:
            self.View.postrotate(a, v)
            self.CallAfterInit(self.GL_deferred_Project)

    def GL_deferred_Project(self):
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glPopMatrix()
        GL.glPushMatrix()
        GL.glMultMatrixd(self.View.ToGLMatrix())

    def OnLeftDown(self, event):
        self.Dragging = True
        self.MousePos = event.GetPosition()

    def OnLeftUp(self, event):
        self.Dragging = False

    @staticmethod
    def cathetus(v):
        vs = 1 - v.dot(v)
        if vs > 0:
            return np.sqrt(vs)
        else:
            return 0

    def OnMotion(self, event):
        if not (self.ThreeD and self.Dragging):
            return
        NewPos = event.GetPosition()
        if NewPos == self.MousePos:
            return
        scale = float(min(self.GetSize()))
        P0 = (2 * np.array(self.MousePos, dtype=np.float) - self.GetSize()) / scale
        P1 = (2 * np.array(NewPos, dtype=np.float) - self.GetSize()) / scale
        r0 = np.hypot(*P0)
        r1 = np.hypot(*P1)
        if r1 >= 1:
            dtheta = (r0 - r1) * np.pi / 2
            q = Quaternion([np.cos(dtheta) * r0, np.sin(dtheta) * P0[1], np.sin(dtheta) * P0[0], 0])
            q.Normalize()
            dphi = np.arctan2(P0[1] * P1[0] - P0[0] * P1[1], P0.dot(P1)) / 2
            q = Quaternion([np.cos(dphi), 0, 0, np.sin(dphi)]) * q
        else:
            a0 = r0 * np.pi / 2
            q = Quaternion([np.cos(a0) * r0, np.sin(a0) * P0[1], np.sin(a0) * P0[0], 0])
            q.Normalize()
            a1 = r1 * np.pi / -2
            q = Quaternion([np.cos(a1) * r1, np.sin(a1) * P1[1], np.sin(a1) * P1[0], 0]) * q
            q.Normalize()
            q._a[3] *= .5
            q._a[0] = self.cathetus(q._a[1:])

        self.View = q * self.View
        self.View.Normalize()
        self.CallAfterInit(self.GL_deferred_Project)
        self.Refresh()
        self.MousePos = NewPos

    def OnMousewheel(self, event):
        # self.ScalePointSize(1.1**( \
        #        float(event.GetWheelRotation())/event.GetWheelDelta()))
        self.ChangePointSize(\
            round(float(event.GetWheelRotation()) / event.GetWheelDelta()))
        self.Refresh()

    def GL_deferred_Display3d(self):
        if not self.ThreeD:
            self.ThreeD = True
            self.View = Quaternion()
            self.View.postrotate(30, (0, 1, 0))
            self.View.postrotate(-25, (1, 0, 0))
            self.GL_deferred_Project()
            #self.OnSize()

        Datamin = self.Data.min(axis=0)
        Datamax = self.Data.max(axis=0)
        Center = .5 * (Datamin + Datamax)
        Data_centered = self.Data - Center
        MaxR = np.sqrt((Data_centered * Data_centered).sum(axis=1).max())
        del Data_centered
        WxGLCanvas.GL_deferred_SetModelview(Center, MaxR)

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glClearDepth(1.)

        GL.glDeleteLists(self.DisplayListStart, 3)

        GL.glNewList(self.DisplayListStart + 2, GL.GL_COMPILE)
        GL.glColor(.7, .7, .7, 1)
        GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
        vertices = np.array(((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
                             (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)))
        vertices = vertices * Datamax + (1 - vertices) * Datamin
        GL.glVertexPointer(3, GL.GL_DOUBLE, 0, vertices)
        lines = np.array((0, 1, 0, 2, 1, 3, 2, 3, 0, 4, 2, 6, 4, 6, 1, 5, 4, 5, 5, 7, 3, 7, 6, 7),
                         dtype=np.uint8)
        GL.glDrawElements(GL.GL_LINES, len(lines), GL.GL_UNSIGNED_BYTE, lines)
        GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
        GL.glEndList()

    def MakeColormap(self, Filter):
        self.Colors = self.Colormap(Filter)

    def GL_deferred_Display(self):
        dim = self.Data.shape[1]
        if dim==2:
            self.GL_deferred_Display2d()
        elif dim==3:
            self.GL_deferred_Display3d()
        else:
            raise AssertionError('Unreachable code. Please file a bug report.')

    def GL_deferred_Display2d(self):
        self.ThreeD = False
        self.View = Quaternion()
        self.GL_deferred_Project()

        Datamin = self.Data.min(axis=0)
        Datamax = self.Data.max(axis=0)
        Center = np.hstack((.5 * (Datamin + Datamax), 0))
        self.Max = .5 * (Datamax - Datamin)
        WxGLCanvas.GL_deferred_SetModelview(Center, 1)

        GL.glDisable(GL.GL_DEPTH_TEST)

        GL.glDeleteLists(self.DisplayListStart, 3)
        #self.OnSize()

    def Recolor(self, Filter):
        self.MakeColormap(Filter)
        self.CallAfterInit(self.GL_deferred_ShowDataPoints)

    def SetData(self, data):
        self.Data = data
        self.CallAfterInit(self.GL_deferred_Display)

    def SetPList(self, plist):
        self.plist = plist
        self.CallAfterInit(self.GL_deferred_ShowDataPoints)

    def GL_deferred_ShowDataPoints(self):
        GL.glDeleteLists(self.DisplayListStart, 2)
        self.highlight = (self.plist is not None)
        if self.highlight:
            N = np.alen(self.Data)
            compl = np.ones(N, dtype=np.bool)
            compl[self.plist] = False
            Data = self.Data[compl, :]
            alpha = .1
            gray = .2
            Colors = self.Colors[compl, :] * alpha + \
                (0, 0, 0, gray) + (1 - alpha - gray)

            GL.glNewList(self.DisplayListStart, GL.GL_COMPILE)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.TextureIndices[1])
            GL.glEnable(GL.GL_TEXTURE_2D)
            GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
            GL.glEnableClientState(GL.GL_COLOR_ARRAY)
            GL.glVertexPointer(Data.shape[1], GL.GL_DOUBLE, 0, Data)
            GL.glColorPointer(4, GL.GL_DOUBLE, 0, Colors)
            GL.glDrawArrays(GL.GL_POINTS, 0, Data.shape[0])
            GL.glDisableClientState(GL.GL_COLOR_ARRAY)
            GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
            GL.glDisable(GL.GL_TEXTURE_2D)
            GL.glEndList()

            Data = self.Data[self.plist]
            Colors = self.Colors[self.plist]
        else:
            Data = self.Data
            Colors = self.Colors

        GL.glNewList(self.DisplayListStart + 1, GL.GL_COMPILE)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.TextureIndices[0])
        GL.glEnable(GL.GL_TEXTURE_2D)
        GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
        GL.glEnableClientState(GL.GL_COLOR_ARRAY)
        GL.glVertexPointer(Data.shape[1], GL.GL_DOUBLE, 0, Data)
        GL.glColorPointer(4, GL.GL_DOUBLE, 0, Colors)
        GL.glDrawArrays(GL.GL_POINTS, 0, Data.shape[0])
        GL.glDisableClientState(GL.GL_COLOR_ARRAY)
        GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
        GL.glDisable(GL.GL_TEXTURE_2D)
        GL.glEndList()

        self.Refresh()

    # TBD: Remove
    @staticmethod
    def GL_deferred_SetModelview(Center, MaxR):
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()
        scale = 1. / MaxR
        GL.glScaled(scale, scale, scale)
        GL.glTranslate(*(-Center))

    @staticmethod
    def Colormap(Filter):
        # tbd: simplify
        import matplotlib.cm as mcm
        cmap = mcm.get_cmap()
        norm = mcm.colors.Normalize(vmin=Filter.min(), vmax=Filter.max())
        return mcm.ScalarMappable(norm=norm, cmap=cmap).to_rgba(Filter)

    def Snapshot(self):
        # http://stackoverflow.com/questions/2195792/saving-an-image-of-what-a-dc-drew-wxpython
        size = self.Size
        bmp = wx.EmptyBitmap(size.width, size.height)
        memDC = wx.MemoryDC()
        memDC.SelectObject(bmp)

        # Blit (in this case copy) the actual screen on the memory DC
        # and thus the Bitmap
        memDC.Blit(0,  # Copy to this X coordinate
                    0,  # Copy to this Y coordinate
                    size.width,  # Copy this width
                    size.height,  # Copy this height
                    wx.WindowDC(self),  # From where do we copy?
                    0,  # What's the X offset in the original DC?
                    0  # What's the Y offset in the original DC?
                    )

        # Select the Bitmap out of the memory DC by selecting a new
        # uninitialized Bitmap
        memDC.SelectObject(wx.NullBitmap)

        return bmp.ConvertToImage()

class Quaternion(object):
    def __init__(self, val=None):
        if val is None:
            val = (1, 0, 0, 0)
        self._a = np.array(val, dtype=np.float)

    def __getitem__(self, idx):
        return self._a[idx]

    def __mul__(self, other):
        assert isinstance(other, Quaternion)
        return Quaternion((
                self[0] * other[0] - self[1] * other[1] \
                    - self[2] * other[2] - self[3] * other[3],
                self[0] * other[1] + self[1] * other[0] \
                    + self[2] * other[3] - self[3] * other[2],
                self[0] * other[2] + self[2] * other[0] \
                    + self[3] * other[1] - self[1] * other[3],
                self[0] * other[3] + self[3] * other[0] \
                    + self[1] * other[2] - self[2] * other[1]))

    def prerotate(self, a, v):
        a2 = a * np.pi / 360
        sa2 = np.sin(a2)
        q = Quaternion((np.cos(a2), sa2 * v[0], sa2 * v[1], sa2 * v[2]))
        self._a = (self * q)._a
        self.Normalize()

    def postrotate(self, a, v):
        a2 = a * np.pi / 360
        sa2 = np.sin(a2)
        q = Quaternion((np.cos(a2), sa2 * v[0], sa2 * v[1], sa2 * v[2]))
        self._a = (q * self)._a
        self.Normalize()

    def RotateVector(self, v):
        V = Quaternion((0, v[0], v[1], v[2]))
        return self * V * self.conjugate()

    def ToGLMatrix(self):
        M = np.eye(4)
        M[0, 0:3] = self.RotateVector((1, 0, 0))[1:4]
        M[1, 0:3] = self.RotateVector((0, 1, 0))[1:4]
        M[2, 0:3] = self.RotateVector((0, 0, 1))[1:4]
        return M

    def Normalize(self):
        norm = np.sqrt(np.dot(self._a, self._a))
        if norm == 0:
            self._a = np.array([1., 0., 0., 0.])
        else:
            self._a /= norm

    def conjugate(self):
        return Quaternion((self[0], -self[1], -self[2], -self[3]))

    def __str__(self):
        return '{0}+{1}i+{2}j+{3}k'.format(*self)


'''##############################################################
   ###  Collapsible panes for the GUI - great on small screens
   ##############################################################'''
plusicon = [
"11 11 2 1",
"  c #FFFFFF",
"X c #000000",
"XXXXXXXXXXX",
"X         X",
"X         X",
"X    X    X",
"X    X    X",
"X  XXXXX  X",
"X    X    X",
"X    X    X",
"X         X",
"X         X",
"XXXXXXXXXXX"]

minusicon = [
"11 11 2 1",
"  c #FFFFFF",
"X c #000000",
"XXXXXXXXXXX",
"X         X",
"X         X",
"X         X",
"X         X",
"X  XXXXX  X",
"X         X",
"X         X",
"X         X",
"X         X",
"XXXXXXXXXXX"]

class CollapsiblePane(wx.Panel):
    def __init__(self, parent, **kwargs):
        wx.Panel.__init__(self, parent)
        label = kwargs['label']

        captionpanel = wx.Panel(self)
        captionpanel.SetBackgroundColour(\
            wx.SystemSettings.GetColour(wx.SYS_COLOUR_HIGHLIGHT))

        caption = wx.StaticText(captionpanel, label=label)
        caption.SetForegroundColour(
            wx.SystemSettings.GetColour(wx.SYS_COLOUR_HIGHLIGHTTEXT))
        font = self.GetFont()
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        font.SetPointSize(font.GetPointSize() * 1.1)
        caption.SetFont(font)

        plusbmp = wx.BitmapFromXPMData(plusicon)
        minusbmp = wx.BitmapFromXPMData(minusicon)
        self.Plus = wx.StaticBitmap(captionpanel, bitmap=plusbmp)
        self.Minus = wx.StaticBitmap(captionpanel, bitmap=minusbmp)


        self.hbox = Hbox()
        self.hbox.AddSpacer((BORDER, -1))
        self.hbox.Add(caption, flag=wx.ALIGN_CENTER_VERTICAL | wx.ALL,
                      border=BORDER / 2)
        self.hbox.AddStretchSpacer()
        self.hbox.Add(self.Plus, flag=wx.ALIGN_CENTER_VERTICAL)
        self.hbox.Add(self.Minus, flag=wx.ALIGN_CENTER_VERTICAL)
        self.hbox.AddSpacer((BORDER, -1))
        captionpanel.SetSizer(self.hbox)

        self.Pane = wx.Panel(self)
        vbox = Vbox()
        vbox.Add(captionpanel, flag=wx.EXPAND | wx.ALL, border=2)
        vbox.Add(self.Pane, flag=wx.EXPAND)
        self.SetSizer(vbox)

        captionpanel.Bind(wx.EVT_LEFT_DOWN, self.OnLeftDown)
        caption.Bind(wx.EVT_LEFT_DOWN, self.OnLeftDown)
        self.Plus.Bind(wx.EVT_LEFT_DOWN, self.OnLeftDown)
        self.Minus.Bind(wx.EVT_LEFT_DOWN, self.OnLeftDown)

        self.Collapse(False)
        self.Bind(EVT_CONTENT_CHANGE, self.OnContentChange)

    def Collapse(self, collapse):
        self.IsCollapsed = collapse
        self.Plus.Show(collapse)
        self.Minus.Show(not collapse)
        self.Pane.Show(not collapse)
        event = EvtContentChange(id=self.GetId())
        wx.PostEvent(self, event)

    def Expand(self):
        self.Collapse(False)

    def OnLeftDown(self, event):
        self.Collapse(not self.IsCollapsed)

    def OnContentChange(self, event):
        #self.SetMinSize(self.GetBestSize())
        event.Skip()

'''##############################################################
   ###  The actual GUI
   ##############################################################'''

import json
import wx.lib.newevent
from itertools import izip

EvtDataChoice, EVT_DATA_CHOICE = wx.lib.newevent.NewCommandEvent()
EvtMetricChoice, EVT_METRIC_CHOICE = wx.lib.newevent.NewCommandEvent()
EvtUpdateStatus, EVT_UPDATE_STATUS = wx.lib.newevent.NewCommandEvent()
EvtMapperAnswer, EVT_MAPPER_ANSWER = wx.lib.newevent.NewEvent()
EvtContentChange, EVT_CONTENT_CHANGE = wx.lib.newevent.NewCommandEvent()
EvtHighlightNodes, EVT_HIGHLIGHT_NODES = wx.lib.newevent.NewEvent()

class MapperJob:
    EvtMapperJob, EVT_MAPPER_JOB = wx.lib.newevent.NewCommandEvent()

    def PostMapperJob(self, cmd, data):
        if not wx.IsBusy(): wx.BeginBusyCursor()
        event = MapperJob.EvtMapperJob(id=self.GetId(), job=(cmd, data))
        wx.PostEvent(self, event)

def ErrorDialog(parent, text):
    lines = text.split('\n')
    if len(lines) > 10:
        text = '\n'.join(lines[:5] + ['\n... (long message, showing only beginning and end) ...\n'] + lines[-5:])
    dlg = wx.MessageDialog(parent, unicode_if_str(text),
                           'Error',
                           wx.OK | wx.ICON_ERROR)
    dlg.ShowModal()
    dlg.Destroy()

def unicode_if_str(text):
    if isinstance(text, unicode):
        return text
    else:
        return  unicode(text, errors='replace')

def Vbox():
    return wx.BoxSizer(wx.VERTICAL)

def Hbox():
    return wx.BoxSizer(wx.HORIZONTAL)

class ChoicePanel(wx.Panel):
    def __init__(self, parent, ChoicesLabels, ChoicesPanels, label=None,
                 default=0):
        wx.Panel.__init__(self, parent)

        stdflags = wx.ALIGN_CENTER_VERTICAL | wx.LEFT
        if not isinstance(parent, wx.Notebook):
            stdflags |= wx.TOP | wx.BOTTOM

        hbox = Hbox()
        self.SetSizer(hbox)
        InnerPanel = wx.Panel(self)
        # InnerPanel.SetBackgroundColour(wx.WHITE)
        hbox.Add(InnerPanel, flag=wx.ALIGN_CENTER_VERTICAL, proportion=1)

        self.box = wx.GridBagSizer()
        InnerPanel.SetSizer(self.box)
        col = 0

        if label:
            textlabel = wx.StaticText(InnerPanel, label=label)
            font = self.GetFont()
            font.SetWeight(wx.FONTWEIGHT_BOLD)
            textlabel.SetFont(font)

            self.box.Add(textlabel, pos=(0, col),
                         flag=stdflags,
                         border=BORDER)
            col += 1

        self.Choices = wx.Choice(InnerPanel, choices=ChoicesLabels)
        # self.Choices.SetBackgroundColour(np.mean(
        #        (self.Choices.GetBackgroundColour(),wx.WHITE), axis=0))
        self.Choices.SetSelection(default)
        self.Choices.Bind(wx.EVT_CHOICE, self.OnChoice)
        self.box.Add(self.Choices, pos=(0, col),
                flag=stdflags | wx.RIGHT,
                border=BORDER)
        self.ParameterPanels = [p(InnerPanel) for p in ChoicesPanels]
        # Extra empty panel for empty choice (-1)
        self.ParameterPanels.append(NilPanel(InnerPanel))
        col += 1

        for i, panel in enumerate(self.ParameterPanels):
            panel.Show(i == default and not isinstance(panel, NilPanel))
        self.CurrentPanel = self.ParameterPanels[default]
        self.box.Add(self.CurrentPanel,
                     pos=(0, col), span=(2, 1),
                     flag=stdflags | wx.RIGHT,
                     border=BORDER)
        self.ReLayout()

        self.Choices.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown)
        self.Choices.Bind(wx.EVT_CHAR, self.OnChar)

    def ChangePanels(self, item):
        NewPanel = self.ParameterPanels[item]
        if NewPanel == self.CurrentPanel:
            return
        NewPanel.Show(not isinstance(NewPanel, NilPanel))  # prevent focus
        self.CurrentPanel.Show(False)
        self.box.Replace(self.CurrentPanel, NewPanel)
        self.CurrentPanel = NewPanel
        self.ReLayout()

    def ReLayout(self):
        self.SetMinSize((-1,-1))
        self.box.Layout()
        NewSize = self.GetBestSize()
        self.SetMinSize(NewSize)
        event = EvtContentChange(id=self.GetId(), size=NewSize)
        wx.PostEvent(self, event)

    def OnChoice(self, event=None):
        self.ChangePanels(self.Choices.GetSelection())

    def GetValue(self):
        Choice = self.Choices.GetStringSelection()
        Parameters = \
            self.ParameterPanels[self.Choices.GetSelection()].GetValue()
        return (Choice, Parameters)

    def GetAllValues(self):
        Choice = self.Choices.GetSelection()
        Values = [Panel.GetAllValues() for Panel in self.ParameterPanels]
        return { 'Choice' : Choice,
                 'Values' : Values }

    def SetValues(self, Config):
        for Panel, PConfig in izip(self.ParameterPanels, Config['Values']):
            Panel.SetValues(PConfig)
        Choice = Config['Choice']
        self.ChangePanels(Choice)
        self.Choices.SetSelection(Choice)

    def OnKeyDown(self, event):
        keycode = event.GetKeyCode()
        if keycode == wx.WXK_UP:
            item = self.Choices.GetSelection()
            if item > 0:
                self.Choices.SetSelection(item - 1)
                self.OnChoice()
        elif keycode == wx.WXK_DOWN:
            item = self.Choices.GetSelection()
            if item < self.Choices.GetCount() - 1:
                self.Choices.SetSelection(item + 1)
                self.OnChoice()
        else:
            event.Skip()

    def OnChar(self, event):
        if event.HasModifiers():
            event.Skip()
            return
        keycode = event.GetKeyCode()
        if keycode == wx.WXK_HOME:
            self.Choices.SetSelection(0)
            self.OnChoice()
        elif keycode == wx.WXK_END:
            self.Choices.SetSelection(self.Choices.GetCount() - 1)
            self.OnChoice()
        else:
            event.Skip()

class ExampleShapes(ChoicePanel, StatusUpdate):
    ShapeNames = ('Camel',
                  'Cat',
                  'Elephant',
                  'Face',
                  'Flamingo',
                  'Head',
                  'Horse',
                  'Lion',
    )
    ShapeFiles = ('camel-reference.csv.gz',
                  'cat-reference.csv.gz',
                  'elephant-reference.csv.gz',
                  'face-reference.csv.gz',
                  'flam-reference.csv.gz',
                  'head-reference.csv.gz',
                  'horse-reference.csv.gz',
                  'lion-reference.csv.gz',
    )

    ShapeDir = None

    def __init__(self, parent):
        ChoicesLabels = self.ShapeNames
        ChoicesPanels = [NilPanel] * len(ChoicesLabels)
        ChoicePanel.__init__(self, parent, ChoicesLabels, ChoicesPanels)

    @staticmethod
    def SetShapeDir(MapperPath):
        ExampleShapes.ShapeDir = os.path.normpath(os.path.join(
            os.path.realpath(MapperPath[0]), 'exampleshapes'))

    @staticmethod
    def DetectShapeDir():
        import mapper
        ExampleShapes.SetShapeDir(mapper.__path__)

    def OnChoice(self, event=None):
        self.PostData()
        ChoicePanel.OnChoice(self, event)

    def PostData(self):
        if self.ShapeDir is None:
            self.DetectShapeDir()
        item = self.Choices.GetSelection()
        self.PostDataChoice('' if item == -1
                            else os.path.join(self.ShapeDir,
                                              self.ShapeFiles[item]))

    def PostDataChoice(self, filename):
        event = EvtDataChoice(self.GetId(), job=('File', filename))
        wx.PostEvent(self, event)

    def GetValue(self):
        if self.ShapeDir is None:
            self.DetectShapeDir()
        item = self.Choices.GetSelection()
        if item == -1:
            raise ParameterError('Please choose an input data set.', self)
        return  ('File', os.path.join(self.ShapeDir, self.ShapeFiles[item]))

class SyntheticShapes(ChoicePanel, StatusUpdate):
    def __init__(self, parent):
        ChoicesLabels = ('Circle', '2-Torus')
        ChoicesPanels = (CircleParPanel, TorusParPanel)
        ChoicePanel.__init__(self, parent, ChoicesLabels, ChoicesPanels)

        self.Bind(wx.EVT_TEXT_ENTER, self.OnTextEnter)

    def OnTextEnter(self, event):
        self.PostData()

    def OnChoice(self, event=None):
        self.PostData()
        ChoicePanel.OnChoice(self, event)

    def PostData(self):
        if self.Choices.GetSelection() >= 0:
            self.PostDataChoice(self.GetValue())

    def PostDataChoice(self, job):
        event = EvtDataChoice(self.GetId(), job=job)
        wx.PostEvent(self, event)

    def GetValue(self):
        Shape = self.Choices.GetStringSelection()
        Parameters = \
            self.ParameterPanels[self.Choices.GetSelection()].GetValue()
        return ('Shape', (Shape, Parameters))

class CircleParPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        # self.SetBackgroundColour((100,0,0))

        hbox = Hbox()
        self.SetSizer(hbox)
        hbox.Add(wx.StaticText(self, label='Sample points'),
                 flag=wx.ALIGN_CENTER_VERTICAL)
        hbox.AddSpacer((BORDER, 0))
        self.samples = NumCtrl(self,
                               value=1000,
                               size=(54, -1),
                               style=wx.TE_RIGHT | wx.TE_PROCESS_ENTER,
                               integerWidth=6,
                               fractionWidth=0,
                               groupDigits=False,
                               min=10,
                               max=100000,
                               )
        self.samples.SetFont(self.GetFont())
        hbox.Add(self.samples, flag=wx.ALIGN_CENTER_VERTICAL)

    def GetValue(self):
        return { 'samples' : int(self.samples.Value) }

    GetAllValues = GetValue

    def SetValues(self, Config):
        self.samples.SetValue(Config['samples'])

    def OnTextEnter(self, event):
        self.Parent.PostData()

class NumCtrl(wx.TextCtrl):
    def __init__(self, parent, **kwargs):
        wx.TextCtrl.__init__(self, parent,
                             size=kwargs['size'],
                             style=kwargs['style'],
                             value=str(kwargs['value']),
                             )
        self.Integer = (kwargs['fractionWidth'] == 0)
        self.min = kwargs['min']
        self.max = kwargs['max']
        self.Bind(wx.EVT_TEXT, self.OnText)

    def SetValue(self, value):
        super(NumCtrl, self).SetValue(str(value))

    def GetValue(self):
        v = super(NumCtrl, self).GetValue()
        try:
            if self.Integer:
                f = int(v)
            else:
                f = float(v)
            assert f >= self.min
            assert f <= self.max
        except (ValueError, AssertionError):
            self.SetBackgroundColour((255, 100, 100))
            f = 'Error'
        return f

    def OnText(self, event):
        v = super(NumCtrl, self).GetValue()
        try:
            if self.Integer:
                f = int(v)
            else:
                f = float(v)
            assert f >= self.min
            assert f <= self.max
            self.SetBackgroundColour(
                wx.SystemSettings_GetColour(wx.SYS_COLOUR_WINDOW))
        except (ValueError, AssertionError):
            self.SetBackgroundColour((255, 100, 100))
        event.Skip()

def EpsCtrl(parent):
    return NumCtrl(parent,
                   value=1.,
                   size=(54, -1),
                   style=wx.TE_RIGHT,
                   integerWidth=4,
                   fractionWidth=3,
                   groupDigits=False,
                   min=0.,
                   max=np.inf,
                   )

class TorusParPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)

        box = wx.FlexGridSizer(2, 2, hgap=BORDER)
        self.SetSizer(box)

        box.Add(wx.StaticText(self, label='Sample points'),
                 flag=wx.ALIGN_CENTER_VERTICAL)
        self.samples = NumCtrl(self,
                               value=1000,
                               size=(54, -1),
                               style=wx.TE_RIGHT | wx.TE_PROCESS_ENTER,
                               integerWidth=6,
                               fractionWidth=0,
                               groupDigits=False,
                               min=10,
                               max=100000,
                               )
        self.samples.SetFont(self.GetFont())
        box.Add(self.samples, flag=wx.ALIGN_CENTER_VERTICAL)

        box.Add(wx.StaticText(self, label='Minor radius'),
                 flag=wx.ALIGN_CENTER_VERTICAL)
        self.rminor = NumCtrl(self,
                              value=.4,
                              size=(54, -1),
                              style=wx.TE_RIGHT | wx.TE_PROCESS_ENTER,
                              integerWidth=1,
                              fractionWidth=3,
                              groupDigits=False,
                              min=0.,
                              max=1.,
                              )
        self.rminor.SetFont(self.GetFont())
        box.Add(self.rminor, flag=wx.ALIGN_CENTER_VERTICAL)

    def GetValue(self):
        return {'samples' : int(self.samples.Value),
                'rminor' : float(self.rminor.Value) }

    GetAllValues = GetValue

    def SetValues(self, Config):
        self.samples.SetValue(Config['samples'])
        self.rminor.SetValue(Config['rminor'])

    def OnTextEnter(self, event):
        self.Parent.PostData()

class LoadData(wx.Panel, StatusUpdate):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        hbox = Hbox()
        self.SetSizer(hbox)
        self.LastDir = '.'

        P1 = wx.Panel(self)
        P1.SetMinSize((360, -1))
        P1.SetMaxSize((600, -1))
        hbox.Add(P1, flag=wx.ALIGN_CENTER_VERTICAL, proportion=1)

        self.InputMask = wx.TextCtrl(P1, style=wx.TE_PROCESS_ENTER)
        FileOpenIcon = wx.ArtProvider.GetBitmap(id=wx.ART_FILE_OPEN,
                                                client=wx.ART_BUTTON)
        SelectFileButton = wx.BitmapButton(P1, bitmap=FileOpenIcon)
        SelectFileButton.Bind(wx.EVT_BUTTON, self.OnSelectFileButton)

        hbox2 = Hbox()
        hbox2.Add(self.InputMask, proportion=1,
                 flag=wx.ALIGN_CENTER_VERTICAL | wx.LEFT,
                 border=BORDER)
        hbox2.Add(SelectFileButton, proportion=0,
                 flag=wx.ALIGN_CENTER_VERTICAL | wx.LEFT | wx.RIGHT,
                 border=BORDER)
        P1.SetSizer(hbox2)

        self.SetMinSize(self.GetBestSize())
        self.InputMask.Bind(wx.EVT_TEXT_ENTER, self.OnTextEnter)

    def OnSelectFileButton(self, event):
        if os.path.isdir(self.LastDir):
            defaultDir = self.LastDir
        else:
            defaultDir = '.'
        FileDialog = wx.FileDialog(self, 'Choose a data set', style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
                                   defaultDir=defaultDir,
                                   wildcard="All files (*.*)|*.*|"
                                   "NumPy binary file (*.npy)|*.npy|"
                                   "Comma-separated values (*.csv)|*.csv|"
                                   "Comma-separated values, gzipped (*.csv.gz)|*.csv.gz|"
                                   "Pickle file (*.pkl;*.pickle)|"
                                   "*.pkl;*.pickle"
                                   )
        if FileDialog.ShowModal() == wx.ID_OK:
            path = FileDialog.GetPath()
            filename = os.path.basename(path)
            self.PostStatusUpdate('New input file selection: {0}.'.\
                                      format(filename))
            self.InputMask.SetValue(path)
            self.PostDataChoice(path)
            self.LastDir = os.path.dirname(path)
        else:
            self.PostStatusUpdate("Input file selection was canceled.")
        FileDialog.Destroy()

    def OnTextEnter(self, event):
        path = self.InputMask.GetValue()
        filename = os.path.basename(path)
        if filename == '':
            self.PostStatusUpdate('Please choose an input file.')
        else:
            self.PostStatusUpdate('New input file selection: {0}.'.\
                                      format(filename))
        self.InputMask.SetValue(path)
        self.PostDataChoice(path)

    def PostData(self):
        filename = self.InputMask.GetValue()
        self.PostDataChoice(filename)

    def PostDataChoice(self, filename):
        event = EvtDataChoice(self.GetId(), job=('File', filename))
        wx.PostEvent(self, event)

    def GetValue(self):
        File = self.InputMask.GetValue()
        if File:
            return ('File', File)
        else:
            raise ParameterError('Please choose an input data set.', self)

    def GetAllValues(self):
        File = self.InputMask.GetValue()
        return  ('File', File)

    def SetValues(self, Config):
        assert Config[0] == 'File'
        self.InputMask.SetValue(Config[1])
        LastDir = os.path.dirname(Config[1])
        if os.path.isdir(LastDir):
            self.LastDir = LastDir

class NotebookDataChoice(wx.Notebook, StatusUpdate):
    '''docstring'''
    def __init__(self, parent):
        wx.Notebook.__init__(self, parent)

        ExampleShapesPanel = ExampleShapes(self)
        self.AddPage(ExampleShapesPanel, 'Example Shapes')

        SyntheticShapesPanel = SyntheticShapes(self)
        self.AddPage(SyntheticShapesPanel, 'Synthetic Shapes')

        LoadDataPanel = LoadData(self)
        self.AddPage(LoadDataPanel, 'Load Data')

        self.Bind(EVT_CONTENT_CHANGE, self.OnContentChange)
        self.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self.OnNotebookPageChanged)

    def GetValue(self):
        return self.GetCurrentPage().GetValue()

    def GetAllValues(self):
        Choice = self.GetSelection()
        Values = [self.GetPage(i).GetAllValues() \
                          for i in xrange(self.GetPageCount())]
        return { 'Choice' : Choice,
                 'Values' : Values }

    def SetValues(self, Config):
        Values = Config['Values']
        for i, PConfig in enumerate(Values):
            self.GetPage(i).SetValues(PConfig)
        self.SetSelection(Config['Choice'])

    def OnContentChange(self, event):
        size = self.CalcSizeFromPage(self.GetCurrentPage().GetMinSize())
        self.SetMinSize(size)
        # For OS X
        self.GetCurrentPage().Show(True)
        event.Skip()

    def OnNotebookPageChanged(self, event):
        self.GetCurrentPage().PostData()
        wx.PostEvent(self, EvtContentChange(id=self.GetId()))
        event.Skip()

class MyInputPane(CollapsiblePane, MapperJob):
    def __init__(self, parent):
        CollapsiblePane.__init__(self, parent, label='Step 1: Input')
        Pane = self.Pane
        PaneBox = Vbox()
        Pane.SetSizer(PaneBox)

        self.Notebook = NotebookDataChoice(Pane)
        self.Notebook.Bind(EVT_DATA_CHOICE, self.OnDataChoice)
        PaneBox.Add(self.Notebook, flag=wx.ALL | wx.EXPAND, border=BORDER)

        hbox = Hbox()
        self.PreprocessCheckBox = \
            wx.CheckBox(Pane, label='Preprocessing')
        hbox.Add(self.PreprocessCheckBox,
                 flag=wx.ALIGN_CENTER_VERTICAL | wx.LEFT, border=BORDER)
        self.Preprocess = wx.TextCtrl(Pane)
        hbox.Add(self.Preprocess,
                 flag=wx.ALIGN_CENTER_VERTICAL | wx.LEFT | wx.RIGHT,
                 proportion=1, border=BORDER)

        self.PreprocessCheckBox.Bind(wx.EVT_CHECKBOX, self.OnCheckBox)
        PaneBox.Add(hbox, flag=wx.EXPAND)

        self.DataInfo = wx.TextCtrl(Pane,
                                    style=wx.TE_READONLY | wx.BORDER_SUNKEN)
        lightbg = np.mean(
            (wx.SystemSettings.GetColour(wx.SYS_COLOUR_BACKGROUND),
             wx.WHITE), axis=0)
        self.DataInfo.SetBackgroundColour(lightbg)
        PaneBox.Add(self.DataInfo, flag=wx.EXPAND | wx.ALL, border=BORDER)

    def OnDataChoice(self, event):
        self.DataInfo.SetValue('Reading data set... please wait.')
        self.PostMapperJob('LoadData', {'Input' : event.job})
        event.Skip()

    def OnCheckBox(self, event=None):
        self.Preprocess.Enable(self.PreprocessCheckBox.GetValue())

    def ReceiveDataInfo(self, data):
        if data is None:
            self.DataInfo.SetValue('')
            return

        IsVector = data['IsVector']
        if IsVector:
            self.DataInfo.SetValue('Data type: vector data, number of ' \
              'points: {0}, dimensionality: {1}.'.format(*data['Shape']))
        else:
            self.DataInfo.SetValue('Data type: pairwise distances for ' \
              '{0} points'.format(data['NumObs']))

    def GetValue(self):
        return { 'PreprocessEnabled' : self.PreprocessCheckBox.GetValue(),
                 'Preprocess' : self.Preprocess.GetValue() }

    def GetAllValues(self):
        return { 'IsCollapsed' : self.IsCollapsed,
                 'PreprocessEnabled' : self.PreprocessCheckBox.GetValue(),
                 'Preprocess' : self.Preprocess.GetValue(),
                 'NotebookValues' : self.Notebook.GetAllValues() }

    def SetValues(self, Config):
        self.Collapse(Config['IsCollapsed'])
        self.Notebook.SetValues(Config['NotebookValues'])
        self.PreprocessCheckBox.SetValue(Config['PreprocessEnabled'])
        self.Preprocess.SetValue(Config['Preprocess'])
        self.OnCheckBox()

class MetricChoicePanel(ChoicePanel):
    def __init__(self, parent):
        ChoicesLabels = ('Euclidean', 'Minkowski', 'Chebychev')
        ChoicesPanels = (NilPanel, MetricExponentPanel, NilPanel)
        ChoicePanel.__init__(self, parent, ChoicesLabels, ChoicesPanels)

    def OnChoice(self, event=None):
        event = EvtMetricChoice(self.GetId())
        wx.PostEvent(self, event)
        ChoicePanel.OnChoice(self, event)

    def GetValue(self):
        Metric = self.Choices.GetStringSelection()
        Parameters = \
            self.ParameterPanels[self.Choices.GetSelection()].GetValue()
        # Special case!
        if Metric == 'Minkowski' and Parameters['exponent'] in ('inf', 'Inf',
                                                              np.inf):
            Metric = 'Chebychev'
            Parameters = {}
        return (Metric, Parameters)

class MyMetricPane(CollapsiblePane):
    def __init__(self, parent):
        CollapsiblePane.__init__(self, parent, label='Step 2: Metric')
        Pane = self.Pane
        PaneBox = Vbox()
        Pane.SetSizer(PaneBox)

        self.MetricChoices = MetricChoicePanel(Pane)
        PaneBox.Add(self.MetricChoices, flag=wx.EXPAND)

        RB1 = wx.RadioButton(Pane, label='Ambient/original metric',
                             style=wx.RB_GROUP)
        # For OS X
        RB1.SetValue(True)
        PaneBox.Add(RB1, flag=wx.LEFT | wx.RIGHT, border=BORDER)

        self.RB2 = wx.RadioButton(Pane, label='Intrinsic metric')
        hbox2 = Hbox()
        PaneBox.Add(hbox2, flag=wx.LEFT | wx.RIGHT | wx.EXPAND, border=BORDER)

        self.hbox2Panel = wx.Panel(Pane)
        hbox3 = Hbox()
        self.hbox2Panel.SetSizer(hbox3)

        NnghbrText = wx.StaticText(self.hbox2Panel,
                                        label='No. of nearest neighbors k ')
        self.Nnghbrs = wx.SpinCtrl(self.hbox2Panel, value='1', size=(50, -1),
                                   min=1, max=10000)
        EpsText = wx.StaticText(self.hbox2Panel, label=u'ε ')
        self.Eps = EpsCtrl(self.hbox2Panel)

        hbox2.Add(self.RB2, flag=wx.ALIGN_CENTER_VERTICAL | wx.TOP | wx.BOTTOM,
                  border=2)
        hbox2.AddStretchSpacer()
        hbox2.Add(self.hbox2Panel, flag=wx.ALIGN_CENTER_VERTICAL)
        hbox3.Add(NnghbrText, flag=wx.ALIGN_CENTER_VERTICAL)
        hbox3.Add(self.Nnghbrs, flag=wx.ALIGN_CENTER_VERTICAL | wx.RIGHT,
                  border=BORDER)
        hbox3.Add(EpsText, flag=wx.ALIGN_CENTER_VERTICAL | wx.LEFT,
                  border=BORDER)
        hbox3.Add(self.Eps, flag=wx.ALIGN_CENTER_VERTICAL)
        hbox2.AddStretchSpacer()

        hbox4 = Hbox()
        PaneBox.Add(hbox4, flag=wx.RIGHT | wx.EXPAND, border=BORDER)
        self.MinNnghbrsText = wx.StaticText(
            Pane,
            label='Minimal k to make the data set connected: ')
        self.MinNnghbrsValue = wx.StaticText(
            Pane,
            label='(?)')
        self.MinNnghbrsValue.SetMinSize((20, -1))

        self.MinNnghbrsButton = wx.Button(Pane, label='&Compute')

        hbox4.Add(self.MinNnghbrsText, flag=wx.LEFT | wx.ALIGN_CENTER_VERTICAL,
                  border=BORDER)
        hbox4.Add(self.MinNnghbrsValue, flag=wx.ALIGN_CENTER_VERTICAL,
                  border=BORDER)
        hbox4.Add(self.MinNnghbrsButton, flag=wx.ALL, border=BORDER)

        self.OnRadioButton()
        self.Bind(wx.EVT_RADIOBUTTON, self.OnRadioButton)

    def OnRadioButton(self, event=None):
        Intrinsic = self.RB2.GetValue()
        for elt in (self.hbox2Panel,
                    self.MinNnghbrsText,
                    self.MinNnghbrsValue,
                    self.MinNnghbrsButton,):
            elt.Enable(Intrinsic)
        if self.MinNnghbrsValue.GetLabel() == '(?)':
            self.MinNnghbrsValue.Enable(False)

    def ReceiveDataInfo(self, data):
        if data is None:
            self.MetricChoices.Enable(True)
        else:
            self.MetricChoices.Enable(data['IsVector'])

    def ClearMinNnghbrs(self):
        self.MinNnghbrsValue.SetLabel('(?)')
        self.MinNnghbrsValue.Enable(False)

    def DisplayMinNnghbrs(self, data):
        self.MinNnghbrsValue.SetLabel(str(data))
        self.MinNnghbrsValue.Enable(True)

    def GetValue(self):
        return { 'Metric' : self.MetricChoices.GetValue(),
                 'Intrinsic' : self.RB2.GetValue(),
                 'Nnghbrs' : self.Nnghbrs.GetValue(),
                 'eps' : self.Eps.GetValue() }

    def GetAllValues(self):
        return { 'IsCollapsed' : self.IsCollapsed,
                 'Metric' : self.MetricChoices.GetAllValues(),
                 'Intrinsic' : self.RB2.GetValue(),
                 'Nnghbrs' : self.Nnghbrs.GetValue(),
                 'eps' : self.Eps.GetValue() }

    def SetValues(self, Config):
        self.MetricChoices.SetValues(Config['Metric'])
        self.RB2.SetValue(Config['Intrinsic'])
        self.Nnghbrs.SetValue(Config['Nnghbrs'])
        self.Eps.SetValue(Config['eps'])
        self.OnRadioButton()
        self.Collapse(Config['IsCollapsed'])

class MyFilterPane(CollapsiblePane):
    def __init__(self, parent):
        CollapsiblePane.__init__(self, parent, label='Step 3: Filter function')
        Pane = self.Pane
        PaneBox = Vbox()
        Pane.SetSizer(PaneBox)

        self.canvas = None
        self.axes = None

        self.Notebook = NotebookFilterChoice(Pane)

        PaneBox.Add(self.Notebook, flag=wx.EXPAND | wx.ALL, border=BORDER)

        self.FilterTrafo = FilterTrafoCtrl(Pane)
        PaneBox.Add(self.FilterTrafo, flag=wx.EXPAND)

        hbox = Hbox()
        PaneBox.Add(hbox, flag=wx.EXPAND | wx.ALL, border=BORDER)

        vbox1 = Vbox()
        hbox.Add(vbox1, flag=wx.EXPAND, proportion=1)
        hbox.AddSpacer((BORDER, -1))
        vbox2 = Vbox()
        hbox.Add(vbox2, flag=wx.EXPAND)

        self.HistPanel = wx.Panel(Pane, size=(-1, 36))
        self.HistPanel.SetBackgroundColour(wx.WHITE)
        self.HistPanel.Bind(wx.EVT_SIZE, self.OnSize)
        vbox1.Add(self.HistPanel, flag=wx.EXPAND, proportion=1)

        self.HistButton = wx.Button(Pane, label='See &histogram')
        vbox2.Add(self.HistButton,
                  flag=wx.ALIGN_RIGHT | wx.EXPAND | wx.BOTTOM, border=BORDER)
        vbox2.AddStretchSpacer()
        self.ViewButton = wx.Button(Pane, label='&View data')
        vbox2.Add(self.ViewButton,
                  flag=wx.ALIGN_RIGHT | wx.EXPAND)

        self.maxminbox = Hbox()
        self.MinTxt = wx.StaticText(Pane, label='min', style=wx.ALIGN_LEFT)
        self.MaxTxt = wx.StaticText(Pane, label='max', style=wx.ALIGN_RIGHT)
        self.maxminbox.Add(self.MinTxt,
                  flag=wx.ALIGN_BOTTOM | wx.ALIGN_LEFT)
        self.maxminbox.AddStretchSpacer()
        self.maxminbox.Add(self.MaxTxt,
                  flag=wx.ALIGN_BOTTOM | wx.ALIGN_RIGHT)
        vbox1.Add(self.maxminbox, flag=wx.EXPAND)

        self.Bind(EVT_CONTENT_CHANGE, self.OnContentChange, self.Notebook)


    def OnContentChange(self, event):
        self.FilterTrafo.Update()
        self.FilterTrafo.SetTrafo(self.Notebook.GetFilter())
        event.Skip()
        super(type(self), self).OnContentChange(event)

    def DisplayFilterHistogram(self, data):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
        from matplotlib.collections import PolyCollection

        if self.axes is None:
            figure = Figure(facecolor='w', dpi=72)
            self.canvas = FigureCanvasWxAgg(parent=self.HistPanel, id= -1,
                                            figure=figure)
            self.axes = figure.add_axes((0, 0, 1, 1),
                                        frameon=False,
                                        )
        else:
            self.axes.clear()
        self.axes.set_xticks(())
        self.axes.set_yticks(())
        self.axes.set_autoscale_on(False)

        Size = self.HistPanel.GetClientSize()
        self.canvas.SetSize(Size)

        print 'Display histogram.'
        if data is not None:
            self.MinTxt.SetLabel(str(data[1]))
            self.MaxTxt.SetLabel(str(data[2]))
            hist = data[0]

            # barcoor = [((x,0),(x,b)) for x,b in enumerate(data)]
            # bars = mpl.collections.LineCollection(segments=barcoor,
            #                                  linewidths=float(Size.x)/len(hist),
            #                                      )
            barcoor = [((x, 0), (x, b), (x + 1, b), (x + 1, 0)) for x, b in enumerate(hist)]
            bars = PolyCollection(barcoor, edgecolors='b',
                                  linewidths=0.)
            self.axes.add_collection(bars)
            self.axes.set_xlim(0, len(hist))
            self.axes.set_ylim(0, max(hist))
        else:
            self.MinTxt.SetLabel('min')
            self.MaxTxt.SetLabel('max')
        self.maxminbox.Layout()
        self.canvas.draw()

    def OnSize(self, event):
        if self.canvas:
            self.canvas.SetSize(event.Size)
        event.Skip()

    def GetValue(self):
        return self.FilterTrafo.GetValue()

    def GetAllValues(self):
        return { 'IsCollapsed' : self.IsCollapsed,
                 'FilterTrafos' : self.FilterTrafo.GetAllValues() }

    def SetValues(self, Config):
        self.Collapse(Config['IsCollapsed'])
        self.FilterTrafo.SetValues(Config['FilterTrafos'])
        self.FilterTrafo.SetTrafo(self.Notebook.GetFilter())

class NotebookFilterChoice(wx.Notebook):
    def __init__(self, parent):
        wx.Notebook.__init__(self, parent)

        DissimilarityFilterPanel = MyDissimilarityFilterPanel(self)
        self.AddPage(DissimilarityFilterPanel, 'Filters for dissimilarity data')
        VectorFilterPanel = MyVectorFilterPanel(self)
        self.AddPage(VectorFilterPanel, 'Filters for vector data only')

        self.Bind(EVT_CONTENT_CHANGE, self.OnContentChange)
        self.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self.OnNotebookPageChanged)

    def OnContentChange(self, event):
        size = self.CalcSizeFromPage(self.GetCurrentPage().GetMinSize())
        self.SetMinSize(size)
        # For OS X
        self.GetCurrentPage().Show(True)
        event.Skip()

    def OnNotebookPageChanged(self, event):
        wx.PostEvent(self, EvtContentChange(id=self.GetId()))
        event.Skip()

    def GetValue(self):
        return self.GetCurrentPage().GetValue()

    def GetFilter(self):
        return self.GetCurrentPage().GetFilter()

    def GetAllValues(self):
        Choice = self.GetSelection()
        Values = [self.GetPage(i).GetAllValues() \
                          for i in xrange(self.GetPageCount())]
        return {'Choice' : Choice,
                'Values' : Values }

    def SetValues(self, Config):
        for i, PConfig in enumerate(Config['Values']):
            self.GetPage(i).SetValues(PConfig)
        self.SetSelection(Config['Choice'])

class MyVectorFilterPanel(ChoicePanel):
    def __init__(self, parent):
        ChoicesLabels = ('N/A',)
        ChoicesPanels = (NilPanel,)
        ChoicePanel.__init__(self, parent, ChoicesLabels, ChoicesPanels)

    def GetValue(self):
        item = self.Choices.GetSelection()
        if item == -1:
            raise ParameterError('Please choose a filter function.', self)
        Filter = self.Choices.GetStringSelection()
        Parameters = self.ParameterPanels[item].GetValue()
        return (Filter, Parameters)

    def GetFilter(self):
        item = self.Choices.GetSelection()
        if item == -1:
            raise ValueError('No filter chosen.')
        Filter = self.Choices.GetStringSelection()
        return Filter

class NilPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        # Minimum height 22: hack for OS X
        self.SetMinSize((0, 22))

    def GetValue(self):
        return {}

    GetAllValues = GetValue

    def SetValues(self, Config):
        pass

    def AcceptsFocus(self):
        return False

class MyDissimilarityFilterPanel(ChoicePanel):
    def __init__(self, parent):
        ChoicesLabels = ('Eccentricity', 'kNN distance',
                         'Distance to a measure',
                         'Density, Gaussian kernel', 'Graph Laplacian',
                         'Distance matrix eigenvector',
                         'No filter')
        ChoicesPanels = (MetricExponentPanel, kNNFiltParPanel, kNNFiltParPanel,
                         GaussianDensParPanel, GraphLaplacianParPanel,
                         DMEVParPanel, NilPanel)
        ChoicePanel.__init__(self, parent, ChoicesLabels, ChoicesPanels)

    def GetValue(self):
        item = self.Choices.GetSelection()
        if item == -1:
            raise ParameterError('Please choose a filter function.', self)
        Filter = self.Choices.GetStringSelection()
        Parameters = self.ParameterPanels[item].GetValue()
        return (Filter, Parameters)

    def GetFilter(self):
        item = self.Choices.GetSelection()
        if item == -1:
            raise ValueError('No filter chosen.')
        Filter = self.Choices.GetStringSelection()
        return Filter

class MetricExponentPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)

        hbox = Hbox()
        self.SetSizer(hbox)
        hbox.Add(wx.StaticText(self, label='Exponent'),
                 flag=wx.ALIGN_CENTER_VERTICAL)
        hbox.AddSpacer((BORDER, 0))
        self.exponent = NumCtrl(self,
                                value=1.,
                                size=(54, -1),
                                style=wx.TE_RIGHT,
                                integerWidth=2,
                                fractionWidth=2,
                                groupDigits=False,
                                min=0,
                                max=99,
                                )
        self.exponent.SetFont(self.GetFont())
        hbox.Add(self.exponent,
                 flag=wx.ALIGN_CENTER_VERTICAL)
        hbox.AddSpacer((BORDER, 0))
        hbox.Add(wx.StaticText(self, label='or'),
                 flag=wx.ALIGN_CENTER_VERTICAL)
        self.Infinity = wx.CheckBox(self, label='infinity')
        self.Infinity.Bind(wx.EVT_CHECKBOX, self.OnCheckBox)
        hbox.Add(self.Infinity, flag=wx.ALIGN_CENTER_VERTICAL | wx.LEFT,
                 border=BORDER)

    def OnCheckBox(self, event=None):
        self.exponent.Enable(not self.Infinity.GetValue())

    def GetValue(self):
        exponent = np.inf if self.Infinity.IsChecked() \
            else self.exponent.GetValue()
        if exponent == 'Error':
            raise ParameterError(u'Invalid “Exponent” parameter.', self)
        return { 'exponent' : exponent }

    def GetAllValues(self):
        return { 'exponent' : self.exponent.GetValue(),
                 'inf' : self.Infinity.IsChecked() }

    def SetValues(self, Config):
        self.exponent.SetValue(Config['exponent'])
        self.Infinity.SetValue(Config['inf'])
        self.OnCheckBox()

class kNNFiltParPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)

        hbox = Hbox()
        self.SetSizer(hbox)

        hbox.Add(wx.StaticText(self, label='No. of nearest neighbors'),
                 flag=wx.ALIGN_CENTER_VERTICAL)
        hbox.AddSpacer((BORDER, 0))
        self.Value = NumCtrl(self,
                             value=1,
                             size=(54, -1),
                             style=wx.TE_RIGHT,
                             integerWidth=5,
                             fractionWidth=0,
                             groupDigits=False,
                             min=1,
                             max=99999,
                             )
        self.Value.SetFont(self.GetFont())
        hbox.Add(self.Value)

    def GetValue(self):
        return { 'k' : self.Value.GetValue() }

    GetAllValues = GetValue

    def SetValues(self, Config):
        self.Value.SetValue(Config['k'])

class GaussianDensParPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)

        hbox = Hbox()
        self.SetSizer(hbox)

        hbox.Add(wx.StaticText(self, label=u'Bandwidth σ'),
                 flag=wx.ALIGN_CENTER_VERTICAL)
        hbox.AddSpacer((BORDER, 0))
        self.Value = NumCtrl(self,
                             value=1.,
                             size=(54, -1),
                             style=wx.TE_RIGHT,
                             integerWidth=4,
                             fractionWidth=3,
                             groupDigits=False,
                             min=0.,
                             max=9999.9,
                             )
        self.Value.SetFont(self.GetFont())
        hbox.Add(self.Value)

    def GetValue(self):
        return { 'sigma' : self.Value.GetValue() }

    GetAllValues = GetValue

    def SetValues(self, Config):
        self.Value.SetValue(Config['sigma'])

class GraphLaplacianParPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)

        box = wx.FlexGridSizer(3, 2, hgap=BORDER, vgap=BORDER)
        self.SetSizer(box)

        box.Add(wx.StaticText(self, label='Order of the eigenvector'),
                 flag=wx.ALIGN_CENTER_VERTICAL)

        self.eigenvec = wx.SpinCtrl(self, value='1', size=(50, -1),
                                    min=1, max=1000)
        box.Add(self.eigenvec)

        hbox = Hbox()
        hbox.Add(wx.StaticText(self, label='k'),
                 flag=wx.ALIGN_CENTER_VERTICAL)

        self.k = NumCtrl(self,
                         value=1,
                         size=(54, -1),
                         style=wx.TE_RIGHT,
                         integerWidth=6,
                         fractionWidth=0,
                         groupDigits=False,
                         min=1,
                         max=999999,
                         )
        self.k.SetFont(self.GetFont())
        hbox.Add(self.k, flag=wx.LEFT | wx.RIGHT,
                 border=BORDER)

        hbox.AddStretchSpacer()
        hbox.Add(wx.StaticText(self, label=u'ε'),
                 flag=wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_LEFT | wx.ALIGN_RIGHT)
        box.Add(hbox, flag=wx.EXPAND)

        self.eps = EpsCtrl(self)
        self.eps.SetFont(self.GetFont())
        box.Add(self.eps)

        hbox = Hbox()
        box.Add(hbox)

        self.weighted_edges = wx.CheckBox(self, label='Weighted edges,')

        hbox.Add(self.weighted_edges)
        self.weighted_edges.Bind(wx.EVT_CHECKBOX, self.OnCheckBox)

        self.sigma_eps_text = wx.StaticText(self, label=u'σ/ε')
        hbox.Add(self.sigma_eps_text,
                 flag=wx.ALIGN_CENTER_VERTICAL | wx.LEFT, border=BORDER)

        self.sigma_eps = NumCtrl(self,
                             value=1.,
                             size=(54, -1),
                             style=wx.TE_RIGHT,
                             integerWidth=2,
                             fractionWidth=3,
                             groupDigits=False,
                             min=0.,
                             max=99.999,
                             )
        self.sigma_eps.SetFont(self.GetFont())
        box.Add(self.sigma_eps)

    def OnCheckBox(self, event):
        self.sigma_eps.Enable(self.weighted_edges.GetValue())
        self.sigma_eps_text.Enable(self.weighted_edges.GetValue())


    def GetValue(self):
        ret = { 'n' : self.eigenvec.GetValue(),
                'k' : self.k.GetValue(),
                'eps' : self.eps.GetValue(),
                'weighted_edges' : bool(self.weighted_edges.GetValue()),
                }
        if self.weighted_edges.GetValue():
            ret['sigma_eps'] = self.sigma_eps.GetValue()
        return ret

    def GetAllValues(self):
        return { 'n' : self.eigenvec.GetValue(),
                 'k' : self.k.GetValue(),
                 'eps' : self.eps.GetValue(),
                 'weighted_edges' : bool(self.weighted_edges.GetValue()),
                 'sigma_eps' : self.sigma_eps.GetValue(),
                 }

    def SetValues(self, Config):
        self.eigenvec.SetValue(Config['n'])
        self.k.SetValue(Config['k'])
        self.eps.SetValue(Config['eps'])
        self.weighted_edges.SetValue(Config['weighted_edges'])
        self.sigma_eps.SetValue(Config['sigma_eps'])
        self.OnCheckBox(None)

class DMEVParPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)

        vbox = Vbox()
        self.SetSizer(vbox)

        self.CenterCheckBox = \
            wx.CheckBox(self, label='Mean-center distance matrix')
        vbox.Add(self.CenterCheckBox,
                 flag=wx.ALIGN_CENTER_VERTICAL, border=BORDER)

        hbox = Hbox()
        vbox.Add(hbox)

        hbox.Add(wx.StaticText(self, label='Order of the eigenvector'),
                 flag=wx.ALIGN_CENTER_VERTICAL)
        hbox.AddSpacer((BORDER, 0))
        self.Value = NumCtrl(self,
                             value=0,
                             size=(54, -1),
                             style=wx.TE_RIGHT,
                             integerWidth=5,
                             fractionWidth=0,
                             groupDigits=False,
                             min=0,
                             max=99999,
                             )
        self.Value.SetFont(self.GetFont())
        hbox.Add(self.Value)

    def GetValue(self):
        return { 'k' : self.Value.GetValue(),
                 'mean_center' : self.CenterCheckBox.GetValue() }

    GetAllValues = GetValue

    def SetValues(self, Config):
        self.Value.SetValue(Config['k'])
        self.CenterCheckBox.SetValue(Config['mean_center'])

class DissimilarityFilterListCtrl(wx.ListCtrl):
    def __init__(self, parent):
        wx.ListCtrl.__init__(self, parent,
                             style=wx.LC_REPORT | wx.LC_VRULES | wx.LC_HRULES |
                             wx.LC_EDIT_LABELS)
        self.InsertColumn(0, 'Parameter')
        self.InsertColumn(1, 'Value')
        self.InsertStringItem(0, 'Exponent')
        self.SetStringItem(0, 1, '1')
        self.InsertStringItem(1, 'Nnghbrs')
        self.SetStringItem(1, 1, '6')

class FilterTrafoCtrl(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        self.Trafos = {}
        self.CurrFilter = None

        hbox = Hbox()
        self.SetSizer(hbox)

        self.TrafoCheckBox = wx.CheckBox(self, label='Filter transformation')
        hbox.Add(self.TrafoCheckBox,
                     flag=wx.ALIGN_CENTER_VERTICAL | wx.LEFT, border=BORDER)
        self.FilterTrafo = wx.TextCtrl(self)
        hbox.Add(self.FilterTrafo,
                 flag=wx.ALIGN_CENTER_VERTICAL | wx.LEFT | wx.RIGHT,
                 proportion=1, border=BORDER)

        self.TrafoCheckBox.Bind(wx.EVT_CHECKBOX, self.OnCheckBox)

    def OnCheckBox(self, event=None):
        self.FilterTrafo.Enable(self.TrafoCheckBox.GetValue())

    def GetValue(self):
        return (self.TrafoCheckBox.GetValue(), self.FilterTrafo.GetValue())

    def GetAllValues(self):
        self.Update()
        return self.Trafos

    def SetValues(self, Trafos):
        self.Trafos = Trafos
        self.OnCheckBox()

    def Update(self):
        if self.CurrFilter:
            self.Trafos[self.CurrFilter] = self.GetValue()

    def SetTrafo(self, Filter):
        if Filter in self.Trafos:
            checked, trafo = self.Trafos[Filter]
        else:
            checked, trafo = False, ''
        self.TrafoCheckBox.SetValue(checked)
        self.FilterTrafo.SetValue(trafo)
        self.FilterTrafo.Enable(checked)
        self.CurrFilter = Filter

class MyMapperParPane(CollapsiblePane):
    def __init__(self, parent):
        CollapsiblePane.__init__(self, parent,
                                 label='Step 4: Mapper parameters')
        Pane = self.Pane
        PaneBox = Vbox()
        Pane.SetSizer(PaneBox)

        # PaneBox.Add(wx.StaticText(Pane, label='Cover'),
        #            flag=wx.TOP|wx.LEFT, border=BORDER)
        self.Cover = CoverPanel(Pane)
        PaneBox.Add(self.Cover, flag=wx.EXPAND)
        PaneBox.Add(wx.StaticLine(Pane), flag=wx.EXPAND | wx.ALL,
                    border=BORDER)
        self.Clustering = ClusteringPanel(Pane)
        PaneBox.Add(self.Clustering, flag=wx.EXPAND)
        PaneBox.Add(wx.StaticLine(Pane), flag=wx.EXPAND | wx.ALL,
                    border=BORDER)
        # PaneBox.Add(wx.StaticText(Pane, label='Cutoff'),
        #            flag=wx.TOP|wx.LEFT, border=BORDER)
        self.Cutoff = CutoffPanel(Pane)
        PaneBox.Add(self.Cutoff, flag=wx.EXPAND)

    def GetValue(self):
        return { 'Cover' : self.Cover.GetValue(),
                 'Clustering' : self.Clustering.GetValue(),
                 'Cutoff' : self.Cutoff.GetValue() }

    def GetAllValues(self):
        return { 'IsCollapsed' : self.IsCollapsed,
                 'Cover' : self.Cover.GetAllValues(),
                 'Clustering' : self.Clustering.GetAllValues(),
                 'Cutoff' :  self.Cutoff.GetAllValues() }

    def SetValues(self, MapperPar):
        self.Collapse(MapperPar['IsCollapsed'])
        self.Cover.SetValues(MapperPar['Cover'])
        self.Clustering.SetValues(MapperPar['Clustering'])
        self.Cutoff.SetValues(MapperPar['Cutoff'])

class CoverPanel(ChoicePanel):
    def __init__(self, parent):
        ChoicesLabels = ('Uniform 1-d cover', 'Balanced 1-d cover',
                         'Subrange decomposition')
        ChoicesPanels = (Uniform1dCoverPanel, Uniform1dCoverPanel,
                         SubrangeDecompositionCover1dPanel)
        ChoicePanel.__init__(self, parent, ChoicesLabels, ChoicesPanels,
                             label='Cover')

class ClusteringPanel(ChoicePanel):
    def __init__(self, parent):
        ChoicesLabels = ['Single', 'Complete', 'Average', 'Weighted',
                         'Median', 'Centroid', 'Ward']
        ChoicesPanels = [NilPanel] * len(ChoicesLabels)
        ChoicePanel.__init__(self, parent, ChoicesLabels, ChoicesPanels,
                             label='Clustering')

class CutoffPanel(ChoicePanel):
    def __init__(self, parent):
        ChoicesLabels = ('First gap', 'Histogram method',
                         'Biggest gap', 'Biggest gap 2',
                         'Scale graph algorithm',
                         )
        ChoicesPanels = (FirstGapParPanel, HistogramMethodParPanel,
                          BiggestGapParPanel, BiggestGapParPanel,
                          ScaleGraphParPanel,
                          )
        ChoicePanel.__init__(self, parent, ChoicesLabels, ChoicesPanels,
                             label='Cutoff')

    def GetValue(self):
        Choice = self.Choices.GetStringSelection()
        Parameters = \
            self.ParameterPanels[self.Choices.GetSelection()].GetValue()
        return (Choice, Parameters)

class ScaleGraphParPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        vbox = Vbox()
        self.SetSizer(vbox)

        hbox1 = Hbox()
        vbox.Add(hbox1)

        hbox1.Add(wx.StaticText(self, label='Exponent'),
                  flag=wx.ALIGN_CENTER_VERTICAL)
        self.exponent = NumCtrl(self,
                                value=0.,
                                size=(54, -1),
                                style=wx.TE_RIGHT,
                                integerWidth=1,
                                fractionWidth=3,
                                groupDigits=False,
                                min=0.,
                                max=20.,
                                )
        self.exponent.SetFont(self.GetFont())
        hbox1.Add(self.exponent,
                  flag=wx.ALIGN_CENTER_VERTICAL | wx.LEFT | wx.RIGHT,
                  border=BORDER)


        self.expand_intervals = \
            wx.CheckBox(self, label='Expand intervals')
        hbox1.Add(self.expand_intervals,
                  flag=wx.ALIGN_CENTER_VERTICAL | wx.LEFT | wx.RIGHT,
                  border=BORDER)

        hbox1b = Hbox()
        vbox.AddSizer((0, BORDER))
        vbox.Add(hbox1b)

        hbox1b.Add(wx.StaticText(self, label='Max. # clusters'),
                  flag=wx.ALIGN_CENTER_VERTICAL | wx.RIGHT,
                  border=BORDER)
        self.maxcluster = wx.SpinCtrl(self, value='50', size=(64, -1),
                                      min=2, max=10000)
        hbox1b.Add(self.maxcluster)


        vbox.AddSizer((0, BORDER))
        vbox.Add(wx.StaticText(self,
                               label='Max-weight to min-weight conversion:'))
        hbox2 = Hbox()
        vbox.Add(hbox2)
        hbox2.Add(wx.StaticText(self, label='f(x) ='),
                  flag=wx.ALIGN_CENTER_VERTICAL)

        self.RB1 = wx.RadioButton(self, label='1/x', style=wx.RB_GROUP)
        # For OS X
        self.RB1.SetValue(True)
        hbox2.Add(self.RB1)
        hbox2.AddSpacer((2 * BORDER, -1))
        self.RB2 = wx.RadioButton(self, label='-x')
        hbox2.Add(self.RB2)
        hbox2.AddSpacer((2 * BORDER, -1))
        self.RB3 = wx.RadioButton(self, label='-(x^.5)')
        hbox2.Add(self.RB3)
        hbox2.AddSpacer((2 * BORDER, -1))
        self.RB4 = wx.RadioButton(self, label='log(1/x)')
        hbox2.Add(self.RB4)
        self.choices = { 'inverse' : self.RB1,
                         'linear' : self.RB2,
                         'root' : self.RB3,
                         'log' : self.RB4 }

    def GetValue(self):
        if self.RB1.GetValue():
            w = 'inverse'
        elif self.RB2.GetValue():
            w = 'linear'
        elif self.RB3.GetValue():
            w = 'root'
        else:
            w = 'log'
        return { 'exponent' : self.exponent.GetValue(),
                 'maxcluster' : self.maxcluster.GetValue(),
                 'expand_intervals' : self.expand_intervals.GetValue(),
                 'weighting' : w }

    GetAllValues = GetValue

    def SetValues(self, Config):
        self.exponent.SetValue(Config['exponent'])
        self.maxcluster.SetValue(Config['maxcluster'])
        self.choices[Config['weighting']].SetValue(True)
        self.expand_intervals.SetValue(Config['expand_intervals'])

class FirstGapParPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)

        hbox = Hbox()
        self.SetSizer(hbox)

        hbox.Add(wx.StaticText(self, label='Gap size'),
                 flag=wx.ALIGN_CENTER_VERTICAL)
        hbox.AddSpacer((BORDER, 0))
        self.gap = NumCtrl(self,
                           value=.1,
                           size=(54, -1),
                           style=wx.TE_RIGHT,
                           integerWidth=1,
                           fractionWidth=3,
                           groupDigits=False,
                           min=0.,
                           max=1.,
                           )
        self.gap.SetFont(self.GetFont())
        hbox.Add(self.gap)

    def GetValue(self):
        val = self.gap.GetValue()
        if val == 'Error':
            raise ParameterError(u'Invalid value for the “Gap size”.', self)
        return { 'gap' : self.gap.GetValue() }

    GetAllValues = GetValue

    def SetValues(self, Config):
        self.gap.SetValue(Config['gap'])

class HistogramMethodParPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)

        hbox = Hbox()
        self.SetSizer(hbox)

        hbox.Add(wx.StaticText(self, label='Number of bins'),
                 flag=wx.ALIGN_CENTER_VERTICAL)
        hbox.AddSpacer((BORDER, 0))
        self.num_bins = NumCtrl(self,
                           value=10,
                           size=(54, -1),
                           style=wx.TE_RIGHT,
                           integerWidth=5,
                           fractionWidth=0,
                           groupDigits=False,
                           min=2,
                           max=10000,
                           )
        self.num_bins.SetFont(self.GetFont())
        hbox.Add(self.num_bins)

    def GetValue(self):
        num_bins = self.num_bins.GetValue()
        if num_bins == 'Error':
            raise ParameterError('Invalid number of bins.', self)
        return { 'num_bins' : self.num_bins.GetValue() }

    GetAllValues = GetValue

    def SetValues(self, Config):
        self.num_bins.SetValue(Config['num_bins'])

class BiggestGapParPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)

        box = wx.FlexGridSizer(2, 2, hgap=BORDER, vgap=BORDER)
        self.SetSizer(box)

        box.Add(wx.StaticText(self, label='Exponent'),
                 flag=wx.ALIGN_CENTER_VERTICAL)
        self.exponent = NumCtrl(self,
                                value=0.,
                                size=(54, -1),
                                style=wx.TE_RIGHT,
                                integerWidth=1,
                                fractionWidth=3,
                                groupDigits=False,
                                min=0.,
                                max=20.,
                                )
        self.exponent.SetFont(self.GetFont())
        box.Add(self.exponent)
        box.Add(wx.StaticText(self, label='Max. # clusters'),
                 flag=wx.ALIGN_CENTER_VERTICAL)
        self.maxcluster = wx.SpinCtrl(self, value='50', size=(64, -1),
                                      min=2, max=10000)
        box.Add(self.maxcluster)

    def GetValue(self):
        return { 'exponent' : self.exponent.GetValue(),
                 'maxcluster' : self.maxcluster.GetValue(),
                 }

    GetAllValues = GetValue

    def SetValues(self, Config):
        self.exponent.SetValue(Config['exponent'])
        self.maxcluster.SetValue(Config['maxcluster'])

class Uniform1dCoverPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        hbox = Hbox()
        self.SetSizer(hbox)
        hbox.Add(wx.StaticText(self, label='Intervals'),
                  flag=wx.ALIGN_CENTER_VERTICAL)
        hbox.AddSpacer((BORDER, 0))
        self.Intervals = wx.SpinCtrl(self, value='15',
                                     size=(50, -1),
                                     min=1, max=999)
        hbox.Add(self.Intervals, flag=wx.ALIGN_CENTER_VERTICAL)
        hbox.AddSpacer((2 * BORDER, 0))
        hbox.Add(wx.StaticText(self, label='Overlap'),
                  flag=wx.ALIGN_CENTER_VERTICAL)
        hbox.AddSpacer((BORDER, 0))
        self.Overlap = NumCtrl(self,
                               value=50.,
                               size=(40, -1),
                               style=wx.TE_RIGHT,
                               integerWidth=5,
                               fractionWidth=2,
                               groupDigits=False,
                               min=0,
                               max=100,
                               )
        self.Overlap.SetFont(self.GetFont())
        hbox.Add(self.Overlap, flag=wx.ALIGN_CENTER_VERTICAL)
        hbox.Add(wx.StaticText(self, label=' %'),
                  flag=wx.ALIGN_CENTER_VERTICAL)

    def GetValue(self):
        intervals = self.Intervals.GetValue()
        if intervals == 'Error':
            raise ParameterError(u'Invalid “Intervals” specification.', self)
        overlap = self.Overlap.GetValue()
        if overlap == 'Error':
            raise ParameterError(u'Invalid “Overlap” specification.', self)
        return { 'intervals' : intervals,
                 'overlap'   : overlap }

    GetAllValues = GetValue

    def SetValues(self, Config):
        self.Intervals.SetValue(Config['intervals'])
        self.Overlap.SetValue(Config['overlap'])

class CsvTextCtrl(wx.TextCtrl):

    def __init__(self, parent, dtype=np.int, **kwargs):

        textctrl_kwargs = {}
        for k in ['size', 'style', 'value']:
            try:
                textctrl_kwargs[k] = kwargs[k]
            except KeyError:
                pass
        wx.TextCtrl.__init__(self, parent, **textctrl_kwargs)
        self.Bind(wx.EVT_TEXT, self.OnText)
        # self.Bind(wx.EVT_TEXT_ENTER, self.OnTextEnter)

        self.dtype = dtype

    def GetValue(self):
        t = super(CsvTextCtrl, self).GetValue()
        tl = t.split(',')
        tl = [s.strip(' ') for s in tl]
        # ignore trailing comma
        if tl[-1] == '': tl.pop()
        try:
            values = [ self.dtype(x) for x in tl ]
        except ValueError:
            self.SetBackgroundColour((255, 100, 100))
            values = 'Error'
        return values

    def SetValue(self, values):
        if values == 'Error':
            super(CsvTextCtrl, self).SetValue(values)
            self.SetBackgroundColour((255, 100, 100))
        else:
            s = [str(i) for i in values]
            super(CsvTextCtrl, self).SetValue(','.join(s))

    def OnText(self, event):
        t = super(CsvTextCtrl, self).GetValue()
        tl = t.split(',')
        tl = [s.strip(' ') for s in tl]
        # ignore trailing comma
        if tl[-1] == '': tl.pop()
        try:
            values = [ self.dtype(x) for x in tl ]
            self.SetBackgroundColour(
                wx.SystemSettings_GetColour(wx.SYS_COLOUR_WINDOW))
        except ValueError:
            self.SetBackgroundColour((255, 100, 100))


class SubrangeDecompositionCover1dPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)

        vbox = Vbox()
        self.SetSizer(vbox)

        hbox_1 = Hbox()
        hbox_1.Add(wx.StaticText(self, label='Intervals by subrange'),
                  flag=wx.ALIGN_CENTER_VERTICAL)
        hbox_1.AddSpacer((BORDER, 0))
        self.IntervalsBySubrange = CsvTextCtrl(self, value='15,15',
                                               size=(46, -1))
        hbox_1.Add(self.IntervalsBySubrange, flag=wx.ALIGN_CENTER_VERTICAL)
        vbox.Add(hbox_1)
        vbox.AddSpacer((2 * BORDER, 0))

        hbox_2 = Hbox()
        hbox_2.Add(wx.StaticText(self, label='Overlaps by subrange'),
                  flag=wx.ALIGN_CENTER_VERTICAL)
        hbox_2.AddSpacer((BORDER, 0))
        self.OverlapsBySubrange = CsvTextCtrl(self, value='50,50',
                                              size=(54, -1),
                                              style=wx.TE_RIGHT)
        self.OverlapsBySubrange.SetFont(self.GetFont())
        hbox_2.Add(self.OverlapsBySubrange, flag=wx.ALIGN_CENTER_VERTICAL)
        hbox_2.Add(wx.StaticText(self, label=' %'),
                  flag=wx.ALIGN_CENTER_VERTICAL)
        vbox.Add(hbox_2)
        vbox.AddSpacer((2 * BORDER, 0))

        hbox_3 = Hbox()
        hbox_3.Add(wx.StaticText(self, label='Subranges'),
                  flag=wx.ALIGN_CENTER_VERTICAL)
        hbox_3.AddSpacer((BORDER, 0))
        self.Subranges = CsvTextCtrl(self, value='', size=(54, -1),
                                     style=wx.TE_RIGHT, dtype=np.float)
        self.Subranges.SetFont(self.GetFont())
        hbox_3.Add(self.Subranges, flag=wx.ALIGN_CENTER_VERTICAL)
        vbox.Add(hbox_3)

    def GetValue(self):
        return { 'intervals_by_subrange' : self.IntervalsBySubrange.GetValue(),
                 'overlaps_by_subrange'   : self.OverlapsBySubrange.GetValue(),
                 'subrange_boundaries' : self.Subranges.GetValue() }

    GetAllValues = GetValue

    def SetValues(self, Config):
        self.IntervalsBySubrange.SetValue(Config['intervals_by_subrange'])
        self.OverlapsBySubrange.SetValue(Config['overlaps_by_subrange'])
        self.Subranges.SetValue(Config['subrange_boundaries'])

class MinSizesPanelClass(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)

        hbox = Hbox()
        self.SetSizer(hbox)

        label = wx.StaticText(self, label=('Cleanup: minimal number of '
                                           'elements per simplex'))
        hbox.Add(label, flag=wx.ALIGN_CENTER_VERTICAL | wx.LEFT | wx.RIGHT,
                 border=BORDER)

        self.MinSizes = wx.TextCtrl(self)
        hbox.Add(self.MinSizes, proportion=1)

        self.MinSizes.Bind(wx.EVT_TEXT, self.OnText)

    def OnText(self, event):
        t = self.MinSizes.GetValue()
        tl = t.split(',')
        tl = [s.strip(' ') for s in tl]
        # ignore trailing comma
        if tl[-1] == '': tl.pop()
        try:
            if np.any(np.array(tl, dtype=np.int) < 1):
                raise AssertionError
            self.MinSizes.SetBackgroundColour(
                wx.SystemSettings_GetColour(wx.SYS_COLOUR_WINDOW))
        except (ValueError, AssertionError):
            self.MinSizes.SetBackgroundColour((255, 100, 100))

    def GetValue(self):
        t = self.MinSizes.GetValue()
        tl = t.split(',')
        tl = [s.strip(' ') for s in tl]
        # ignore trailing comma
        if tl[-1] == '': tl.pop()
        try:
            f = [int(s) for s in tl]
            for i in f:
                if i < 1: raise AssertionError
        except (ValueError, AssertionError):
            self.MinSizes.SetBackgroundColour((255, 100, 100))
            raise ParameterError ('Invalid value for the minimal number of '
                 'elements per simplex.', self)
        return f

    GetAllValues = GetValue

    def SetValues(self, MinSizes):
        if MinSizes == 'Error':
            self.MinSizes.SetValue(MinSizes)
            self.MinSizes.SetBackgroundColour((255, 100, 100))
        else:
            s = [str(i) for i in MinSizes]
            self.MinSizes.SetValue(', '.join(s))

class SimpleComplexDummy:
    def __init__(self, CheckBox):
        self.CheckBox = CheckBox

    def GetValue(self):
        return self.CheckBox.IsChecked()

    GetAllValues = GetValue

    def SetValues(self, SimpleComplex):
        self.CheckBox.Check(SimpleComplex)

class MyDisplayParPane(CollapsiblePane):
    def __init__(self, parent):
        CollapsiblePane.__init__(self, parent,
                                 label='Step 5: Display parameters')
        Pane = self.Pane
        PaneBox = Vbox()
        Pane.SetSizer(PaneBox)

        hbox = Hbox()
        self.NodeColorCheckBox = \
            wx.CheckBox(Pane, label='Node coloring')
        hbox.Add(self.NodeColorCheckBox,
                 flag=wx.ALIGN_CENTER_VERTICAL | wx.LEFT, border=BORDER)
        self.NodeColor = wx.TextCtrl(Pane)
        hbox.Add(self.NodeColor,
                 flag=wx.ALIGN_CENTER_VERTICAL | wx.LEFT | wx.RIGHT,
                 proportion=1, border=BORDER)

        self.NodeColorCheckBox.Bind(wx.EVT_CHECKBOX, self.OnCheckBox)
        PaneBox.Add(hbox, flag=wx.EXPAND)
        self.MinSizesPanel = MinSizesPanelClass(Pane)
        PaneBox.Add(self.MinSizesPanel, flag=wx.EXPAND | wx.ALL,
                    border=BORDER)

    def GetValue(self):
        return { 'NodeColorEnabled' : self.NodeColorCheckBox.GetValue(),
                 'NodeColor' : self.NodeColor.GetValue(),
                 'MinSizes' : self.MinSizesPanel.GetValue() }

    def GetAllValues(self):
        return { 'IsCollapsed' : self.IsCollapsed,
                 'NodeColorEnabled' : self.NodeColorCheckBox.GetValue(),
                 'NodeColor' : self.NodeColor.GetValue(),
                 'MinSizes' : self.MinSizesPanel.GetAllValues() }

    def SetValues(self, Config):
        self.Collapse(Config['IsCollapsed'])
        self.NodeColorCheckBox.SetValue(Config['NodeColorEnabled'])
        self.NodeColor.SetValue(Config['NodeColor'])
        self.MinSizesPanel.SetValues(Config['MinSizes'])
        self.OnCheckBox()

    def OnCheckBox(self, event=None):
        self.NodeColor.Enable(self.NodeColorCheckBox.GetValue())

class MainPanel(wx.Panel, StatusUpdate, MapperJob):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)

        self.ValidDataDisplay = False
        self.ValidMapperDisplay = False
        self.MapperOutputFrame = None

        Box = Vbox()

        self.InputPane = MyInputPane(parent=self)
        Box.Add(self.InputPane, flag=wx.EXPAND | wx.TOP, border=BORDER)
        self.MetricPane = MyMetricPane(parent=self)
        Box.Add(self.MetricPane, flag=wx.EXPAND)
        self.FilterPane = MyFilterPane(parent=self)
        Box.Add(self.FilterPane, flag=wx.EXPAND)
        self.MapperParPane = MyMapperParPane(parent=self)
        Box.Add(self.MapperParPane, flag=wx.EXPAND)
        self.DisplayParPane = MyDisplayParPane(parent=self)
        Box.Add(self.DisplayParPane, flag=wx.EXPAND)

        hbox = Hbox()
        self.ExitButton = wx.Button(self, id=wx.ID_EXIT)
        self.InterruptButton = wx.Button(self, label='&Interrupt')
        self.InterruptButton.Enable(False)
        ScaleGraphButton = wx.Button(self, label='Generate &Scale Graph')
        ScaleGraphButton.Bind(wx.EVT_BUTTON, self.OnGenerateScaleGraph)

        self.MapperButton = wx.Button(self, id=wx.ID_OK, label='Run &Mapper')
        self.MapperButton.Bind(wx.EVT_BUTTON, self.OnRunMapper)

        hbox.AddStretchSpacer()
        hbox.Add(self.ExitButton, flag=wx.ALL, border=BORDER)
        hbox.Add(self.InterruptButton, flag=wx.ALL, border=BORDER)
        hbox.Add(ScaleGraphButton, flag=wx.ALL, border=BORDER)
        hbox.Add(self.MapperButton, flag=wx.ALL, border=BORDER)
        Box.Add(hbox, flag=wx.EXPAND | wx.BOTTOM, border=BORDER)
        Box.AddStretchSpacer()

        self.SetSizer(Box)

        self.SimpleComplex = SimpleComplexDummy(self.GetParent().Simple)

        self.InterruptButton.Bind(wx.EVT_BUTTON, self.OnInterrupt)

        self.SetMinSize((400, -1))
        self.Fit()

        self.PostStatusUpdate('Initialize the Mapper worker process...')
        self.MapperThread = MapperInterfaceThread(self, EvtMapperAnswer)
        self.Bind(EVT_MAPPER_ANSWER, self.OnMapperAnswer)
        self.Bind(MapperJob.EVT_MAPPER_JOB, self.OnMapperJob)
        self.FilterPane.HistButton.Bind(wx.EVT_BUTTON,
                                         self.OnRequestHistogram)
        self.MetricPane.MinNnghbrsButton.Bind(
            wx.EVT_BUTTON, self.OnMinNnghbrs)
        self.FilterPane.ViewButton.Bind(wx.EVT_BUTTON, self.OnViewData)
        self.InputPane.Bind(EVT_DATA_CHOICE, self.OnDataChoice)
        self.MetricPane.Bind(EVT_METRIC_CHOICE, self.OnMetricChoice)

        self.PostMapperJob('Start', {})

    def __del__(self):
        # Stop the worker thread and process before exiting.
        self.MapperThread.StopInterface()
        wx.Panel.__del__(self)

    def GetValue(self):
        Input = self.InputPane.Notebook.GetValue()
        PreprocessPar = self.InputPane.GetValue()
        PreprocessEnabled = PreprocessPar['PreprocessEnabled']
        Preprocess = PreprocessPar['Preprocess']
        Metric = self.MetricPane.GetValue()
        FilterFn = self.FilterPane.Notebook.GetValue()
        FilterTrafo = self.FilterPane.GetValue()
        MapperPar = self.MapperParPane.GetValue()
        MapperParameters = { 'Cover' : MapperPar['Cover'],
                             'Clustering' : MapperPar['Clustering'] }
        Cutoff = MapperPar['Cutoff']
        SimpleComplex = self.SimpleComplex.GetValue()
        DisplayPar = self.DisplayParPane.GetValue()
        MinSizes = DisplayPar['MinSizes']
        NodeColorEnabled = DisplayPar['NodeColorEnabled']
        NodeColor = DisplayPar['NodeColor']
        return { 'Input' : Input,
                 'PreprocessEnabled' : PreprocessEnabled,
                 'Preprocess' : Preprocess,
                 'Metric' : Metric,
                 'FilterFn' : FilterFn,
                 'FilterTrafo' : FilterTrafo,
                 'MapperParameters' : MapperParameters,
                 'Cutoff' : Cutoff,
                 'SimpleComplex' : SimpleComplex,
                 'MinSizes' : MinSizes,
                 'NodeColorEnabled' : NodeColorEnabled,
                 'NodeColor' : NodeColor }

    def GetAllValues(self):
        Input = self.InputPane.GetAllValues()
        Metric = self.MetricPane.GetAllValues()
        FilterFn = self.FilterPane.Notebook.GetAllValues()
        FilterTrafo = self.FilterPane.GetAllValues()
        MapperPar = self.MapperParPane.GetAllValues()
        SimpleComplex = self.SimpleComplex.GetAllValues()
        DisplayPar = self.DisplayParPane.GetAllValues()
        return { 'Input' : Input,
                 'Metric' : Metric,
                 'FilterFn' : FilterFn,
                 'FilterTrafo' : FilterTrafo,
                 'MapperPar' : MapperPar,
                 'SimpleComplex' : SimpleComplex,
                 'DisplayPar' : DisplayPar }

    def SetValues(self, Config):
        self.InputPane.SetValues(Config['Input'])
        self.MetricPane.SetValues(Config['Metric'])
        self.FilterPane.Notebook.SetValues(Config['FilterFn'])
        self.FilterPane.SetValues(Config['FilterTrafo'])
        self.MapperParPane.SetValues(Config['MapperPar'])
        self.SimpleComplex.SetValues(Config['SimpleComplex'])
        self.DisplayParPane.SetValues(Config['DisplayPar'])

    def OnInterrupt(self, event):
        print 'Restart worker process.'
        self.InterruptButton.Enable(False)
        self.PostStatusUpdate('Restart worker process.')
        wx.CallAfter(self.OnInterruptPhase2)

    def OnInterruptPhase2(self):
        self.MapperThread.RestartWorker()
        self.PostMapperJob('Start', {})

    def OnRequestHistogram(self, event):
        try:
            Parameter = self.GetValue()
        except ParameterError:
            return
        Parameter['Bins'] = 50
        self.SetCursor(wx.StockCursor(wx.CURSOR_WAIT))

        self.PostMapperJob('FilterHistogram', Parameter)
        print 'Request filter histogram'

    def OnRunMapper(self, event):
        try:
            Parameter = self.GetValue()
        except ParameterError:
            return
        self.PostMapperJob('RunMapper', Parameter)
        print 'Run Mapper'

    def OnSimpleCheckbox(self, event):
        self.PostStatusUpdate('Generate simple Mapper output (no triangles).' \
                                  if event.Checked() else
                              'Generate full Mapper output.')
        self.SimpleComplex.SetValues(event.Checked())

    def ReloadData(self):
        try:
            Parameter = self.GetValue()
        except ParameterError:
            return
        self.PostMapperJob('LoadData', Parameter)
        print 'Reload data'

    def OnGenerateScaleGraph(self, event):
        try:
            Parameter = self.GetValue()
        except ParameterError:
            return
        self.PostMapperJob('GenerateScaleGraph', Parameter)
        print 'Generate Scale Graph'

    def OnDataChoice(self, event):
        self.MetricPane.ClearMinNnghbrs()
        self.ValidDataDisplay = False
        self.ValidMapperDisplay = False
        event.Skip()

    def OnMetricChoice(self, event):
        self.MetricPane.ClearMinNnghbrs()

    def OnMinNnghbrs(self, event):
        try:
            Parameter = self.GetValue()
        except ParameterError:
            return
        self.PostMapperJob('MinNnghbrs', Parameter)
        print 'Find minimal number of neighbors'

    def OnMapperJob(self, event):
        self.MapperThread.SendJob(event.job)
        self.InterruptButton.Enable(True)

    def GenerateScript(self):
        try:
            Parameter = self.GetValue()
        except ParameterError:
            return
        self.PostMapperJob('GenerateScript', Parameter)

    def OnMapperAnswer(self, event):
        answer_type, data = event.answer
        if answer_type == 'Progress':
            self.ProgressUpdate(data)
        elif answer_type == 'Data info':
            self.InputPane.ReceiveDataInfo(data)
            self.MetricPane.ReceiveDataInfo(data)
        elif answer_type == 'StartupSuccess':
            ExampleShapes.SetShapeDir(data['MapperPath'])
            self.PostStatusUpdate('Ready.')
        elif answer_type == 'Error':
            self.PostStatusUpdate('Mapper error: ' + data.splitlines()[0])
            ErrorDialog(self, data)
        elif answer_type == 'ScaleGraph':
            self.M = data['MapperOutput']
            self.DisplayScaleGraph()
        elif answer_type == 'MapperOutput':
            self.M = data['MapperOutput']
            self.minsizes = data['MinSizes']
            self.DisplayMapperOutput(minsizes=self.minsizes,
                                     node_color=self.node_color,
                                     node_color_scheme=self.node_color_scheme)
        elif answer_type == 'FilterHistogram':
            self.FilterPane.DisplayFilterHistogram(data)
        elif answer_type == 'MinNnghbrs':
            self.MetricPane.DisplayMinNnghbrs(data)
        elif answer_type == 'InputData':
            self.OnInputData(data)
        elif answer_type == 'Script':
            self.OnScript(data)
        elif answer_type == 'NoPending':
            self.InterruptButton.Enable(False)
            # BeginBusyCursor and EndBusyCursor are nesting commands.
            # We circumvent this mechanism, as the 'NoPending' signal
            # detects independently when the application is not busy
            # any more.
            while wx.IsBusy():
                wx.EndBusyCursor()
        elif answer_type == 'Node colors':
            self.node_color, self.node_color_scheme = data
        else:
            raise ValueError('Unknown worker return value.')

    def OnInputData(self, data):
        Data, Filter, Mask = data
        self.PostStatusUpdate('Display input data.')
        self.DisplayData(Data, Filter, Mask)

    def DisplayData(self, *args):
        try:
            NewFrame = not isinstance(self.DataFrame, DataFrame)
        except AttributeError:
            NewFrame = True
        if NewFrame:
            self.DataFrame = DataFrame(self, title='Mapper input data',
                                           size=(800, 600))
            self.DataFrame.Show()
        else:
            self.DataFrame.Iconize(False)

        if self.MapperOutputFrame and self.ValidMapperDisplay:
            nodes = self.MapperOutputFrame.GetHighlightedNodes()
            plist, highlighted_nodes = self.HighlightNodes(nodes)
        else:
            plist = None
        self.DataFrame.Display(*args)
        self.ValidDataDisplay = True
        self.DataFrame.ShowDataPoints(plist)  # xxx
        self.DataFrame.Raise()

    def DisplayScaleGraph(self):
        self.PostStatusUpdate('Display the scale graph.')
        try:
            NewFrame = not isinstance(self.ScaleGraphFrame, ScaleGraphFrame)
        except AttributeError:
            NewFrame = True
        if NewFrame:
            self.ScaleGraphFrame = ScaleGraphFrame(self)
        else:
            self.ScaleGraphFrame.Clear()
        self.ScaleGraphFrame.Display(self.M.scale_graph_data)

    def DisplayMapperOutput(self, minsizes=(), node_color=None, node_color_scheme=None):
        self.PostStatusUpdate('Display the Mapper output.')
        try:
            NewFrame = not isinstance(self.MapperOutputFrame,
                                      MapperOutputFrame)
        except AttributeError:
            NewFrame = True
        if self.MapperOutputFrame:
            self.MapperOutputFrame.Clear()
        else:
            self.MapperOutputFrame = MapperOutputFrame(self)

        self.MapperOutputFrame.ResetHighlightedVertices()
        self.MapperOutputFrame.Display(self.M,
                                       minsizes=minsizes,
                                       node_color=node_color,
                                       node_color_scheme=node_color_scheme)
        if NewFrame:
            self.MapperOutputFrame.Bind(EVT_HIGHLIGHT_NODES,
                                        self.OnHighlightNodes)
            self.MapperOutputFrame.AdjustFrameSize()
        else:
            self.MapperOutputFrame.AdjustFigureSize()

        self.MapperOutputFrame.ShowFrame()
        self.ValidMapperDisplay = True

    def HighlightNodes(self, nodes):
        if len(nodes):
            # TBD: Rewrite so that event.nodes is renamed to be
            # event.node_indices or better just have event.nodes actually
            # contain nodes.
            highlighted_nodes = [self.M.nodes[node_index] \
                                     for node_index in nodes]
            plist = np.unique(np.hstack([n.points for n in highlighted_nodes]))
            # plist is sorted, according to the "unique" specification.
        else:
            highlighted_nodes = []
            plist = None

        return plist, highlighted_nodes

    def OnHighlightNodes(self, event):
        plist, highlighted_nodes = self.HighlightNodes(event.nodes)
        if self.ValidDataDisplay and self.ValidMapperDisplay:
            try:
                self.DataFrame.ShowDataPoints(plist)
            except AttributeError:
                pass

    def OnViewData(self, event):
        try:
            Parameter = self.GetValue()
        except ParameterError:
            return
        self.PostMapperJob('ViewData', Parameter)

    def OnScript(self, script):
        FileDialog = wx.FileDialog(self, 'Python script name',
                                   defaultDir=".",
                                   style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
                                   wildcard='Python files (*.py)|*.py')
        if FileDialog.ShowModal() == wx.ID_OK:
            path = FileDialog.GetPath()
            if path[-3:] != '.py':
                path += '.py'
            with open(path, 'w') as f:
                f.write(script)
            self.PostStatusUpdate("Python script was saved to {0}.".\
                                      format(path))
        else:
            self.PostStatusUpdate("Canceled.")
        FileDialog.Destroy()

class DataFrame(wx.Frame, ResizeableFrame):

    LastSaveDir = '.'
    LastSaveFilename = 'MapperScreenshot.png'

    def __init__(self, parent, **kwargs):
        wx.Frame.__init__(self, parent, **kwargs)
        try:
            self.Canvas = WxGLCanvas(self)
            self.valid_frame = True
        except EnvironmentError:
            self.valid_frame = False

        MenuBar = wx.MenuBar()
        View = wx.Menu()

        BalanceId = wx.NewId()
        self.Balance = wx.MenuItem(View, BalanceId, '&Balance filter values\t'
                                   'Ctrl-B', '', wx.ITEM_CHECK)
        View.AppendItem(self.Balance)
        self.Bind(wx.EVT_MENU, self.OnBalanceCheckbox, id=BalanceId)

        ResizeId = wx.NewId()
        View.Append(ResizeId, "&Resize window\tCtrl-R",
                            "Resize window to a given width and height")
        self.Bind(wx.EVT_MENU, self.OnManualResize, id=ResizeId)

        SnapshotId = wx.NewId()
        View.Append(SnapshotId, "&Snapshot\tCtrl-S",
                        "Save a snapshot as as bitmap file")
        self.Bind(wx.EVT_MENU, self.OnSnapshot, id=SnapshotId)

        MenuBar.Append(View, '&View')
        self.SetMenuBar(MenuBar)

    def OnBalanceCheckbox(self, event=None):
        self.BalanceFilter()
        self.Canvas.Recolor(self.FilterFinal)

    def OnSnapshot(self, event):
        FileDialog = wx.FileDialog(self, 'Save screenshot',
                                   defaultDir=DataFrame.LastSaveDir,
                                   style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
                                   wildcard="Portable Network Graphics (*.png)|*.png|"
                                   "All files (*.*)|*.*",
                                   defaultFile=DataFrame.LastSaveFilename)

        if FileDialog.ShowModal() == wx.ID_OK:
            path = FileDialog.GetPath()
            try:
                DataFrame.LastSaveDir, DataFrame.LastSaveFilename = \
                    os.path.split(path)
                img = self.Canvas.Snapshot()
                img.SaveFile(path, wx.BITMAP_TYPE_PNG)
                self.Parent.PostStatusUpdate(\
                    "Screenshot was saved to {0}.".format(path))
            except Exception as e:
                msg = 'The bitmap could not be saved: {0}'.format(e)
                print msg
                traceback.print_exc(None, sys.stderr)
                ErrorDialog(self, msg)
                self.Parent.PostStatusUpdate(msg.split('\n')[0])
        else:
            self.Parent.PostStatusUpdate("Canceled.")
        FileDialog.Destroy()

    def Display(self, Data, Filter, Mask):
        if not self.valid_frame: return
        assert Data.ndim == 2
        if Mask is None:
            self.Filter = Filter
            self.Data = Data
        else:
            self.Filter = Filter[Mask]
            self.Data = Data[Mask]
        self.BalanceFilter()

        dim = self.Data.shape[1]
        if dim not in (2,3):
            self.PostStatusUpdate('Cannot display {0}-dimensional data.' \
                                      .format(Data.shape[1]))
            del self.Data
            del self.Filter
            del self.FilterFinal
            return
        self.Canvas.MakeColormap(self.FilterFinal)
        self.Canvas.SetData(self.Data)

    def BalanceFilter(self):
        if self.Balance.IsChecked():
            self.FilterFinal = np.empty_like(self.Filter)
            self.FilterFinal[self.Filter.argsort()] = \
                np.arange(np.alen(self.Filter), dtype=np.float)
        else:
            self.FilterFinal = self.Filter

    def ShowDataPoints(self, plist):
        if self.valid_frame:
            self.Canvas.SetPList(plist)

    def OptimalHeight(self, resx):
        dim = self.Data.shape[1]
        if dim == 2:
            # to do: take the point size into account
            Range = self.Canvas.Max
            # 2 pixel extra margin
            corr = self.Canvas.PointSize + self.Canvas.border2d
            # S *= (m-self.PointSize-4)/m # 2 pixel extra margin
            # print self.Canvas.PointSize

            return int(np.ceil(float(resx - corr) * Range[1] / Range[0] + corr))
        elif dim == 3:
            # 3d data: quadratic window is optimal
            return resx

class MapperOutputFrame(FigureFrame):
    startsize = 640
    minsizes = ()
    LastNodeListDir = '.'
    LastNodeListFilename = 'NodeList.txt'

    def __init__(self, parent):
        FigureFrame.__init__(self, parent, title='Mapper output',
            size=(MapperOutputFrame.startsize, MapperOutputFrame.startsize))

        self.ResetHighlightedVertices()
        self.VertexCoords = None

        menuBar = wx.MenuBar()

        file_menu = wx.Menu()
        ToFileId = wx.NewId()
        file_menu.Append(ToFileId, "&Save figure\tCtrl-S",
                            "Save Mapper output figure to file")
        self.Bind(wx.EVT_MENU, self.OnToFile, id=ToFileId)
        SaveHighlightedNodesId = wx.NewId()
        file_menu.Append(SaveHighlightedNodesId, "Save &highlighted nodes\tCtrl-Shift-S",
                            "Save highlighted nodes to file")
        self.Bind(wx.EVT_MENU, self.OnSaveHighlightedNodes, id=SaveHighlightedNodesId)
        menuBar.Append(file_menu, "&File")

        options_menu = wx.Menu()
        RelabelId = wx.NewId()
        options_menu.Append(RelabelId, "Re&label\tAlt-L", "Relabel nodes")
        self.Bind(wx.EVT_MENU, self.OnNodesRelabel, id=RelabelId)
        menuBar.Append(options_menu, "&Options")

        view_menu = wx.Menu()
        ResetViewId = wx.NewId()
        view_menu.Append(ResetViewId, "&Reset\t1", "Reset view")
        self.Bind(wx.EVT_MENU, self.OnResetView, id=ResetViewId)
        ShowLabelsId = wx.NewId()
        self.ShowLabels = wx.MenuItem(view_menu, ShowLabelsId,
                                      '&Show Labels\tCtrl-L', '', wx.ITEM_CHECK)
        view_menu.AppendItem(self.ShowLabels)
        view_menu.Check(ShowLabelsId, True)
        self.Bind(wx.EVT_MENU, self.OnShowLabels, id=ShowLabelsId)
        ResizeId = wx.NewId()
        view_menu.Append(ResizeId, "&Resize window\tCtrl-R",
                            "Resize window to a given width and height")
        self.Bind(wx.EVT_MENU, self.OnManualResize, id=ResizeId)
        menuBar.Append(view_menu, "&View")
        self.SetMenuBar(menuBar)

        self.Bind(wx.EVT_CHAR, self.OnChar)

        self.M = None
        self.node_labels = None
        self.node_labels_scheme = None
        self.node_color = None
        self.node_color_scheme = None

    def Display(self, M=None, minsizes=(), node_color=None, node_color_scheme=None):
        self.minsizes = minsizes

        if M is not None:
            self.M = M

        if self.ShowLabels.IsChecked():
            node_labels = self.node_labels
            node_scale = 1000
        else:
            node_labels = 'empty'
            node_scale = 300

        bbox, vertices, vertex_pos, circles = \
            self.M.draw_2D(ax=self.axes, legend=True,
                           minsizes=minsizes,
                           node_labels=node_labels,
                           node_scale=node_scale,
                           node_labels_scheme=self.node_labels_scheme,
                           node_color=node_color,
                           node_color_scheme=node_color_scheme)

        self.SetBbox(bbox)
        self.VertexToNode = vertices
        self.VertexCoords = vertex_pos
        self.circles = circles
        self.node_facecolors = circles.get_facecolors()
        self.SetHighlightedVertices()

    def OnChar(self, event):
        keycode = event.GetKeyCode()
        if not event.HasModifiers():
            if keycode == ord('l'):
                self.HighlightLevelsetOffset(0)
            elif keycode == ord('>'):
                self.HighlightLevelsetOffset(1)
            elif keycode == ord('<'):
                self.HighlightLevelsetOffset(-1)
        super(type(self), self).OnChar(event)

    def ResetHighlightedVertices(self):
        self.HighlightedVertices = set()

    def HighlightLevelsetOffset(self, offset):
        if not self.M.nodes:
            return
        nodes = self.GetHighlightedNodes()
        levels = [self.M.nodes[n].level for n in nodes]
        levels1D = [l + offset for l, in levels]

        minlevel = min([n.level[0] for n in self.M.nodes])
        maxlevel = max([n.level[0] for n in self.M.nodes])

        levels = set([(l,) for l in levels1D if l >= minlevel and l <= maxlevel])

        if not len(levels1D) and offset != 0:
            if offset > 0:
                levels = ((minlevel,),)
            else:
                levels = ((maxlevel,),)

        self.HighlightedVertices = set((i for i, n in
                                        enumerate(self.VertexToNode)
                                        if self.M.nodes[n].level in levels))
        msg = 'Highlighted level{0}: {1}{2}'.\
            format('' if len(levels) == 1 else 's',
                   ', '.join([str(l) for l, in sorted(levels)]),
                   '' if len(self.HighlightedVertices) else
                   ' (no nodes)')
        print(msg)
        self.PostStatusUpdate(msg)
        self.DrawHighlightedVertices()

    def OnClick(self, coor, event):
        ds = self._cdist(np.array((coor,)), self.VertexCoords)
        i = ds.argmin()
        d = np.sqrt(ds[i])
        dp = (self.axes.transData.transform((d, 0)) \
                  - self.axes.transData.transform((0, 0)))[0]
        dp = dp * 72. / self.axes.get_figure().dpi
        if np.pi * dp * dp > 1200 * 1.1:
            vertex = []
        else:
            vertex = [i]

        if event.ShiftDown():
            self.HighlightedVertices = \
                self.HighlightedVertices.symmetric_difference(vertex)
        else:
            self.HighlightedVertices = set(vertex)

        self.DrawHighlightedVertices()
        event.Skip()

    def DrawHighlightedVertices(self):
        self.SetHighlightedVertices()
        self.canvas.draw()

    def SetHighlightedVertices(self):
        active = list(self.HighlightedVertices)
        newevent = EvtHighlightNodes(nodes=self.VertexToNode[active])
        wx.PostEvent(self, newevent)

        if active:
            facecolors = np.ones_like(self.node_facecolors)  # White color
            facecolors[active] = self.node_facecolors[active]
        else:
            facecolors = self.node_facecolors
        self.circles.set_facecolors(facecolors)

        '''
        # Experimental
        facecolors[0] = [0,0,0,1]
        self.circles.set_facecolors(facecolors)
        self.axes.draw_artist(self.circles)
        self.canvas.blit(self.axes.bbox)
        '''

    def GetHighlightedNodes(self):
        return self.VertexToNode[list(self.HighlightedVertices)]


    def OnSaveHighlightedNodes(self, event):
        endings = ['txt']
        endingslist = ';'.join(['*.' + e for e in endings])
        wildcard = 'Text file ({0})|{0}'.format(endingslist)
        FileDialog = wx.FileDialog(self, 'Data points in highlighted nodes',
                                   defaultDir=FigureFrame.LastSaveDir,
                                   style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
                                   wildcard=wildcard,
                                   defaultFile=MapperOutputFrame.LastNodeListFilename)
        if FileDialog.ShowModal() == wx.ID_OK:
            path = FileDialog.GetPath()
            try:
                MapperOutputFrame.LastNodeListDir, MapperOutputFrame.LastNodeListFilename = \
                    os.path.split(path)
                self.SaveNodeList(path)
                self.Parent.PostStatusUpdate(\
                    "Highlighted nodes were saved to {0}.".format(path))
            except Exception as e:
                msg = 'The highlighted nodes could not be saved: {0}'.format(e)
                print msg
                traceback.print_exc(None, sys.stderr)
                ErrorDialog(self, msg)
                self.Parent.PostStatusUpdate(msg.split('\n')[0])
        else:
            self.Parent.PostStatusUpdate("Canceled.")
        FileDialog.Destroy()

    def SaveNodeList(self, path):
        nodes_indices = self.GetHighlightedNodes()
        if len(nodes_indices):
            nodes_to_save = [self.M.nodes[node_index] \
                             for node_index in nodes_indices]
        else:
            nodes_to_save = self.M.nodes
        plist = np.unique(np.hstack([n.points for n in nodes_to_save]))
        # plist is sorted, according to the "unique" specification.

        with open(path, "w") as outfile:
            outfile.write('\n'.join(map(str, plist)))

    @staticmethod
    def _cdist(X, Y):
        '''
        Pairwise squared Euclidean distance between all row vectors in X
        and all row vectors in Y.
        '''
        d = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
        return np.ravel((d * d).sum(axis=2))

    def OnResetView(self, event):
        self.OriginalView()

    def OnNodesRelabel(self, event):
        FileDialog = wx.FileDialog(self, 'Choose a Python script for node labels',
                                   style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
                                   defaultDir=".",
                                   wildcard="Python file (*.py)|*.py|"
                                   "All files (*.*)|*.*"
                                   )
        if FileDialog.ShowModal() == wx.ID_OK:
            path = FileDialog.GetPath()
            try:
                import imp
                node_labels_module = imp.load_source("node_labels_module", path)
                #node_labels_module.init(self.M)
                self.node_labels = node_labels_module.label
                self.node_labels_scheme = node_labels_module.name
                self.Clear()
                self.Display(minsizes=self.minsizes)
            except Exception as e:
                traceback.print_exc(None, sys.stderr)
                ErrorDialog(self, repr(e))
                self.node_labels = None
                self.node_labels_scheme = None
                self.Clear()
                self.Display(minsizes=self.minsizes)
            self.ShowFrame()
        FileDialog.Destroy()

    def OnShowLabels(self, event):
        self.Clear()
        self.Display(minsizes=self.minsizes)
        self.canvas.draw()

    def OptimalHeight(self, resx):
        x1, y1, x2, y2 = self.MarginCoords((resx, resx))
        # The Mapper output is rotated for landscape format, so we can set
        # the y-size equal to the x-size, and the resulting coordinates
        # are as if we had not specified a desired y-size.
        return int(np.ceil(float(y2 - y1) / (x2 - x1) * resx))

class ScaleGraphFrame(FigureFrame):
    LastSaveFilename = 'ScaleGraph.svg'

    def __init__(self, parent):
        from mapper import scale_graph_axes_args, scale_graph_axes_kwargs
        FigureFrame.__init__(self, parent, title='Scale graph',
                             axesargs=scale_graph_axes_args,
                             axeskwargs=scale_graph_axes_kwargs)

        menuBar = wx.MenuBar()

        options_menu = wx.Menu()
        LogscaleId = wx.NewId()
        self.Logscale = wx.MenuItem(options_menu, LogscaleId,
                                    '&Logarithmic y-axis\t'
                                   'Ctrl-L', '', wx.ITEM_CHECK)
        options_menu.AppendItem(self.Logscale)
        self.Bind(wx.EVT_MENU, self.OnLogscale, id=LogscaleId)
        ResizeId = wx.NewId()
        options_menu.Append(ResizeId, "&Resize window\tCtrl-R",
                            "Resize window to a given width and height")
        self.Bind(wx.EVT_MENU, self.OnManualResize, id=ResizeId)
        ToFileId = wx.NewId()
        options_menu.Append(ToFileId, "&Save figure\tCtrl-S",
                            "Save scale graph figure to file")
        self.Bind(wx.EVT_MENU, self.OnToFile, id=ToFileId)

        ToPDFId = wx.NewId()
        options_menu.Append(ToPDFId, "Save figure as &PDF\tCtrl-P",
                            "Save scale graph figure to a compact PDF file. "
                            "The file size can be much smaller than the PDF "
                            "files which are generated by matplotlib.")
        self.Bind(wx.EVT_MENU, self.OnToPDFFile, id=ToPDFId)

        menuBar.Append(options_menu, "&Options")

        self.SetMenuBar(menuBar)

    def Display(self, sgd):
        self.sgd = sgd
        Bbox = sgd.draw_scale_graph(ax=self.axes,
                                    log=self.Logscale.IsChecked(),
                                    verbose=True)
        self.SetBbox(Bbox)
        self.ShowFrame()

    def OnLogscale(self, event=None):
        self.sgd.set_yaxis(self.axes, self.Logscale.IsChecked())
        self.canvas.draw()

    def OriginalView(self):
        M = self.MarginCoords()
        if self.aspect_ratio_1:
            self.ExpandToFullSize(M)
        self.axes.set_xlim(M[0], M[2])
        self.AtMargin[:] = True
        self.OnLogscale()

    def OnToPDFFile(self, event):
        wildcard = ('Portable Document Format (*.pdf)|*.pdf'
                   '|PDF + TeX (*.pdf, *.tex)|*.pdf;*.tex')
        FileDialog = wx.FileDialog(
            self, 'Mapper output figure',
            defaultDir=FigureFrame.LastSaveDir,
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
            wildcard=wildcard,
            defaultFile=ScaleGraphFrame.LastSaveFilename)
        if FileDialog.ShowModal() == wx.ID_OK:
            path = FileDialog.GetPath()
            try:
                FigureFrame.LastSaveDir, ScaleGraphFrame.LastSaveFilename = \
                    os.path.split(path)

                x0, x1 = self.axes.get_xlim()
                y0, y1 = self.axes.get_ylim()
                w, h = self.GetClientSize()
                # dpi = float(self.axes.get_figure().dpi)
                dpifactor = .5

                from mapper import save_scale_graph_as_pdf
                save_scale_graph_as_pdf(self.sgd, path,
                    width=w * dpifactor, height=h * dpifactor,
                    log_yaxis=self.Logscale.IsChecked(),
                    maxvertices=None,
                    bbox=(x0, y0, x1, y1),
                    tex_labels=(FileDialog.GetFilterIndex() == 1))
                self.Parent.PostStatusUpdate(\
                    "Mapper figure was saved to {0}.".format(path))
            except Exception as e:
                msg = 'The figure could not be saved: {0}'.format(e)
                print(msg)
                traceback.print_exc(None, sys.stderr)
                ErrorDialog(self, msg)
                self.Parent.PostStatusUpdate(msg.split('\n')[0])
        else:
            self.Parent.PostStatusUpdate("Canceled.")
        FileDialog.Destroy()

class MainFrame(wx.Frame, StatusUpdate):
    def __init__(self, parent, **kwargs):
        # super(type(self), self).__init__(parent, **kwargs)
        wx.Frame.__init__(self, parent, **kwargs)

        self.LastConfigDir = '.'

        MainMenuBar = wx.MenuBar()
        File = wx.Menu()
        ReloadId = wx.NewId()
        File.Append(ReloadId, '&Reload data\tCtrl-R')
        self.Bind(wx.EVT_MENU, self.OnReloadData, id=ReloadId)
        LoadConfigId = wx.NewId()
        File.Append(LoadConfigId, 'Load &Configuration')
        self.Bind(wx.EVT_MENU, self.OnLoadConfig, id=LoadConfigId)
        SaveConfigId = wx.NewId()
        File.Append(SaveConfigId, 'Save &Configuration')
        self.Bind(wx.EVT_MENU, self.OnSaveConfig, id=SaveConfigId)
        GenerateId = wx.NewId()
        File.Append(GenerateId, '&Generate standalone Python script\tCtrl-G')
        self.Bind(wx.EVT_MENU, self.GenerateScript, id=GenerateId)
        QuitId = wx.NewId()
        File.Append(QuitId, '&Quit\tCtrl-Q')
        self.Bind(wx.EVT_MENU, self.OnClose, id=QuitId)
        MainMenuBar.Append(File, '&File')
        Options = wx.Menu()
        SimpleId = wx.NewId()
        self.Simple = wx.MenuItem(Options, SimpleId, '&Simple Mapper output '
                       '(no threefold intersections)', '', wx.ITEM_CHECK)
        Options.AppendItem(self.Simple)
        MainMenuBar.Append(Options, '&Options')
        Help = wx.Menu()
        AboutId = wx.NewId()
        Help.Append(AboutId, "&About")
        self.Bind(wx.EVT_MENU, self.OnAbout, id=AboutId)
        MainMenuBar.Append(Help, 'H&elp')
        self.SetMenuBar(MainMenuBar)

        StatusBar = self.CreateStatusBar(style=0, number=1)
        self.Bind(EVT_UPDATE_STATUS, self.OnUpdateStatus)

        self.Panel = MainPanel(self)
        self.Bind(wx.EVT_MENU, self.Panel.OnSimpleCheckbox, id=SimpleId)

        self.LoadConfig()

        self.Panel.ExitButton.Bind(wx.EVT_BUTTON, self.OnClose)
        self.Bind(wx.EVT_CLOSE, self.OnClose)

        # Call this after all the CONTENT_CHANGE events have passed.
        wx.CallAfter(self.FinalizeFrame)

        self.Panel.MapperButton.SetFocus()

    def OnAbout(self, event):
        from mapper import __version__, __date__
        wx.MessageBox('GUI for Python Mapper\n\n'
                      'Copyright Daniel Müllner and '
                      'Aravindakshan Babu, 2011–2015\n\n'
                      'Mapper version {}, dated {}\n'
                      'GUI version {}, dated {}'.\
                      format(__version__, __date__, GUIversion, GUIdate),
                      'About',
                      wx.OK | wx.ICON_INFORMATION)

    def FinalizeFrame(self):
        self.OnContentChange()
        self.Bind(EVT_CONTENT_CHANGE, self.OnContentChange)
        self.Show()

    def OnClose(self, event):
        self.SetStatusText('Exiting...')
        self.SaveConfig()
        self.Destroy()

    def OnReloadData(self, event):
        self.Panel.ReloadData()

    def OnLoadConfig(self, event):
        FileDialog = wx.FileDialog(self, 'Choose a configuration file',
                                   style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
                                   defaultDir=self.LastConfigDir,
                                   wildcard="All files (*.*)|*.*"
                                   "|JSON file (*.json)|*.json"
                                   )
        if FileDialog.ShowModal() == wx.ID_OK:
            path = FileDialog.GetPath()
            rslt = self.LoadConfig(path=path)
            if rslt is True:
                msg = 'Loaded configuration from {0}.'.format(path)
            else:
                ErrorDialog(self, 'Could not load configuration file: ' +
                            str(rslt))
                msg = 'Could not load configuration from {0}.'.format(path)
            print(msg)
            self.PostStatusUpdate(msg)
            self.LastConfigDir = os.path.dirname(path)
        else:
            self.PostStatusUpdate("Configuration file selection was canceled.")
        FileDialog.Destroy()

    def OnSaveConfig(self, event):
        FileDialog = wx.FileDialog(self, 'Choose a configuration file',
                                   style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
                                   defaultDir=self.LastConfigDir,
                                   wildcard="All files (*.*)|*.*"
                                   "|JSON file (*.json)|*.json"
                                   )
        if FileDialog.ShowModal() == wx.ID_OK:
            path = FileDialog.GetPath()
            filename = os.path.basename(path)
            if filename[-5:] != '.json':
                path += '.json'
            Config = self.Panel.GetAllValues()
            try:
                with open(path, 'w') as f:
                    json.dump(Config, f)
                self.PostStatusUpdate('Saved configuration to {0}.'.
                                      format(path))
            except IOError as e:
                traceback.print_exc(None, sys.stderr)
                ErrorDialog(self, str(e))
                self.PostStatusUpdate('Could not save configuration to {0}.'.
                                      format(path))
            self.LastConfigDir = os.path.dirname(path)
        else:
            self.PostStatusUpdate("Configuration file selection was canceled.")
        FileDialog.Destroy()

    def OnUpdateStatus(self, event):
        if event.message:
            self.SetStatusText(unicode_if_str(event.message))
        else:
            if isinstance(event.gauge, str):
                self.SetStatusText(event.gauge)
            else:
                self.SetStatusText('Completed: {0}%'.format(int(event.gauge)))

    def OnContentChange(self, event=None):
        size = self.Panel.GetBestSize()
        self.SetMinSize((-1, -1))
        self.SetClientSize(size)
        newsize = self.GetSize()
        # Glitch in wxPython 2.8: The reported size is too small since it
        # disregards the menu bar.
        if oldwx and newsize.y - size[1] < 35:
            newsize += (0, 23)
            self.SetSize(newsize)
        self.SetMinSize(newsize)

    def SaveConfig(self):
        ParameterError.ShowErrorDialog = False
        try:
            Config = self.Panel.GetAllValues()
            with open('gui_config.json', 'w') as f:
                json.dump(Config, f)
        except (ParameterError, IOError) as e:
            print(u'-----------------------------------------------------\n'
                  u'Warning: The configuration file could not be written.\n'
                  u'{0}\n'
                  u'-----------------------------------------------------'.\
                      format(unicode(e)))
            if not isinstance(e, ParameterError):
                traceback.print_exc(None, sys.stderr)
        ParameterError.ShowErrorDialog = True

    def LoadConfig(self, event=None, path='gui_config.json'):
        try:
            self.SetStatusText('Load configuration.')
            with open(path, 'r') as f:
                Config = json.load(f)
            self.Panel.SetValues(Config)
            return True
        except (KeyError, TypeError) as e:
            traceback.print_exc(None, sys.stderr)
            self.SetStatusText('Incomplete configuration due to error.')
            ErrorDialog(self, u"The configuration file ‘gui_config.json’ "
                        "could not be processed completely. If the Mapper GUI "
                        "was updated before the last restart, this is "
                        "normal; otherwise search for bugs.\n\n"
                        "In any case, the Mapper GUI is fully usable, only "
                        "the last used parameters are lost.")
            return e
        except (IOError, ValueError) as e:
            print("No valid configuration file found. "
                  "Use default configuration.")
            self.SetStatusText('Default configuration.')
            return e

    def GenerateScript(self, event):
        self.Panel.GenerateScript()

class MapperGUI(wx.App):
    '''Main application'''
    def OnInit(self):
        sys.excepthook = self._excepthook
        self.GUIMainFrame = MainFrame(None, title='Python Mapper',
            style=wx.DEFAULT_FRAME_STYLE & (~wx.MAXIMIZE_BOX))

        self.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown)
        return True

    def OnKeyDown(self, event):
        key = event.GetKeyCode()
        if event.CmdDown() and key == ord('C'):
            self.GUIMainFrame.Close()
        else:
            event.Skip()

    def OnExit(self):
        print 'Exit'

    def _excepthook(self, etype, value, tb):
        try:
            import traceback
            traceback.print_exception(etype, value, tb)
            # _prev_excepthook(etype, value, traceback)
        except:
            pass
        finally:
            try:
                self.GUIMainFrame.Close()
            except:
                try:
                    print "Terminate worker process."
                    WorkerProcess.terminate()
                    WorkerProcess.join()
                except:
                    pass
                finally:
                    print 'Exit the main process.'
                    os._exit(1)

# _prev_excepthook = sys.excepthook

if __name__=='__main__':
    # Try to adjust the Python search path so that users do not need
    # to change it.
    mapperpath = os.path.abspath(os.path.join(os.path.split(
        os.path.realpath(__file__))[0],
        '..', '..'))
    if mapperpath not in sys.path:
        if os.path.isfile(os.path.join(mapperpath, 'mapper', '_mapper.py')):
            sys.path.append(mapperpath)
            print('Added ' + mapperpath + ' to the Python search path.')

    from multiprocessing import freeze_support
    freeze_support()
    app = MapperGUI(0)
    app.MainLoop()
