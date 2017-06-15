import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pyaudio
import sys
import serial
from matplotlib.colors import LogNorm
from scipy.signal import welch
import mysql.connector
import datetime
import time
import atexit

# Mic Constants
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

# Serial Port Constants
SERIAL_PATH = '/dev/tty.usbmodem1421'
UART_BAUD_RATE = 9600
TIMEOUT = 1

# DB Constants
DB_NAME = 'swm'

# Flow Meter Constants
N_TEMPLATES = 12
TEMPLATE_LISTEN_TIME = 10
MEASURE_LISTEN_TIME = 3


def chiSquare(a, b):
    assert len(a) == len(b)
    return np.sum((a - b) ** 2)


def unix_time_millis(dt):
    epoch = datetime.datetime.utcfromtimestamp(0)
    return (dt - epoch).total_seconds() * 1000.0


class FlowMeter():
    def __init__(self):
        self.init_mic()
        self.init_database()
        self.ser = None
        self.init_serial(SERIAL_PATH)
        atexit.register(self.cleanup)
        self.templates = None
        self.templateLevels = None

    '''
    Runs at start of program.  By default loads templates, listens to pipe,
    makes a guess at the flow rate, then stores it into the database
    '''
    def start(self, make_templates=False):
        print("Templating...")
        if make_templates:
            self.make_templates()
        else:
            self.load_templates()
            print(self.templateLevels)
            print(self.templates)
        print("Logging...")
        while(1):
            self.make_measurement()
            # time.sleep(3)

    '''
    Generate new templates from mic data and known flows from inline meter
    '''
    def make_templates(self):
        self.templateLevels = np.empty(N_TEMPLATES, dtype='float64')
        self.templates = np.empty((N_TEMPLATES, 257), dtype='float64')

        for tNdx in range(N_TEMPLATES):
            raw_input("Ready?")
            ampl = self.listen(TEMPLATE_LISTEN_TIME)
            flow = self.get_flow()
            f, tedge, power, nwindows = self.get_power(None,
                                                       ampl.astype(np.float))
            template = self.make_template(f, tedge, power)
            self.templates[tNdx] = template
            self.templateLevels[tNdx] = flow
            self.store_template(template, flow)

        print(self.templateLevels)

    '''
    Load existing templates from a database
    '''
    def load_templates(self):
        self.templateLevels = np.empty(N_TEMPLATES, dtype='float64')
        self.templates = np.empty((N_TEMPLATES, 257), dtype='float64')

        try:
            self.cursor.execute("SELECT * FROM Templates")
        except mysql.connector.Error as err:
            print("Failed querying templates: {}".format(err))
            exit(1)

        tNdx = 0
        for (array, flow) in self.cursor:
            self.templateLevels[tNdx] = flow
            self.templates[tNdx] = np.loads(str(array))
            tNdx += 1

    '''
    Listen to the pipe, guess the flow, and store it
    '''
    def make_measurement(self):
        ampl = self.listen(MEASURE_LISTEN_TIME)
        f, tedge, powers, nwindows = self.get_power(None, ampl.astype(np.float))

        # powers = [self.templates[0] * 1.01]

        estimatedFlow = 0
        for power in powers:
            estimatedFlow += self.guess(power)
        estimatedFlow /= len(powers)

        print(estimatedFlow)
        self.store_estimated_flow(estimatedFlow, storeActualFlow=False)

    '''
    Use PyAudio to setup a USB mic
    '''
    def init_mic(self):
        self.audio = pyaudio.PyAudio()

        self.stream = self.audio.open(format=FORMAT, channels=CHANNELS,
                                      rate=RATE, input=True,
                                      frames_per_buffer=CHUNK)
        self.stream.stop_stream()

    '''
    Record using the USB mic
    '''
    def listen(self, seconds):
        self.stream.start_stream()

        print('Listening')
        sys.stdout.flush()

        frames = np.empty(((RATE / CHUNK * seconds), CHUNK), dtype='int16')

        for i in range(0, int(RATE / CHUNK * seconds)):
            data = self.stream.read(CHUNK)
            frames[i] = np.fromstring(data, np.int16)

        self.stream.stop_stream()

        print('Finished Listening')

        return frames.flatten()

    '''
    Cleanup USB mic resources
    '''
    def cleanup_mic(self):
        print("Cleaning up mic")
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

    '''
    Setup MySQL database connection
    '''
    def init_database(self):
        print("Connecting to database")
        cnx = mysql.connector.connect(user='root', password='YOUR_PW_HERE',
                                      host='127.0.0.1')
        self.cursor = cnx.cursor()
        self.cursor.cnx = cnx

        # Create database if it doesn't exist
        try:
            cnx.database = DB_NAME
        except mysql.connector.Error as err:
            if err.errno == mysql.connector.errorcode.ER_BAD_DB_ERROR:
                self.fresh_database()
                cnx.database = DB_NAME
            else:
                print(err)
                exit(1)

    '''
    Make a fresh database instance
    '''
    def fresh_database(self):
        print("Making fresh database")
        try:
            self.cursor.execute(
                "CREATE DATABASE {} DEFAULT CHARACTER SET 'utf8'"
                .format(DB_NAME))
            self.cursor.cnx.database = DB_NAME
            self.cursor.execute(
                "CREATE TABLE `Templates` ("
                "PRIMARY KEY(flow),"
                "`array` BLOB NOT NULL,"
                "`flow` DOUBLE NOT NULL"
                ")"
            )
            self.cursor.execute(
                "CREATE TABLE `FlowData` ("
                "PRIMARY KEY(whenRecorded),"
                "`whenRecorded` DATETIME NOT NULL,"
                "`estimatedFlow` DOUBLE NOT NULL,"
                "`actualFlow` DOUBLE"
                ")"
            )
        except mysql.connector.Error as err:
            print("Failed creating database: {}".format(err))
            exit(1)

    '''
    Close database connection
    '''
    def cleanup_database(self):
        print("Cleaning up database")
        self.cursor.close()
        self.cursor.cnx.close()

    '''
    Setup serial connection for inline meter
    '''
    def init_serial(self, serialPort):
        if self.ser:
            print("Closing existing connection")
            self.ser.close()
            self.ser = None
        self.ser = serial.Serial(serialPort, UART_BAUD_RATE, timeout=TIMEOUT)
        self.ser.reset_input_buffer()
        print("Opening Device {}".format(self.ser.name))

    '''
    Read flow from inline meter Reasonable figures are
    from about 300 (GPH) to about 10 (GPH)
    '''
    def get_flow(self):
        retVal = None
        print("Reading flow from serial port")
        self.ser.write(b'f')
        line = self.ser.readline()
        if line is '':
            print("Read no flow")
            return 0
        try:
            retVal = float(line[:-4])
            if retVal > 400:
                raise ValueError("Unreasonable flow value larger than 400 GPH")
        except ValueError as err:
            print("Failed to convert {} to float: {}".format(line, err))
            return self.get_flow()
        print("Read flow of {} GPH".format(retVal))
        return retVal

    '''
    Return a raw serial reading
    '''
    def get_raw(self):
        # self.ser.reset_input_buffer()
        print(self.ser.readline())

    '''
    Close the serial port
    '''
    def cleanup_serial(self):
        print("Cleaning up serial port")
        # self.ser.reset_input_buffer()
        self.ser.close()

    '''
    Convert time domain audio file to power spectrum
    '''
    def get_power(self, t, x, window_length_ms=200., segment_length=512):
        # Calculate the sampling frequency from the time stamps.
        sampling_frequency = RATE
        # Calculate the window size in samples
        window_size = int(round(1e-3 * window_length_ms * sampling_frequency))
        # Calculate how many complete windows we have.
        nwindows = int(np.floor(x.size / window_size))
        # Calculate the timestamp at each window edge.
        tedge = 1e-3 * window_length_ms * np.arange(nwindows + 1)
        # Allocate memory for the log-power in each window.
        nfreq = segment_length // 2 + 1
        power = np.empty((nwindows, nfreq))
        # Loop over windows.
        for i in range(nwindows):
            xwin = x[i * window_size: (i + 1) * window_size]
            f, power[i] = welch(xwin, fs=sampling_frequency,
                                nperseg=segment_length)
            assert len(f) == nfreq
        return f, tedge, power, nwindows

    '''
    Collapse power spectra to a single template and optionally plot it
    '''
    def make_template(self, f, tedge, power, plot=False):
        template = np.mean(power, axis=0)

        if (plot):
            # So it displays :)
            thisSegmentAvg = np.array((template, template))

            plt.figure(figsize=(12, 8))
            max_power = np.max(thisSegmentAvg)
            plt.imshow(thisSegmentAvg.T, interpolation='none',
                       norm=LogNorm(vmin=1e-5 * max_power, vmax=max_power),
                       extent=(tedge[0], tedge[-1], f[0], f[-1] / 1e3),
                       aspect='auto', cmap='magma', origin='lower')

            plt.xlabel('Time [sec]')
            plt.ylabel('Frequency [KHz]')
            plt.xlim(tedge[0], tedge[-1])
            plt.ylim(0, 2)
            plt.colorbar(pad=0.02).set_label(
                'Power Spectral Density [ADU/sqrt[Hz)]]')

        return template

    '''
    Store a newly created template in the database
    '''
    def store_template(self, template, flow):
        try:
            self.cursor.execute(
                "INSERT INTO Templates(array, flow) VALUES(%s, %s)",
                (template.dumps(), flow))
        except mysql.connector.Error as err:
            print("Failed inserting template: {}".format(err))
            exit(1)
        self.cursor.cnx.commit()

    '''
    Store the flow guess in the database.  Optionall store the flow
    from the inline meter.
    '''
    def store_estimated_flow(self, estimatedFlow, storeActualFlow=False):
        actualFlow = None
        if storeActualFlow:
            actualFlow = float(self.get_flow())
        try:
            self.cursor.execute(
                "INSERT INTO FlowData(whenRecorded, estimatedFlow, actualFlow)"
                "VALUES(%s, %s, %s)",
                (datetime.datetime.now(), float(estimatedFlow), actualFlow))
        except mysql.connector.Error as err:
            print("Failed inserting estimated flow: {}".format(err))
            exit(1)
        self.cursor.cnx.commit()

    '''
    Generate a figure comparing guessed flow to inline meter flow
    '''
    def plot_flow(self):
        timestamp = []
        estimated = []
        actual = []

        try:
            self.cursor.execute("SELECT * FROM FlowData")
        except mysql.connector.Error as err:
            print("Failed querying estimated flow: {}".format(err))
            exit(1)

        for (whenRecorded, estimatedFlow, actualFlow) in self.cursor:
            print((unix_time_millis(whenRecorded), estimatedFlow, actualFlow))
            if estimatedFlow < 300 and actualFlow < 300:
                ts = datetime.datetime.fromtimestamp(unix_time_millis(whenRecorded) / 1000.0)
                timestamp.append(ts)
                estimated.append(estimatedFlow)
                actual.append(actualFlow)

        fig = plt.figure()
        # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
        # plt.gca().xaxis.set_major_locator(mdates.DayLocator())

        plt.plot(timestamp, estimated, label='Estimated Flow')
        plt.plot(timestamp, actual, label='Actual Flow')

        plt.gcf().autofmt_xdate()

        plt.legend(loc='upper center', shadow=True)
        fig.savefig('temp.png', dpi=fig.dpi)

    '''
    Guess the flow by comparing an input power spectrum to the known flow
    templates in self.templates
    '''
    def guess(self, power):
        chiSquareds = np.zeros(len(self.templates))

        for i in np.arange(len(self.templates)):
            chiSquareds[i] = chiSquare(self.templates[i], power)

        chiSquareds = 1 - (chiSquareds / np.max(chiSquareds))
        print(chiSquareds)

        return self.templateLevels[np.argmax(chiSquareds)]

    '''
    Closes mic, database connection, and serial port
    '''
    def cleanup(self):
        self.cleanup_mic()
        self.cleanup_database()
        self.cleanup_serial()


'''
If run from the command line as python flowmeter.py, start the meter
'''
if __name__ == "__main__":
    FlowMeter().start(make_templates=False)
    # FlowMeter().plot_flow()
