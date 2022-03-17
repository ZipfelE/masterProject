import struct
import numpy as np

#Klasse um das Bytearray in eine Liste mit float Werten zu konvertieren
class ConvData:
    def __init__(self, PSinput):
        self.PSinput = PSinput


    # Funktion um den Index des aufrufenden Agenten zu extrahieren
    def get_agentIndex(self, PSinput):
        # agentIndex ist immer die erste Zahl im Array
        self.agentIndex = self.PSinput[0]
        return self.agentIndex

    #Funktion um die Aufgabe des RL-Agenten zu extrahieren und abzufragen
    def get_task(self, PSinput):
        #Aufgabennummer ist immer die zweite Zahl im Array
        self.task = self.PSinput[1]
        return self.task

    #Funktion um die Daten zur Initialisierung zu formatieren und auszulesen
    def get_init_data(self, PSinput):
        #Parameter werden auf initialzustand gesetzt
        self.rawdata = PSinput[2:]
        self.intNum = 0
        #4 ist die Größe des Datenpaketes für eine Zahl
        self.intNumTo = 4
        self.data = []
        self.arrayLength = len(self.rawdata)
        self.arrayLength = int(self.arrayLength/4) #teile durch 4 für 32bit float
        for i in range(self.arrayLength):
            x = bytearray(self.rawdata[i*4:self.intNumTo])
            x = struct.unpack("f", x)
            self.data.append(x)
            self.intNumTo += 4 # 4bytes 32bit
        return self.data

    #Funktion um die Daten für einen Trainingsschritt zu extrahieren und umzuformen
    def get_data(self, PSinput):
        self.rawdata = PSinput[2:]
        self.intNum = 0
        self.intNumTo = 4
        self.data = []
        self.arrayLength = len(self.rawdata)
        self.arrayLength = int(self.arrayLength/4) #divide by 4 32bit float
        for i in range(self.arrayLength):
            x = bytearray(self.rawdata[i*4:self.intNumTo])
            x = struct.unpack("f", x)
            self.data.append(x)
            self.intNumTo += 4 # 4bytes 32bit
        self.data = np.asarray(self.data, dtype=np.float32)
        return self.data

    #Funktion um die Daten für einen Trainingsschritt zu extrahieren und umzuformen
    def get_memory(self, PSinput):
        self.rawdata = PSinput[9:]
        # Werte zum separieren (3=number of state variables, 4=number of actions passed {1}, 5= number of rewards granted {1}, 6= number of new state variables, 7=number of total rewards {1}, 8= number of sim_episodes {1})
        split = (PSinput[3], PSinput[4], PSinput[5], PSinput[6], PSinput[7], PSinput[8])   #Variablen für die Länge jedes Arrays
        #initialisiere alle Listen um Daten aus übermitteltem Paket zu auszulesen             
        self.result = []
        self.state = []
        self.action = []
        self.reward = []
        self.next_state = []
        rNum = 0

        for k in range(6):
            if k == 0:
                intNumTo = 4
            arrayLength = split[k] #Länge des Arrays de Zustands
            for i in range(arrayLength):
                x = bytearray(self.rawdata[rNum*4:intNumTo])
                x = struct.unpack("f", x)
                if k == 0: #lese den aktuellen Zustand aus dem Array
                    self.state.append(x)
                elif k == 1: #lese die Aktion vom Array
                    x = int(x[0])
                    self.action = x
                elif k == 2: #lese die Belohnung aus dem Array
                    x = round(x[0],2)
                    self.reward = x
                elif k == 3: #lese Folgezustand aus Array
                    self.next_state.append(x)
                elif k == 4: #lese Gesamtbelohnung aus Array
                    self.total_reward = x[0]
                elif k == 5:
                    self.sim_episode = x[0]
                intNumTo += 4 # 4bytes 32bit
                rNum += 1 #Schleifenvariable

        #formatiere die Listen der extrahierten Daten in NumPyArrays -> für Keras notwendig
        self.state = np.asarray(self.state, dtype=np.float32)
        self.next_state = np.asarray(self.next_state, dtype=np.float32)
        self.result = (self.state, self.action, self.reward, self.next_state, self.total_reward, self.sim_episode)
        
        #Rückgabewert sind Listen der Zustände, Aktionen, Belohnungen, Folgezustände, Gesamtbelohnung und SimEpisode
        return self.result
        
        
