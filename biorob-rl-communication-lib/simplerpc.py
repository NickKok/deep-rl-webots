#
#  Synchronized subscriber
#
import zmq
#import cPickle as pickle
import pickle
import numpy as np
import copy
import uuid


INIT_DATA = {
    'qpos' : [],
    'qvel' : [],
    'com' : [],
    'mass' : [],
    'phase' : 1.0,
    'timestep' : 0,
    'time' : 0.0,
    'extrainfo' : [],
    'ctrl' : [] # Control signal
}

class SimpleRPCServer:
    '''
    This class is a simple RPC server that can be used to send data
    in a synchronized way with a client
    '''
    context = None
    server = None

    SUBSCRIBERS_EXPECTED = 1
    SUBSCRIBER_UUID = '';


    def __init__(self,port):
        self.context = zmq.Context()
        # Socket to receive signals
        self.server = self.context.socket(zmq.REP)
        self.workerID = port
        try:
            print("Creating server on tcp://*:{}".format(port))
            self.server.bind('tcp://*:{}'.format(port))
        except:
            print("Port : {} in use, Try an other one".format(port))

        #  We wait for N subscribers
        self.SUBSCRIBERS_EXPECTED = 1
        self.sync()

    def sync(self):
        # wait for synchronization request
        msg = self.server.recv()
        if(msg == self.SUBSCRIBER_UUID):
            self.SUBSCRIBER_UUID = ''


class WebotsCommunicatorService(SimpleRPCServer):
    '''
    This class is a child of SimpleRPCServer.
    The main differences is that it specifically ensures
    a structure in what is received from and sent to the client.
    '''

    def __init__(self,obs_dim,actuator_ctrlrange,port):
        SimpleRPCServer.__init__(self,port)
        self.actuator_ctrlrange = actuator_ctrlrange
        self.actuator_number = actuator_ctrlrange.shape[0]
        self.nq = obs_dim
        self.nv = obs_dim
        # Data about the current step
        self.lastData = None
        self.firstValidData = None
        self.data = INIT_DATA

        self.opt = {
            'timestep' : 0.001
        }
        self.runState = {
            'time' : 0.0
        }

        self.fast_mode = True
        self.reset_model = False
        self.dataThere = False
        self.syncstatus = True


    def update(self,*args):
        # this is called by do_simulation
        # a n_frames number of steps
        # this means that the simulation will use the same
        # actions for those n_steps

        # We want to send the actions to the client
        #try:
            act = []
            if(len(args) == 1):
                act = args[0]

            act = self.data['ctrl']
            send = {
            'act': act,
            'fast_mode': self.fast_mode,
            'reset_model': self.reset_model,
            'uuid' : self.SUBSCRIBER_UUID
            }
            serialized_send = pickle.dumps(send, protocol=0)
            self.server.send(serialized_send)

            if(self.reset_model == True):
                self.sync()
                self.runState = {
                    'time' : 0.0
                }
                self.reset_model = False
                self.data = copy.deepcopy(INIT_DATA)
                self.syncstatus = True
                return False
            else:
                # We want to get back info from the client
                serialized_obs = self.server.recv()
                if(serialized_obs != b""):
                    self.runState["time"] += self.opt["timestep"]
                    try:
                        obs = pickle.loads(serialized_obs)
                    except:
                        import ipdb; ipdb.set_trace()


                    # self.lastData = copy.deepcopy(self.data)
                    # We need from webots :
                    self.SUBSCRIBER_UUID = obs['uuid']
                    self.data['qpos'] = np.array(obs['qpos'])
                    self.data['qvel'] = np.array(obs['qvel'])
                    self.data['extrainfo'] = np.array(obs['extrainfo'])
                    self.data['com'] = np.array(obs['com'])
                    self.data['mass'] = np.array(obs['mass'])
                    self.data['phase'] = obs['phase']
                    self.data['timestep'] = np.array([obs['timestep']])
                    self.data['time'] = np.array([obs['time']])

                    if(self.data['time'][0] != self.runState['time']):
                        # Desychronization between server and client !
                        # We therefore require a reset
                        self.syncstatus = False
                    if(self.dataThere):
                        if(len(self.firstValidData.keys()) != sum([self.data[k].size == self.firstValidData[k].size for k in self.firstValidData.keys()])):
                            self.data = copy.deepcopy(self.firstValidData);
                        else:
                            self.lastData = copy.deepcopy(self.data)

                    if not self.dataThere and len(self.data['extrainfo']) != 0 :
                        self.firstValidData = copy.deepcopy(self.data)
                        self.lastData = copy.deepcopy(self.data)
                        self.dataThere = True
                return True

        # except:
        #     print("connection lost")
        #     self.SUBSCRIBER_UUID = ''
        #     self.server.send(pickle.dumps({}, protocol=0))
        #     self.sync()


class DumpBrainConnection:
    def __init__(self):
        pass
    def get(self):
        return {
            "fast_mode": False,
            "reset_model" : False,
            "act": np.random.randn(10)*10
            }
    def send(self,actions):
        pass


class SomebodyIsAlreadyConnected(Exception):
    pass

DEFAULT_PORT = 5662
class SimpleRPCClient:
    '''
    This class is a simple RPC client that can be used to send data
    in a synchronized way with a client
    '''
    context = None
    server = None
    uuid = None;
    port = 0
    port_offset = 0;
    automatic_discovery = False;

    def __init__(self,port=0):
        if port == 0: # e.g. this should be the opposite. If you don't want automatic_discovery you set the port.
            self.automatic_discovery = True
            self.port = DEFAULT_PORT
        else:
            self.port = port



        self.uuid = str(uuid.uuid4())
        self.context = zmq.Context()
        # Second, synchronize with server
        self.connect()

    def connect(self):
        self.syncclient = self.context.socket(zmq.REQ)
        print("Connected to tcp://localhost:{}".format(self.port+self.port_offset))
        self.syncclient.connect('tcp://localhost:{}'.format(self.port+self.port_offset))
        self.sync()

    def sync(self):
        # send a synchronization request
        self.syncclient.send(b'')

    def syncReset(self):
        # send a synchronization request
        self.syncclient.send(b'{}'.format(self.uuid))

    def get(self):
        data = pickle.loads(self.syncclient.recv())
        if(self.automatic_discovery and data['uuid'] != '' and data['uuid'] != self.uuid):
            self.port_offset = self.port_offset + 1
            self.connect()
            #raise SomebodyIsAlreadyConnected('UUID mismatch, server excepts : {}, got : {}'.format(data['uuid'],self.uuid))
        return data

    def send(self,obs):
        try:
            obs['uuid'] = self.uuid
            self.syncclient.send(pickle.dumps(obs, protocol=0))
        except:
            print("I will restart soon")
