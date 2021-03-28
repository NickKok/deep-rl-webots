import matplotlib.pyplot as plt
import numpy as np

class ContinousStateToDiscreMap(object):
    """
    A simple class to create a discreteMap
    """
    def __init__(self, size, center):
        self.map = np.zeros([int(size[0]),int(size[1])]);
        self.size = [float(size[0]),float(size[1])];
        self.center = center;

    def drawpoint(self,P,r):
        PN = P/self.size;
        pCN = self.center + PN;
        pC = pCN*self.size;
        xRange = np.arange(int(pC[0])-r,int(pC[0])+r+1,1)
        yRange = np.arange(int(pC[1])-r,int(pC[1])+r+1,1)
        [xNew,yNew] = np.meshgrid(xRange,yRange);
        self.map[xNew,yNew] = 255;

# Schema
# [torso,hips,knees,ankles,foots]

# points = [
#     0,0
#     30,0
#     60,20
#     60,-15
#     90,10
#     90,-30
#     90,5
#     90,-35
#     ];

pointsFromWebots = np.array([
    [1.4316434391768058, 0.2652967474702148],
    [1.041002046812994, 0.17931899279901475],
    [0.5551639712372729, 0.06120646336175953],
    [0.12456667052413116, -0.19289200172290702],
    [1.0409887605138592, 0.17932448748154078],
    [0.5827596631783166, 0.37939026680981286],
    [0.1988565214376072, 0.059046221424305954],
]);

# convert in centimeters and adapt axis and reference
pointsFromWebots = 100*(pointsFromWebots - pointsFromWebots[0,:])
pointsFromWebots[:,0] = pointsFromWebots[:,0]*-1;


mymap = ContinousStateToDiscreMap([200,200], [0.15,0.55])




for p in range(0,len(pointsFromWebots)):
    mymap.drawpoint(pointsFromWebots[p,:],3);

plt.imshow(mymap.map)
plt.show()
