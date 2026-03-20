###############################################################################
# MCIST: A Cistercian Numerals Image Dataset
# See: https://en.wikipedia.org/wiki/Cistercian_numerals
# Wout Mommen
# 13.03.2026
#
# https://github.com/WoutMommen/MCIST
# https://zenodo.org/records/19135381 with DOI: https://doi.org/10.5281/zenodo.19135381
#
# Data set: CC BY 4.0 license
# Code: MIT License
###############################################################################


###############
# Imports
###############
import os,sys,pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter,rotate,shift

###############
# Parameters
###############
NTRAIN = 60000 #60
NTEST = 10000  #10
SIZEY,SIZEX = 28,28 #MNIST size
JITTERC = 0.2 #central line jitter
JITTERL = 1.0 #outer line jitters
LINEWIDTH = 2
PICKLEFN = "mcist.pcl"
GFSIGMA = 1 #gaussian filter sigma
GFRAD = 1 #gaussian filter radius
MAXROTATION = 10 #degrees
MAXSHIFTX = 1.5
MAXSHIFTY = 1.5
THRESHOLD = 0.1

###############
# Classes
###############
class bc:
    HEADER = '\033[95m'
    RED   = "\033[1;31m"  
    BLUE  = "\033[1;34m"
    CYAN  = "\033[1;36m"
    GREEN = "\033[0;32m"
    RESET = "\033[0;0m"
    BOLD    = "\033[;1m"
    REVERSE = "\033[;7m"
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
#print(bc.RED + 'This is red.' + bc.ENDC)

###############
# Functions
###############
def GenLine(frompoint,topoint,jitter=JITTERL):
    t = np.linspace(0., 1., max(SIZEY,SIZEX)//2)
    #midpoint = (frompoint+topoint)/2 + np.random.uniform(low=-jitter,high=jitter,size=(1,2))
    #points = np.array([(1-t_)*frompoint + t_*topoint + 0.1*(1-t_)*t_*midpoint for t_ in t])
    midpoint1 = 2/3*frompoint + 1/3*topoint + np.random.uniform(low=-jitter,high=jitter,size=(1,2))
    midpoint2 = 1/3*frompoint + 2/3*topoint + np.random.uniform(low=-jitter,high=jitter,size=(1,2))
    points = np.array([ (1-t_)**3*frompoint + 3*midpoint1*t_*(1-t_)**2 + 3*midpoint2*t_**2*(1-t_) + t_**3*topoint  for t_ in t])
    points = points.round().astype('int').squeeze()
    #print('GenLine:', points.shape)
    return points

def GenDigit(digit,in1,in2,out1,out2):
    # in1----out1
    # |        |
    # |        |
    # in2----out2
    points = []
    if digit in [1,5,7,9]: #top hor line
        points.extend(GenLine(in1,out1))
    if digit in [2,8,9]: # bottom hor line
        points.extend(GenLine(in2,out2))
    if digit in [6,7,8,9]: #vert line
        points.extend(GenLine(out1,out2))
    if digit in [4,5]: # diagonal line bot to top
        points.extend(GenLine(in2,out1))
    if digit in [3]: #diagonal line top to bot
        points.extend(GenLine(in1,out2))
    points = np.array(points)
    #print('GenDigit:', points.shape)
    return points
        
def GenPoint(central,jitter=JITTERC):
    point = np.array(central) + np.random.uniform(low=-jitter,high=jitter,size=2)
    return point
    
def GenerateImage(label,size=(SIZEX,SIZEY)):
    assert  label>=0 and label<=9999
    units = label%10
    tens = (label//10)%10
    hundreds = (label//100)%10
    thousands = (label//1000)%10
    image = np.zeros(shape=(SIZEY,SIZEX))
    
    #generate the central line control points
    c1 = GenPoint([0.75/5*SIZEY,SIZEX/2])
    c2 = GenPoint([2/5*SIZEY,SIZEX/2])
    c3 = GenPoint([3/5*SIZEY,SIZEX/2])
    c4 = GenPoint([4.25/5*SIZEY,SIZEX/2])
    
    #generate the top left control points
    tl1 = GenPoint([0.75/5*SIZEY,SIZEX/2 - 1.25/5*SIZEX])
    tl2 = GenPoint([2/5*SIZEY,SIZEX/2 - 1.25/5*SIZEX])
    
    #generate the top right control points
    tr1 = GenPoint([0.75/5*SIZEY,SIZEX/2 + 1.25/5*SIZEX])
    tr2 = GenPoint([2/5*SIZEY,SIZEX/2 + 1.25/5*SIZEX])
    
    #generate the bottom left control points
    bl1 = GenPoint([3/5*SIZEY,SIZEX/2 - 1.25/5*SIZEX])
    bl2 = GenPoint([4.25/5*SIZEY,SIZEX/2 - 1.25/5*SIZEX])
    
    #generate the bottom right control points
    br1 = GenPoint([3/5*SIZEY,SIZEX/2 + 1.25/5*SIZEX])
    br2 = GenPoint([4.25/5*SIZEY,SIZEX/2 + 1.25/5*SIZEX])
    
    #generate the central vertical line
    c = np.vstack((c1,c2,c3,c4))
    y,x = c[:,0],c[:,1]
    spl = CubicSpline(y,x)
    ys = np.linspace(y[0],y[-1],SIZEY)
    xs = spl(ys)
    for x_,y_ in zip(xs,ys):
        image[int(y_),int(x_)] = 1.0
    
    #order of control points determined by position of digit
    #draw the units (top right)
    if units!=0:
        points = GenDigit(units,c1,c2,tr1,tr2)
        image[points[:,0],points[:,1]] = 1.0
    #draw the tens (top left)    
    if tens!=0:
        points = GenDigit(tens,c1,c2,tl1,tl2)
        image[points[:,0],points[:,1]] = 1.0
    #draw the hundreds (bottom right)
    if hundreds!=0:
        points = GenDigit(hundreds,c4,c3,br2,br1)
        image[points[:,0],points[:,1]] = 1.0
    #draw the thousands (bottom left)
    if thousands!=0:
        points = GenDigit(thousands,c4,c3,bl2,bl1)
        image[points[:,0],points[:,1]] = 1.0
        
    return image        

    
def To8bit(image):
    maxval = 2**8-1
    image = image - image.min()
    image = maxval * image/image.max()
    return image.astype('uint8')
 
def MakeOneNumeral(label): 
    image = GenerateImage(label)
    #blur
    image = gaussian_filter(image,sigma=GFSIGMA,radius=GFRAD)
    #rotate
    angle = np.random.uniform(low=-MAXROTATION,high=MAXROTATION)
    image = rotate(image,angle=angle,reshape=False)
    #shift
    shiftx = np.random.uniform(low=-MAXSHIFTX,high=MAXSHIFTX)
    shifty = np.random.uniform(low=-MAXSHIFTY,high=MAXSHIFTY)
    image = shift(image,shift=(shifty,shiftx))
    #normalize pixels
    image = image - image.min()
    image = image/image.max()
    image = image * (image>=THRESHOLD)
    image = np.power(image,0.75) #make pixels a bit brighter
    #to chars
    image = To8bit(image)
    return image
    
def MakeSet(numlabels):
    images = []
    labels = np.random.permutation([i%10000 for i in range(numlabels)]) #balanced shuffled dataset
    for n,label in enumerate(labels):
        if n%1000==0:
            print('.',end='')
        image = MakeOneNumeral(label)
        images.append(image)
    print()
    return images,labels


###############
# Main
###############
if __name__ == '__main__':
    np.random.seed(42)
    
    #If the dataset doesn't exist yet, build and save it
    if not os.path.exists(PICKLEFN):
    
        print(bc.BLUE + f'Building trainset of {NTRAIN} images.' + bc.ENDC)
        trainimages,trainlabels = MakeSet(NTRAIN)
        
        print(bc.BLUE + f'Building testset of {NTEST} images.' + bc.ENDC)
        testimages,testlabels = MakeSet(NTEST)
        
        print(bc.BLUE + f'Pickling dataset to file {PICKLEFN}.' + bc.ENDC)
        with open(PICKLEFN,'wb') as theFile:
            pickle.dump((trainimages,trainlabels,testimages,testlabels),theFile)
        print(bc.BLUE+'Done building the MCIST dataset.'+bc.ENDC)
    
    #If the dataset exists, load it
    else: 
        print(bc.BLUE + f'Loading MCIST dataset from file {PICKLEFN}.'+bc.ENDC)
        with open(PICKLEFN,'rb') as theFile:
            trainimages,trainlabels,testimages,testlabels = pickle.load(theFile)
        

    #Plot some examples
    nx,ny = 4,5
    fig, axs = plt.subplots(nx, ny)
    for px in range(nx):
        for py in range(ny):
            ax = axs[px,py]
            label = trainlabels[px*ny+py]
            image = trainimages[px*ny+py]
            #label = (py+px)*10**px
            #image = MakeOneNumeral(label)
            ax.imshow(image,cmap='Greys_r')
            ax.set_title(f'({label:04d})',fontsize=9)
            ax.set_xlim(0,SIZEX-1)
            ax.set_ylim(SIZEY-1,0)
            ax.set_xticks([0,SIZEX//2,SIZEX-1])
            ax.set_yticks([0,SIZEY//2,SIZEY-1])
            ax.tick_params(axis='both',labelsize=7)
    plt.tight_layout()
    plt.savefig('mcist_examples.png',dpi=100)
    plt.show()