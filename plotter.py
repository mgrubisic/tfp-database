############################################################################
#             	Plotter utility

# Created by: 	Huy Pham
# 				University of California, Berkeley

# Date created:	May 2020

# Description:  Plotter utility used to plot csv files in /outputs/

############################################################################

import pandas as pd
import matplotlib.pyplot as plt

plt.close('all')

dispColumns = ['time', 'isol1', 'isol2', 'isol3', 'isol4', 'isolLC']

isolDisp = pd.read_csv('./outputs/isolDisp.csv', sep=' ', header=None, names=dispColumns)
isolVert = pd.read_csv('./outputs/isolVert.csv', sep=' ', header=None, names=dispColumns)
isolRot  = pd.read_csv('./outputs/isolRot.csv', sep=' ', header=None, names=dispColumns)

story1Disp = pd.read_csv('./outputs/story1Disp.csv', sep=' ', header=None, names=dispColumns)
story2Disp = pd.read_csv('./outputs/story2Disp.csv', sep=' ', header=None, names=dispColumns)
story3Disp = pd.read_csv('./outputs/story3Disp.csv', sep=' ', header=None, names=dispColumns)

forceColumns = ['time', 'iAxial', 'iShearX', 'iShearY', 'iMomentX', 'iMomentY', 'iMomentZ', 'jAxial', 'jShearX', 'jShearY', 'jMomentX', 'jMomentY', 'jMomentZ']

isol1Force = pd.read_csv('./outputs/isol1Force.csv', sep = ' ', header=None, names=forceColumns)
isol2Force = pd.read_csv('./outputs/isol2Force.csv', sep = ' ', header=None, names=forceColumns)
isol3Force = pd.read_csv('./outputs/isol3Force.csv', sep = ' ', header=None, names=forceColumns)
isol4Force = pd.read_csv('./outputs/isol4Force.csv', sep = ' ', header=None, names=forceColumns)
isolLCForce = pd.read_csv('./outputs/isolLCForce.csv', sep = ' ', header=None, names=forceColumns)

outercolForce = pd.read_csv('./outputs/colForce1.csv', sep=' ', header=None, names=forceColumns)
innercolForce = pd.read_csv('./outputs/colForce2.csv', sep=' ', header=None, names=forceColumns)

diaphragmForce1 = pd.read_csv('./outputs/diaphragmForce1.csv', sep = ' ', header=None, names=forceColumns)
diaphragmForce2 = pd.read_csv('./outputs/diaphragmForce2.csv', sep = ' ', header=None, names=forceColumns)
diaphragmForce3 = pd.read_csv('./outputs/diaphragmForce3.csv', sep = ' ', header=None, names=forceColumns)

force1Normalize = -isol1Force['iShearX']/isol1Force['iAxial']
force2Normalize = -isol2Force['iShearX']/isol2Force['iAxial']
force3Normalize = -isol3Force['iShearX']/isol3Force['iAxial']
force4Normalize = -isol4Force['iShearX']/isol4Force['iAxial']
forceLCNormalize = -isolLCForce['iShearX']/isolLCForce['iAxial']

story1DriftOuter 	= (story1Disp['isol1'] - isolDisp['isol1'])/(13*12)
story1DriftInner 	= (story1Disp['isol2'] - isolDisp['isol2'])/(13*12)

story2DriftOuter 	= (story2Disp['isol1'] - story1Disp['isol1'])/(13*12)
story2DriftInner 	= (story2Disp['isol2'] - story1Disp['isol2'])/(13*12)

story3DriftOuter 	= (story3Disp['isol1'] - story2Disp['isol1'])/(13*12)
story3DriftInner 	= (story3Disp['isol2'] - story2Disp['isol2'])/(13*12)

sumAxial 		= isol1Force['iAxial'] + isol2Force['iAxial'] + isol3Force['iAxial'] + isol4Force['iAxial']

# fig = plt.figure()
# plt.plot(diaphragmForce1['time'], diaphragmForce1['iAxial'])
# plt.title('Diaphragm 1 axial')
# plt.xlabel('Time (s)')
# plt.ylabel('Axial force (kip)')
# plt.grid(True)

# fig = plt.figure()
# plt.plot(diaphragmForce1['time'], diaphragmForce1['iMomentZ'])
# plt.title('Diaphragm 1 moment')
# plt.xlabel('Time (s)')
# plt.ylabel('Moment Z (kip)')
# plt.grid(True)

# # Outer column hysteresis
# fig = plt.figure()
# plt.plot(isolDisp['isol1'], force1Normalize)
# plt.title('Isolator 1 hysteresis')
# plt.xlabel('Displ (in)')
# plt.ylabel('V/N')
# plt.grid(True)

# All hystereses
fig = plt.figure()
plt.plot(isolDisp['isol1'], force1Normalize)
plt.title('Isolator hystereses')
plt.xlabel('Displ (in)')
plt.ylabel('V/N')
plt.grid(True)

plt.plot(isolDisp['isol2'], force2Normalize)
plt.xlabel('Displ (in)')
plt.ylabel('V/N')
plt.grid(True)

plt.plot(isolDisp['isolLC'], forceLCNormalize)
plt.xlabel('Displ (in)')
plt.ylabel('V/N')
plt.grid(True)

# Displacement history
fig = plt.figure()
plt.plot(isolDisp['time'], isolDisp['isol1'])
plt.title('Bearing 1 disp history')
plt.xlabel('Time (s)')
plt.ylabel('Displ (in)')
plt.grid(True)

# # Displacement history
# fig = plt.figure()
# plt.plot(isolDisp['time'], isolDisp['isol4'])
# plt.title('Bearing 4 disp history')
# plt.xlabel('Time (s)')
# plt.ylabel('Displ (in)')
# plt.grid(True)

# Vertical displacement
fig = plt.figure()
plt.plot(isolDisp['time'], isolVert['isol1'])
plt.title('Bearing vertical disp history')
plt.xlabel('Time (s)')
plt.ylabel('Displ z (in)')
plt.grid(True)

plt.plot(isolDisp['time'], isolVert['isol2'])
plt.xlabel('Time (s)')
plt.ylabel('Displ z (in)')
plt.grid(True)

plt.plot(isolDisp['time'], isolVert['isol3'])
plt.xlabel('Time (s)')
plt.ylabel('Displ z (in)')
plt.grid(True)

plt.plot(isolDisp['time'], isolVert['isol4'])
plt.xlabel('Time (s)')
plt.ylabel('Displ z (in)')
plt.grid(True)

# Drift history
fig = plt.figure()
plt.plot(isolDisp['time'], story1DriftOuter)
plt.title('Story 1 outer drift history')
plt.xlabel('Time (s)')
plt.ylabel('Drift ratio')
plt.grid(True)

fig = plt.figure()
plt.plot(isolDisp['time'], story2DriftOuter)
plt.title('Story 2 outer drift history')
plt.xlabel('Time (s)')
plt.ylabel('Drift ratio')
plt.grid(True)

fig = plt.figure()
plt.plot(isolDisp['time'], story3DriftOuter)
plt.title('Story 3 outer drift history')
plt.xlabel('Time (s)')
plt.ylabel('Drift ratio')
plt.grid(True)


# Rotation history
# fig = plt.figure()
# plt.plot(isolDisp['isol1'], isolRot['isol1'])
# plt.title('Bearing rotation history, outer')
# plt.xlabel('Displ x (in)')
# plt.ylabel('Rotation (in/in)')
# plt.grid(True)

# fig = plt.figure()
# plt.plot(isolDisp['isol2'], isolRot['isol2'])
# plt.title('Bearing rotation history, inner')
# plt.xlabel('Displ x (in)')
# plt.ylabel('Rotation (in/in)')
# plt.grid(True)

# plt.plot(isolDisp['isol3'], isolRot['isol3'])
# plt.xlabel('Displ x (in)')
# plt.ylabel('Rotation (in/in)')
# plt.grid(True)

# plt.plot(isolDisp['isol4'], isolRot['isol4'])
# plt.xlabel('Displ x (in)')
# plt.ylabel('Rotation (in/in)')
# plt.grid(True)

# # Axial force history
# fig = plt.figure()
# plt.plot(isolDisp['time'], isol1Force['iAxial'])
# plt.title('Bearing 1 axial force')
# plt.xlabel('Time (s)')
# plt.ylabel('Axial force (k)')
# plt.grid(True)
# plt.ylim([0,300])

# fig = plt.figure()
# plt.plot(isolDisp['time'], isol2Force['iAxial'])
# plt.title('Bearing 2 axial force')
# plt.xlabel('Time (s)')
# plt.ylabel('Axial force (k)')
# plt.grid(True)
# plt.ylim([0,300])

# fig = plt.figure()
# plt.plot(isolDisp['time'], isol3Force['iAxial'])
# plt.title('Bearing 3 axial force')
# plt.xlabel('Time (s)')
# plt.ylabel('Axial force (k)')
# plt.grid(True)
# plt.ylim([0,300])

# fig = plt.figure()
# plt.plot(isolDisp['time'], isol4Force['iAxial'])
# plt.title('Bearing 4 axial force')
# plt.xlabel('Time (s)')
# plt.ylabel('Axial force (k)')
# plt.grid(True)
# plt.ylim([0,300])

# fig = plt.figure()
# plt.plot(isolDisp['time'], isolLCForce['iAxial'])
# plt.title('Bearing LC axial force')
# plt.xlabel('Time (s)')
# plt.ylabel('Axial force (k)')
# plt.grid(True)

# fig = plt.figure()
# plt.plot(isolDisp['time'], sumAxial)
# plt.title('Total axial force')
# plt.xlabel('Time (s)')
# plt.ylabel('Axial force (k)')
# plt.grid(True)

# Shear force history
fig = plt.figure()
plt.plot(outercolForce['time'], outercolForce['iShearX'])
plt.title('Column shear force, outer')
plt.xlabel('Time (s)')
plt.ylabel('Shear force (k)')
plt.grid(True)

# fig = plt.figure()
# plt.plot(outercolForce['time'], outercolForce['iAxial'])
# plt.title('Column axial force, outer')
# plt.xlabel('Time (s)')
# plt.ylabel('Axial force (k)')
# plt.grid(True)

# fig = plt.figure()
# plt.plot(innercolForce['time'], innercolForce['iShearX'])
# plt.title('Column shear force, inner')
# plt.xlabel('Time (s)')
# plt.ylabel('Axial force (k)')
# plt.grid(True)

# fig = plt.figure()
# plt.plot(innercolForce['time'], innercolForce['iAxial'])
# plt.title('Column axial force, inner')
# plt.xlabel('Time (s)')
# plt.ylabel('Axial force (k)')
# plt.grid(True)

# fig = plt.figure()
# plt.plot(isolDisp['isol1'], isol1Force['iShearX'])
# plt.title('Bearing 1 shear force')
# plt.xlabel('Displ (in)')
# plt.ylabel('shear force (k)')
# plt.grid(True)

# fig = plt.figure()
# plt.plot(isolDisp['isol2'], isol2Force['iShearX'])
# plt.title('Bearing 2 shear force')
# plt.xlabel('Displ (in)')
# plt.ylabel('shear force (k)')
# plt.grid(True)

# fig = plt.figure()
# plt.plot(isolDisp['isol3'], isol3Force['iShearX'])
# plt.title('Bearing 3 shear force')
# plt.xlabel('Displ (in)')
# plt.ylabel('shear force (k)')
# plt.grid(True)

# fig = plt.figure()
# plt.plot(isolDisp['isol4'], isol4Force['iShearX'])
# plt.title('Bearing 4 shear force')
# plt.xlabel('Displ (in)')
# plt.ylabel('shear force (k)')
# plt.grid(True)

# fig = plt.figure()
# plt.plot(beamRot['rot'], -outerbeamForce['jMomentZ'])
# plt.title('Outer beam moment curvature')
# plt.xlabel('Curvature')
# plt.ylabel('Moment')
# plt.grid(True)