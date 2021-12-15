############################################################################
#             	Build model

# Created by: 	Huy Pham
# 				University of California, Berkeley

# Date created:	May 2020

# Description: 	Script models designed structure (superStructDesign.py) in OpenSeesPy.
#				Returns loading if function called

# Open issues: 	(1) Nonlinear beamWithHinges needs verification for HingeRadau
#				(2) Bearing kv is arbitrary
#				(3) Moat wall parameters arbitrary
# 				(4) Impact element not stopping movement

# following example on http://opensees.berkeley.edu/wiki/index.php/Elastic_Frame_Example
# nonlinear model example from https://opensees.berkeley.edu/wiki/index.php/OpenSees_Example_5._2D_Frame,_3-story_3-bay,_Reinforced-Concrete_Section_%26_Steel_W-Section#Elastic_Element
# using Ex5.Frame2D.build.InelasticSection.tcl
############################################################################


# import OpenSees and libraries
import numpy as np
import matplotlib.pyplot as plt
import math

from openseespy.opensees import *
from openseespy.postprocessing.Get_Rendering import * 

############################################################################
#              Utilities
############################################################################

# get shape properties
def getProperties(shape):
	Ag 		= float(shape.iloc[0]['A'])
	Ix 		= float(shape.iloc[0]['Ix'])
	Iy 		= float(shape.iloc[0]['Iy'])
	Zx 		= float(shape.iloc[0]['Zx'])
	Sx 		= float(shape.iloc[0]['Sx'])
	d 		= float(shape.iloc[0]['d'])
	bf 		= float(shape.iloc[0]['bf'])
	tf 		= float(shape.iloc[0]['tf'])
	tw 		= float(shape.iloc[0]['tw'])
	return(Ag, Ix, Iy, Zx, Sx, d, bf, tf, tw)

# create or destroy fixed base, for eigenvalue analysis
def refix(nodeTag, action):
	for j in range(1,7):
		remove('sp', nodeTag, j)
	if(action == "fix"):
		fix(nodeTag,  1, 1, 1, 1, 1, 1)
	if(action == "unfix"):
		fix(nodeTag,  0, 1, 0, 1, 0, 1)

# returns load to analysis script
def giveLoads():
	return(w0, w1, w2, w3, pLc0, pLc1, pLc2, pLc3)

# add superstructure damping (dependent on eigenvalue anly)
def provideSuperDamping(regTag, omega1, zetaTarget):
	alphaM 		= 0.0
	betaK 		= 0.0
	betaKInit 	= 0.0
	a1 			= 2*zetaTarget/omega1
	region(regTag, '-eleRange', 1, 21, '-rayleigh', alphaM, betaK, betaKInit, a1)

############################################################################
#              Start model
############################################################################

def build():

	# remove existing model
	wipe()
	wipeAnalysis()
	plt.close('all')

	############################################################################
	#              Definitions
	############################################################################

	# units: in, kip, s
	# dimensions
	inch 	= 1.0
	in2 	= inch*inch
	in4 	= inch*inch*inch*inch
	ft 		= 12.0*inch
	sec 	= 1.0
	g 		= 386.4*inch/(sec**2)
	pi 		= math.pi
	kip 	= 1.0
	ksi 	= kip/(inch**2)

	# set modelbuilder
	# command: model('basic', '-ndm', ndm, '-ndf', ndf=ndm*(ndm+1)/2)
	model('basic', '-ndm', 3, '-ndf', 6)

	import superStructDesign as sd
	(mu1, mu2, mu3, R1, R2, R3, moatGap, selectedBeam, selectedRoofBeam, selectedCol) = sd.design()

	(AgCol, IzCol, IyCol, ZxCol, SxCol, dCol, bfCol, tfCol, twCol) = getProperties(selectedCol)
	(AgBeam, IzBeam, IyBeam, ZxBeam, SxBeam, dBeam, bfBeam, tfBeam, twBeam) = getProperties(selectedBeam)
	(AgRoofBeam, IzRoofBeam, IyRoofBeam, ZxRoofBeam, SxRoofBeam, dRoofBeam, bfRoofBeam, tfRoofBeam, twRoofBeam) = getProperties(selectedRoofBeam)

	############################################################################
	#              Model construction
	############################################################################

	global w0, w1, w2, w3, pLc0, pLc1, pLc2, pLc3

	# assuming mass only includes a) dead load, no weight, no live load, unfactored, or b) dead load + live load, factored
	# masses represent half the building's mass

	# m0Inner = 46.9*kip/g
	# m1Inner = 46.9*kip/g
	# m2Inner = 46.9*kip/g
	# m3Inner = 35.6*kip/g

	# m0Outer = 23.4*kip/g
	# m1Outer = 23.4*kip/g
	# m2Outer = 23.4*kip/g
	# m3Outer = 17.8*kip/g

	m0Inner = 81.7*kip/g
	m1Inner = 81.7*kip/g
	m2Inner = 81.7*kip/g
	m3Inner = 58.1*kip/g

	m0Outer = 40.9*kip/g
	m1Outer = 40.9*kip/g
	m2Outer = 40.9*kip/g
	m3Outer = 29.0*kip/g

	w0 = 2.72*kip/(1*ft)
	w1 = 2.72*kip/(1*ft)
	w2 = 2.72*kip/(1*ft)
	w3 = 1.94*kip/(1*ft)

	pOuter = (w0 + w1 + w2 + w3)*15*ft
	pInner = (w0 + w1 + w2 + w3)*30*ft

	# Leaning column loads

	pLc0 = 482.0*kip
	pLc1 = 482.0*kip
	pLc2 = 482.0*kip
	pLc3 = 340.0*kip

	pLc  = pLc0 + pLc1 + pLc2 + pLc3

	# mLc0 = 281.2*kip/g
	# mLc1 = 281.2*kip/g
	# mLc2 = 281.2*kip/g
	# mLc3 = 213.7*kip/g

	mLc0 = 490.4*kip/g
	mLc1 = 490.4*kip/g
	mLc2 = 490.4*kip/g
	mLc3 = 348.4*kip/g

	# create nodes
	# command: node(nodeID, x-coord, y-coord, z-coord)
	node(1, 	0.0*ft,  	0.0*ft,		1.0*ft)
	node(2, 	30.0*ft, 	0.0*ft,		1.0*ft)
	node(3, 	60.0*ft, 	0.0*ft,		1.0*ft)
	node(4, 	90.0*ft, 	0.0*ft,		1.0*ft)

	node(5, 	0.0*ft,  	0.0*ft,		14.0*ft)
	node(6, 	30.0*ft,	0.0*ft,		14.0*ft)
	node(7, 	60.0*ft,	0.0*ft,		14.0*ft)
	node(8, 	90.0*ft,	0.0*ft,		14.0*ft)

	node(9, 	0.0*ft, 	0.0*ft,		27.0*ft)
	node(10,	30.0*ft, 	0.0*ft,		27.0*ft)
	node(11,	60.0*ft, 	0.0*ft,		27.0*ft)
	node(12,	90.0*ft, 	0.0*ft,		27.0*ft)

	node(13,	0.0*ft, 	0.0*ft,		40.0*ft)
	node(14,	30.0*ft, 	0.0*ft,		40.0*ft)
	node(15,	60.0*ft, 	0.0*ft,		40.0*ft)
	node(16,	90.0*ft, 	0.0*ft,		40.0*ft)

	# Base isolation layer node i
	node(17,	0.0*ft, 	0.0*ft,		0.0*ft)
	node(18,	30.0*ft, 	0.0*ft,		0.0*ft)
	node(19,	60.0*ft, 	0.0*ft,		0.0*ft)
	node(20,	90.0*ft, 	0.0*ft,		0.0*ft)

	# Leaning columns
	node(21, 120.0*ft, 	0.0*ft,	0.0*ft)

	node(22, 120.0*ft, 	0.0*ft,	1.0*ft)

	node(23, 120.0*ft,	0.0*ft,	14.0*ft)
	node(24, 120.0*ft,	0.0*ft,	14.0*ft)
	node(25, 120.0*ft,	0.0*ft,	14.0*ft)

	node(26, 120.0*ft, 	0.0*ft,	27.0*ft)
	node(27, 120.0*ft, 	0.0*ft,	27.0*ft)
	node(28, 120.0*ft, 	0.0*ft,	27.0*ft)

	node(29, 120.0*ft, 	0.0*ft,	40.0*ft)
	node(30, 120.0*ft, 	0.0*ft,	40.0*ft)

	# Moat wall
	node(31, 0.0*ft,  	0.0*ft,		1.0*ft)
	node(32, 90.0*ft, 	0.0*ft,		1.0*ft)

	# Make 3DOF nodes for zeroLengthImpact3D
	model('basic', '-ndm', 3, '-ndf', 3)

	node(311, 0.0*ft,  	0.0*ft,		1.0*ft)			# restrained/retained
	node(312, 0.0*ft,  	0.0*ft,		1.0*ft)			# constrained

	node(321, 90.0*ft, 	0.0*ft,		1.0*ft)			# constrained
	node(322, 90.0*ft, 	0.0*ft,		1.0*ft)			# restrained/retained

	# return to 6DOF
	model('basic', '-ndm', 3, '-ndf', 6)


	# assign masses, in direction of motion and stiffness
	# DOF list: X, Y, Z, rotX, rotY, rotZ
	mass(1,		m0Outer,	m0Outer,	0.0, 	0.0,	0.0,	0.0)
	mass(2,		m0Inner,	m0Inner,	0.0, 	0.0,	0.0,	0.0)
	mass(3,		m0Inner,	m0Inner,	0.0, 	0.0,	0.0,	0.0)
	mass(4,		m0Outer,	m0Outer,	0.0, 	0.0,	0.0,	0.0)

	mass(5,		m1Outer,	m1Outer,	0.0, 	0.0,	0.0,	0.0)
	mass(6,		m1Inner,	m1Inner,	0.0, 	0.0,	0.0,	0.0)
	mass(7,		m1Inner,	m1Inner,	0.0, 	0.0,	0.0,	0.0)
	mass(8,		m1Outer,	m1Outer,	0.0, 	0.0,	0.0,	0.0)

	mass(9,		m2Outer,	m2Outer,	0.0, 	0.0,	0.0,	0.0)
	mass(10,	m2Inner,	m2Inner,	0.0, 	0.0,	0.0,	0.0)
	mass(11,	m2Inner,	m2Inner,	0.0, 	0.0,	0.0,	0.0)
	mass(12,	m2Outer,	m2Outer,	0.0, 	0.0,	0.0,	0.0)

	mass(13,	m3Outer,	m3Outer,	0.0, 	0.0,	0.0,	0.0)
	mass(14,	m3Inner,	m3Inner,	0.0, 	0.0,	0.0,	0.0)
	mass(15,	m3Inner,	m3Inner,	0.0, 	0.0,	0.0,	0.0)
	mass(16,	m3Outer,	m3Outer,	0.0, 	0.0,	0.0,	0.0)

	mass(22,	mLc0,	mLc0,	0.0, 	0.0,	0.0,	0.0)
	mass(24,	mLc1,	mLc1,	0.0, 	0.0,	0.0,	0.0)
	mass(27,	mLc2,	mLc2,	0.0, 	0.0,	0.0,	0.0)
	mass(30,	mLc3,	mLc3,	0.0, 	0.0,	0.0,	0.0)

	# restraints
	# command: fix(nodeID, DOF1, DOF2, DOF3) 0 = free, 1 = fixed
	for j in range(17,21):
	    fix(j, 1, 1, 1, 1, 1, 1)

	fix(21, 1, 1, 1, 1, 1, 1)

	# stopgap solution: restrain all nodes from moving in y-plane and rotating about X & Z axes
	for j in range(1,5):
		fix(j, 0, 1, 0, 1, 0, 1)

	for j in range(5,17):
		fix(j, 0, 1, 0, 1, 0, 1)

	fix(22, 0, 1, 0, 1, 0, 1)
	fix(24, 0, 1, 0, 1, 0, 1)
	fix(27, 0, 1, 0, 1, 0, 1)
	fix(30, 0, 1, 0, 1, 0, 1)

	fix(31, 1, 1, 1, 1, 1, 1)
	fix(32, 1, 1, 1, 1, 1, 1)

	# geometric transformation for beam-columns
	# command: geomTransf('Type', TransfTag)
	# command: geomTransf('Linear', transfTag, *vecxz, '-jntOffset', *dI, *dJ) for 3d

	beamTransfTag 	= 1
	colTransfTag 	= 2

	geomTransf('Linear', beamTransfTag, 0, -1, 0) #beams
	geomTransf('PDelta', colTransfTag, 0, -1, 0) #columns

	# define elements and section

	# Columns
	colSecTag 		= 1		# column section tag

	# Beams
	beamSecTag 		= 2		# beam section tag
	roofSecTag 		= 3

	# Torsion
	torsionSecTag 	= 4

	# Fiber section tags
	colFiberTag 	= 5

	# General elastic section (non-plastic beam columns, leaning columns)
	elasticColSecTag 		= 10
	elasticBeamSecTag 		= 11
	elasticRoofBeamSecTag 	= 12
	LCSpringMatTag 			= 13
	frameLinkTag 			= 14

	# Steel material tag
	steelMatTag			= 31

	# Isolation layer tags
	frn1ModelTag 	= 41
	frn2ModelTag 	= 42
	frn3ModelTag 	= 43
	fpsMatPTag 		= 44
	fpsMatMzTag 	= 45

	# Plastic hinge model tags
	hingeColTag 	= 90
	hingeBeamTag 	= 91
	hingeRoofTag 	= 92


	# define material: Steel01
	# command: uniaxialMaterial('Steel01', matTag, Fy, E0, b, a1, a2, a3, a4)
	Fy 	= 50*ksi		# yield strength
	Es 	= 29000*ksi		# initial elastic tangent
	nu 	= 0.3			# Poisson's ratio
	Gs 	= Es/2/(1 + nu) # Torsional stiffness modulus
	J 	= 1e10			# Set large torsional stiffness
	b  	= 0.1			# hardening ratio
	uniaxialMaterial('Steel01', steelMatTag, Fy, Es, b)
	uniaxialMaterial('Elastic', torsionSecTag, J)

	# Fiber section parameters
	nfw			= 4		# number of fibers in web
	nff			= 8		# number of fibers in each flange

	# column section: fiber wide flange section
	# command:  section('WFSection2d', secTag, matTag, d, tw, bf, tf, Nfw, Nff)
	section('WFSection2d', colFiberTag, steelMatTag, dCol, twCol, bfCol, tfCol, nfw, nff)
	section('Aggregator', colSecTag, torsionSecTag, 'T', '-section', colFiberTag)

	# # beam section: fiber wide flange section
	# # command:  section('WFSection2d', secTag, matTag, d, tw, bf, tf, Nfw, Nff)
	# section('WFSection2d', beamFiberTag, steelMatTag, dBeam, twBeam, bfBeam, tfBeam, nfw, nff)
	# section('Aggregator', beamSecTag, torsionSecTag, 'T', '-section', beamFiberTag)

	# column section: uniaxial section
	# colMatFlexTag 	= 6
	# colMatAxialTag 	= 7
	# EICol 			= Es*IzCol				# EI, for moment-curvature relationship
	# EACol 			= Es*AgCol				# EA, for axial-force-strain relationship
	# MyCol 			= SxCol*Fy				# yield moment kip*in
	# bCol 			= b						# strain-hardening ratio (ratio between post-yield tangent and initial elastic tangent)
	# uniaxialMaterial('Steel01', colMatFlexTag, MyCol, EICol, bCol) 				# bilinear behavior for flexure
	# uniaxialMaterial('Elastic', colMatAxialTag, EACol)							# this is not used as a material, this is an axial-force-strain response
	# section('Aggregator', colSecTag, colMatAxialTag, 'P', colMatFlexTag, 'Mz', torsionSecTag, 'T')	# combine axial and flexural behavior into one section (no P-M interaction here)

	# roof beam section: uniaxial section
	roofMatFlexTag 	= 6
	roofMatAxialTag = 7
	EIRoof 			= Es*IzRoofBeam				# EI, for moment-curvature relationship
	EARoof 			= Es*AgRoofBeam				# EA, for axial-force-strain relationship
	MyRoof 			= SxRoofBeam*Fy				# yield moment kip*in
	bRoof 			= b							# strain-hardening ratio (ratio between post-yield tangent and initial elastic tangent)
	uniaxialMaterial('Steel01', roofMatFlexTag, MyRoof, EIRoof, bRoof) 				# bilinear behavior for flexure
	uniaxialMaterial('Elastic', roofMatAxialTag, EARoof)							# this is not used as a material, this is an axial-force-strain response
	section('Aggregator', roofSecTag, roofMatAxialTag, 'P', roofMatFlexTag, 'Mz', torsionSecTag, 'T')	# combine axial and flexural behavior into one section (no P-M interaction here)

	# beam section: uniaxial section
	beamMatFlexTag 	= 8
	beamMatAxialTag = 9
	EIBeam 			= Es*IzBeam				# EI, for moment-curvature relationship
	EABeam 			= Es*AgBeam				# EA, for axial-force-strain relationship
	MyBeam 			= SxBeam*Fy				# yield moment kip*in
	bBeam 			= b						# strain-hardening ratio (ratio between post-yield tangent and initial elastic tangent)
	uniaxialMaterial('Steel01', beamMatFlexTag, MyBeam, EIBeam, bBeam) 				# bilinear behavior for flexure
	uniaxialMaterial('Elastic', beamMatAxialTag, EABeam)							# this is not used as a material, this is an axial-force-strain response
	section('Aggregator', beamSecTag, beamMatAxialTag, 'P', beamMatFlexTag, 'Mz', torsionSecTag, 'T')	# combine axial and flexural behavior into one section (no P-M interaction here)

	# General elastic section
	# command: section('Elastic', secTag, E, A, Iz, Iy, G, J, alphaY=0.0, alphaZ=0.0)
	section('Elastic', elasticColSecTag, Es, AgCol, IzCol, IyCol, Gs, J)
	section('Elastic', elasticBeamSecTag, Es, AgBeam, IzBeam, IyBeam, Gs, J)
	section('Elastic', elasticRoofBeamSecTag, Es, AgRoofBeam, IzRoofBeam, IyRoofBeam, Gs, J)

	# Frame link
	ARigid = 1000.0			# define area of truss section (make much larger than A of frame elements)
	IRigid = 1e6*in4		# moment of inertia for p-delta columns  (make much larger than I of frame elements)
	uniaxialMaterial('Elastic', frameLinkTag, Es)


	# Isolator parameters
	uy 			= 0.00984*inch 			# 0.025cm from Scheller & Constantinou
	# DSlider 	= 4*inch				# diameter of slider
	# ASlider 	= pi*DSlider**2/4		# area of slider
	# aspectRatio = 0.52					# defined as height slider / diameter slider
	# hSlider 	= aspectRatio*DSlider	# height of slider
	# EASlider 	= Es*ASlider			# used for vertical stiffness of slider

	dSlider1 	= 4	*inch				# slider diameters
	dSlider2 	= 11*inch
	dSlider3 	= 11*inch

	d1 		= 10*inch	- dSlider1		# displacement capacities
	d2 		= 37.5*inch	- dSlider2
	d3 		= 37.5*inch	- dSlider3

	h1 		= 1*inch					# half-height of sliders
	h2 		= 4*inch
	h3 		= 4*inch

	L1 		= R1 - h1
	L2 		= R2 - h2
	L3 		= R3 - h3

	uLim 	= 2*d1 + d2 + d3 + L1*d3/L3 - L1*d2/L2

	H0 		= 12*inch					# total height of bearing

	# friction pendulum system
	# kv = EASlider/hSlider
	kv = 6*1000*kip/inch
	uniaxialMaterial('Elastic', fpsMatPTag, kv)
	uniaxialMaterial('Elastic', fpsMatMzTag, 10.0)


	# Define friction model for FP elements
	# command: frictionModel Coulomb tag mu
	frictionModel('Coulomb', frn1ModelTag, mu1)
	frictionModel('Coulomb', frn2ModelTag, mu2)
	frictionModel('Coulomb', frn3ModelTag, mu3)

	# define elements

	# define plastic integration
	# choosing Radau because it allows for P.H.

	# command:  beamIntegration(type, tag, *args)
	# args:	 beamIntegration('HingeRadau', secI, lpI, secJ, lpJ, secE)
	beamIntegration('HingeRadau', hingeColTag, colSecTag, 0.1, colSecTag, 0.1, elasticColSecTag)
	beamIntegration('HingeRadau', hingeBeamTag, beamSecTag, 0.1, beamSecTag, 0.1, elasticBeamSecTag)
	beamIntegration('HingeRadau', hingeRoofTag, roofSecTag, 0.1, roofSecTag, 0.1, elasticRoofBeamSecTag)

	# create force-based beam-column elements, with BeamWithHinges integration tags
	# command:  element('forceBeamColumn', eleTag, iNode, jNode, transfTag, integrationTag, '-iter', maxIter=10, tol=1e-12, '-mass', mass=0.0)

	# define the columns
	element('forceBeamColumn',	1 ,	 1,	 5,	colTransfTag,	hingeColTag)
	element('forceBeamColumn',	2 ,	 5,	 9,	colTransfTag,	hingeColTag)
	element('forceBeamColumn',	3 ,	 9,	13,	colTransfTag,	hingeColTag)
	element('forceBeamColumn',	4 ,	 2,	 6,	colTransfTag,	hingeColTag)
	element('forceBeamColumn',	5 ,	 6,	10,	colTransfTag,	hingeColTag)
	element('forceBeamColumn',	6 ,	10,	14,	colTransfTag,	hingeColTag)
	element('forceBeamColumn',	7 ,	 3,	 7,	colTransfTag,	hingeColTag)
	element('forceBeamColumn',	8 ,	 7,	11,	colTransfTag,	hingeColTag)
	element('forceBeamColumn',	9 ,	11,	15,	colTransfTag,	hingeColTag)
	element('forceBeamColumn',	10,	 4,	 8,	colTransfTag,	hingeColTag)
	element('forceBeamColumn',	11,	 8,	12,	colTransfTag,	hingeColTag)
	element('forceBeamColumn',	12,	12,	16,	colTransfTag,	hingeColTag)

	# Define the beams
	element('forceBeamColumn',	13,	 5,	 6,	beamTransfTag,	hingeBeamTag)
	element('forceBeamColumn',	14,	 6,	 7,	beamTransfTag,	hingeBeamTag)
	element('forceBeamColumn',	15,	 7,	 8,	beamTransfTag,	hingeBeamTag)
	element('forceBeamColumn',	16,	 9,	10,	beamTransfTag,	hingeBeamTag)
	element('forceBeamColumn',	17,	10,	11,	beamTransfTag,	hingeBeamTag)
	element('forceBeamColumn',	18,	11,	12,	beamTransfTag,	hingeBeamTag)
	element('forceBeamColumn',	19,	13,	14,	beamTransfTag,	hingeRoofTag)
	element('forceBeamColumn',	20,	14,	15,	beamTransfTag,	hingeRoofTag)
	element('forceBeamColumn',	21,	15,	16,	beamTransfTag,	hingeRoofTag)


	# define 2-D isolation layer 
	# command: element TripleFrictionPendulum $eleTag $iNode $jNode $frnTag1 $frnTag2 $frnTag3 $vertMatTag $rotZMatTag $rotXMatTag $rotYMatTag $L1 $L2 $L3 $d1 $d2 $d3 $W $uy $kvt $minFv $tol
	kvt 	= 0.01*kv
	minFv 	= 1.0
	element('TripleFrictionPendulum', 22, 17, 1, frn1ModelTag, frn2ModelTag, 
		frn3ModelTag, fpsMatPTag, fpsMatMzTag, fpsMatMzTag, fpsMatMzTag, 
		L1, L2, L3, d1, d2, d3, pOuter, uy, kvt, minFv, 1e-5)
	element('TripleFrictionPendulum', 23, 18, 2, frn1ModelTag, frn2ModelTag, 
		frn3ModelTag, fpsMatPTag, fpsMatMzTag, fpsMatMzTag, fpsMatMzTag, 
		L1, L2, L3, d1, d2, d3, pInner, uy, kvt, minFv, 1e-5)
	element('TripleFrictionPendulum', 24, 19, 3, frn1ModelTag, frn2ModelTag, 
		frn3ModelTag, fpsMatPTag, fpsMatMzTag, fpsMatMzTag, fpsMatMzTag, 
		L1, L2, L3, d1, d2, d3, pInner, uy, kvt, minFv, 1e-5)
	element('TripleFrictionPendulum', 25, 20, 4, frn1ModelTag, frn2ModelTag, 
		frn3ModelTag, fpsMatPTag, fpsMatMzTag, fpsMatMzTag, fpsMatMzTag, 
		L1, L2, L3, d1, d2, d3, pOuter, uy, kvt, minFv, 1e-5)

	element('TripleFrictionPendulum', 26, 21,22, frn1ModelTag, frn2ModelTag, 
		frn3ModelTag, fpsMatPTag, fpsMatMzTag, fpsMatMzTag, fpsMatMzTag, 
		L1, L2, L3, d1, d2, d3, pLc, uy, kvt, minFv, 1e-5)


	# define leaning columns, all beam sizes
	# truss beams
	# command: element('TrussSection', eleTag, *eleNodes, secTag[, '-rho', rho][, '-cMass', cFlag][, '-doRayleigh', rFlag])
	element('Truss', 27, 8, 24, ARigid, frameLinkTag)
	element('Truss', 28,12, 27, ARigid, frameLinkTag)
	element('Truss', 29,16, 30, ARigid, frameLinkTag)

	# p-Delta columns
	# command: element('elasticBeamColumn', eleTag, *eleNodes, Area, E_mod, G_mod, Jxx, Iy, Iz, transfTag[, '-mass', massPerLength][, '-cMass'])
	element('elasticBeamColumn', 30, 22, 23, 
		ARigid, Es, Gs, J, IRigid, IRigid, colTransfTag)
	element('elasticBeamColumn', 31, 25, 26, 
		ARigid, Es, Gs, J, IRigid, IRigid, colTransfTag)
	element('elasticBeamColumn', 32, 28, 29, 
		ARigid, Es, Gs, J, IRigid, IRigid, colTransfTag)

	# Rotational hinge at leaning column joints via zeroLength elements
	kLC = 1e-9*kip/inch

	# Create the material (spring)
	uniaxialMaterial('Elastic', LCSpringMatTag, kLC)

	# Create moment releases at leaning column ends
	# command: element('zeroLength', eleTag, *eleNodes, '-mat', *matTags, '-dir', *dirs[, '-doRayleigh', rFlag=0][, '-orient', *vecx, *vecyp])
	# command: equalDOF(retained, constrained, DOF_1, DOF_2)
	def rotLeaningCol(eleID, nodeI, nodeJ):
		element('zeroLength', eleID, nodeI, nodeJ, '-mat', LCSpringMatTag, '-dir', 5)			# Create zero length element (spring), rotations allowed about Y axis
		equalDOF(nodeI, nodeJ, 1, 2, 3, 4, 6)													# Constrain the translational DOFs and out-of-plane rotations

	rotLeaningCol(33, 24, 23)
	rotLeaningCol(34, 24, 25)
	rotLeaningCol(35, 27, 26)
	rotLeaningCol(36, 27, 28)
	rotLeaningCol(37, 30, 29)

	# define rigid 'diaphragm' at bottom layer
	# command: element('elasticBeamColumn', eleTag, *eleNodes, Area, E_mod, G_mod, Jxx, Iy, Iz, transfTag[, '-mass', massPerLength][, '-cMass'])
	element('elasticBeamColumn', 38, 1, 2, 
		ARigid, Es, Gs, J, IRigid, IRigid, beamTransfTag)
	element('elasticBeamColumn', 39, 2, 3, 
		ARigid, Es, Gs, J, IRigid, IRigid, beamTransfTag)
	element('elasticBeamColumn', 40, 3, 4, 
		ARigid, Es, Gs, J, IRigid, IRigid, beamTransfTag)

	# element('elasticBeamColumn', 41, 4,22, AgBeam, Es, Gs, J, IyBeam, IzBeam, beamTransfTag)
	element('Truss', 41, 4, 22, ARigid, frameLinkTag)

	# define impact moat as ZeroLengthImpact3D elements
	# https://opensees.berkeley.edu/wiki/index.php/Impact_Material
	khWall 		= 25000*kip*inch 										# impact stiffness parameter from Muthukumar, 2006
	e 			= 0.7													# coeff of restitution (1.0 = perfectly elastic collision)
	delM 		= 0.025*inch 											# maximum penetration during pounding event, from Hughes paper
	kEffWall	= khWall*math.sqrt(delM)								# effective stiffness
	a 			= 0.1													# yield coefficient
	delY 		= a*delM												# yield displacement
	nImpact 	= 3/2													# Hertz power rule exponent
	EImpact 	= khWall*delM**(nImpact+1)*(1 - e**2)/(nImpact+1)		# energy dissipated during impact
	K1 			= kEffWall + EImpact/(a*delM**2)						# initial impact stiffness
	K2 			= kEffWall - EImpact/((1-a)*delM**2)					# secondary impact stiffness

	dirWall 	= 1								# 1 for out-normal vector pointing towards +X direction
	moatGap 	= float(moatGap)				# rounded up DmPrime
	# moatGap 	= 20*inch
	muWall 		= 0.01							# friction ratio in two tangential directions parallel to restrained+constrained planes
	KtWall 		= 1e5*kip/inch					# tangential stiffness (?)
	cohesionTag = 0								# 0 for no cohesion

	# tie up 3DOF and 6DOF nodes
	# command: equalDOF(rNodeTag, cNodeTag, *dofs)
	equalDOF(1, 312, 1, 2, 3)
	equalDOF(4, 321, 1, 2, 3)
	equalDOF(31, 311, 1, 2, 3)
	equalDOF(32, 322, 1, 2, 3)

	# impact elements
	# command: element zeroLengthImpact3D $tag $cNode $rNode $direction $initGap $frictionRatio $Kt $Kn $Kn2 $Delta_y $cohesion
	# element('zeroLengthImpact3D', 51, 311, 312, dirWall, moatGap, muWall, KtWall, K1, K2, delY, cohesionTag)
	# element('zeroLengthImpact3D', 52, 322, 321, dirWall, moatGap, muWall, KtWall, K1, K2, delY, cohesionTag)
	element('zeroLengthImpact3D', 51, 312, 311, 
		dirWall, moatGap, muWall, KtWall, K1, K2, delY, cohesionTag)
	# element('zeroLengthImpact3D', 52, 321, 322, dirWall, moatGap, muWall, KtWall, K1, K2, delY, cohesionTag)
	element('zeroLengthImpact3D', 52, 322, 321, 
		dirWall, moatGap, muWall, KtWall, K1, K2, delY, cohesionTag)

	# print("Model built!")
	# plot_model()

# if ran alone, build model and plot
if __name__ == '__main__':
	build()
	print('Model built!')
	plot_model()