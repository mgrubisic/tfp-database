# tfp-database
OpenSeesPy model of triple friction pendulum isolated steel moment frame, packaged for EQ database.
Additional inverse designing capabilities based off of GPML in progress.

buildModel.py: Construct the model

ReadRecord.py: Support script; reads ground motions

superStructDesign.py: Bearing and superstructure design algorithm

Given site parameter, desired period, desired damping, (+guesses): output beams and columns' W-shapes and bearing parameters

LHS.py generates input set

runControl.py writes input set to files, performs the runs over the set, and records the session

postprocessing.py handles outputs and prepares run results

eqAnly.py calls design script, builds model, and performs analysis
