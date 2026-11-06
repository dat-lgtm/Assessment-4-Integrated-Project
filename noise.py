
import numpy as np

P0 = np.diag([.025**2, .025**2, np.deg2rad(2.)**2])

Q_D = np.diag([.004**2, .004**2, np.deg2rad(.4)**2])
R_N = np.diag([.06**2, .06**2, np.deg2rad(5.)**2])
R_LIDAR = np.diag([.025**2, .025**2, np.deg2rad(2.)**2])
ENCODER_SIGMA = .0005
CORNER_SIGMA = .15
TRIAL_CONDITIONS = [(1., 0., 0.), (1.5, .15, .02), (2., .30, -.02)]
