import numpy as np
import osqp
from scipy import sparse

from speed_profiler.ros_interface import Logger as logger


class SpeedProfileOptimizer(object):
    """ Computes an optimal speed profile given a reference path of N waypoints.
    The speed profile is created by solving an optimization problem
    which takes into account the following:
    - An absolute maximum speed.
    - A lateral acceleration that must not be exceeded.
    - The maximum and minimum longitudinal acceleration.
    - A smoothening parameter to penalize fast changes in speed.
    """

    def __init__(self, parameters):

        ## Speed limit
        if parameters['mission'] == 'EBS':
            self._speed_limit = parameters['EBS_speed_limit']
        else:
            self._speed_limit = parameters['speed_limit']
        
        ## Maximum allowed lateral acceleration
        ## (used to determine maximum allowed speed from curvature)
        self._lateral_acceleration_limit = \
            parameters['lateral_acceleration_limit']

        ## Weight for penalizing large accelerations
        self._alpha_smooth = parameters['alpha_smooth']

        ## Minimum desired acceleration along path (negative value)
        self._accel_min = parameters['accel_min']

        ## Maximum desired acceleration along path
        self._accel_max = parameters['accel_max']

        ## Optimization problem dimension (number of waypoints, N)
        self._waypoints_count = 0

    def compute_speed_profile(self, curvatures, distances, v_init, v_final):
        """
        Computes an optimal speed profile along a path
        as introduced in chapter 7.1 of:

        Pedro F. Lima - "Predictive control for autonomous driving"
        http://www.diva-portal.org/smash/get/diva2:925562/FULLTEXT01.pdf

        @param curvatures: List of path curvatures at N waypoints (in 1/m)
        @param distances: List of N-1 distances between waypoints (in m)
        @param v_init (optional): Initial speed on the path (in m/s)
        @param v_final (optional): Final speed on the path (in m/s)
        @return Speed profile (ndarray, N elements) of optimal speeds (in m/s)
        """
        curvatures = np.array(curvatures)
        distances = np.array(distances)

        self._waypoints_count = curvatures.shape[0]

        if curvatures.shape != (self._waypoints_count,):
            raise ValueError(
                'Wrong shape of curvatures. Has to be ' +
                '({},) but was {}'.format(self._waypoints_count, curvatures.shape))

        if distances.shape != (self._waypoints_count - 1,):
            raise ValueError(
                'Wrong shape of distances. Has to be ' +
                '({},) but was {}'.format(self._waypoints_count - 1, distances.shape))

        v_max = self._compute_v_max(curvatures)

        D1 = self._get_D1(distances)
        P = self._osqp_get_P(D1)
        q = self._osqp_get_q(v_max)
        A, l, u = self._osqp_get_constraints(D1, v_max, v_init, v_final)

        # Updating instead of new setup after 1st iteration didn't work
        solver = osqp.OSQP()
        solver.setup(P, q, A, l, u, verbose=False)

        result = solver.solve()

        if result.info.status != 'solved':
            logger.error('[Speed profiler] OSQP: Problem not solved. Status: {}'
                      .format(result.info.status))
            return None

        return np.sqrt(np.maximum(result.x, 0))

    def _compute_v_max(self, kappa_ref):
        """ Computes the maximum speeds on a track with the given curvatures
        without exceeding a defined lateral acceleration and speed limit.

        @param kappa_ref: ndarray (N elements) of path curvatures (in 1/m)
        @return ndarray (N elements) of max speeds (in m/s)
        """

        # We are only interested in absolute curvature, not the direction
        kappa_ref = np.absolute(kappa_ref)

        # Since speed is calculated from curvature with a monotonous function,
        # a maximum speed limit is equivalent to a minimum curvature limit.
        # Specifying the minimum curvature that equals the speed limit prevents
        # division by small numbers.
        kappa_min = self._lateral_acceleration_limit / (self._speed_limit ** 2)
        kappa = np.maximum(kappa_ref, kappa_min * np.ones(kappa_ref.size))
        return np.sqrt(self._lateral_acceleration_limit / kappa)

    def _osqp_get_P(self, D1):
        """ Construct P matrix for OSQP problem formulation.

        @param D1: D1 as sparse matrix
        @return P matrix for OSQP problem as sparse matrix
        """
        # Note: there is a difference between numpy transpose/dot and the ones
        # used below. Don't use numpy functions!
        return 2 * (sparse.identity(self._waypoints_count) +
                    self._alpha_smooth * (D1.transpose().dot(D1)))

    def _osqp_get_q(self, v_max):
        """ Construct q vector for OSQP problem formulation.

        @param v_max: ndarray (N elements) of maximum speeds (in m/s)
        @return q vector for OSPQ problem as ndarray.
        """
        return -2 * np.square(v_max)

    def _osqp_get_constraints(self, D1, v_max, v_init, v_final):
        """ Construct constraints for OSQP problem formulation.

        @param D1: D1 as sparse matrix
        @param v_max: ndarray (N elements) of maximum speeds (in m/s)
        @param v_init (optional): Initial speed at first waypoint (in m/s)
        @param v_final (optional): Target speed at last waypoint (in m/s)
        @return (A, l, u) such that l <= Ax <= u is the constraint for the
            optimization problem (A as compressed sparse column matrix)
        """
        A = np.append(D1.toarray(), np.eye(self._waypoints_count), axis=0)

        a_min = self._accel_min * np.ones(self._waypoints_count - 1)
        a_max = self._accel_max * np.ones(self._waypoints_count - 1)

        w_min = np.zeros(self._waypoints_count)
        w_max = np.square(v_max)

        if v_init is not None:
            w_min[0] = v_init**2
            w_max[0] = v_init**2
        if v_final is not None:
            w_min[self._waypoints_count - 1] = v_final**2
            w_max[self._waypoints_count - 1] = v_final**2

        l = np.append(a_min, w_min)
        u = np.append(a_max, w_max)

        return sparse.csc_matrix(A), l, u

    def _get_D1(self, distances):
        """ Compute matrix operator to get accelerations from speeds.
        Multiplying this matrix with the squared speeds vector
        returns accelerations by using first order differences.

        @param distances: Numpy array of distances between waypoints (in m)
        @return D1 as sparse matrix
        """
        diag = 0.5 / distances
        return sparse.diags([-diag, diag], [0, 1],
                            shape=(self._waypoints_count - 1, self._waypoints_count))