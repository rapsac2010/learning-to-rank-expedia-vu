import numpy as np
import pandas as pd

# Placeholder class
class HelperClass:
    def __init__(self):
        print("Hello World Init")
    
    def test(self):
        print("Hello World Function Call")


class KalmanFilter:
    """
    This class implements a Kalman filter. We have:

    1) The observation equation:
        y_t = Z_t * alpha_t + eps_t
        eps_t ~ N(0, H_t)

    2) The state update equation:
        alpha_{t+1} = T_t * alpha_t + R_t * eta_t
        eta_t ~ N (0, Q_t)

    3) Initialization
        alpha_1 ~ N(a1, P1)

        Stationary      -> a1 = mean(y_t),  P1 = var(y_t)
        Non-stationary  -> a1 = 0,          P1 = 10e7

    NOTE:
        - Z_t, H_t, R_t and Q_t are non-stochastic but can be time varying matrices.
    """

    def __init__(self, y, a_init, P_init, H, Q, R, d=0, c=0):
        self.y = np.array(y)
        self.n = len(y)
        self.a_init = np.array(a_init)
        self.P_init = np.array(P_init)

        # Make vector over time if not already
        self.H = self.as_vector_over_time(H)
        self.Q = self.as_vector_over_time(Q)
        self.R = self.as_vector_over_time(R)
        self.d = self.as_vector_over_time(d)
        self.c = self.as_vector_over_time(c)

        # Keep track of the procedures being ran
        self.smoother_ran = False
        self.filter_ran = False
        self.disturbance_smoother_ran = False

    def run_filter(self, Z, T):
        """
        Return a dictionary with:
            - v
            - F
            - K
            - a_pred (a_t)
            - a_filter (a_t|t)
            - P
        """
        result_dict = {key: "" for key in ["a_pred", "a_filter", "P", "v", "F", "K"]}

        # Initialization
        # a_pred = np.array([np.zeros(shape=self.a_init.shape) for i in range(self.n + 1)])
        a_pred = np.zeros(shape=(self.n+1, self.a_init.shape[0], self.a_init.shape[1]))
        a_pred[0] = self.a_init
        a_filtered = np.array([np.zeros(shape=self.a_init.shape) for i in range(self.n)])

        P = [np.zeros(shape=self.P_init.shape) for i in range(self.n + 1)]
        P[0] = self.P_init

        F = [None] * self.n
        K = [None] * self.n
        v = [np.array([0]).reshape(1, 1) for i in range(self.n)]

        Z = self.as_vector_over_time(Z)
        T = self.as_vector_over_time(T)

        # Loop as in slide 36 week 2
        for t in range(self.n):    
            # Prediction error
            v[t] = self.y[t] - Z[t] @ a_pred[t] - self.d[t]
            
            if np.isnan(self.y[t]):
                v[t] = np.zeros(shape=v[t - 1].shape)

            # Prediction variance
            F[t] = Z[t] @ P[t] @ Z[t].T + self.H[t]

            # print(f"F[{t}] = {F[t]}\n P[{t}] = {P[t]}\n Z[{t}] = {Z[t]}\n")
            PZF1 = P[t] @ Z[t].T @ np.linalg.inv(F[t])

            # Kalman Gain
            if np.isnan(self.y[t]):
                K[t] = np.zeros(shape=(T[t].shape[0], Z[t].shape[0]))
            else:
                K[t] = T[t] @ PZF1

            # Filtered and predicted a
            a_filtered[t] = a_pred[t] + PZF1 @ v[t]
            a_pred[t + 1] = T[t] @ a_filtered[t] + self.c[t] 

            # Update variance
            P[t + 1] = (
                T[t] @ P[t] @ T[t].T
                + self.R[t] @ self.Q[t] @ self.R[t].T
                - K[t] @ F[t] @ K[t].T
            )
        
        result_dict["a_pred"] = np.array(a_pred).reshape(self.n + 1, len(a_pred[0]))
        result_dict["a_filter"] = np.array(a_filtered).reshape(
            self.n, len(a_filtered[0])
        )
        result_dict["P"] = P
        result_dict["v"] = np.array(v).reshape(self.n, len(v[0]))
        result_dict["F"] = F
        result_dict["K"] = K

        # Save all matrices in class using self
        self.Z = Z
        self.T = T
        self.a_pred = result_dict["a_pred"]
        self.a_filter = result_dict["a_filter"]
        self.P = result_dict["P"]
        self.v = result_dict["v"]
        self.F = result_dict["F"]
        self.K = result_dict["K"]

        self.results_filter = result_dict
        self.filter_ran = True

        return result_dict
    
    

    def run_smoother(self, Z, T):
        """
        Return a dictionary with:
            - a_smooth
            - P_smooth (V in the book)
            - N
            - r
        """
        results_dict = {key: "" for key in ["a_smooth", "P_smooth", "N", "r"]}

        # Run filter if not already done
        if not self.filter_ran:
            self.run_filter(Z, T)
        res = self.results_filter

        Z = self.as_vector_over_time(Z)
        T = self.as_vector_over_time(T)

        # Read F from filter results
        # HARDCODED SIZE of v for now
        m = self.a_init.shape[0]
        F = res["F"].copy()
        v = res["v"].copy().reshape(self.n, 1, 1)  
        a_pred = res["a_pred"].reshape(self.n + 1, m, 1)

        # In the case of missing observations, we want F --> inf.
        # NOTE: we do not want this when using missing observations to `forecast`.
        # However: in that case we don't use the smoother/
        for t, y_t in enumerate(self.y):
            if np.isnan(y_t):
                F[t] = F[t] * 10e7

        # Initialize the smoothed values and the cumulants
        a_smooth = np.zeros(shape=(self.n, m, 1))
        P_smooth = np.zeros(shape=(self.n, m, m))
        r = np.zeros(shape=(self.n + 1, m, 1))
        N = np.zeros(shape=(self.n + 1, m, m))

        # Backwards recursive loop as in slide 45 week 2
        for t in range(self.n - 1, -1, -1):
            L = T[t] - res["K"][t] @ Z[t]
            r[t] = Z[t].T @ np.linalg.inv(F[t]) @ v[t] + L.T @ r[t + 1]
            N[t] = Z[t].T @ np.linalg.inv(F[t]) @ Z[t] + L.T @ N[t + 1] @ L

        # Forwards recursive loop as in slide 45 week 2
        for t in range(0, self.n):
            a_smooth[t] = a_pred[t] + res["P"][t] @ r[t]
            P_smooth[t] = res["P"][t] - res["P"][t].T @ N[t] @ res["P"][t]

        results_dict["a_smooth"] = np.array(a_smooth).reshape(self.n, len(a_smooth[0]))
        results_dict["P_smooth"] = P_smooth
        results_dict["N"] = N[1:]
        results_dict["r"] = r[1:]

        # Save all matrices in class using self
        self.a_smooth = results_dict["a_smooth"]
        self.P_smooth = results_dict["P_smooth"]
        self.N = results_dict["N"]
        self.r = results_dict["r"]

        self.results_smoother = results_dict
        self.smoother_ran = True

        return results_dict

    def run_disturbance_smoother(self):
        """This methods runs the disturbance smoothing recursions found on
        p. 96 of the DK book
        """
        if not self.smoother_ran:
            raise ValueError("Run the smoother first")
        if not self.filter_ran:
            raise ValueError("Run the filter first")

        results_dict = {
            key: ""
            for key in [
                "eps_smoothed",
                "eta_smoothed",
                "u",
                "D",
                "var_eps_smoothed",
                "var_eta_smoothed",
            ]
        }

        results_dict["N"] = self.N
        results_dict["r"] = self.r

        # Initialize empty arrays for the result arrays
        self.u = np.zeros(shape=(self.n, 1))
        self.eps_smoothed = np.zeros(shape=(self.n, 1))
        self.eta_smoothed = np.zeros(shape=(self.n, 1))
        self.D = [np.zeros(shape=self.F[0].shape) for _ in range(self.n)]
        self.var_eps_smoothed = np.zeros(shape=(self.n, 1))
        self.var_eta_smoothed = np.zeros(shape=(self.n, 1))

        for t in range(self.n):
            self.u[t] = np.linalg.inv(self.F[t]) @ self.v[t] - self.K[t].T @ self.r[t]
            self.D[t] = np.linalg.inv(self.F[t]) + self.K[t].T @ self.N[t] @ self.K[t]

            # Smoothed disturbances and variances
            self.eps_smoothed[t] = self.H[t] @ self.u[t]
            self.eta_smoothed[t] = self.Q[t] @ self.R[t].T @ self.r[t]
            self.var_eps_smoothed[t] = self.H[t] - self.H[t] @ self.D[t] @ self.H[t]
            self.var_eta_smoothed[t] = (
                self.Q[t] - self.Q[t] @ self.R[t].T @ self.N[t] @ self.R[t] @ self.Q[t]
            )

        # Save results
        results_dict["eps_smoothed"] = self.eps_smoothed
        results_dict["eta_smoothed"] = self.eta_smoothed
        results_dict["var_eps_smoothed"] = self.var_eps_smoothed
        results_dict["var_eta_smoothed"] = self.var_eta_smoothed
        results_dict["u"] = self.u
        results_dict["D"] = self.D

        self.results_disturbance_smoother = results_dict
        self.disturbance_smoother_ran = True

        return results_dict

    def run_simul(self, eps_hat):

        results_dict = {key: "" for key in ["eps_plus", "eta_plus", "y_plus", "A_plus"]}

        self.eps_plus = np.zeros(shape=(self.n, 1))
        self.eta_plus = np.zeros(shape=(self.n, 1))

        self.eps_hat_plus = np.zeros(shape=(self.n, 1))

        self.eps_tilde = np.zeros(shape=(self.n, 1))
        self.eta_tilde = np.zeros(shape=(self.n, 1))

        self.y_plus = np.zeros(shape=(self.n, 1))
        self.A_plus = np.zeros(shape=(self.n, 1))

        self.A_plus[0] = self.y[0]

        for t in range(self.n):
            self.eps_plus[t] = np.random.normal(0, np.sqrt(self.H[t]))
            self.eta_plus[t] = np.random.normal(0, np.sqrt(self.Q[t]))
            self.y_plus[t] = self.eps_plus[t] + self.A_plus[t]
            if t != self.n - 1:
                self.A_plus[t + 1] = self.A_plus[t] + self.eta_plus[t]

        results_dict["eps_plus"] = self.eps_plus
        results_dict["eta_plus"] = self.eta_plus
        results_dict["y_plus"] = self.y_plus
        results_dict["A_plus"] = self.A_plus

        return results_dict

    def run_simul_comp():
        results_dict = {key: "" for key in ["eps_plus", "eta_plus", "y_plus", "A_sim"]}

        return results_dict

    def as_vector_over_time(self, matrix):
        # If the 'matrix' is already a list of length n, this is not needed
        if type(matrix) == list and len(matrix) == self.n:
            return matrix

        # Return an error if the length is not good
        if type(matrix) == list and len(matrix) != self.n:
            raise ValueError("The vector of matrices needs to be of length n")

        if type(matrix) == np.ndarray and matrix.shape[0] == self.n:
            return matrix
        
        if type(matrix) == np.ndarray:
            return [matrix for i in range(self.n)]
        
        if (
            type(matrix) == int
            or type(matrix) == float
            or type(matrix) == np.float64
            or type(matrix) == np.int64
            or type(matrix) == np.int32
            or type(matrix) == np.float32
            or type(matrix) == np.int16
            or type(matrix) == np.float16
        ):
            return [np.array(matrix).reshape(1, 1) for i in range(self.n)]

        else:
            raise TypeError(
                f"matrix needs to be of type np.ndarray or list, type given: {type(matrix)}"
            )
