"""
Adaptive Filter System for Motion Capture Data

This module provides an intelligent filtering system that combines multiple online filters
(Kalman, One Euro, EMA) and automatically adapts based on detected motion states.

Author: V-AIM Project
"""

import numpy as np
from typing import Tuple, Dict, List, Optional


# ============================================================================
# FILTER IMPLEMENTATIONS
# ============================================================================

class KalmanFilter:
    """
    A simple Kalman filter for a 1D signal, assuming a constant velocity model.
    The state is [position, velocity].
    """
    def __init__(self, dt: float, process_noise: float, measurement_noise: float):
        """
        Initializes the Kalman Filter.
        
        Args:
            dt (float): The time step between frames (1.0 / frequency).
            process_noise (float): The variance of the process noise (q). 
                                   Represents uncertainty in the motion model.
            measurement_noise (float): The variance of the measurement noise (r).
                                       Represents uncertainty in the measurement.
        """
        self.dt = dt
        
        # State transition matrix A: x(t) = A * x(t-1)
        self.A = np.array([[1, dt], 
                           [0, 1]])

        # Measurement matrix H: z(t) = H * x(t)
        self.H = np.array([[1, 0]])

        # Process noise covariance Q
        q_val = process_noise
        self.Q = q_val * np.array([[(dt**4)/4, (dt**3)/2],
                                   [(dt**3)/2,  dt**2]])
        
        # Measurement noise covariance R
        self.R = np.array([[measurement_noise]])

        # Initial state covariance P (high uncertainty)
        self.P = np.eye(2) * 1000.0
        
        # Initial state x (position=0, velocity=0)
        self.x = np.zeros((2, 1))
        
        self.initialized = False

    def __call__(self, z: float) -> float:
        """
        Processes a new measurement (z) and returns the filtered position.
        """
        if not self.initialized:
            self.x[0, 0] = z
            self.x[1, 0] = 0.0  # Initial velocity is zero
            self.initialized = True
            return self.x[0, 0]

        # --- Prediction Step ---
        x_pred = self.A @ self.x
        P_pred = self.A @ self.P @ self.A.T + self.Q

        # --- Update Step ---
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        self.x = x_pred + K * (z - self.H @ x_pred)
        self.P = (np.eye(2) - K @ self.H) @ P_pred
        
        return self.x[0, 0]
    
    def get_velocity(self) -> float:
        """Returns the current estimated velocity."""
        return self.x[1, 0] if self.initialized else 0.0

    def reset(self):
        """Resets the filter's internal state."""
        self.initialized = False
        self.P = np.eye(2) * 1000.0
        self.x = np.zeros((2, 1))


class OneEuroFilter:
    """
    One Euro Filter - An adaptive low-pass filter that adjusts its cutoff frequency
    based on the velocity of the signal. Excellent for filtering noisy signals while
    maintaining responsiveness to rapid changes.
    
    Reference: Casiez, G., Roussel, N., & Vogel, D. (2012). 1€ filter: a simple 
    speed-based low-pass filter for noisy input in interactive systems.
    """
    def __init__(self, freq: float, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0):
        """
        Initialize the One Euro Filter.
        
        Args:
            freq (float): Sampling frequency (Hz)
            min_cutoff (float): Minimum cutoff frequency (Hz). Lower = more smoothing
            beta (float): Cutoff slope. Higher = more responsive to velocity changes
            d_cutoff (float): Cutoff frequency for derivative (Hz)
        """
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        
        self.x_prev = None
        self.dx_prev = 0.0
        self.initialized = False
    
    def _smoothing_factor(self, cutoff: float) -> float:
        """Calculate the smoothing factor (alpha) from cutoff frequency."""
        tau = 1.0 / (2.0 * np.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)
    
    def __call__(self, x: float) -> float:
        """
        Filter a new measurement.
        
        Args:
            x (float): New measurement
            
        Returns:
            float: Filtered value
        """
        if not self.initialized:
            self.x_prev = x
            self.initialized = True
            return x
        
        # Estimate derivative (velocity)
        dx = (x - self.x_prev) * self.freq
        
        # Filter the derivative
        alpha_d = self._smoothing_factor(self.d_cutoff)
        dx_filtered = alpha_d * dx + (1.0 - alpha_d) * self.dx_prev
        
        # Adaptive cutoff frequency based on velocity
        cutoff = self.min_cutoff + self.beta * abs(dx_filtered)
        
        # Filter the signal
        alpha = self._smoothing_factor(cutoff)
        x_filtered = alpha * x + (1.0 - alpha) * self.x_prev
        
        # Update state
        self.x_prev = x_filtered
        self.dx_prev = dx_filtered
        
        return x_filtered
    
    def get_velocity(self) -> float:
        """Returns the current estimated velocity."""
        return self.dx_prev if self.initialized else 0.0
    
    def reset(self):
        """Reset the filter state."""
        self.x_prev = None
        self.dx_prev = 0.0
        self.initialized = False


class ExponentialMovingAverage:
    """
    Simple Exponential Moving Average (EMA) filter.
    Fast and lightweight, but with fixed responsiveness.
    """
    def __init__(self, alpha: float = 0.3):
        """
        Initialize EMA filter.
        
        Args:
            alpha (float): Smoothing factor (0-1). Higher = more responsive, less smooth
        """
        self.alpha = alpha
        self.value = None
        self.initialized = False
    
    def __call__(self, x: float) -> float:
        """
        Filter a new measurement.
        
        Args:
            x (float): New measurement
            
        Returns:
            float: Filtered value
        """
        if not self.initialized:
            self.value = x
            self.initialized = True
            return x
        
        self.value = self.alpha * x + (1.0 - self.alpha) * self.value
        return self.value
    
    def reset(self):
        """Reset the filter state."""
        self.value = None
        self.initialized = False


# ============================================================================
# MOTION STATE DETECTION
# ============================================================================

class MotionStateDetector:
    """
    Detects motion characteristics to inform adaptive filtering decisions.
    Tracks velocity, acceleration, and jitter for each signal dimension.
    """
    def __init__(self, dt: float, velocity_window: int = 5):
        """
        Initialize motion state detector.
        
        Args:
            dt (float): Time step between frames
            velocity_window (int): Window size for velocity estimation
        """
        self.dt = dt
        self.velocity_window = velocity_window
        
        self.position_history = []
        self.velocity = 0.0
        self.acceleration = 0.0
        self.prev_velocity = 0.0
    
    def update(self, position: float) -> Dict[str, float]:
        """
        Update motion state with new position measurement.
        
        Args:
            position (float): Current position
            
        Returns:
            dict: Motion state metrics (velocity, acceleration, jitter_score)
        """
        self.position_history.append(position)
        
        # Keep only recent history
        if len(self.position_history) > self.velocity_window:
            self.position_history.pop(0)
        
        # Calculate velocity
        if len(self.position_history) >= 2:
            self.velocity = (self.position_history[-1] - self.position_history[-2]) / self.dt
        
        # Calculate acceleration
        self.acceleration = (self.velocity - self.prev_velocity) / self.dt
        self.prev_velocity = self.velocity
        
        # Calculate jitter score (high-frequency oscillation indicator)
        jitter_score = 0.0
        if len(self.position_history) >= 3:
            # Count direction changes
            direction_changes = 0
            for i in range(len(self.position_history) - 2):
                diff1 = self.position_history[i+1] - self.position_history[i]
                diff2 = self.position_history[i+2] - self.position_history[i+1]
                if diff1 * diff2 < 0:  # Sign change indicates direction reversal
                    direction_changes += 1
            jitter_score = direction_changes / max(1, len(self.position_history) - 2)
        
        return {
            'velocity': abs(self.velocity),
            'acceleration': abs(self.acceleration),
            'jitter_score': jitter_score
        }
    
    def reset(self):
        """Reset the detector state."""
        self.position_history = []
        self.velocity = 0.0
        self.acceleration = 0.0
        self.prev_velocity = 0.0


# ============================================================================
# ADAPTIVE FILTER MANAGER
# ============================================================================

class AdaptiveFilterManager:
    """
    Manages multiple filters and adaptively blends them based on motion state.
    Uses weighted combination of Kalman, One Euro, and EMA filters.
    """
    def __init__(self, 
                 dt: float,
                 # Kalman parameters
                 kalman_process_noise: float = 1e-2,
                 kalman_measurement_noise: float = 3e-5,
                 # One Euro parameters
                 one_euro_min_cutoff: float = 1.0,
                 one_euro_beta: float = 0.007,
                 # EMA parameters
                 ema_alpha: float = 0.3,
                 # Adaptive parameters
                 velocity_threshold_slow: float = 0.1,
                 velocity_threshold_fast: float = 1.0):
        """
        Initialize the adaptive filter manager.
        
        Args:
            dt (float): Time step between frames (1/fps)
            kalman_process_noise (float): Kalman process noise variance
            kalman_measurement_noise (float): Kalman measurement noise variance
            one_euro_min_cutoff (float): One Euro minimum cutoff frequency
            one_euro_beta (float): One Euro velocity sensitivity
            ema_alpha (float): EMA smoothing factor
            velocity_threshold_slow (float): Threshold for slow motion
            velocity_threshold_fast (float): Threshold for fast motion
        """
        self.dt = dt
        freq = 1.0 / dt
        
        # Initialize filters
        self.kalman = KalmanFilter(dt, kalman_process_noise, kalman_measurement_noise)
        self.one_euro = OneEuroFilter(freq, one_euro_min_cutoff, one_euro_beta)
        self.ema = ExponentialMovingAverage(ema_alpha)
        
        # Motion state detector
        self.motion_detector = MotionStateDetector(dt)
        
        # Thresholds
        self.velocity_threshold_slow = velocity_threshold_slow
        self.velocity_threshold_fast = velocity_threshold_fast
        
        # State tracking
        self.last_weights = {'kalman': 0.33, 'one_euro': 0.34, 'ema': 0.33}
    
    def calculate_weights(self, motion_state: Dict[str, float], confidence: float) -> Dict[str, float]:
        """
        Calculate adaptive weights for filter blending based on motion state.
        
        Args:
            motion_state (dict): Motion characteristics (velocity, acceleration, jitter)
            confidence (float): Detection confidence (0-1)
            
        Returns:
            dict: Weights for each filter (sum to 1.0)
        """
        velocity = motion_state['velocity']
        jitter = motion_state['jitter_score']
        
        # Base weights
        w_kalman = 0.3
        w_one_euro = 0.4
        w_ema = 0.3
        
        # Adjust based on velocity
        if velocity < self.velocity_threshold_slow:
            # STATIC: Heavy smoothing with EMA
            w_kalman = 0.2
            w_one_euro = 0.2
            w_ema = 0.6
        elif velocity < self.velocity_threshold_fast:
            # SLOW: Balanced, favor One Euro
            w_kalman = 0.25
            w_one_euro = 0.55
            w_ema = 0.2
        else:
            # FAST: Favor Kalman for prediction
            w_kalman = 0.5
            w_one_euro = 0.4
            w_ema = 0.1
        
        # Adjust based on jitter
        if jitter > 0.5:
            # High jitter: Increase smoothing (EMA and One Euro)
            w_ema += 0.1
            w_one_euro += 0.1
            w_kalman -= 0.2
        
        # Adjust based on confidence
        if confidence < 0.5:
            # Low confidence: Rely more on Kalman prediction
            w_kalman += 0.2
            w_one_euro -= 0.1
            w_ema -= 0.1
        
        # Normalize to sum to 1.0
        total = w_kalman + w_one_euro + w_ema
        weights = {
            'kalman': w_kalman / total,
            'one_euro': w_one_euro / total,
            'ema': w_ema / total
        }
        
        return weights
    
    def __call__(self, measurement: float, confidence: float = 1.0) -> Tuple[float, Dict[str, float]]:
        """
        Process a new measurement through the adaptive filter system.
        
        Args:
            measurement (float): New measurement value
            confidence (float): Detection confidence (0-1)
            
        Returns:
            tuple: (filtered_value, weights_used)
        """
        # Update motion state
        motion_state = self.motion_detector.update(measurement)
        
        # Run all filters in parallel
        kalman_out = self.kalman(measurement)
        one_euro_out = self.one_euro(measurement)
        ema_out = self.ema(measurement)
        
        # Calculate adaptive weights
        weights = self.calculate_weights(motion_state, confidence)
        self.last_weights = weights
        
        # Weighted combination
        filtered_value = (weights['kalman'] * kalman_out +
                         weights['one_euro'] * one_euro_out +
                         weights['ema'] * ema_out)
        
        return filtered_value, weights
    
    def reset(self):
        """Reset all filters and motion detector."""
        self.kalman.reset()
        self.one_euro.reset()
        self.ema.reset()
        self.motion_detector.reset()
        self.last_weights = {'kalman': 0.33, 'one_euro': 0.34, 'ema': 0.33}


# ============================================================================
# BATCH PROCESSING UTILITIES
# ============================================================================

def apply_adaptive_filter_to_keypoints(kpts3d: np.ndarray, 
                                       fps: float = 30.0,
                                       **filter_params) -> Tuple[np.ndarray, List[Dict]]:
    """
    Apply adaptive filtering to 3D keypoint data.
    
    Args:
        kpts3d (np.ndarray): Shape (n_frames, n_keypoints, 4) where last dim is [x, y, z, confidence]
        fps (float): Frames per second
        **filter_params: Additional parameters for AdaptiveFilterManager
        
    Returns:
        tuple: (filtered_keypoints, filter_weights_history)
            - filtered_keypoints: Same shape as input
            - filter_weights_history: List of weight dicts for each frame
    """
    n_frames, n_keypoints, _ = kpts3d.shape
    dt = 1.0 / fps
    
    # Initialize output
    kpts3d_filtered = np.zeros_like(kpts3d)
    weights_history = []
    
    # Create filter bank: one AdaptiveFilterManager per keypoint per dimension
    print(f"Initializing {n_keypoints} x 3 adaptive filter bank (fps={fps})...")
    filter_bank = [
        [AdaptiveFilterManager(dt, **filter_params) for _ in range(3)]
        for _ in range(n_keypoints)
    ]
    
    # Process frame by frame
    print(f"Applying adaptive filter to {n_frames} frames...")
    
    for t in range(n_frames):
        frame_weights = {}
        
        for j in range(n_keypoints):
            confidence = kpts3d[t, j, 3]
            is_valid = confidence > 0.0
            
            for d in range(3):  # x, y, z dimensions
                val = kpts3d[t, j, d]
                
                if is_valid:
                    val_filtered, weights = filter_bank[j][d](val, confidence)
                    # Store weights for first keypoint only (for analysis)
                    if j == 0 and d == 0:
                        frame_weights = weights
                else:
                    val_filtered = 0.0
                    filter_bank[j][d].reset()
                
                kpts3d_filtered[t, j, d] = val_filtered
        
        # Copy confidence
        kpts3d_filtered[t, :, 3] = kpts3d[t, :, 3]
        weights_history.append(frame_weights)
    
    print("Adaptive filtering complete.")
    return kpts3d_filtered, weights_history
