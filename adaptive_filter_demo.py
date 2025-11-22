"""
Adaptive Filter Demonstration Script

This script demonstrates the adaptive filtering system for motion capture data.
Run this instead of the notebook if you prefer a Python script.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import json
import os
import sys

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mokka.postprocess.adaptive_filter import apply_adaptive_filter_to_keypoints, KalmanFilter


def calculate_smoothness(trajectory):
    """Calculate jerk (third derivative) as smoothness metric"""
    velocity = np.diff(trajectory)
    acceleration = np.diff(velocity)
    jerk = np.diff(acceleration)
    return np.mean(np.abs(jerk))


def main():
    print("="*60)
    print("ADAPTIVE FILTER DEMONSTRATION")
    print("="*60)
    
    # 1. Load Input Data
    print("\n1. Loading input data...")
    input_file = 'motion-correction/footage_Day08_Mon10_Yr2025_Hr11_Min11_Sec50.json'
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    frames = data['frames']
    n_frames = len(frames)
    n_keypoints = len(frames[0])
    
    print(f"   Loaded {n_frames} frames with {n_keypoints} keypoints each")
    
    # Convert to numpy array
    kpts3d = np.array(frames, dtype=np.float32)
    print(f"   Data shape: {kpts3d.shape}")
    
    # Check confidence values
    confidence_values = kpts3d[:, :, 3]
    print(f"   Confidence range: [{confidence_values.min():.3f}, {confidence_values.max():.3f}]")
    
    # 2. Configure Adaptive Filter
    print("\n2. Configuring adaptive filter...")
    FPS = 30.0
    
    filter_params = {
        'kalman_process_noise': 1e-2,
        'kalman_measurement_noise': 3e-5,
        'one_euro_min_cutoff': 1.0,
        'one_euro_beta': 0.007,
        'ema_alpha': 0.3,
        'velocity_threshold_slow': 0.1,
        'velocity_threshold_fast': 1.0,
    }
    
    print("   Filter parameters:")
    for key, value in filter_params.items():
        print(f"     {key}: {value}")
    
    # 3. Apply Adaptive Filter
    print("\n3. Applying adaptive filter...")
    start_time = time.time()
    
    kpts3d_filtered, weights_history = apply_adaptive_filter_to_keypoints(
        kpts3d, 
        fps=FPS,
        **filter_params
    )
    
    elapsed_time = time.time() - start_time
    
    print(f"   Filtering completed in {elapsed_time:.2f} seconds")
    print(f"   Processing speed: {n_frames / elapsed_time:.1f} fps")
    
    # 4. Calculate Metrics
    print("\n4. Calculating smoothness metrics...")
    keypoint_idx = 0
    
    print(f"   Smoothness comparison for keypoint {keypoint_idx}:")
    for d, dim in enumerate(['X', 'Y', 'Z']):
        orig_smoothness = calculate_smoothness(kpts3d[:, keypoint_idx, d])
        filt_smoothness = calculate_smoothness(kpts3d_filtered[:, keypoint_idx, d])
        improvement = ((orig_smoothness - filt_smoothness) / orig_smoothness) * 100
        
        print(f"     {dim}: Original={orig_smoothness:.6f}, Filtered={filt_smoothness:.6f}, Improvement={improvement:.1f}%")
    
    # 5. Analyze Filter Weights
    print("\n5. Analyzing filter weight distribution...")
    kalman_weights = [w.get('kalman', 0) for w in weights_history]
    one_euro_weights = [w.get('one_euro', 0) for w in weights_history]
    ema_weights = [w.get('ema', 0) for w in weights_history]
    
    print(f"   Kalman  - Mean: {np.mean(kalman_weights):.3f}, Std: {np.std(kalman_weights):.3f}")
    print(f"   One Euro - Mean: {np.mean(one_euro_weights):.3f}, Std: {np.std(one_euro_weights):.3f}")
    print(f"   EMA     - Mean: {np.mean(ema_weights):.3f}, Std: {np.std(ema_weights):.3f}")
    
    # 6. Compare with Simple Kalman
    print("\n6. Comparing with simple Kalman filter...")
    dt = 1.0 / FPS
    kalman_only = KalmanFilter(dt, process_noise=1e-2, measurement_noise=3e-5)
    
    kalman_result = []
    for t in range(n_frames):
        val = kpts3d[t, keypoint_idx, 0]  # X dimension
        if kpts3d[t, keypoint_idx, 3] > 0:
            filtered_val = kalman_only(val)
        else:
            filtered_val = 0.0
            kalman_only.reset()
        kalman_result.append(filtered_val)
    
    kalman_smoothness = calculate_smoothness(np.array(kalman_result))
    adaptive_smoothness = calculate_smoothness(kpts3d_filtered[:, keypoint_idx, 0])
    
    print(f"   Kalman only smoothness: {kalman_smoothness:.6f}")
    print(f"   Adaptive filter smoothness: {adaptive_smoothness:.6f}")
    print(f"   Difference: {((kalman_smoothness - adaptive_smoothness) / kalman_smoothness * 100):.1f}%")
    
    # 7. Export Results
    print("\n7. Exporting results...")
    output_frames = kpts3d_filtered.tolist()
    output_data = {'frames': output_frames}
    
    output_filename = 'motion-correction/' + time.strftime("d%d_mo%m_y%Y_%Hh_%Mm_%Ss_") + 'adaptive-filtered_output.json'
    
    with open(output_filename, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"   Successfully saved to: {output_filename}")
    print(f"   File size: {os.path.getsize(output_filename) / 1024:.1f} KB")
    
    # 8. Create Visualizations
    print("\n8. Creating visualizations...")
    
    # Plot 1: Trajectory comparison
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(f'Adaptive Filter Results - Keypoint {keypoint_idx}', fontsize=16)
    
    dimensions = ['X', 'Y', 'Z']
    colors_orig = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    colors_filt = ['#C44569', '#218C8D', '#2E86AB']
    
    for d, (ax, dim, c_orig, c_filt) in enumerate(zip(axes, dimensions, colors_orig, colors_filt)):
        ax.plot(kpts3d[:, keypoint_idx, d], label='Original', alpha=0.5, color=c_orig, linewidth=1)
        ax.plot(kpts3d_filtered[:, keypoint_idx, d], label='Filtered', color=c_filt, linewidth=2)
        
        ax.set_ylabel(f'{dim} Position', fontsize=11)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Frame', fontsize=11)
    plt.tight_layout()
    plt.savefig('motion-correction/adaptive_filter_trajectory.png', dpi=150, bbox_inches='tight')
    print("   Saved: adaptive_filter_trajectory.png")
    
    # Plot 2: Filter weights evolution
    fig, ax = plt.subplots(figsize=(14, 5))
    
    ax.plot(kalman_weights, label='Kalman Weight', color='#E74C3C', linewidth=2)
    ax.plot(one_euro_weights, label='One Euro Weight', color='#3498DB', linewidth=2)
    ax.plot(ema_weights, label='EMA Weight', color='#2ECC71', linewidth=2)
    
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Filter Weight', fontsize=12)
    ax.set_title('Adaptive Filter Weight Evolution Over Time', fontsize=14)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('motion-correction/adaptive_filter_weights.png', dpi=150, bbox_inches='tight')
    print("   Saved: adaptive_filter_weights.png")
    
    # Plot 3: Comparison with Kalman
    fig, ax = plt.subplots(figsize=(14, 5))
    
    ax.plot(kpts3d[:, keypoint_idx, 0], label='Original', alpha=0.4, color='gray', linewidth=1)
    ax.plot(kalman_result, label='Kalman Only', color='#E74C3C', linewidth=2, alpha=0.7)
    ax.plot(kpts3d_filtered[:, keypoint_idx, 0], label='Adaptive Filter', color='#2ECC71', linewidth=2)
    
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('X Position', fontsize=12)
    ax.set_title(f'Comparison: Kalman vs Adaptive Filter (Keypoint {keypoint_idx})', fontsize=14)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('motion-correction/adaptive_vs_kalman.png', dpi=150, bbox_inches='tight')
    print("   Saved: adaptive_vs_kalman.png")
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE!")
    print("="*60)
    print("\nThe adaptive filter successfully combines:")
    print("  • Kalman Filter: Prediction and missing data handling")
    print("  • One Euro Filter: Velocity-adaptive smoothing")
    print("  • EMA: Strong smoothing for static poses")
    print("\nFilter weights automatically adjust based on motion state.")


if __name__ == "__main__":
    main()
