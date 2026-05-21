import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
import os

# ============================================================================
# 1. Physical Constants and Simulation Parameters
# ============================================================================
c0 = 299792458.0                # Speed of light in vacuum (m/s)
Z0 = 377.0                      # Impedance of free space (Ohm)

# Simulation settings
f_min = 0.5e9                   # 0.5 GHz
f_max = 4.0e9                   # 4.0 GHz
num_freq = 501                  # Number of frequency points
freqs = np.linspace(f_min, f_max, num_freq)  # Frequency array (Hz)
omega = 2 * np.pi * freqs

# Time domain settings (for pulse and A‑scan)
fs_time = 50e9                  # Sampling frequency (50 GHz)
t_max = 10e-9                   # Time window (10 ns)
dt = 1 / fs_time
n_time = int(round(t_max * fs_time))
if n_time % 2 != 0:
    n_time += 1
t_max = n_time * dt
time = np.linspace(0, t_max, n_time, endpoint=False)  # Time array

# Antenna scan positions (linear scan along x) - exactly as in the study
x_start = -0.05                 # -5 cm
x_end = 0.05                    # +5 cm
num_antennas = 51               # 51 positions as in the study
x_positions = np.linspace(x_start, x_end, num_antennas)

# ============================================================================
# DATA MATCHING THE STUDY (PDF)
# ============================================================================
# Tumor dimensions from the study: diameter 5 mm, depth 2.5 cm
tumor_diameter_cm = 1.48       # cm (5 mm)
tumor_depth_cm = 1.46          # cm (from skin surface)
tumor_center_x = 0.0            # cm (centered under antenna)

# Convert to meters
tumor_radius = (tumor_diameter_cm / 2) / 100     # m
tumor_depth = tumor_depth_cm / 100               # m

# ============================================================================
# 2. Tissue Properties using Cole‑Cole Model (Gabriel et al. 1996)
# Parameters as in the study: [ε∞, Δε, τ (ps), α, σ (S/m)]
# ============================================================================
cole_params = {
    'skin':      [4.0,  32.0, 7.23e-12, 0.00, 0.2],
    'fat':       [2.5,   3.0, 7.96e-12, 0.10, 0.05],
    'muscle':    [8.0,  45.0, 7.96e-12, 0.10, 0.7],
    'tumor':     [10.0, 55.0, 7.00e-12, 0.10, 1.2]   # From study (Table, page 7)
}

def permittivity_cole_cole(f, params):
    eps_inf, delta_eps, tau, alpha, sigma = params
    omega = 2 * np.pi * f
    term = delta_eps / (1 + (1j * omega * tau) ** (1 - alpha))
    sigma_term = 1j * sigma / (omega * 8.8541878128e-12)
    return eps_inf + term - sigma_term

# Pre‑compute complex permittivities for each tissue at all frequencies
eps_complex = {}
for tissue, params in cole_params.items():
    eps_complex[tissue] = permittivity_cole_cole(freqs, params)

# ============================================================================
# 3. Transmission Line Method for Layered Structure (Vectorized)
# ============================================================================
def reflection_coefficient_layers(thicknesses, eps_list, Z0, omega):
    """
    thicknesses: list of layer thicknesses (m), length L
    eps_list: list of complex permittivity arrays for each layer, each shape (num_freqs,)
    returns: Gamma array shape (num_freqs,)
    """
    num_layers = len(thicknesses)
    eps_stack = np.array(eps_list)  # shape (num_layers, num_freqs)
    Z = Z0 / np.sqrt(eps_stack)      # characteristic impedance per layer (complex)
    gamma = 1j * omega * np.sqrt(eps_stack) / c0   # propagation constant

    # Start from the last layer (semi‑infinite)
    Z_in = Z[-1, :]   # shape (num_freqs,)
    # Iterate backwards
    for i in range(num_layers-2, -1, -1):
        d = thicknesses[i]
        phase = gamma[i, :] * d
        tanh_phase = np.tanh(phase)
        Z_in = Z[i, :] * (Z_in + Z[i, :] * tanh_phase) / (Z[i, :] + Z_in * tanh_phase)
    Gamma = (Z_in - Z0) / (Z_in + Z0)
    return Gamma

def build_layer_list(include_tumor=False, tumor_position=None, tumor_thickness=None):
    """
    Build thicknesses and permittivity list for the layered breast model.
    If include_tumor, the tumor can be placed in ANY layer (skin, fat, or muscle).
    """
    thickness_skin = 0.002
    thickness_fat = 0.010
    total_muscle = 0.050

    if not include_tumor:
        thicknesses = [thickness_skin, thickness_fat, total_muscle]
        eps_list = [eps_complex['skin'], eps_complex['fat'], eps_complex['muscle']]
        return thicknesses, eps_list

    # Tumor boundaries
    tumor_half = tumor_thickness / 2.0
    tumor_top = tumor_position - tumor_half
    tumor_bottom = tumor_position + tumor_half

    # Layer boundaries
    skin_bottom = thickness_skin
    fat_bottom = skin_bottom + thickness_fat
    muscle_bottom = fat_bottom + total_muscle

    thicknesses = []
    eps_list = []

    # Skin layer
    if tumor_top < skin_bottom and tumor_bottom > 0:
        d_before = max(0, tumor_top - 0)
        d_tumor = min(skin_bottom, tumor_bottom) - max(0, tumor_top)
        d_after = max(0, skin_bottom - tumor_bottom)
        if d_before > 0:
            thicknesses.append(d_before); eps_list.append(eps_complex['skin'])
        thicknesses.append(d_tumor); eps_list.append(eps_complex['tumor'])
        if d_after > 0:
            thicknesses.append(d_after); eps_list.append(eps_complex['skin'])
    else:
        thicknesses.append(thickness_skin); eps_list.append(eps_complex['skin'])

    # Fat layer
    if tumor_top < fat_bottom and tumor_bottom > skin_bottom:
        d_before = max(0, tumor_top - skin_bottom)
        d_tumor = min(fat_bottom, tumor_bottom) - max(skin_bottom, tumor_top)
        d_after = max(0, fat_bottom - tumor_bottom)
        if d_before > 0:
            thicknesses.append(d_before); eps_list.append(eps_complex['fat'])
        thicknesses.append(d_tumor); eps_list.append(eps_complex['tumor'])
        if d_after > 0:
            thicknesses.append(d_after); eps_list.append(eps_complex['fat'])
    else:
        if not (tumor_top < skin_bottom and tumor_bottom > 0):
            thicknesses.append(thickness_fat); eps_list.append(eps_complex['fat'])

    # Muscle layer
    if tumor_top < muscle_bottom and tumor_bottom > fat_bottom:
        d_before = max(0, tumor_top - fat_bottom)
        d_tumor = min(muscle_bottom, tumor_bottom) - max(fat_bottom, tumor_top)
        d_after = max(0, muscle_bottom - tumor_bottom)
        if d_before > 0:
            thicknesses.append(d_before); eps_list.append(eps_complex['muscle'])
        thicknesses.append(d_tumor); eps_list.append(eps_complex['tumor'])
        if d_after > 0:
            thicknesses.append(d_after); eps_list.append(eps_complex['muscle'])
    else:
        if not (tumor_top < fat_bottom and tumor_bottom > skin_bottom):
            thicknesses.append(total_muscle); eps_list.append(eps_complex['muscle'])

    return thicknesses, eps_list

# Compute healthy and tumor reflection coefficients
thick_healthy, eps_healthy = build_layer_list(include_tumor=False)
Gamma_healthy = reflection_coefficient_layers(thick_healthy, eps_healthy, Z0, omega)

thick_tumor, eps_tumor = build_layer_list(include_tumor=True,
                                          tumor_position=tumor_depth,
                                          tumor_thickness=2*tumor_radius)
Gamma_tumor = reflection_coefficient_layers(thick_tumor, eps_tumor, Z0, omega)
Gamma_diff = Gamma_tumor - Gamma_healthy

# ============================================================================
# 4. Transmitted Pulse (Modulated Gaussian) with delay t0 = 2 ns
# ============================================================================
def transmitted_pulse(time, fc=1.5e9, t0=2e-9, sigma=0.3e-9):
    envelope = np.exp(-((time - t0) / sigma)**2)
    carrier = np.sin(2 * np.pi * fc * (time - t0))
    return envelope * carrier

pulse_t = transmitted_pulse(time, fc=1.5e9, t0=2e-9, sigma=0.3e-9)
Pulse_f = fft(pulse_t)
freqs_fft = fftfreq(n_time, dt)

# Interpolate pulse spectrum onto our frequency grid (positive only)
mask_pos = (freqs_fft >= f_min) & (freqs_fft <= f_max)
Pulse_f_interp = np.interp(freqs, freqs_fft[mask_pos], np.abs(Pulse_f[mask_pos])) * \
                 np.exp(1j * np.interp(freqs, freqs_fft[mask_pos], np.angle(Pulse_f[mask_pos])))

# ============================================================================
# 5. Helper: Frequency‑domain to Time‑domain via IFFT
# ============================================================================
def freq_to_time(transfer_function, pulse_spectrum, freqs, fft_freqs, n_time, dt):
    """
    Convert frequency‑domain transfer function (at given freqs) to time‑domain
    received signal.
    """
    product = pulse_spectrum * transfer_function
    pos_fft_freqs = fft_freqs[:n_time//2+1]
    product_interp = np.interp(pos_fft_freqs, freqs, product, left=0, right=0)
    full_spectrum = np.zeros(n_time, dtype=complex)
    full_spectrum[0] = product_interp[0]
    full_spectrum[1:n_time//2] = product_interp[1:n_time//2]
    if n_time % 2 == 0:
        full_spectrum[n_time//2] = np.real(product_interp[n_time//2])
    for k in range(1, n_time//2):
        full_spectrum[n_time - k] = np.conj(full_spectrum[k])
    received = np.real(ifft(full_spectrum))
    return received

# Compute healthy A‑scan (same for all antenna positions)
healthy_ascan = freq_to_time(Gamma_healthy, Pulse_f_interp, freqs, freqs_fft, n_time, dt)

# ============================================================================
# 6. Compute A‑scans for All Antenna Positions (Tumor Case)
# ============================================================================
idx_fc = np.argmin(np.abs(freqs - 1.5e9))
eps_muscle_fc = eps_complex['muscle'][idx_fc].real
v_eff = c0 / np.sqrt(eps_muscle_fc)   # effective wave speed in muscle (m/s)

ascans_healthy = np.tile(healthy_ascan, (num_antennas, 1))
ascans_tumor = np.zeros((num_antennas, n_time))

for i, x in enumerate(x_positions):
    dx = x - tumor_center_x
    r = np.sqrt(dx**2 + tumor_depth**2)
    delay = 2 * r / v_eff
    phase = np.exp(-1j * omega * delay)
    amp = 1.0 / (r + 0.01)   # spherical spreading factor
    tumor_transfer = Gamma_diff * phase * amp
    tumor_ascan = freq_to_time(tumor_transfer, Pulse_f_interp, freqs, freqs_fft, n_time, dt)
    ascans_tumor[i] = healthy_ascan + tumor_ascan

# ============================================================================
# 7. Add AWGN (SNR = 20 dB as in the study)
# ============================================================================
def add_awgn(signal, snr_db):
    signal_power = np.mean(signal**2)
    snr_linear = 10**(snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power) * np.random.randn(*signal.shape)
    return signal + noise

np.random.seed(42)
snr_db = 20   # as in the study
if snr_db is not None:
    ascans_healthy_noisy = np.array([add_awgn(ascans_healthy[i], snr_db) for i in range(num_antennas)])
    ascans_tumor_noisy   = np.array([add_awgn(ascans_tumor[i], snr_db) for i in range(num_antennas)])
    ascans_healthy_used = ascans_healthy_noisy
    ascans_tumor_used   = ascans_tumor_noisy
else:
    ascans_healthy_used = ascans_healthy
    ascans_tumor_used   = ascans_tumor

# ============================================================================
# 8. Delay‑and‑Sum Beamforming (VECTORIZED for speed)
# ============================================================================
x_img = np.linspace(-0.06, 0.06, 101)   # 101 points as in study
z_img = np.linspace(0.005, 0.08, 101)   # 101 points
img = np.zeros((len(z_img), len(x_img)))

ant_indices = np.arange(num_antennas)

for ix, xp in enumerate(x_img):
    for iz, zp in enumerate(z_img):
        distances = np.sqrt((xp - x_positions)**2 + zp**2)
        delays = 2 * distances / v_eff
        idx = (delays / dt).astype(int)
        valid = (idx >= 0) & (idx < n_time)
        if not np.any(valid):
            continue
        diff_vals = (ascans_tumor_used[ant_indices[valid], idx[valid]] -
                     ascans_healthy_used[ant_indices[valid], idx[valid]])
        img[iz, ix] = np.sum(diff_vals) / num_antennas

# ============================================================================
# 9. Plot Results
# ============================================================================
output_dir = "study_case_results"
os.makedirs(output_dir, exist_ok=True)

# Print case info
print("="*60)
print("Simulating the STUDY case (5 mm tumor at 2.5 cm depth, t0 = 2 ns)")
print(f"Tumor diameter: {tumor_diameter_cm:.2f} cm, depth: {tumor_depth_cm:.2f} cm")
print(f"Effective wave speed in muscle: {v_eff:.3f} m/ns")
expected_echo_time = 2e-9 + 2*tumor_depth/v_eff
print(f"Expected echo time: {expected_echo_time*1e9:.2f} ns (including t0=2 ns)")
print(f"SNR = {snr_db} dB")
print("="*60)

# Figure 1: Reflection coefficient
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(freqs/1e9, np.abs(Gamma_healthy), label='Healthy')
plt.plot(freqs/1e9, np.abs(Gamma_tumor), label='Tumor')
plt.xlabel('Frequency (GHz)'); plt.ylabel('|Γ|'); plt.legend(); plt.grid(True)
plt.title('Reflection Coefficient Magnitude')
plt.subplot(1,2,2)
plt.plot(freqs/1e9, np.angle(Gamma_healthy), label='Healthy')
plt.plot(freqs/1e9, np.angle(Gamma_tumor), label='Tumor')
plt.xlabel('Frequency (GHz)'); plt.ylabel('Phase (rad)'); plt.legend(); plt.grid(True)
plt.title('Reflection Coefficient Phase')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Reflection_Coefficient.png'), dpi=300)
plt.close()

# Figure 2: A‑scan at antenna over tumor
ant_idx = np.argmin(np.abs(x_positions - tumor_center_x))
plt.figure(figsize=(10,6))
plt.plot(time*1e9, ascans_healthy_used[ant_idx], 'g-', label='Healthy', alpha=0.8)
plt.plot(time*1e9, ascans_tumor_used[ant_idx], 'r-', label='Tumor', alpha=0.8)
plt.xlabel('Time (ns)'); plt.ylabel('Amplitude'); plt.legend(); plt.grid(True)
plt.title(f'A‑scan at Antenna over Tumor (Depth={tumor_depth_cm:.1f} cm)')
plt.savefig(os.path.join(output_dir, 'Ascan_Healthy_vs_Tumor.png'), dpi=300)
plt.close()

# Difference A‑scan
diff_ascan = ascans_tumor_used[ant_idx] - ascans_healthy_used[ant_idx]
plt.figure(figsize=(10,6))
plt.plot(time*1e9, diff_ascan, 'b-', label='Difference (Tumor - Healthy)')
plt.xlabel('Time (ns)'); plt.ylabel('Amplitude'); plt.legend(); plt.grid(True)
plt.title('A‑scan Difference')
plt.savefig(os.path.join(output_dir, 'Ascan_Difference.png'), dpi=300)
plt.close()

# B‑scan difference
diff_bscan = ascans_tumor_used - ascans_healthy_used
extent = [x_positions[0]*100, x_positions[-1]*100, time[-1]*1e9, time[0]*1e9]
plt.figure(figsize=(10,6))
plt.imshow(diff_bscan, aspect='auto', cmap='RdBu_r', extent=extent)
plt.xlabel('Position (cm)'); plt.ylabel('Time (ns)'); plt.colorbar(label='Amplitude')
plt.title('B‑scan Difference (Tumor - Healthy)')
plt.savefig(os.path.join(output_dir, 'Bscan_Difference.png'), dpi=300)
plt.close()

# Beamformed image
extent_img = [x_img[0]*100, x_img[-1]*100, z_img[-1]*100, z_img[0]*100]
plt.figure(figsize=(8,6))
plt.imshow(img, aspect='auto', cmap='hot', extent=extent_img)
plt.xlabel('X (cm)'); plt.ylabel('Depth (cm)'); plt.colorbar(label='Intensity')
plt.title('Delay‑and‑Sum Image (Tumor Detection)')
plt.plot(tumor_center_x*100, tumor_depth*100, 'wo', markersize=8, label='Tumor center')
plt.legend()
plt.savefig(os.path.join(output_dir, 'Beamformed_Image.png'), dpi=300)
plt.close()

# Tissue permittivity
plt.figure(figsize=(8,6))
for tissue, eps in eps_complex.items():
    plt.plot(freqs/1e9, eps.real, label=tissue.capitalize())
plt.xlabel('Frequency (GHz)'); plt.ylabel("Real Permittivity (ε')"); plt.legend(); plt.grid(True)
plt.title('Frequency‑Dependent Tissue Permittivity (Cole‑Cole)')
plt.savefig(os.path.join(output_dir, 'Tissue_Properties.png'), dpi=300)
plt.close()

print(f"\nSimulation complete. Results saved in '{output_dir}'.")