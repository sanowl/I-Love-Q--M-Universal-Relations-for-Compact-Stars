import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fsolve, minimize

G, c, M_sun = 6.67430e-11, 299792458, 1.989e30

params = {
    'I': [1.47, 0.0817, 0.0149, 2.87e-4, -3.64e-5],
    'Q': [0.194, 0.0936, 0.0474, -4.21e-3, 1.23e-4],
    'δM': [-1.619, 0.255, -0.0195, -1.08e-4, 1.81e-5]
}

def universal_relation(x, y):
    return np.exp(sum(params[y][i] * (np.log(x))**i for i in range(5)))

def extract_M0(lambda_S, M_S, Omega_S):
    I_bar = universal_relation(lambda_S, 'I')
    δM_bar = universal_relation(lambda_S, 'δM')
    return fsolve(lambda M0: np.log((M_S - M0) / (Omega_S**2 * M0**3 * I_bar**2)) - np.log(δM_bar), M_S)[0]

def calculate_I_Q(lambda_S, M_S, Omega_S, approach='extended'):
    I_bar, Q_bar = universal_relation(lambda_S, 'I'), universal_relation(lambda_S, 'Q')
    M0 = extract_M0(lambda_S, M_S, Omega_S) if approach == 'extended' else M_S
    return M0**3 * I_bar, Omega_S**2 * M0**5 * I_bar**2 * Q_bar, M0

class EoS:
    def pressure(self, rho): pass
    def energy_density(self, rho): pass

class PolytropicEoS(EoS):
    def __init__(self, K, gamma):
        self.K, self.gamma = K, gamma
    def pressure(self, rho): return self.K * rho**self.gamma
    def energy_density(self, rho): return rho * c**2 + self.pressure(rho) / (self.gamma - 1)

class MITBagEoS(EoS):
    def __init__(self, B): self.B = B
    def pressure(self, rho): return (rho * c**2 - 4*self.B) / 3
    def energy_density(self, rho): return rho * c**2

def TOV_equations(y, r, eos):
    m, P, nu = y
    rho = (P / eos.K)**(1/eos.gamma) if isinstance(eos, PolytropicEoS) else 3*P/c**2 + 4*eos.B/c**2
    E = eos.energy_density(rho)
    dPdr = -G * (E/c**2 + P/c**2) * (m + 4*np.pi*r**3*P/c**2) / (r * (r - 2*G*m/c**2))
    dmdr = 4 * np.pi * r**2 * E / c**2
    dnudr = 2 * G * (m + 4*np.pi*r**3*P/c**2) / (r * (r - 2*G*m/c**2) * c**2)
    return [dmdr, dPdr, dnudr]

def solve_TOV(P_c, eos):
    r = np.logspace(-6, 2, 1000) * 1000
    return r, odeint(TOV_equations, [0, P_c, 0], r, args=(eos,)).T

def compute_star_properties(P_c, eos):
    r, m, P, _ = solve_TOV(P_c, eos)
    R = r[np.argmin(np.abs(P))]
    return m[np.argmin(np.abs(P))], R

def compute_Love_number(M, R, eos):
    def love_ode(y, r, m, P, nu):
        H, dHdr = y
        rho = (P / eos.K)**(1/eos.gamma) if isinstance(eos, PolytropicEoS) else 3*P/c**2 + 4*eos.B/c**2
        E = eos.energy_density(rho)
        f = 2 * (1 - 2*G*m/(r*c**2))
        dEdr = -G * (E + P) * (m + 4*np.pi*r**3*P/c**2) / (r**2 * f)
        d2Hdr2 = -2 * (1/r + G/(r*c**2) * (m/(r*f) + 4*np.pi*r**2*P/c**2)) * dHdr + \
                 (2*G/(c**2*r) * (dm/dr - 4*np.pi*r**2*E/c**2) - \
                  4*np.pi*G*r/c**4 * (5*E + 9*P + (E + P) * dP/dE)) * H / f
        return [dHdr, d2Hdr2]
    
    r, m, P, nu = solve_TOV(P_c, eos)
    sol = odeint(love_ode, [r[0]**2, 2*r[0]], r, args=(m, P, nu))
    y, R = sol[-1, 0], r[-1]
    C = G * M / (R * c**2)
    y_R = y[-1] / R**2
    k2 = (8*C**5/5) * (1-2*C)**2 * (2+2*C*(y_R-1)-y_R) * \
         (2*C*(6-3*y_R+3*C(5*y_R-8))+4*C**3(13-11*y_R+C(3*y_R-2)+2*C**2(1+y_R))+3*(1-2*C)**2(2-y_R+2*C(y_R-1))*np.log(1-2*C))**(-1)
    return 2 * k2 * R**5 / (3 * G)

def generate_star_models(eos, P_c_range):
    return [(P_c, *compute_star_properties(P_c, eos), compute_Love_number(*compute_star_properties(P_c, eos), eos)) for P_c in P_c_range]

def infer_EoS_parameters(lambda_S, M_S, Omega_S, eos_type, param_ranges):
    def error_function(params):
        eos = PolytropicEoS(*params) if eos_type == 'polytropic' else MITBagEoS(params[0])
        models = generate_star_models(eos, np.logspace(30, 36, 100))
        best_model = min(models, key=lambda m: abs(m[1] - M_S) / M_S + abs(m[3] - lambda_S) / lambda_S)
        M_0, R_0 = best_model[1], best_model[2]
        I_ext, Q_ext, M0_ext = calculate_I_Q(lambda_S, M_S, Omega_S, 'extended')
        I_std, Q_std, M0_std = calculate_I_Q(lambda_S, M_S, Omega_S, 'standard')
        error_ext = abs(M0_ext - M_0) / M_0 + abs(I_ext - M_0 * R_0**2) / (M_0 * R_0**2)
        error_std = abs(M0_std - M_0) / M_0 + abs(I_std - M_0 * R_0**2) / (M_0 * R_0**2)
        return error_ext, error_std
    
    bounds = param_ranges if eos_type == 'polytropic' else [param_ranges[0]]
    x0 = [(b[0] + b[1])/2 for b in bounds]
    result_ext = minimize(lambda x: error_function(x)[0], x0, bounds=bounds)
    result_std = minimize(lambda x: error_function(x)[1], x0, bounds=bounds)
    return result_ext.x, result_std.x, result_ext.fun, result_std.fun

def analyze_star_model(eos, P_c, Omega_S):
    M, R = compute_star_properties(P_c, eos)
    lambda_S = compute_Love_number(M, R, eos)
    I_ext, Q_ext, M0_ext = calculate_I_Q(lambda_S, M, Omega_S, 'extended')
    I_std, Q_std, M0_std = calculate_I_Q(lambda_S, M, Omega_S, 'standard')
    return abs(I_ext - I_std) / I_ext, abs(Q_ext - Q_std) / Q_ext, abs(M0_ext - M0_std) / M0_ext

def plot_errors_vs_frequency(eos, P_c):
    M, R = compute_star_properties(P_c, eos)
    lambda_S = compute_Love_number(M, R, eos)
    Omega_range = np.linspace(0, np.sqrt(G*M / R**3), 100)
    errors = [analyze_star_model(eos, P_c, Omega) for Omega in Omega_range]
    plt.figure(figsize=(10, 6))
    for i, label in enumerate(['E_I', 'E_Q', 'E_M0']):
        plt.plot(Omega_range / (2 * np.pi), [e[i] for e in errors], label=label)
    plt.xlabel('Rotation frequency (Hz)')
    plt.ylabel('Relative error')
    plt.title('Relative errors vs. rotation frequency')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    plt.show()

def analyze_EoS_inference(true_eos, true_params, param_ranges, Omega_S):
    P_c = 6e34
    M, R = compute_star_properties(P_c, true_eos)
    lambda_S = compute_Love_number(M, R, true_eos)
    inferred_params_ext, inferred_params_std, error_ext, error_std = infer_EoS_parameters(
        lambda_S, M, Omega_S, type(true_eos).__name__.lower().replace('eos', ''), param_ranges
    )
    rel_errors_ext = np.abs(np.array(inferred_params_ext) - np.array(true_params)) / np.array(true_params)
    rel_errors_std = np.abs(np.array(inferred_params_std) - np.array(true_params)) / np.array(true_params)
    return rel_errors_ext, rel_errors_std, error_ext, error_std

def plot_EoS_inference(eos, true_params, param_ranges, Omega_S):
    if isinstance(eos, PolytropicEoS):
        K_range, gamma_range = param_ranges
        K_values, gamma_values = [np.linspace(*r, 50) for r in param_ranges]
        K_grid, gamma_grid = np.meshgrid(K_values, gamma_values)
        errors_ext = np.zeros_like(K_grid)
        errors_std = np.zeros_like(K_grid)
        for i in range(len(K_values)):
            for j in range(len(gamma_values)):
                K, gamma = K_grid[i, j], gamma_grid[i, j]
                test_eos = PolytropicEoS(K, gamma)
                P_c = 6e34
                M, R = compute_star_properties(P_c, test_eos)
                lambda_S = compute_Love_number(M, R, test_eos)
                _, _, errors_ext[i, j], errors_std[i, j] = infer_EoS_parameters(
                    lambda_S, M, Omega_S, 'polytropic', param_ranges
                )
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        for ax, errors, title in zip([ax1, ax2], [errors_ext, errors_std], ['Extended Approach', 'Standard Approach']):
            c = ax.contourf(K_grid, gamma_grid, np.log10(errors), levels=20)
            ax.set_xlabel('K')
            ax.set_ylabel('gamma')
            ax.set_title(title)
            fig.colorbar(c, ax=ax, label='log10(error)')
            ax.plot(true_params[0], true_params[1], 'r*', markersize=10)
    else:
        B_range = param_ranges[0]
        B_values = np.linspace(*B_range, 100)
        errors_ext = []
        errors_std = []
        for B in B_values:
            test_eos = MITBagEoS(B)
            P_c = 6e34
            M, R = compute_star_properties(P_c, test_eos)
            lambda_S = compute_Love_number(M, R, test_eos)
            _, _, error_ext, error_std = infer_EoS_parameters(
                lambda_S, M, Omega_S, 'mit_bag', param_ranges
            )
            errors_ext.append(error_ext)
            errors_std.append(error_std)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        for ax, errors, title in zip([ax1, ax2], [errors_ext, errors_std], ['Extended Approach', 'Standard Approach']):
            ax.plot(B_values, np.log10(errors))
            ax.set_xlabel('B (g/cm^3)')
            ax.set_ylabel('log10(error)')
            ax.set_title(title)
            ax.axvline(true_params[0], color='r', linestyle='--')
    plt.tight_layout()
    plt.show()

def analyze_rotation_effect(eos, true_params, param_ranges, Omega_range):
    P_c = 6e34
    M, R = compute_star_properties(P_c, eos)
    lambda_S = compute_Love_number(M, R, eos)
    errors = [infer_EoS_parameters(lambda_S, M, Omega, type(eos).__name__.lower().replace('eos', ''), param_ranges)[2:] for Omega in Omega_range]
    plt.figure(figsize=(10, 6))
    for i, label in enumerate(['Extended Approach', 'Standard Approach']):
        plt.plot(Omega_range / (2 * np.pi), [e[i] for e in errors], label=label)
    plt.xlabel('Rotation frequency (Hz)')
    plt.ylabel('Total error')
    plt.title('EoS Inference Error vs. Rotation Frequency')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    plt.show()

# Main analysis and execution
if __name__ == "__main__":
    K, gamma, B = 100, 2, 1e14
    poly_eos, mit_eos = PolytropicEoS(K, gamma), MITBagEoS(B)
    Omega_max = lambda M, R: np.sqrt(G*M / R**3)

    # Analyze polytropic models
    P_c_values = np.logspace(33, 35, 6)
    results = [(*compute_star_properties(P_c, poly_eos), *analyze_star_model(poly_eos, P_c, Omega_max(*compute_star_properties(P_c, poly_eos)))) for P_c in P_c_values]

    print("Polytropic Models (AU0-AU5):")
    print("Model | P_c (g/cm^3) | M (M_sun) | R (km) | E_I | E_Q | E_M0")
    for i, (M, R, E_I, E_Q, E_M0) in enumerate(results):
        print(f"AU{i} | {P_c_values[i]:.2e} | {M/M_sun:.3f} | {R/1000:.3f} | {E_I:.2e} | {E_Q:.2e} | {E_M0:.2e}")

    # Analyze MIT bag model
    P_c = 5e34
    M, R = compute_star_properties(P_c, mit_eos)
    Omega_S = Omega_max(M, R)
    E_I, E_Q, E_M0 = analyze_star_model(mit_eos, P_c, Omega_S)

    print("\nMIT Bag Model:")
    print(f"P_c = {P_c:.2e} g/cm^3, M = {M/M_sun:.3f} M_sun, R = {R/1000:.3f} km")
    print(f"E_I = {E_I:.2e}, E_Q = {E_Q:.2e}, E_M0 = {E_M0:.2e}")

    # Plot errors vs. frequency
    plot_errors_vs_frequency(poly_eos, P_c_values[3])  # Plot for AU3 model
    plot_errors_vs_frequency(mit_eos, P_c)  # Plot for MIT bag model

    # Analyze EoS parameter inference
    true_poly_params = [K, gamma]
    poly_param_ranges = [(50, 150), (1.8, 2.2)]
    Omega_S_poly = 2 * np.pi * 500  # Example rotation frequency

    rel_errors_ext_poly, rel_errors_std_poly, error_ext_poly, error_std_poly = analyze_EoS_inference(
        poly_eos, true_poly_params, poly_param_ranges, Omega_S_poly
    )

    print("\nPolytropic EoS Parameter Inference:")
    print(f"Extended approach relative errors: K: {rel_errors_ext_poly[0]:.2e}, gamma: {rel_errors_ext_poly[1]:.2e}")
    print(f"Standard approach relative errors: K: {rel_errors_std_poly[0]:.2e}, gamma: {rel_errors_std_poly[1]:.2e}")
    print(f"Extended approach total error: {error_ext_poly:.2e}")
    print(f"Standard approach total error: {error_std_poly:.2e}")

    true_mit_params = [B]
    mit_param_ranges = [(0.5e14, 1.5e14)]
    Omega_S_mit = 2 * np.pi * 700  # Example rotation frequency

    rel_errors_ext_mit, rel_errors_std_mit, error_ext_mit, error_std_mit = analyze_EoS_inference(
        mit_eos, true_mit_params, mit_param_ranges, Omega_S_mit
    )

    print("\nMIT Bag EoS Parameter Inference:")
    print(f"Extended approach relative error: B: {rel_errors_ext_mit[0]:.2e}")
    print(f"Standard approach relative error: B: {rel_errors_std_mit[0]:.2e}")
    print(f"Extended approach total error: {error_ext_mit:.2e}")
    print(f"Standard approach total error: {error_std_mit:.2e}")

    # Plot EoS inference results
    plot_EoS_inference(poly_eos, true_poly_params, poly_param_ranges, Omega_S_poly)
    plot_EoS_inference(mit_eos, true_mit_params, mit_param_ranges, Omega_S_mit)

    # Analyze the effect of rotation rate on EoS inference accuracy
    Omega_range = np.linspace(0, 2 * np.pi * 1000, 50)
    analyze_rotation_effect(poly_eos, true_poly_params, poly_param_ranges, Omega_range)
    analyze_rotation_effect(mit_eos, true_mit_params, mit_param_ranges, Omega_range)

    print("\nAnalysis complete. Please review the generated plots and printed results.")
    