# I-Love-Q-δM Universal Relations for Compact Stars

This repository contains an implementation of the extended I-Love-Q-δM universal relations for compact stars, as described in the paper "I-Love-Q, and δM too: The role of the mass in universal relations of compact stars" by Eneko Aranguren, José A. Font, Nicolas Sanchis-Gual, and Raül Vera.

## Paper Summary

The paper introduces an extended version of the I-Love-Q universal relations for rotating compact stars, incorporating a fourth parameter δM. This extension aims to improve the accuracy of inferring neutron star properties, especially for rapidly rotating stars.

Key points:

1. Standard I-Love-Q relations use a background mass M0, which is not directly observable.
2. Most studies approximate M0 as equal to the actual stellar mass MS, which can be inconsistent for rapidly rotating stars.
3. The extended relations include δM, proportional to MS - M0, allowing unambiguous extraction of M0.
4. The extended approach yields more accurate predictions, especially for rapidly rotating stars.
5. The paper demonstrates improved accuracy in inferring equation of state (EoS) parameters using the extended relations.

## Code Implementation

The provided Python script implements the key concepts from the paper:

1. Universal relations for I-Love-Q-δM
2. Polytropic and MIT Bag equations of state
3. Simplified TOV equation solver
4. Functions to compute star properties and Love numbers
5. EoS parameter inference method comparing extended and standard approaches
6. Visualization of relative errors vs. rotation frequency

### Key Functions

- `universal_relation(x, y)`: Calculates the universal relation for a given quantity
- `extract_M0(lambda_S, M_S, Omega_S)`: Extracts M0 using the extended approach
- `calculate_I_Q(lambda_S, M_S, Omega_S, approach)`: Calculates I and Q using either extended or standard approach
- `solve_TOV(P_c, eos)`: Solves TOV equations for given central pressure
- `compute_star_properties(P_c, eos)`: Computes star properties for given central pressure
- `infer_EoS_parameters(lambda_S, M_S, Omega_S, eos_type, param_ranges)`: Infers EoS parameters using both approaches

### Usage

The script demonstrates the use of these functions for both Polytropic and MIT Bag EoS. It compares the accuracy of the extended and standard approaches in inferring EoS parameters and plots the relative errors as a function of rotation frequency.

## Limitations

This implementation is a simplified version of the full astrophysical models used in the paper. Notable simplifications include:

- Simplified TOV equation solver and Love number calculation
- Incomplete implementation of the perturbative approach for rotating stars
- Potential differences in numerical methods for equation solving and optimization

To exactly replicate the paper's results, one would need access to the specific numerical methods, full perturbative approach implementation, and detailed EoS models used by the authors.

## Conclusion

This code demonstrates the key concepts and approach described in the paper, showing how the extended I-Love-Q-δM relations can improve the accuracy of inferring neutron star properties, especially for rapidly rotating stars. The results highlight the importance of incorporating the δM parameter in universal relations for compact stars.