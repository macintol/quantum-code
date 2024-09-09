#python import statements

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize

#qiskit import statements

from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import AerSimulator
from qiskit.visualization import plot_histogram, plot_state_city
from qiskit.quantum_info import random_statevector, Statevector
import qiskit.quantum_info as qi
from qiskit.circuit import Parameter #allows us to have parameterized gates
from qiskit.algorithms.optimizers import SPSA

# Use Aer's qasm_simulator
#simulator = AerSimulator()

def add_single_proj_gate(phi, theta, cr, qubits): 
    """Helper function to insert Bmatrix to a circuit for a given phi and theta angle"""
    cr.rz(-1*phi, qubits)
    cr.ry(-1*theta, qubits)
    cr.rz(phi, qubits)

def add_single_proj_gate_conj(phi, theta, cr, qubits):
    """Helper function to insert Bmatrix conjugate to a circuit for a given phi and theta angle"""
    cr.rz(-1*phi, qubits)
    cr.ry(theta, qubits)
    cr.rz(phi, qubits)

def add_beamsplitter(top_index, angle, cr):
    """Helper function to insert beamsplitter for a given angle (representing the reflectivity)"""
    cr.cz(top_index, top_index+1) 
    cr.cx(top_index, top_index+1)
    cr.cry(angle, top_index+1, top_index)
    cr.cx(top_index, top_index+1)
    cr.cz(top_index, top_index+1)

def add_mult_proj_bot(top_index, ancilla, cr):
    """Helper function to insert projector for bottom arm of interferomter (both B and Bconj)"""
    cr.cz(top_index, top_index+1)
    cr.cx(top_index, top_index+1)
    cr.cx(top_index+1, top_index)
    cr.cz(top_index+1, top_index)
    cr.crx(math.pi, top_index, top_index+1)
    cr.cx(top_index, ancilla)
    cr.cx(top_index+1, ancilla+1)
    cr.crx(math.pi, top_index, top_index+1)
    cr.cz(top_index+1, top_index)
    cr.cx(top_index+1, top_index)
    cr.cx(top_index, top_index+1)
    cr.cz(top_index, top_index+1)

def add_mult_proj_top(top_index, ancilla, cr): 
    """Helper function to insert projector for top arm of interferomter (both B and Bconj)"""
    cr.x(top_index+1)
    cr.cx(top_index+1, top_index)
    cr.x(top_index+1)
    cr.cx(top_index, ancilla)
    cr.cx(top_index+1, ancilla+1)
    cr.x(top_index+1)
    cr.cx(top_index+1, top_index)
    cr.x(top_index+1)

def make_spin_circuits(phi_1, phi_2, theta_1, theta_2, time):
    circuit = QuantumCircuit(6) #need 2N qubits in order to calculate the cost function at the end
    circuit.h([0, 3])
    circuit.rz(time, [0, 3]) #first time evolution
    add_single_proj_gate_conj(phi_1, theta_1, circuit)
    circuit.cx(0, 1)
    circuit.cx(3, 4)
    add_single_proj_gate(phi_1, theta_1, circuit)
    circuit.rz(time, [0,3]) #second time evolution
    add_single_proj_gate_conj(phi_2, theta_2, circuit)
    circuit.cx(0, 2)
    circuit.cx(3, 5)
    add_single_proj_gate(phi_2, theta_2, circuit)

    measPartial = QuantumCircuit(6, 2) #6 qubits, and 4 classical bits to record outcome
    measPartial.barrier(range(6)) #just draws a barrier on the circuit
    measPartial.cx(1, 4)
    measPartial.cx(2, 5)

    measPartial.measure([4, 5], range(2)) #add the two measurements

    #smooshing together the acutal circuit and the measurements on the end
    qcPartial = measPartial.compose(circuit, range(6), front=True)

    measFull = QuantumCircuit(6, 4) #6 qubits, and 4 classical bits to record outcome
    measFull.barrier(range(6)) #just draws a barrier on the circuit
    measFull.cx(1, 4)
    measFull.cx(2, 5)
    measFull.h([1, 2])
    #the hadamards are to calculate full cost

    measFull.measure([1, 2, 4, 5], range(4)) #add the four measurements
    #smooshing together the acutal circuit and the measurements on the end
    qcFull = measFull.compose(circuit, range(6), front=True)

    return(qcPartial, qcFull)

def make_interfer_test_circuits(phi_1, phi_2, theta_1, theta_2, start, meas):
    """
    Use 'none' for angles of projectors that won't be added. Use 'proj' or 'system' to define what to measure
    """
    circuit = QuantumCircuit(6) #need 2N qubits in order to calculate the cost function at the end
    if start == 'bot':
        circuit.x([0, 3]) #set which arm it starts in
    circuit.ry(math.pi/2, [0, 3]) #first beamsplitter
    if phi_1 != 'none':
        add_single_proj_gate_conj(phi_1, theta_1, circuit)
        circuit.cx(0, 1)
        circuit.cx(3, 4)
        add_single_proj_gate(phi_1, theta_1, circuit)
    circuit.ry(-1*math.pi/2, [0,3]) #second beamsplitter
    if phi_2 != 'none':
        add_single_proj_gate_conj(phi_2, theta_2, circuit)
        circuit.cx(0, 2)
        circuit.cx(3, 5)
        add_single_proj_gate(phi_2, theta_2, circuit)

    if meas == 'proj':
        measTest = QuantumCircuit(6, 2) #6 qubits, and 4 classical bits to record outcome
        measTest.barrier(range(6)) #just draws a barrier on the circuit
        measTest.measure([4, 5], range(2)) #add the two measurements
    elif meas == 'system':
        measTest = QuantumCircuit(6, 1) #6 qubits, and 4 classical bits to record outcome
        measTest.barrier(range(6)) #just draws a barrier on the circuit
        measTest.measure([3], range(1)) #add the two measurements

    #smooshing together the acutal circuit and the measurements on the end
    qcTest = measTest.compose(circuit, range(6), front=True)

    return(qcTest)

def bind_parameters(varying_params, qcFull, qcPartial):
    first_angle = varying_params[0][0]
    second_angle = varying_params[1][0]
    #first_range = varying_params[0][1]
    #second_range = varying_params[1][1]
    #angle_range = list(itertools.product(first_range, second_range))

    full_circuits = qcFull.assign_parameters({first_angle: first_angle_val, second_angle: second_angle_val})

    partial_circuits = qcPartial.assign_parameters({first_angle: first_angle_val, second_angle: second_angle_val})
            
    return(full_circuits, partial_circuits)

def run_spin_simulation(in_phi_1, in_phi_2, in_theta_1, in_theta_2, time, simulator):
    #this first bit sets what the variable parameters are and sets the rest to the input values
    varying_params = []
    cost_matrix = np.zeros(32**2)

    (phi_1, phi_2, theta_1, theta_2) = (in_phi_1, in_phi_2, in_theta_1, in_theta_2)
    if in_phi_1 == "var":
        phi_1 = Parameter('phi_1')
        phi_1_range = np.linspace(-3, 3, 32)
        varying_params.append((phi_1, phi_1_range))
    if in_phi_2 == "var":
        phi_2 = Parameter('phi_2')
        phi_2_range = np.linspace(-3, 3, 32)
        varying_params.append((phi_2, phi_2_range))
    if in_theta_1 == "var":
        theta_1 = Parameter('theta_1')
        theta_1_range = np.linspace(0, np.pi, 32)
        varying_params.append((theta_1, theta_1_range))
    if in_theta_2 == "var":
        theta_2 = Parameter('theta_2')
        theta_2_range = np.linspace(0, np.pi, 32)
        varying_params.append((theta_2, theta_2_range))
    #this is where we make the circuit
    qcPartial, qcFull = make_spin_circuits(phi_1, phi_2, theta_1, theta_2, time)
    #bind parameters
    full_circuits, partial_circuits = bind_parameters(varying_params, qcFull, qcPartial)
    compiled_circuit_partial = transpile(partial_circuits, simulator)

    # Execute the circuit on the qasm simulator
    job_partial = simulator.run(compiled_circuit_partial, shots=1024)

    # Grab results from the job
    result_partial = job_partial.result()

    # Returns counts
    counts_partial = result_partial.get_counts()

    # compile the circuit down to low-level QASM instructions
    # supported by the backend (not needed for simple circuits)
    compiled_circuit_full = transpile(full_circuits, simulator)

    # Execute the circuit on the qasm simulator
    job_full = simulator.run(compiled_circuit_full, shots=1024)

    # Grab results from the job
    result_full = job_full.result()

    # Returns counts
    counts_full = result_full.get_counts()

    for i in range(32**2):
        cost_matrix[i] = compute_cost(counts_full[i], counts_partial[i])
    cost_matrix = np.reshape(cost_matrix, (32, 32))
    cost_matrix = np.rot90(cost_matrix)
    return cost_matrix, varying_params

def run_interfer_test(in_phi_1, in_phi_2, in_theta_1, in_theta_2, start_in, meas_in, simulator):
    
    qcTest = make_interfer_test_circuits(in_phi_1, in_phi_2, in_theta_1, in_theta_2, start_in, meas_in)

    compiled_circuit_test = transpile(qcTest, simulator)

    # Execute the circuit on the qasm simulator
    job_test = simulator.run(compiled_circuit_test, shots=1024)

    # Grab results from the job
    result_test = job_test.result()

    # Returns counts
    counts_test = result_test.get_counts()

    return(counts_test)

def run_basic_circuit(circuit, simulator):
    """just runs any circuti and returns the measured outputs, defaults to using AerSimulator and 1024 shots"""
    compiled_circuit = transpile(circuit, simulator)

    # Execute the circuit on the qasm simulator
    job = simulator.run(compiled_circuit, shots=1024)

    # Grab results from the job
    result = job.result()

    # Returns counts
    counts = result.get_counts()

    return(compiled_circuit, counts)

def run_basic_circuit_sv(circuit, simulator):
    """just runs any circuti and returns the measured outputs, defaults to using AerSimulator and 1024 shots"""
    compiled_circuit = transpile(circuit, simulator)

    # Execute the circuit on the qasm simulator
    job = simulator.run(compiled_circuit)

    # Grab results from the job
    result = job.result()

    # Returns counts
    counts = result.get_counts()

    return(compiled_circuit, result)

fails = {'0101', '0111', '1010', '1011', '1101', '1110'}
shots = 1024

def compute_cost(full_counts, partial_counts):
    p_full = 0
    overlap_partial = 0
    for i in fails:
        if i in full_counts:
            p_full += full_counts[i]
    overlap_full = 1-2*(p_full/shots)
    overlap_partial = partial_counts['00']/shots
    return overlap_full-overlap_partial

def cost_function(var_array, simulator):
    #this first bit sets what the variable parameters are and sets the rest to the input values
    in_phi_1 = var_array[0]
    in_phi_2 = var_array[1]
    (phi_1, phi_2, theta_1, theta_2) = (in_phi_1, in_phi_2, 0.5*math.pi, 0.5*math.pi)
    time = 2
    varying_params = [phi_1, phi_2]
    #this is where we make the circuit
    qcPartial, qcFull = make_spin_circuits(phi_1, phi_2, theta_1, theta_2, time)
    #bind parameters
    full_circuits = qcFull.assign_parameters({phi_1: in_phi_1, phi_2: in_phi_2})
    partial_circuits = qcPartial.assign_parameters({phi_1: in_phi_1, phi_2: in_phi_2})

    compiled_circuit_partial = transpile(partial_circuits, simulator)

    # Execute the circuit on the qasm simulator
    job_partial = simulator.run(compiled_circuit_partial, shots=1024)

    # Grab results from the job
    result_partial = job_partial.result()

    # Returns counts
    counts_partial = result_partial.get_counts()

    # compile the circuit down to low-level QASM instructions
    # supported by the backend (not needed for simple circuits)
    compiled_circuit_full = transpile(full_circuits, simulator)

    # Execute the circuit on the qasm simulator
    job_full = simulator.run(compiled_circuit_full, shots=1024)

    # Grab results from the job
    result_full = job_full.result()

    # Returns counts
    counts_full = result_full.get_counts()

    cost = compute_cost(counts_full, counts_partial)
    return cost

def visualize_cost(cost_matrix, varying_params):
    first_angle = varying_params[0][0]
    second_angle = varying_params[1][0]
    first_range = varying_params[0][1]
    second_range = varying_params[1][1]
    plt.imshow(cost_matrix, cmap='hot', interpolation='none', extent=[min(first_range), max(first_range), min(second_range), max(second_range)], vmin=0, vmax=0.4)
    plt.xlabel(first_angle)
    plt.ylabel(second_angle)
    plt.colorbar()
    plt.show() 


#sample inputs: 
#initial_guess = np.array([0,1])
#result = minimize(cost_function, initial_guess, method='COBYLA')
#print(result)