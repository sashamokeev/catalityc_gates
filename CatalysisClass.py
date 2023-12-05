import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
#from scipy.optimize import Bounds
from numpy.linalg import norm 
from copy import deepcopy
from scipy.optimize import OptimizeResult
#from joblib import Parallel, delayed






#testing
def random_self_conjugate_matrix(size=3):
    # Random real matrix
    real_part = np.random.rand(size, size)
    
    # Random imaginary matrix
    imag_part = np.random.rand(size, size)
    
    # Upper triangular part (without diagonal)
    upper_triangle = np.triu(imag_part, 1)
    
    # Construct the Hermitian matrix using real_part and upper_triangle
    matrix = real_part + 1j * upper_triangle
    matrix = matrix + matrix.T.conj() - np.diag(matrix.diagonal().imag)
    
    return np.matrix(matrix)



def generate_quasirandom_points(N, dimension):
    # Generate N quasirandom points in the given dimension
    return np.random.rand(N, dimension)

def sm_multistart_eps(measurer,total_num_params, N = 10, q = 4, s = 2, epsilon = 1e-3, MFCALL = 100000):
    #N = 10 number of points to work with
    #q = 4 number of best points
    #s = 2 how many times should point be presented in the best q points to start local search
    #epsilon = 1e-3 tolerance of zero
    #MFCALL = 100 maximum number of function calls
    # Step 1: Generate N points
    x = generate_quasirandom_points(N, total_num_params)
    F_BEST = float('inf')
    k = 0  # Total function evaluations
    NTIX = [0] * N  # Number of times x[j] has been in the top q
    success_x = None  # To store the successful point
    
    maxiter_small = 10
    maxiter_big = 1000

    grad_fun = lambda x: np.float64(measurer.fro_norm_grad(x))
    cost_fun = lambda x: np.float64(measurer.cost_fun(x))


    while k < MFCALL:
        y = []
        for i in range(N):
            #result = measurer.bfgs(x[i], max_iter=maxiter_small, g_tol=1e-3, x_tol=1e-8)
            result = minimize(cost_fun, x[i], method='CG', jac=grad_fun, tol=1e-8, options={'maxiter': maxiter_small})
            y.append([result.x, result.fun])
            k += result.nit  # Increment iterations

        # Step 2: Reduce
        sorted_indices = sorted(range(N), key=lambda i: y[i][1])
        I = sorted_indices[:q]

        for j in range(N):
            if j in I:
                NTIX[j] += 1
            else:
                NTIX[j] = 0

        # Step 3: Find Local Minimum
        for j in I:
            if NTIX[j] >= s:
                NTIX[j] = 0
                z = measurer.bfgs(y[j][0], max_iter=maxiter_big, g_tol=1e-3, x_tol=1e-8)
                k += z.nit# Increment iterations
                if z.fun < F_BEST:
                    F_BEST = z.fun
                    success_x = z.x
                    if F_BEST < epsilon:
                        return success_x, 'Success'

        # Step 4: Sample Additional Points
        for j in range(N):
            if NTIX[j] == 0:
                x[j] = generate_quasirandom_points(1, total_num_params)[0]
            else:
                x[j] = y[j][0]

    return success_x if success_x is not None else F_BEST, 'Failure'

def sm_multistart(measurer,total_num_params, N = 10, q = 4, s = 2, epsilon = 1e-3, MFCALL = 100000):
    #N = 10 number of points to work with
    #q = 4 number of best points
    #s = 2 how many times should point be presented in the best q points to start local search
    #epsilon = 1e-3 tolerance of zero
    #MFCALL = 100 maximum number of function calls
    # Step 1: Generate N points
    r = 10
    x = generate_quasirandom_points(N, total_num_params)
    F_BEST = float('inf')
    k = 0  # Total function evaluations
    NTIX = [0] * N  # Number of times x[j] has been in the top q
    success_x = None  # To store the successful point


    NSP = 0
    NSWP = 0
    epss = 1e-1
    
    maxiter_small = 10
    maxiter_big = 1000

    grad_fun = lambda x: np.float64(measurer.fro_norm_grad(x))
    cost_fun = lambda x: np.float64(measurer.cost_fun(x))


    while k < MFCALL:
        y = []
        for i in range(N):
            #result = measurer.bfgs(x[i], max_iter=maxiter_small, g_tol=1e-3, x_tol=1e-8)
            result = minimize(cost_fun, x[i], method='CG', jac=grad_fun, tol=1e-8, options={'maxiter': maxiter_small})
            y.append([result.x, result.fun])
            k += result.nit  # Increment iterations

        # Step 2: Reduce
        sorted_indices = sorted(range(N), key=lambda i: y[i][1])
        I = sorted_indices[:q]

        for j in range(N):
            if j in I:
                NTIX[j] += 1
            else:
                NTIX[j] = 0

        # Step 3: Find Local Minimum
        for j in I:
            if NTIX[j] >= s:
                NTIX[j] = 0
                if NSP == 0 or F_BEST + epss >= y[j][1]:
                    z = measurer.bfgs(y[j][0], max_iter=maxiter_big, g_tol=1e-3, x_tol=1e-8)

                    if z.fun < epsilon:
                        #    success_x = z
                            return z, 'Success'

                    k += z.nit# Increment iterations
                    if z.fun < F_BEST:
                        NSP += 1
                        NWSP = 0
                        F_BEST = z.fun
                        x_best = z.x
                        
                    else:
                        NWSP += 1
                else:
                    NSWP += 1


                if NWSP >= r * NSP:
                  #  success_x = z
                    return z, 'Success'

        # Step 4: Sample Additional Points
        for j in range(N):
            if NTIX[j] == 0:
                x[j] = generate_quasirandom_points(1, total_num_params)[0]
            else:
                x[j] = y[j][0]

    return (F_BEST, x_best, 'Failure')


def multikron(M_list):
    res = np.eye(1, dtype=np.cdouble)
    for M in M_list:
        res = np.kron(res, M)
    return res
#


"""
def expm(A):
    # Ensure the matrix is square
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square")

    # Diagonalize A
    eigenvalues, P = np.linalg.eig(A)
    D = np.diag(eigenvalues)

    # Calculate the exponential of D
    expD = np.diag(np.exp(np.diagonal(D)))

    # Compute the matrix exponential
    expA = P @ expD @ np.linalg.inv(P)

    return expA
"""
def exp_valuator(group, coef_list, amp = -1j):
        return expm(sum([amp*coef_list[k]*group[k] for k in group.keys()]))

def vec_marriator(vec, ord_gr_keys):
        return dict(zip(ord_gr_keys, vec))


def suzuki_trotter_approximation(A_list, t_list, B, exp_ind = False):#testing
    result_l = np.eye(A_list[0].shape[0])
    for i in range(len(A_list)):
        result_l = np.dot(result_l, expm(-1j * t_list[i] * A_list[i] / 2))
    
    
    result_r = np.eye(A_list[0].shape[0])
    for i in reversed(range(len(A_list))):
        result_r = np.dot(result_r, expm(-1j * t_list[i] * A_list[i] / 2))

    if exp_ind:
        result = np.dot(np.dot(result_l, expm(-1j * B)), result_r)
    else:
        result = np.dot(np.dot(result_l, B),result_r)
    
    return result, result_l, result_r


# Define the Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]], dtype=np.cdouble)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.cdouble)
sigma_z = np.array([[1, 0], [0, -1]], dtype=np.cdouble)
I = np.array([[1, 0], [0, 1]], dtype=np.cdouble)




def generate_pauli_group(n):
# Generate the Pauli group as dictionary {'code': matrix}
    if n == 1:
        return {
            'I': I,
            'X': sigma_x,
            'Y': sigma_y,
            'Z': sigma_z
        }
    
    smaller_group = generate_pauli_group(n-1)
    new_group = {}
    
    for pauli, code in [(I, 'I'), (sigma_x, 'X'), (sigma_y, 'Y'), (sigma_z, 'Z')]:
        for smaller_code, smaller_pauli in smaller_group.items():
            new_code = code + smaller_code
            new_group[new_code] = np.kron(pauli, smaller_pauli)
    
    return new_group



def qft_matrix(n):
    # Generate the matrix of the quantum Fourier transform on n qubits
    N = 2**n
    indices = np.arange(N)
    i, j = np.meshgrid(indices, indices)
    matrix = np.exp(2j * np.pi * i * j / N)
    return matrix / np.sqrt(N)


def inverse_permutation(permutation):

    inverse = np.zeros_like(permutation)
    for i, p in enumerate(permutation):
        inverse[p] = i
    return inverse


def qubit_permutation(qubit_length, gate_descriptions):
    pointer = 0
    involved_qubits = np.zeros(qubit_length)
    Permutation = np.zeros(qubit_length, dtype=int)  # Ensure integer type for indices
    for gate in gate_descriptions:
        for qubit in gate:
            if involved_qubits[qubit] == 1:
                raise ValueError("The qubit is already involved in a gate")
            else:
                involved_qubits[qubit] = 1  # Mark the qubit as involved
                Permutation[qubit] = pointer  # Assign qubit to its new position
                pointer += 1
    for i in range(qubit_length):
        if involved_qubits[i] == 0:
            Permutation[i] = pointer  # Assign uninvolved qubits to remaining positions
            pointer += 1
    return inverse_permutation(Permutation)



def int_to_qubit_basis(index, num_qubits):
    
    
    return format(index, f'0{num_qubits}b')



def qubit_basis_to_int(qubit_basis):

    return int(qubit_basis, 2)

def get_permutation_matrix(pattern):

    n = len(pattern)
    norm_perm = []
    for i in range(2**n):
        itq = int_to_qubit_basis(i, n)
        itq_perm = [itq[pattern[j]] for j in range(n)]
        itq_perm = ''.join(itq_perm)
        norm_perm.append(qubit_basis_to_int(itq_perm))
        
    return np.eye(2**n)[norm_perm]

def qubit_perm_to_perm(qubit_perm):
    # Convert the permutation of qubits to the permutation of the computational basis states
    qubit_perm_mat = get_permutation_matrix(qubit_perm)

    return np.array(qubit_perm_mat@ np.arange(2**len(qubit_perm)), dtype=int)

def EV_r(E,k, r):
    # Generate the matrix E\otimes I_r
    diag_values = np.concatenate([np.ones(r), np.zeros(k - r)])
    I_r = np.diag(diag_values)
    result = np.kron(E, I_r)

    return result

def VE_r(E,k, r):
    # Generate the matrix E\otimes I_r
    diag_values = np.concatenate([np.ones(r), np.zeros(k - r)])
    I_r = np.diag(diag_values)
    result = np.kron(I_r, E)

    return result


def id_permutator(A):
        return A


def generate_permutator_gen(perm):
    # Generate the function that permutes the rows and columns of a matrix according to the given permutation(general case)
    def permutator(A):
            return A[perm, :][:, perm]
        
    return permutator
    
def generate_permutator(q_perm):
    # Generate the function that permutes the rows and columns of a matrix according to the given permutation(qubit case)
    perm = qubit_perm_to_perm(q_perm)
    def permutator(A):
            return A[perm, :][:, perm]
        
    return permutator



class UpdaterHolder:
    
    

    def __init__(self, params, N, group, ord_gr_keys, Pauli=False):
        self.N = N
        self.params = params

        self.coeffs = {}
        self.Iind = 'I' * len(ord_gr_keys[0])
        #self.coeffs[self.Iind] = -1j * np.sin(params[self.Iind]) + np.cos(params[self.Iind])

        self.group = group
        self.Pauli = Pauli
        self.ord_gr_keys = ord_gr_keys


    def pauli_mat_mul(self, pauli_1, pauli_2):
        # Multiply two Pauli matrices
        if pauli_1 == 'I':
            return 1, pauli_2
        elif pauli_2 == 'I':
            return 1, pauli_1
        elif pauli_1 == pauli_2:
            return 1, 'I'
        elif pauli_1 == 'X':
            if pauli_2 == 'Y':
                return 1j, 'Z'
            elif pauli_2 == 'Z':
                return -1j, 'Y'
        elif pauli_1 == 'Y':
            if pauli_2 == 'X':
                return -1j, 'Z'
            elif pauli_2 == 'Z':
                return 1j, 'X'
        elif pauli_1 == 'Z':
            if pauli_2 == 'X':
                return 1j, 'Y'
            elif pauli_2 == 'Y':
                return -1j, 'X'

    def pauli_str_mult(self, pauli_str_1, pauli_str_2):
        # Multiply two Pauli strings
        if len(pauli_str_1) != len(pauli_str_2):
            raise ValueError("Pauli strings must have the same length")
        
        phase = 1
        pauli_str_mult = ""
        for i in range(len(pauli_str_1)):
            ph_upt, multpl = self.pauli_mat_mul(pauli_str_1[i],pauli_str_2[i])
            pauli_str_mult +=  multpl
            phase *= ph_upt
        
        return phase, pauli_str_mult

    def evaluator(self, coef_list, amp=1):
        return sum([amp * coef_list[k] * self.group[k] for k in self.ord_gr_keys]) + amp * coef_list[self.Iind] * self.group[self.Iind]

    def calculer(self):

        coeffs_linear = {}
        coeffs_linear[self.Iind] = -1j * np.sin(self.params[self.Iind]) + np.cos(self.params[self.Iind])#deepcopy(self.coeffs)
        coeffs_second = {}
        for PSN in self.ord_gr_keys:
            coeffs_second_tmp = {}

            for k in coeffs_linear.keys():
                pau_amp, psm_tmp = self.pauli_str_mult(k, PSN)
                coeffs_second_tmp[psm_tmp] = -1j * np.sin(self.params[PSN]) * pau_amp * coeffs_linear[k]

            for k in coeffs_second.keys():
                coeffs_second[k] = np.cos(self.params[PSN]) * coeffs_second[k]

            for k in coeffs_second_tmp.keys():
                if k in coeffs_second.keys():
                    coeffs_second[k] += coeffs_second_tmp[k]
                else:
                    coeffs_second[k] = coeffs_second_tmp[k]

            for k in coeffs_linear.keys():
                coeffs_linear[k] = np.cos(self.params[PSN]) * coeffs_linear[k]

            coeffs_linear[PSN] = -1j * np.sin(self.params[PSN]) * coeffs_linear[self.Iind]
            coeffs_linear[self.Iind] = np.cos(self.params[PSN]) * coeffs_linear[self.Iind]

        for k, v in coeffs_linear.items():
            self.coeffs[k] = v

        for k, v in coeffs_second.items():
            if k in self.coeffs:
                self.coeffs[k] += v
            else:
                self.coeffs[k] = v

        del coeffs_linear, coeffs_second, coeffs_second_tmp

    


    def get_unitary(self):

        if self.Pauli:
            self.calculer()
            res = self.evaluator(self.coeffs)            
            
        
        else:
            res = self.evaluator(self.params, amp=-1j)
            res = expm(res)
            
        return res

    def apply_updt(self,U):
        # Apply the update to the given unitary matrix
        H = self.get_unitary()
        return U @ H
    




class cat_class:

    
    def __init__(self, E,k,r, scheme, init_X, init_Qs, pattern):

        
        self.E = E
        if k < r:
            raise ValueError("k must be greater than r")
        else:
            self.k = k
        if r < 1:
            raise ValueError("r must be greater than 1")
        else:
            self.r = r
        self.n = E.shape[0]
        self.p =  int(np.log2(self.n))
        self.s = int(np.log2(self.k))
        #define LHS of the equation
        self.Er = EV_r(E,k,r)#11111!!!!!!!
        self.InIr = EV_r(np.eye(self.n, dtype=np.cdouble),self.k,self.r) #11111!!!!!!!
        #self.Er = VE_r(E,k,r)
        #self.InIr = VE_r(np.eye(self.n),self.k,self.r)

        
        #scheme is a list of dictionaries of the form {'num_params':num_params,'updaters':updaters, 'permutation':permutation_function},
        # updaters is a list of tuples of the form (num_params,dim,updater_group).
        self.scheme = scheme
        self.total_num_params = sum([self.scheme[i]["num_params"] for i in range(len(self.scheme))])
        #U_list is a list of dictionaries of the form {'Unitary':unitary,'params':params}, where unitary is a matrix and params is a list of parameters for the unitary
        self.U_list = []
        self.actual_X = init_X
        for q in range(len(init_Qs)):
            init_Qs[q] = np.matrix(init_Qs[q])
        self.generate_Us(init_X, init_Qs)
        #pattern is a tuple of tuples (int, "label"), int is number of unitary to use and the "label" says to apply hermitian congugate if the label is "H".
        self.pattern = pattern
        self.cache_Q_list = init_Qs
        self.generate_indeces()        

        self.cache_Unitaries = []
        self.cache_F = None #self.expr_evaluator()
        self.cache_F_actual = False
        self.cache_F_old = None #deepcopy(self.cache_F)
        self.cache_eps = None #np.zeros(self.total_num_params)
        self.cache_B = None #self.direct_jac()
        self.Us_cache = None #deepcopy(self.U_list)
        self.cache_norm = None
        self.cache_norm_grad = None
        self.cache_norm_Hess = None
        #deepcopy(init_Qs)



    @staticmethod
    def get_unitary(N, group, params):
        # Calculate the unitary matrix corresponding to the current parameters
        if group == {}:
            #return np.eye(N, dtype=complex)
            raise ValueError("group is empty")

        else:
            res = np.zeros((N, N), dtype=np.cdouble)
            for key in params.keys():
                res += -1j * params[key] * group[key]

            res = expm(res) 

            return np.matrix(res)
        
    def generate_indeces(self):
        self.indeces = np.zeros(len(self.scheme), dtype=int)
        for p in self.pattern:
            self.indeces[p[0]] += 1
 #   """
    def generate_Us(self, X, Q_list = None):
        if Q_list is None:
            Q_list = self.cache_Q_list
        #generate the list of unitaries and parameters
        self.actual_X = X
        #for each cell of the scheme corresponds a unitary with parameters which is dictionary of the form {'Unitary':unitary,'params':params}
        self.U_list = []
        counter_Q = 0
        count = 0
        for un_flag in range(len(self.scheme)):
            if self.scheme[un_flag]["num_params"] == 0:
                #Q_list is a list of unitaries which are constant, they are given as matrix and parameters are empty
                #first in Q_list will go to the first cell of the scheme with no parameters, i.e. self.scheme[un_flag]["num_params"] == 0
                self.U_list.append({'Unitary':[Q_list[counter_Q],],'params':[]})
                counter_Q += 1
            else:
                #else we generate the unitary with some initial parameters
                #scheme element is a dictionary of the form {'num_params':num_params,'updaters':updaters, 'permutation':permutation_function},
                # updaters is a list of tuples of the form (num_params,dim,updater_group,ord_gr_keys).

                cut = X[count:count + self.scheme[un_flag]["num_params"]]
                cut1 = []
                count += self.scheme[un_flag]["num_params"]
                in_count = 0
                u = []
                for up_element in self.scheme[un_flag]["updaters"]:
                    
                    if (up_element[0] == 0): #or np.allclose(cut[in_count:in_count+up_element[0]], np.zeros(up_element[0])) :
                        

                        u.append(np.eye(up_element[1]))
                        cut1.append([])


                    else:
                        


                        un = cat_class.get_unitary(up_element[1], up_element[2], dict( zip( up_element[3],
                                                        cut[in_count:in_count+up_element[0]] ) ) )
                        u.append(un)
                        cut1.append(cut[in_count:in_count+up_element[0]])
                        in_count += up_element[0]

                #set unitary and parameters
                #u is a list of unitaries which we need to Kronecker multiply
                self.U_list.append({'Unitary':u,'params':cut1})


    
    """
    #parallel US generation
    # some plug for parallel shit
    @staticmethod
    def compute_unitary(up_element, cut):
        unitary_params = dict(zip(up_element[3], cut))
        return cat_class.get_unitary(up_element[1], up_element[2], unitary_params)

    def generate_Us(self, X, Q_list=None):#parallel
        if Q_list is None:
            Q_list = self.cache_Q_list
        #generate the list of unitaries and parameters
        self.actual_X = X
        #for each cell of the scheme corresponds a unitary with parameters which is dictionary of the form {'Unitary':unitary,'params':params}
        count = 0
        parallel_args = {}
        for un_flag in range(len(self.scheme)):
            if self.scheme[un_flag]["num_params"] == 0:
                #Q_list is a list of unitaries which are constant, they are given as matrix and parameters are empty
                #first in Q_list will go to the first cell of the scheme with no parameters, i.e. self.scheme[un_flag]["num_params"] == 0
                pass
            else:
                #else we generate the unitary with some initial parameters
                #scheme element is a dictionary of the form {'num_params':num_params,'updaters':updaters, 'permutation':permutation_function},
                # updaters is a list of tuples of the form (num_params,dim,updater_group,ord_gr_keys).

                cut = X[count:count + self.scheme[un_flag]["num_params"]]
                count += self.scheme[un_flag]["num_params"]
                in_count = 0
                for index, up_element in enumerate(self.scheme[un_flag]["updaters"]):
                    
                    if (up_element[0] == 0): #or np.allclose(cut[in_count:in_count+up_element[0]], np.zeros(up_element[0])) :
                        pass
                    else:
                        
                        parallel_args[f"{un_flag},{index}"] =  (up_element, cut[in_count:in_count+up_element[0]])

                        in_count += up_element[0]


        parallel_results = Parallel(n_jobs=-1)(delayed(cat_class.compute_unitary)(*args) for args in parallel_args.values())

        parallel_results = dict(zip(parallel_args.keys(), parallel_results))


        self.U_list = []
        counter_Q = 0
        count = 0
        for un_flag in range(len(self.scheme)):
            if self.scheme[un_flag]["num_params"] == 0:
                #Q_list is a list of unitaries which are constant, they are given as matrix and parameters are empty
                #first in Q_list will go to the first cell of the scheme with no parameters, i.e. self.scheme[un_flag]["num_params"] == 0
                self.U_list.append({'Unitary':[Q_list[counter_Q],],'params':[]})
                counter_Q += 1
            else:
                #else we generate the unitary with some initial parameters
                #scheme element is a dictionary of the form {'num_params':num_params,'updaters':updaters, 'permutation':permutation_function},
                # updaters is a list of tuples of the form (num_params,dim,updater_group,ord_gr_keys).

                cut = X[count:count + self.scheme[un_flag]["num_params"]]
                cut1 = []
                count += self.scheme[un_flag]["num_params"]
                in_count = 0
                u = []
                for index, up_element in enumerate(self.scheme[un_flag]["updaters"]):
                    
                    if (up_element[0] == 0): #or np.allclose(cut[in_count:in_count+up_element[0]], np.zeros(up_element[0])) :
                        u.append(np.eye(up_element[1]))
                        cut1.append([])

                    else:
                        un = parallel_results[f"{un_flag},{index}"]
                        u.append(un)
                        cut1.append(cut[in_count:in_count+up_element[0]])
                        in_count += up_element[0]

                #set unitary and parameters
                #u is a list of unitaries which we need to Kronecker multiply
                self.U_list.append({'Unitary':u,'params':cut1})





    
    #"""
    

    
    def U_Supdater(self, X, update = True, pauli = False):


    #update the unitaries
        count = 0
    #if update is True we update the unitaries in self.U_list inplace, if update is False we create a new list of unitaries and write updated unitaries there.
    #update is for "Update??!!??11!"
        if update:
            tmp = self.U_list#in the case of update tmp is a pointer to self.U_list
        else:
            tmp = []  #in the case of not update tmp is a new list of unitaries

        for un_flag in range(len(self.scheme)):
            
            if self.scheme[un_flag]["num_params"] != 0:

                cut = X[count:count + self.scheme[un_flag]["num_params"]]
                cut1 = []
                overall_uptd = []
                #cut the piece of X for the current unitary
                count += self.scheme[un_flag]["num_params"]
                #create updaters
                in_count = 0
                for index, up_element in enumerate(self.scheme[un_flag]["updaters"]):
                #up_element is a tuple of the form (num_params,dim,updater_group,ord_gr_keys)  
                    if (up_element[0] == 0):# or ( np.allclose(np.zeros(up_element[0]), cut[in_count:in_count+up_element[0]], rtol=1e-15, atol=1e-18) ):


                        #if we don't need to update with parameters we add identity
                        overall_uptd.append(self.U_list[un_flag]["Unitary"][index])
                        cut1.append([])

                    else:
                        #if we need to update with parameters
                        #KOLHOZ KOSTYL WARNING
                        factor = 1/2


                        if pauli:
                            updator = UpdaterHolder( dict( zip( up_element[3], factor*cut[in_count:in_count+up_element[0]] ) ),
                                                            up_element[1], up_element[2], up_element[3][1:], Pauli = True)
                            updator_rev = UpdaterHolder( dict( zip( up_element[3], factor*cut[in_count:in_count+up_element[0]] ) ),
                                                            up_element[1], up_element[2], list(reversed(up_element[3][1:])),Pauli = True)
                        else:
                            updator = UpdaterHolder( dict( zip( up_element[3], factor*cut[in_count:in_count+up_element[0]] ) ),
                                                            up_element[1], up_element[2], up_element[3][1:], Pauli = False)
                            updator_rev = updator
                        
                            #apply the updater
                        overall_uptd.append(updator.get_unitary() @ self.U_list[un_flag]["Unitary"][index] @ updator_rev.get_unitary())
                        cut1.append(cut[in_count:in_count+up_element[0]])
                        in_count += up_element[0]
                        
                        del updator, updator_rev


                if update:
      
                    new_params = [self.U_list[un_flag]['params'][k] + cut1[k] for k in range(len(cut1))]
                    tmp[un_flag] = {'Unitary':overall_uptd,'params': new_params}#self.U_list[un_flag]['params'] + cut}#update the U_list since tmp is a pointer to it. If not we write the updated unitaries in new dictionary tmp
                else:
                
                    new_params = [self.U_list[un_flag]['params'][k] + cut1[k] for k in range(len(cut1))]
                    tmp.append({'Unitary':overall_uptd,'params': new_params})#self.U_list[un_flag]['params'] + cut}))
                del overall_uptd

        self.cache_F_actual = False

        

        if not update:
            return tmp#return the new list of unitaries if update is False
    
    """#parallel US update
    def _compute_updated_unitaries(self, up_element, cut_piece, current_unitary):
        # This function will be executed in parallel.
        factor = 1/2
        params = dict(zip(up_element[3], factor * cut_piece))
        updator = UpdaterHolder(params, up_element[1], up_element[2], up_element[3][1:], Pauli=True)
        updator_rev = UpdaterHolder(params, up_element[1], up_element[2], list(reversed(up_element[3][1:])), Pauli=True)
        
        # Return the updated unitary
        return updator.get_unitary() @ current_unitary @ updator_rev.get_unitary()

    def U_Supdater_parallel(self, X, update=True):
        if update:
            tmp = self.U_list  # in the case of update, tmp is a pointer to self.U_list
        else:
            tmp = []  # in the case of not update, tmp is a new list of unitaries

        # Prepare arguments for parallel computation
        parallel_args = []

        for un_flag in range(len(self.scheme)):
            if self.scheme[un_flag]["num_params"] != 0:
                cut = X[count:count + self.scheme[un_flag]["num_params"]]
                count += self.scheme[un_flag]["num_params"]
                in_count = 0
                for index, up_element in enumerate(self.scheme[un_flag]["updaters"]):
                    if up_element[0] > 0:
                        # Prepare arguments for parallel computation
                        cut_piece = cut[in_count:in_count + up_element[0]]
                        current_unitary = self.U_list[un_flag]["Unitary"][index]
                        parallel_args.append((up_element, cut_piece, current_unitary))
                    in_count += up_element[0]

        # Use a pool of workers to compute updated unitaries in parallel
        with Pool() as pool:
            updated_unitaries = pool.starmap(self._compute_updated_unitaries, parallel_args)

        # Integrate the results back into self.U_list or tmp
        result_index = 0
        for un_flag in range(len(self.scheme)):
            if self.scheme[un_flag]["num_params"] != 0:
                overall_uptd = []
                for index, up_element in enumerate(self.scheme[un_flag]["updaters"]):
                    if up_element[0] > 0:
                        overall_uptd.append(updated_unitaries[result_index])
                        result_index += 1
                    else:
                        overall_uptd.append(self.U_list[un_flag]["Unitary"][index])
                
                if update:
                    self.U_list[un_flag]["Unitary"] = overall_uptd
                else:
                    tmp.append({'Unitary': overall_uptd, 'params': self.U_list[un_flag]['params'] + cut})

        self.cache_F_actual = False

        if not update:
            return tmp  # return the new list of unitaries if update is False
    """

    
    def variable_locator(self, j):
        # Initialize the running sum to keep track of the current parameter index
        param_index = 0
        
        # Iterate over each unitary (U_k) definition in the scheme
        for k, unitary in enumerate(self.scheme):
            if unitary["num_params"] == 0:
                # If the unitary has no parameters, skip it
                continue
            else:
                # Otherwise, check if the parameter index is in the current unitary
                if param_index <= j < param_index + unitary["num_params"]:
                    u_flag = k
                    break
                else:
                    # If not, increment the running sum and continue
                    param_index += unitary["num_params"]
                    continue
        for index, up_element in enumerate(self.scheme[u_flag]["updaters"]):
            if (up_element[0] == 0):
                continue
            else:
                if param_index <= j < param_index + up_element[0]:
                    return u_flag, index, j - param_index
                else:
                    param_index += up_element[0]
                    continue

    def local_updator(self, X, position):
        
        up_element = self.scheme[position[0]]["updaters"][position[1]]
        updator = UpdaterHolder( dict( zip( up_element[3], 1/2 * X ) ),
                                                        up_element[1], up_element[2], up_element[3][1:], Pauli = True)
        updator_rev = UpdaterHolder( dict( zip( up_element[3], 1/2 * X ) ),
                                                        up_element[1], up_element[2], list(reversed(up_element[3][1:])),Pauli = True)
        
        res = updator.get_unitary()
        res1 = updator_rev.get_unitary()
        
        return res @ self.U_list[position[0]]["Unitary"][position[1]] @ res1

    def local_evaluator(self, x, position):
        #scheme is a list of dictionaries of the form {'num_params':num_params,'updaters':updaters, 'permutation':permutation_function},
        # updaters is a list of tuples of the form (num_params,dim,updater_group,ord_gr_keys).
        #U_list is a list of dictionaries of the form {'Unitary':unitary,'params':params}, where unitary the list of unitaries and params is the list of lists of parameters for the unitaries
        up_element = self.scheme[position[0]]["updaters"][position[1]]
        sum_list = [-1j*x[i] * up_element[2][up_element[3][i]] for i in range(len(x))]
        U_d = expm(np.sum(sum_list, axis=0))
        return U_d
    """
    def local_ev_test(self):
        self.generate_Us(self.actual_X)
        test = []
        for k in range(self.total_num_params):
            position = self.variable_locator(k)
            x = self.U_list[position[0]]['params'][position[1]].copy()
            test_p = np.allclose(self.local_evaluator(x, position), self.U_list[position[0]]["Unitary"][position[1]])
            test.append(test_p)
        return test
    """    
    def local_derivative(self, position, eps = 1e-4):
                
        j = position[2]
        x = self.U_list[position[0]]['params'][position[1]].copy()
        x[j] += eps
        U_d = self.local_evaluator(x, position)
        x[j] -= 2*eps
        U_d1 = self.local_evaluator(x, position)
        dir = (U_d - U_d1)/(2*eps)


        #dir = (U_d - self.U_list[position[0]]["Unitary"][position[1]])/eps

        return np.matrix(dir)
    
    def getF(self):
        if self.cache_F_actual:
            return self.cache_F
        else:
            self.cache_F = self.expr_evaluator()
            self.cache_F_actual = True
            return self.cache_F
    

    @staticmethod
    def multikron(M_list):
        if len(M_list) == 1:
            return M_list[0]
        else:
            res = M_list[0]
            for M in M_list[1:]:
                res = np.kron(res, M) 
            return res
        
    def unitary_cacher(self):
        self.cache_Unitaries = []
        for uni_num in range(len(self.U_list)):
            self.cache_Unitaries.append(cat_class.multikron(self.U_list[uni_num]["Unitary"]))
        
    

    def expr_evaluator(self, Oplist = None, cache = False, dirr = False, replacer = None, repl_pos = None):
        #evaluate the expression in the LHS without I\otimesI_r mult picking the unitaries from U_lis according to the pattern
        res = np.eye(self.n*self.k, dtype=np.cdouble)
        if Oplist is None and cache:
            Oplist = self.cache_Unitaries
        elif Oplist is None and not cache:
            self.unitary_cacher()
            Oplist = self.cache_Unitaries
            cache = True
        else:
            pass

        

        if dirr and self.indeces[repl_pos] > 1:
            interm_prods = [np.eye(self.n*self.k, dtype=np.cdouble),]
            st_prod = np.eye(self.n*self.k, dtype=np.cdouble)
            for U_flag in range(len(self.pattern)):
                op_num = self.pattern[U_flag][0]
                herm_id = self.pattern[U_flag][1]
                perm_tmp = self.scheme[op_num]["permutation"]
                T = Oplist[op_num]
                if op_num != repl_pos:
                    
                    if herm_id == 'H':
                        st_prod = st_prod @ perm_tmp(T.H)
                    else:
                        st_prod = st_prod @ perm_tmp(T)
                else:

                    if herm_id == 'H':
                        interm_prods.append(interm_prods[0]@st_prod@perm_tmp(replacer.H))
                    else:
                        interm_prods.append(interm_prods[0]@st_prod@perm_tmp(replacer))
                    
                    for flag in range(len(interm_prods)-1):
                        if herm_id == 'H':
                            interm_prods[flag] = interm_prods[flag] @ st_prod @ perm_tmp(T.H)
                        else:
                            interm_prods[flag] = interm_prods[flag] @ st_prod @ perm_tmp(T)
                    
                    
                    st_prod = np.eye(self.n*self.k, dtype=np.cdouble)


            interm_prods = [np.matrix(el)@st_prod for el in interm_prods[1:]]
            res = sum(interm_prods)
            del interm_prods, st_prod
            return np.matrix(res)

        elif dirr and self.indeces[repl_pos] == 1:
            Oplist_tmp = Oplist.copy()
            Oplist_tmp[repl_pos] = replacer
            return self.expr_evaluator(Oplist = Oplist_tmp, dirr = False)

        else:
            for U_flag in range(len(self.pattern)):
                op_num = self.pattern[U_flag][0]
                herm_id = self.pattern[U_flag][1]
                perm_tmp = self.scheme[op_num]["permutation"]
                T = Oplist[op_num]
                if herm_id == 'H':
                    #self.pattern[U_flag][0] is the number of unitary to use, self.pattern[U_flag][1] is the label which says to apply hermitian congugate
                    # res = perm_tmp(np.matrix(self.U_list[op_num]["Unitary"]).H) @ res 
                    res = res @ perm_tmp(T.H)
                else:
                    #res = perm_tmp(self.U_list[op_num]["Unitary"]) @ res
                    res = res @ perm_tmp(T)
            
        return np.matrix(res)
    



    def first_derivative(self, j, eps = 1e-4):
        
        
        #self.generate_Us(self.actual_X)#test
        #numbers in pos are: number of unitary, number of updater, number of parameter in updater
        pos = self.variable_locator(j)
        dir_op = self.local_derivative(pos, eps = eps)
        tmp_in_U_list = deepcopy(self.U_list[pos[0]]["Unitary"])
        tmp_in_U_list[pos[1]] = dir_op
        replacer = cat_class.multikron(tmp_in_U_list)
        del tmp_in_U_list

        dirD = self.expr_evaluator(dirr=True, replacer = replacer, repl_pos = pos[0])
        #test good eps 1e-4
        """
        dir_g1 = self.expr_evaluator()
        x = deepcopy(self.actual_X)
        U_L_Tmp = deepcopy(self.U_list)
        self.actual_X[j] += eps
        self.generate_Us(self.actual_X)
        dir_g2 = self.expr_evaluator()
        dir_g = (dir_g2 - dir_g1)/eps
        self.U_list = deepcopy(U_L_Tmp)
        self.actual_X = x
        print(norm(dir_g - dirD, 'fro'))
        """
        return dirD
    
    
    def test_jac(self):
        res = []
        for j in range(self.total_num_params):
            dj  = self.first_derivative(j)
            res.append(dj)
        return res
    
   
    def jac(self):
        #Jacobian
        #res = np.zeros((self.total_num_params, (self.n*self.k)**2), dtype=complex)
        res = []      
        self.unitary_cacher()
        pos_old = self.variable_locator(0) 
        left_in_prod = np.eye(1,  dtype=np.cdouble)
        counter = 0
        #self scheme is a list of dictionaries of the form {'num_params':num_params,'updaters':updaters, 'permutation':permutation_function},
        # updaters is a list of tuples of the form (num_params,dim,updater_group,ord_gr_keys).
        while self.scheme[pos_old[0]]['updaters'][counter][0] == 0:
            left_in_prod = np.kron(left_in_prod, np.eye(self.scheme[pos_old[0]]['updaters'][counter][1]))
       
            counter += 1

        middle_in = self.local_derivative(pos_old)
        if len(self.U_list[pos_old[0]]["Unitary"]) - pos_old[1] > 1:
            right_in_prod = self.multikron(self.U_list[pos_old[0]]["Unitary"][pos_old[1]+1:])#??????
        else:
            right_in_prod = np.eye(1,  dtype=np.cdouble)
        
        
        U_tmp = cat_class.multikron([left_in_prod, middle_in, right_in_prod])
        res.append(self.expr_evaluator(cache=True, dirr = True, replacer = U_tmp, repl_pos = pos_old[0]))
        for j in range(self.total_num_params)[1:]:
            pos_new = self.variable_locator(j)
            if pos_new[0] > pos_old[0]:
                left_in_prod = np.eye(1,  dtype=np.cdouble)
                counter = 0
                while self.scheme[pos_new[0]]['updaters'][counter][0] == 0:
                    left_in_prod = np.kron(left_in_prod, np.eye(self.scheme[pos_new[0]]['updaters'][counter][1]))
                    counter += 1
                middle_in = self.local_derivative(pos_new)
                if len(self.U_list[pos_new[0]]["Unitary"]) - pos_new[1] > 1:
                    right_in_prod = self.multikron(self.U_list[pos_new[0]]["Unitary"][pos_new[1]+1:])#???????
                else:
                    right_in_prod = np.eye(1,  dtype=np.cdouble)
                U_tmp = cat_class.multikron([left_in_prod, middle_in, right_in_prod])
                res.append(self.expr_evaluator(cache=True, dirr = True, replacer = U_tmp, repl_pos = pos_new[0]))
                pos_old = pos_new
            elif pos_new[1] > pos_old[1]:
                left_in_prod = np.kron(left_in_prod, self.U_list[pos_old[0]]["Unitary"][pos_old[1]])
                counter = pos_old[1] + 1
                while self.scheme[pos_new[0]]['updaters'][counter][0] == 0:
                    left_in_prod = np.kron(left_in_prod, np.eye(self.scheme[pos_new[0]]['updaters'][counter][1]))
                    counter += 1
                middle_in = self.local_derivative(pos_new)
                if len(self.U_list[pos_new[0]]["Unitary"]) - pos_new[1] > 1:
                    right_in_prod = self.multikron(self.U_list[pos_new[0]]["Unitary"][pos_new[1]+1:])
                else:
                    right_in_prod = np.eye(1,  dtype=np.cdouble)
                U_tmp = cat_class.multikron([left_in_prod, middle_in, right_in_prod])
                res.append(self.expr_evaluator(cache=True, dirr = True, replacer = U_tmp, repl_pos = pos_new[0]))
                pos_old = pos_new
            else:
                middle_in = self.local_derivative(pos_new)
                U_tmp = cat_class.multikron([left_in_prod, middle_in, right_in_prod])
                res.append(self.expr_evaluator(cache=True, dirr = True, replacer = U_tmp, repl_pos = pos_new[0]))
                pos_old = pos_new
        return res

    
    def expr_evaluator_norm(self, Oplist = None):
        #norm of LHS - RHS
        #return norm(self.expr_evaluator_mEr(Oplist), 'fro')**2

        expr = self.expr_evaluator(Oplist)

        P1 = self.InIr
        P2 = np.eye(self.InIr.shape[0], dtype=np.cdouble) - self.InIr
        S = P1 @ expr @ P2
        Q = P2 @ expr @ P1


        expr_evaluator_mEr = P1@expr - self.Er#P1@expr - self.Er
        norm1 = norm(expr_evaluator_mEr, 'fro')**2
        norm2 = norm(S, 'fro')**2
        norm3 = norm(Q, 'fro')**2
    
        return np.float64(norm1 + norm2 + norm3)
    
    
    def close_point_evaluator(self, X):
        
        #evaluation in the close point
        U_cache = self.U_list.copy()
        x_cache = self.actual_X.copy()
        self.generate_Us(X)
        res = self.expr_evaluator_norm()
        self.U_list = U_cache.copy()
        self.actual_X = x_cache.copy()
        
        #del tmp        
        return res
    
    """
    def close_point_evaluator(self, X):
        
        #evaluation in the close point
        tmp = self.U_Supdater(X, update = False)
        res = self.expr_evaluator_norm(tmp)
        
        del tmp        
        return res
    """
    
    def cost_fun(self, x):
        self.generate_Us(x)
        return self.expr_evaluator_norm()

  
    def fro_norm_dir_1(self,S_dir = None, direction = None, eps = 1e-4):
        expr = self.expr_evaluator()
        P1 = self.InIr
        S = P1 @ expr
        P2 = np.eye(self.InIr.shape[0], dtype=np.cdouble) - self.InIr
        Q = P2 @ expr

        if S_dir is not None:
            pass
        elif direction is not None:
            S_dir = self.first_derivative(direction, eps = eps)
        else:
            raise ValueError("Specify direction or S_dir")
        
        Q_dir = np.matrix(P2 @ S_dir)
        S_dir = np.matrix(self.InIr @ S_dir)
        M1 = - S_dir.H @ self.Er
        M2 =  S_dir.H @ S

        dir1 =  np.trace(M1 + M1.H + M2 + M2.H)


        M3 =  S_dir @ P2 @ S.H
        dir2 = np.trace(M3 + M3.H)#np.trace(S_dir @ P2 @ S.H + S@P2@S_dir.H)

        M4 = Q_dir @ P1 @ Q.H
        dir3 = np.trace(M4 + M4.H)#np.trace(Q_dir @ P1 @ Q + Q @ P1 @ Q_dir)


        return np.float64(dir1 + dir2 + dir3)
    
    def fro_norm_grad(self, X = None):
        if X is not None:
            self.generate_Us(X)
        tmp_grad = np.zeros(self.total_num_params, dtype=np.cdouble)
        jac = self.jac()
        for j in range(self.total_num_params):
            tmp_grad[j] = self.fro_norm_dir_1(S_dir = jac[j])
        
        return tmp_grad
    

    def wolfe_condition_neg(self, x, p, gradient, alpha, f_x, c1=1e-4, c2=0.9):
        armijo_neg = (self.close_point_evaluator(x + alpha * p) > f_x + c1 * alpha * np.dot(gradient, p))
        if armijo_neg:
            return True
        else:
            grad = self.fro_norm_grad(x + alpha * p)
            wolfe_neg = (np.dot(grad, p) < c2 * np.dot(gradient, p))
            return wolfe_neg
            

    def backtracking(self, x, p, gradient, c1=1e-4, alpha=1):
        # Calculate the current function value for comparison
        f_x = self.expr_evaluator_norm()  
        #print(f_x)
        k=0
        # Loop to find the step size that satisfies the Amijo condition
        #while self.close_point_evaluator(x + alpha * p) > f_x + c1 * alpha * np.dot(gradient, p):
        while self.wolfe_condition_neg(x, p, gradient, alpha, f_x, c1=1e-4, c2=0.9):
            alpha *= 0.5  # Reduce step size by half
            if alpha < 1e-8:  # Terminate if step size becomes too small
                break
            k+=1
        return alpha

    def bfgs(self, x_0, max_iter=10000, g_tol=1e-3, x_tol=1e-8):
        x = x_0  # Initialize the position
        n = len(x_0)  # Dimensionality of the input
        
        
        # Evaluate the function at the start and print the value
        self.generate_Us(x, self.cache_Q_list)  # Update the state/parameters
        gradient = self.fro_norm_grad()  # Compute the gradient
        H_inv = np.eye(n)  # Initial inverse Hessian approximation (identity matrix)
        function_value_start = self.expr_evaluator_norm()
        print(f"Start: function value = {function_value_start}")
        
        for iteration in range(max_iter):
            
            
            # Break if the gradient norm is below the threshold
            if np.linalg.norm(gradient) < g_tol:
                break
            
            # Calculate the search direction
            p = -np.dot(H_inv, gradient)
            
            # Find the step size using backtracking line search
            step_size = self.backtracking(x, p, gradient)
           # print("st.size", step_size, iteration)
            # Update the position
            s = step_size * p
            if np.linalg.norm(s) < x_tol:
                break
            x_new = x + s
            
            # BFGS formula to update the inverse Hessian approximation
            self.generate_Us(x_new, self.cache_Q_list)  # Update the state/parameters to the new position
            self.cache_norm_grad = gradient.copy()  # Save the gradient at the old position
            gradient = self.fro_norm_grad()  # Compute the gradient at the new position
            y = gradient - self.cache_norm_grad  # Change in the gradient
            if np.dot(y, s) <= 1e-10:
                break
            rho = 1.0 / np.dot(y, s)
            I = np.eye(n)
            H_inv = (I - rho * np.outer(s, y)) @ H_inv @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)
            
            # Assign the new position
            x = x_new
        
        
        success = True
        if iteration >= max_iter - 1 and np.linalg.norm(gradient) > g_tol:
            success = False
        # Evaluate and print the function value at the end
        function_value_end = self.expr_evaluator_norm()
        print(f"End of BFGS: Iteration {iteration}: gradient norm = {np.linalg.norm(gradient)}, function value = {function_value_end}")
        #return x
        result = OptimizeResult(x=self.actual_X, fun=function_value_end, jac = gradient, hess_inv = H_inv, success=success, nit=iteration)
        #print(result)
        return result
    
    @staticmethod
    def TABU_dist(x, tabu_list):
        dist = 1
        for s in tabu_list:
            dist = dist * np.linalg.norm(x - s, ord = 2)
        return dist**2
    
    @staticmethod
    def TABU_dist_grad(x, tabu_list, eps = 1e-4):
        grad = np.zeros(len(x))
        x_eps = deepcopy(x)
        for j in range(len(x)):

            x_eps[j] += eps
            f1 = cat_class.TABU_dist(x_eps, tabu_list)
            x_eps[j] -= 2*eps
            f2 = cat_class.TABU_dist(x_eps, tabu_list)
            x_eps[j] += eps
            grad[j] = (f1 - f2)/(2*eps)

        return grad

    def TABU_ttf(self, asp, tabu_list, x = None):
        if x is None:
            x = self.actual_X
        else:
            self.generate_Us(x, self.cache_Q_list)
        TTF = ((self.expr_evaluator_norm() - asp)**2) / cat_class.TABU_dist(x, tabu_list)
        return TTF
    
    def TABU_ttf_grad(self, asp, tabu_list):

        x = self.actual_X
        grad_cf = self.fro_norm_grad()
        tabu_grad = cat_class.TABU_dist_grad(x, tabu_list)
        tabu_fun = self.TABU_dist(x,tabu_list)
        print("tabu_fun", tabu_fun)
        cost_a = self.expr_evaluator_norm() -asp
        grad = []
        for j in range(len(x)):
            dfdxj = (2*cost_a*grad_cf[j]*tabu_fun - (cost_a**2)*tabu_grad[j])/(tabu_fun**2)
            grad.append(dfdxj)

        return np.array(grad)


    def TABU_backtracking(self, x, p, gradient, asp, tabu_list, c1=1e-4, alpha=1):
        # Calculate the current function value for comparison
        f_x = self.TABU_ttf(asp, tabu_list, x = x)
        #print(f_x)
        
        # Loop to find the step size that satisfies the Wolfe condition
        while self.TABU_ttf(asp, tabu_list, x = x + alpha * p) > f_x + c1 * alpha * np.dot(gradient, p):
                
            alpha *= 0.5  # Reduce step size by half
            if alpha < 1e-8:  # Terminate if step size becomes too small
                break
        return alpha

    def TABU_ttf_bfgs(self, x_0, asp, tabu_list, max_iter=10000, g_tol=1e-3, x_tol=1e-8):
        x = x_0  # Initialize the position
        n = len(x_0)  # Dimensionality of the input
        
        
        # Evaluate the function at the start and print the value
        self.generate_Us(x, self.cache_Q_list)  # Update the state/parameters
        gradient = self.TABU_ttf_grad(asp, tabu_list)  # Compute the gradient
        H_inv = np.eye(n)  # Initial inverse Hessian approximation (identity matrix)
        function_value_start = self.TABU_ttf(asp, tabu_list)
        
        for iteration in range(max_iter):
            
            
            # Break if the gradient norm is below the threshold
            if np.linalg.norm(gradient) < g_tol:
                break
            
            # Calculate the search direction
            p = -np.dot(H_inv, gradient)
            
            # Find the step size using backtracking line search
            step_size = self.TABU_backtracking(x, p, gradient, asp, tabu_list)
           # print("st.size", step_size, iteration)
            # Update the position
            s = step_size * p
            if np.linalg.norm(s) < x_tol:
                break
            x_new = x + s
            
            # BFGS formula to update the inverse Hessian approximation
            self.generate_Us(x_new, self.cache_Q_list)  # Update the state/parameters to the new position
            TABU_cache_grad = gradient.copy()  # Save the gradient at the old position
            gradient = self.TABU_ttf_grad(asp, tabu_list)  # Compute the gradient at the new position
            y = gradient - TABU_cache_grad  # Change in the gradient
            if np.dot(y, s) <= 1e-10:
                break
            rho = 1.0 / np.dot(y, s)
            I = np.eye(n)
            H_inv = (I - rho * np.outer(s, y)) @ H_inv @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)
            
            # Assign the new position
            x = x_new
        
        
        success = True
        if iteration >= max_iter - 1 and np.linalg.norm(gradient) > g_tol:
            success = False
        # Evaluate and print the function value at the end
        function_value_end = self.TABU_ttf(asp, tabu_list, x = x)
       # print(f"End of BFGS: Iteration {iteration}: gradient norm = {np.linalg.norm(gradient)}, function value = {function_value_end}")
        #return x
        result = OptimizeResult(x=x, fun=function_value_end, jac = gradient, hess_inv = H_inv, success=success)
        #print(result)
        return result



    def grad_descent(self, x_0, max_iter=1000, g_tol=1e-3):
        x = x_0  # Start from the initial guess
        self.generate_Us(x, self.cache_Q_list)  # Update the state/parameters
        
        # Evaluate and print the function value at the start
        function_value_start = self.expr_evaluator_norm()
        print(f"Start: function value = {function_value_start}")
        #initialization of conjugate gradient method
        gradient = self.fro_norm_grad()
        search_dir = -gradient  # Search direction
        step_size = self.backtracking(x, search_dir, gradient)  # Find the step size using backtracking line search
        x = x + step_size * search_dir  # Take a step
        grad_cache = gradient.copy()  # Save the gradient at the old position
        dir_cache = search_dir.copy()  # Save the search direction at the old position
        
        for iteration in range(max_iter):
            self.generate_Us(x, self.cache_Q_list)  # Update the state/parameters
            gradient = self.fro_norm_grad()  # Compute the gradient
            
            
            # Break if the gradient norm is below the threshold
            if np.linalg.norm(gradient) < g_tol:
                break
            
            beta = np.dot(gradient.T, gradient) / np.dot(grad_cache.T, grad_cache)#beta FR

            # Calculate the search direction
            search_dir = -gradient + beta * dir_cache
            
            # Apply backtracking line search to determine the step size
            #step_size = self.backtracking(x, search_dir, gradient)
            step_size = 10**(-3)
            
            # Update the position by taking a step
            x = x + step_size * search_dir
        
        # Evaluate and print the function value at the end
        if iteration >= max_iter - 1 and np.linalg.norm(gradient) > g_tol:
            success = False
            

        self.generate_Us(x, self.cache_Q_list)
        function_value_end = self.expr_evaluator_norm()
        #print(f"End of Gradient Descent: function value = {function_value_end}")
        #return x
        result = OptimizeResult(x=self.actual_X, fun=function_value_end, jac = gradient, success=success)
        return result

"""
p=  2
n = 2**p
s = 2
k = 2**s
r = 1
E = qft_matrix(p)
Q = qft_matrix(s+p)

id_permutator = lambda A: A
permutation_LU_q_vec = (0,2,1,3)
permutator_LU = generate_permutator(permutation_LU_q_vec)

#permutation_LU_q_vec2 = (1,0)
#permutator_LU2 = generate_permutator(permutation_LU_q_vec2)
 

PauGr2 = generate_pauli_group(2)
#lexicographicalli ordered pauli codes
pau_codes2 = sorted(list(PauGr2.keys()))

PauGr2 = (PauGr2, pau_codes2)
#scheme element is a dictionary of the form {'num_params':num_params,'updaters':updaters, 'permutation':permutation_function},
# updaters is a list of tuples of the form (num_params,dim,updater_group,ord_gr_keys).

scheme = [{'num_params':16,'updaters':((0,4,{},[]),(16,4,*PauGr2)), 'permutation':id_permutator},
          {'num_params':32,'updaters':((16,4,*PauGr2),(16,4,*PauGr2)), 'permutation':permutator_LU},
          {'num_params':0,'updaters':((0,16,{},[]),), 'permutation':id_permutator},
          {'num_params':32,'updaters':((16,4,*PauGr2),(16,4,*PauGr2)), 'permutation':permutator_LU},
         # {'num_params':32,'updaters':((16,4,*PauGr2),(16,4,*PauGr2)), 'permutation':permutator_LU},
          ]

totat_par_calculer = lambda scheme: sum([scheme[i]["num_params"] for i in range(len(scheme))])
total_num_params = totat_par_calculer(scheme)

pattern = [(0,'H'),(1,'I'),(2,'I'),(2,'I'),(0,'I')]

x_0 = np.random.rand(total_num_params)
x_1 = np.random.rand(total_num_params)
eps = 10**(-5)
#PARAMETERS SETTING end
def measurer_creator():
    measurer = cat_class(E,k,r, scheme, x_0, [Q,], pattern)
    return measurer
"""
