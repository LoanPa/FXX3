import matplotlib.pyplot as plt
import numpy as np
import random

from numpy.core.fromnumeric import size

def factorial(n):
    if n <= 0:
        return 1
    res = n*factorial(n-1)
    return res
def plot_results(ux, uy, ox, oy, param, protocol, save_it=True, x_label='x-axis', y_label='y-axis'):
    """ Generic plot:
            ux : mean of x-axis magnitude
            uy : mean of y-axis magnitude
            ox : std of x-axis magnitude
            oy : std of y-axis magnitude
    """
    nodes, lmbd, L, T, t_sim, n_iters = param
    
    plt.figure()
    plt.plot(ux, uy, 'k', zorder=10, linewidth=2, label='Avg.')
    plt.plot(ux+ox, uy+oy, 'k', linestyle='dashed', alpha=0.5, label='Std.')
    plt.plot(ux-ox, uy-oy, 'k', linestyle='dashed', alpha=0.5,)
    y_max = np.max(uy)
    idx_y_max = ux[np.where(uy == y_max)] 
    plt.scatter(idx_y_max, y_max, zorder=11, label=f'Max.')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'{protocol} ({t_sim} s. simulation)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_it: plt.savefig(f'./results_{protocol}_{nodes[-1]}_{lmbd}_{L}_{T}_{t_sim}_{n_iters}.png', dpi=100)
    plt.show()
    plt.close()

def simulate_MAC(time_slots, lmbd, num_nodes, protocol):
    """ Returns number of packets transmitted and number of packets correctly received """
    
    # Choose MAC protocol to simulate
    if protocol == 'S-ALOHA':

        n_transmitted, n_received, n_collided = 0, 0, 0

        # Run discrete time
        for slot in time_slots:
            n_transmitted_slot = 0
            for i in range(num_nodes):
                t = random.expovariate(lmbd)
                if np.ceil(t) == 1:
                    n_transmitted_slot += 1
            
            n_transmitted += n_transmitted_slot
            if n_transmitted_slot > 1:
                n_collided += n_transmitted_slot
            else:
                n_received += n_transmitted_slot
                
            
        # Check that the output is valid and return
        assert n_transmitted == (n_received + n_collided)
        return n_transmitted, n_received

    elif protocol == 'ALOHA':

        n_transmitted, n_received, n_collided = 0, 0, 0
        time_slots*=1000
        nodes_aloha = [0]* num_nodes


        for slot in time_slots:
            n_transmitted_slot = 0


            for j in nodes_aloha:
                if nodes_aloha[j] == 0:
                    continue
                nodes_aloha[j] += random.expovariate(lmbd)*1000
                if np.floor(nodes_aloha[j]) == 0:
                    n_transmitted_slot += 1
            



            
            n_transmitted += n_transmitted_slot
            if n_transmitted_slot > 1:
                n_collided += n_transmitted_slot
            else:
                n_received += n_transmitted_slot
    else:
        raise ValueError(f'Unknown MAC protocol: {protocol}.')

def run(n_time_slots, protocol, param):
    """ Runs several simulations of a MAC protocl and gets the output statistics """
    nodes, lmbd, L, T, t_sim, n_iters = param

    # Pre-alloc needed memory
    results     = np.empty((n_iters, 2), dtype=np.float32)
    results_avg = np.empty((len(nodes), 2), dtype=np.float32)
    results_std = np.empty((len(nodes), 2), dtype=np.float32)


    # EX 2

    # Start simulation
    # we increase the number of nodes in steps of 2 to increase the offered traffic
    for n, N in enumerate(nodes):
        print(f'Simulating for {N} active nodes. Progess: {100*N/nodes[-1]:.1f}% ...')

        # Do n_iters for the same number of nodes 
        for i in range(n_iters):
            results[i] = simulate_MAC(time_slots=np.arange(n_time_slots, dtype=int),
                                      lmbd=lmbd, 
                                      num_nodes=N,
                                      protocol=protocol)

        # Get the statistics of the n_iters
        results_avg[n] = results.mean(axis=0)
        results_std[n] = results.std(axis=0)


    G_avg = results_avg[:, 0]*lmbd*T*L*t_sim
    G_std = results_std[:, 0]*lmbd*T*L

    S_avg = (results_avg[:, 1]/results_avg[:, 0])*G_avg
    S_std = (results_std[:, 1]/results_std[:, 0])*G_std

    # Plot the results
    # first, convert the results to any metric of interest
    sf_x = 1    # scale factor for x-axis magnitude
    sf_y = 1
    
    plot_results(ux= G_avg * sf_x,
                 uy= S_avg * sf_y,
                 ox= G_std * sf_x,
                 oy= S_std * sf_y,
                 param=param,
                 protocol=protocol,
                 x_label='Offered traffic (G)',
                 y_label='Throughput (S)')

    


'''    plot_results(ux=results_avg[:, 0] * sf_x,
                 uy=results_avg[:, 1] * sf_y,
                 ox=results_std[:, 0] * sf_x,
                 oy=results_std[:, 1] * sf_y,
                 param=param,
                 protocol=protocol,
                 x_label='Number of Packets Sent',
                 y_label='Number of Packets Received')'''


if __name__ == '__main__':


    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------
    # Simulation parameters:

    lmbd  = 0.1     # lambda of each node : arrival rate
    L     = 704     # bits of each packet
    T     = 0.034   # duration in seconds of each packet
    t_sim = 60      # time duration to simulate in seconds

    nodes = np.arange(start=1, stop=35, step=2, dtype=int)      
    # end number of active nodes to simulate, starting from 1 (integer)
    # we increase the number of nodes in steps of 2 to increase the offered traffic

    n_iters = 10            # number of repetitions of each experiment
    np.random.seed(1337)    # seed for random processes

    # -----------------------------------------------------------------------
    # Run simulation:
    
    # pack all parameters 
    param = nodes, lmbd, L, T, t_sim, n_iters

    # run the simulation
    run(n_time_slots = int(t_sim/T),    # number of time slots for S-ALOHA
        protocol     = 'S-ALOHA',       # MAC protocol
        param        = param)       
