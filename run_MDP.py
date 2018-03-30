from MDP import MDP

def simple_var_mdp():
    mdp = MDP(mode="SimpleVarMDP")
    # mdp = MDP(random_state=1, X=25, U=2, mode="test_maze")
    # mdp = MDP(random_state=1, X=25, U=2, mode="1D")
    print("policy_before\n" + str(mdp.policy_ml))
    print("v_iterative\n" + str(mdp.iterative_PE()))
    print("v_algebraic\n" + str(mdp.algebraic_PE()))
    # print("v_series\n" + str(mdp.series_PE()))

    mdp.PI()

    print("policy_after\n" + str(mdp.policy_ml))
    print("v_iterative\n" + str(mdp.iterative_PE()))
    print("v_algebraic\n" + str(mdp.algebraic_PE()))
    # print("v_series\n" + str(mdp.series_PE(times = 10)))
    # print("v_series\n" + str(mdp.series_PE(times = 30)))

def simple_var_mdp_tradeoff():
    mdp = MDP(mode="SimpleVarMDP")

simple_var_mdp()